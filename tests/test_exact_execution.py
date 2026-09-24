"""
Validate exact_execution.run_reconstructed_exact against Aer qasm simulator
on reconstructed circuits small enough for Aer to finish quickly.

Run from the project root:
    PYTHONPATH=src python -u tests/test_exact_execution.py
"""
import sys
import time
import unittest
import contextlib
import io
from collections import Counter
from pathlib import Path
import numpy as np

sys.setrecursionlimit(1000000)
SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from fast_analysis import analyze_per_gate_fast, analyze_per_gate_nolock
from batch_partition import partition_nodes
from circuits_selection import (
    dfs_quantum_circuit,
    standard_final_read_nodes,
    transformation,
)
from utilities import (ReplayConstructionError, generate_gate, quantum_path,
                       filter_operation_list,
                       solve_qmon_for_circuit, generate_monitoring_circuit,
                       generate_new_operation_list, execute)
from exact_execution import (ExactExecutionViolation,
                             reconstructed_final_marginal,
                             reconstructed_reduced_density,
                             run_reconstructed_exact)
from rq3_manifest import circuit_qasm_sha256, stable_hash
import rq3_realexec_eval as rq3_real


DEFAULT_SELECTION = "nolock"
DEFAULT_QMAX = 24


def _filter_final_nodes(qc, res, paths):
    """Apply the standard terminal-measurement deduplication to a circuit.

    The shared legacy helper reloads a circuit from ``key``. Doing the small
    filter here is necessary for a genuinely circuit-matched mutant plan.
    """
    selected_identities = {
        (int(node_index), int(qubit))
        for qubit, node_indices in paths.items()
        for node_index in node_indices
    }
    final_reads = standard_final_read_nodes(qc, selected_identities)

    selected = []
    for qubit, node_indices in paths.items():
        for node_index in node_indices:
            if (node_index, qubit) in final_reads:
                continue
            for node in res[node_index]:
                if node[2] == qubit and node not in selected:
                    selected.append(node)
    return selected


def _build_partial_circuit(qc, key, selected_nodes):
    """Build replay operations from ``qc`` without reloading ``key``.

    This mirrors ``circuits_selection.bulid_partial_circuit``. The local form
    is essential when ``qc`` is a mutant: both selection and replay must come
    from that same circuit.
    """
    operation_list = []
    for node in selected_nodes:
        if not node[-1]:
            continue
        index, gate_name, qubit_info, p0, p1 = node[:5]
        replay = transformation(qc, qubit_info, node[-1], [])
        operation_list.append(
            [index, gate_name, qubit_info, p0, p1, replay])
    return {key: operation_list} if operation_list else {}


def _prepare_instrumentation(key, qc, selection, qmax):
    """Return ``(qc, operation_list, constraint)`` for one exact circuit."""
    if qc is None:
        qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    analyzers = {
        "nolock": analyze_per_gate_nolock,
        "lock": analyze_per_gate_fast,
    }
    if selection not in analyzers:
        raise ValueError(
            f"unknown selection {selection!r}; expected one of "
            f"{sorted(analyzers)}")
    if not isinstance(qmax, int) or isinstance(qmax, bool) or qmax <= 0:
        raise ValueError(f"qmax must be a positive integer, got {qmax!r}")
    if qc.num_qubits > qmax:
        raise ValueError(
            f"circuit has {qc.num_qubits} original qubits, exceeding "
            f"qmax={qmax}")
    
    res, _, paths = analyzers[selection](qc)
    selected_nodes = []
    for node in _filter_final_nodes(qc, res, paths):
        index, gname, qubit_info, p0, p1 = node
        path = dfs_quantum_circuit(qc, qubit_info, index, set(), [])
        selected_nodes.append([index, gname, qubit_info, p0, p1, path])
    operation_list = _build_partial_circuit(qc, key, selected_nodes)
    constraint = {'num_oq': qc.num_qubits}
    if key not in operation_list:
        return qc, operation_list, constraint

    _, eq_container = filter_operation_list(key, operation_list, 10000000)
    for node, eq in eq_container.items():
        constraint[node] = {'num_eq': len(eq), 'qubits': []}
        for gate in operation_list[key]:
            if gate[0] == node:
                constraint[node]['qubits'].append(gate[2])
    return qc, operation_list, constraint


def build_instrumented_circuit(key, qc=None, *, selection=DEFAULT_SELECTION,
                              qmax=DEFAULT_QMAX):
    """Build one circuit-matched QMon instrumented subset.

    By default, separability is reconsidered after every gate (``nolock``), so a
    qubit that later disentangles can become monitorable again, with the RQ3
    deployment budget ``Qmax=24``. Passing ``qc`` reconstructs that exact circuit
    (for example, a mutant) while retaining ``key`` as its label.

    This compatibility helper uses ``solve_qmon_for_circuit`` to select ONE
    within-budget subset. It is not the full-coverage batched protocol. Use
    :func:`build_instrumented_batches` when every feasible node must be
    partitioned and executed as in the main RQ3 experiment.

    ``selection="lock"`` explicitly requests the conservative legacy selector
    that permanently retires a qubit after its first entangling gate. It is kept
    only for controlled comparisons and is not the experiment default.
    """
    qc, operation_list, constraint = _prepare_instrumentation(
        key, qc, selection, qmax)
    if key not in operation_list or len(constraint) == 1:
        return qc, qc.copy()
    ans = solve_qmon_for_circuit(key, constraint, Qmax=qmax, two_stage=True)
    return qc, generate_monitoring_circuit(qc, key, operation_list, 10000000, ans)


def build_instrumented_batches(key, qc=None, *,
                              selection=DEFAULT_SELECTION,
                              qmax=DEFAULT_QMAX):
    """Build the main-RQ3 full-coverage batch partition for one exact circuit.

    Feasible gate nodes are partitioned with ``partition_nodes`` and every batch
    is generated as a separate monitored execution. The returned metadata records
    the complete partition, including nodes that are individually infeasible at
    ``qmax``; it never relabels a single selected subset as a batched plan.

    Returns ``(qc, executions, metadata)``. Each execution is a dictionary with
    ``circuit``, ``nodes`` (gate indices), ``cost`` (extra qubits), and
    ``monitor_events`` (gate-qubit reads generated in that batch).
    """
    qc, operation_list, constraint = _prepare_instrumentation(
        key, qc, selection, qmax)
    batches, infeasible = partition_nodes(constraint, qmax)
    executions = []
    gates = operation_list.get(key, [])
    for batch in batches:
        node_set = set(batch['nodes'])
        ans = {'picked_nodes': node_set}
        monitored = generate_monitoring_circuit(
            qc, key, operation_list, 10000000, ans)
        if monitored.num_qubits > qmax:
            raise ValueError(
                f"generated batch exceeds qmax: {monitored.num_qubits}>{qmax}"
            )
        executions.append({
            'circuit': monitored,
            'nodes': tuple(batch['nodes']),
            'cost': int(batch['cost']),
            'emitted_qubits': int(monitored.num_qubits),
            'monitor_events': sum(
                1 for gate in gates if gate[0] in node_set),
            'monitor_groups': tuple(
                monitored.metadata.get('qmon_monitor_groups', ())),
            'replay_fail_closed': bool(
                monitored.metadata.get('qmon_replay_fail_closed', False)),
        })
    metadata = {
        'selection': selection,
        'qmax': qmax,
        'partitioner': 'partition_nodes',
        'original_qubits': qc.num_qubits,
        'extra_qubit_capacity': qmax - qc.num_qubits,
        'candidate_gate_nodes': len(constraint) - 1,
        'candidate_checkpoint_nodes': len(gates),
        'feasible_gate_nodes': sum(len(batch['nodes']) for batch in batches),
        'infeasible_gate_nodes': tuple(infeasible),
        'individually_infeasible_checkpoint_nodes': tuple(
            (int(gate[0]), int(gate[2])) for gate in gates
            if gate[0] in set(infeasible)),
        'coverage_scope': (
            'every individually feasible instrumentable gate node; '
            'infeasible pre-budget candidates are recorded, not claimed covered'),
        'batch_count': len(batches),
        'batches': tuple({
            'nodes': tuple(execution['nodes']),
            'cost': execution['cost'],
            'emitted_qubits': execution['emitted_qubits'],
            'monitor_events': execution['monitor_events'],
            'monitor_groups': execution['monitor_groups'],
            'replay_fail_closed': execution['replay_fail_closed'],
        } for execution in executions),
    }
    plan_value = {
        'circuit': key,
        'source_qasm_sha256': circuit_qasm_sha256(qc),
        'selection': selection,
        'qmax': qmax,
        'original_qubits': qc.num_qubits,
        'batches': metadata['batches'],
        'infeasible_gate_nodes': metadata['infeasible_gate_nodes'],
    }
    metadata['source_qasm_sha256'] = plan_value['source_qasm_sha256']
    metadata['plan_sha256'] = stable_hash(plan_value)
    return qc, executions, metadata


def tvd_counts(c1, c2, shots):
    keys = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in keys) / shots


def check_reconverging_path_deduplicates():
    """A shared ancestor reached from two DFS branches must replay once."""
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.cx(0, 1)  # shared ancestor reached through both downstream branches
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(3, 2)
    nested = dfs_quantum_circuit(qc, 2, 4, set(), [])
    transformed = transformation(qc, 2, nested, [])
    flat = generate_new_operation_list(transformed, 4, {})[4]
    indices = sorted(gate[0] for gate in flat)
    assert indices == [0, 1, 2, 3, 4], (
        f"reconverging cone replayed wrong gate set: {indices}")
    print("Reconverging-path deduplication passed.", flush=True)


def check_batched_instrumentation_partitions():
    """A circuit-matched plan must emit every partition as a separate run."""
    qc = QuantumCircuit(6, 4)
    for offset in (0, 3):
        qc.h(offset)
        qc.h(offset + 1)
        qc.x(offset + 2)
        qc.h(offset + 2)
        qc.cx(offset, offset + 2)
        qc.h(offset)
        qc.cx(offset + 1, offset + 2)
        qc.h(offset + 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(3, 2)
    qc.measure(4, 3)

    # Two groups each need one extra qubit; a capacity of one forces two batches.
    for qmax, expected_batches in ((7, 2), (8, 1)):
        _, executions, metadata = build_instrumented_batches(
            "synthetic_two_dj", qc=qc, qmax=qmax)
        assert metadata["selection"] == "nolock"
        assert metadata["batch_count"] == len(executions) == expected_batches
        nodes = [node for execution in executions for node in execution["nodes"]]
        assert len(nodes) == len(set(nodes)) == 12
        assert metadata["candidate_gate_nodes"] == metadata["feasible_gate_nodes"] == 12
        assert metadata["infeasible_gate_nodes"] == ()
        assert all(execution["circuit"].num_qubits <= qmax
                   for execution in executions)
    print("Batched instrumentation partitioning passed.", flush=True)


def _joint_counter(columns, order):
    return Counter(zip(*(columns[node].tolist() for node in order)))


def _aer_memory_counter(circuit, clbits, shots, seed):
    memory = Aer.get_backend("aer_simulator").run(
        circuit, shots=shots, memory=True, seed_simulator=seed
    ).result().get_memory()
    output = Counter()
    for word in memory:
        little_endian = word.replace(" ", "")[::-1]
        output[tuple(int(little_endian[index]) for index in clbits)] += 1
    return output


def _counter_tvd(first, second, shots):
    support = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(value, 0) - second.get(value, 0))
        for value in support
    ) / shots


class JointTrajectoryTests(unittest.TestCase):
    SHOTS = 4096

    def assert_matches_aer(self, circuit, mid_nodes, final_nodes, clbits,
                           seed=2027, tolerance=0.06):
        sampled = rq3_real.sample_joint_trajectory(
            circuit,
            mid_nodes=mid_nodes,
            final_nodes=final_nodes,
            shots=self.SHOTS,
            seed=seed,
        )
        self.assertTrue(sampled["complete"], sampled["diagnostics"])
        order = list(mid_nodes) + list(final_nodes)
        actual = _joint_counter(sampled["shot_table"], order)
        expected = _aer_memory_counter(
            circuit, clbits, self.SHOTS, seed + 1
        )
        self.assertLess(
            _counter_tvd(actual, expected, self.SHOTS), tolerance
        )

    def test_fully_correlated_checkpoints_share_shot_rows(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.reset(0)
        circuit.measure(1, 1)
        circuit.reset(1)
        nodes = [(1, 0), (3, 1)]
        sampled = rq3_real.sample_joint_trajectory(
            circuit, mid_nodes=nodes, shots=self.SHOTS, seed=7
        )
        self.assertTrue(sampled["complete"])
        np.testing.assert_array_equal(
            sampled["shot_table"][nodes[0]],
            sampled["shot_table"][nodes[1]],
        )
        self.assert_matches_aer(circuit, nodes, (), (0, 1))

    def test_dynamic_differential_suite(self):
        # Correct/multiple-monitor/healing: reset plus replay makes both reads
        # valid while preserving their physical shot-row identity.
        healing = QuantumCircuit(1, 2)
        healing.h(0)
        healing.measure(0, 0)
        healing.reset(0)
        healing.h(0)
        healing.measure(0, 1)
        healing.reset(0)

        mutant = QuantumCircuit(1, 2)
        mutant.x(0)
        mutant.measure(0, 0)
        mutant.reset(0)
        mutant.x(0)
        mutant.measure(0, 1)
        mutant.reset(0)

        entangled = QuantumCircuit(2, 2)
        entangled.h(0)
        entangled.cx(0, 1)
        entangled.measure(0, 0)
        entangled.reset(0)
        entangled.measure(1, 1)
        entangled.reset(1)

        final_read = QuantumCircuit(2, 3)
        final_read.h(0)
        final_read.cx(0, 1)
        final_read.measure(0, 0)
        final_read.reset(0)
        final_read.measure(0, 1)
        final_read.measure(1, 2)

        scenarios = (
            ("correct-healing-multiple", healing,
             ((0, 0), (3, 0)), (), (0, 1)),
            ("mutant", mutant, ((0, 0), (3, 0)), (), (0, 1)),
            ("entangled", entangled, ((1, 0), (3, 1)), (), (0, 1)),
            ("multiple-final-read", final_read, ((1, 0),),
             ((3, 0), (3, 1)), (0, 1, 2)),
        )
        for name, circuit, mids, finals, clbits in scenarios:
            with self.subTest(name=name):
                self.assert_matches_aer(circuit, mids, finals, clbits)

    def test_batches_are_independent_and_cap_is_incomplete(self):
        batch = QuantumCircuit(1, 1)
        batch.h(0)
        batch.measure(0, 0)
        batch.reset(0)
        node = (0, 0)
        first = rq3_real.sample_joint_trajectory(
            batch, mid_nodes=[node], shots=self.SHOTS, seed=31
        )
        second = rq3_real.sample_joint_trajectory(
            batch, mid_nodes=[node], shots=self.SHOTS, seed=32
        )
        correlation = np.corrcoef(
            first["shot_table"][node], second["shot_table"][node]
        )[0, 1]
        self.assertLess(abs(float(correlation)), 0.08)
        self.assertNotEqual(
            first["shot_table_sha256"], second["shot_table_sha256"]
        )

        entangled = QuantumCircuit(2, 2)
        entangled.h(0)
        entangled.cx(0, 1)
        entangled.measure(0, 0)
        entangled.reset(0)
        entangled.measure(1, 1)
        entangled.reset(1)
        capped = rq3_real.sample_joint_trajectory(
            entangled,
            mid_nodes=[(1, 0), (3, 1)],
            shots=256,
            seed=3,
            cap=1,
        )
        self.assertFalse(capped["complete"])
        self.assertEqual(capped["diagnostics"]["cap_hits"], 1)


class ExactChannelAndEmitterTests(unittest.TestCase):
    @staticmethod
    def forced_group(circuit, gate, qubits, key="forced-group"):
        operation_list = {key: []}
        paths = {}
        for qubit in qubits:
            path = dfs_quantum_circuit(circuit, qubit, gate, set(), [])
            paths[qubit] = path
            operation_list[key].append(
                [gate, circuit.data[gate].operation.name, qubit,
                 0.0, 0.0, path])
        with contextlib.redirect_stdout(io.StringIO()):
            emitted = generate_monitoring_circuit(
                circuit, key, operation_list, 10 ** 7,
                {"picked_nodes": {gate}},
            )
        return emitted, paths

    @staticmethod
    def full_fidelity(original, emitted):
        expected, _, _ = reconstructed_reduced_density(
            original, range(original.num_qubits))
        actual, _, violations = reconstructed_reduced_density(
            emitted, range(original.num_qubits))
        return float(state_fidelity(expected, actual)), violations

    def test_bell_measure_reset_uses_exact_branch_mixture_and_fails_certificate(self):
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.reset(0)

        probabilities, mids, violations = reconstructed_final_marginal(
            circuit, [1], require_separable=False)
        np.testing.assert_allclose(probabilities, [0.5, 0.5], atol=1e-12)
        self.assertEqual(len(mids), 1)
        self.assertEqual(len(violations), 1)
        self.assertAlmostEqual(violations[0]["residual"], 0.25, places=12)

        # The removed branch-selection approximation picked |0> at equality,
        # yielding [1, 0] and TVD 0.5 instead of the exact channel's TVD 0.
        self.assertAlmostEqual(
            0.5 * np.abs(probabilities - np.array([1.0, 0.0])).sum(),
            0.5,
        )
        with self.assertRaises(ExactExecutionViolation):
            run_reconstructed_exact(circuit, shots=32, seed=7)

    def test_same_index_joint_monitor_group_preserves_product_basis(self):
        preparations = ("id", "x", "h", "sx")
        for left in preparations:
            for right in preparations:
                with self.subTest(left=left, right=right):
                    circuit = QuantumCircuit(2)
                    getattr(circuit, left)(0)
                    getattr(circuit, right)(1)
                    circuit.swap(0, 1)
                    gate = 2
                    emitted, _ = self.forced_group(
                        circuit, gate, (0, 1), key="joint-product")
                    expected, _, _ = reconstructed_reduced_density(
                        circuit, [0, 1])
                    actual, _, violations = reconstructed_reduced_density(
                        emitted, [0, 1])
                    np.testing.assert_allclose(actual, expected, atol=1e-12)
                    self.assertEqual(violations, [])
                    groups = emitted.metadata["qmon_monitor_groups"]
                    self.assertEqual(len(groups), 1)
                    self.assertEqual(groups[0]["qubits"], (0, 1))
                    self.assertEqual(
                        groups[0]["replay"],
                        "one_shared_causal_cone_replay",
                    )

    def test_reconvergent_all_plus_group_has_identical_deduplicated_cones(self):
        circuit = QuantumCircuit(4)
        for qubit in range(4):
            circuit.h(qubit)
        circuit.cx(0, 2)
        circuit.cx(1, 2)
        circuit.cx(0, 3)
        circuit.cx(1, 3)
        circuit.cx(2, 3)
        gate = len(circuit.data) - 1

        emitted, paths = self.forced_group(
            circuit, gate, (2, 3), key="reconvergent-plus")
        flattened = []
        for qubit in (2, 3):
            replay = generate_new_operation_list(paths[qubit], gate, {})[gate]
            flattened.append(tuple(sorted(item[0] for item in replay)))
        self.assertEqual(flattened[0], flattened[1])

        group = emitted.metadata["qmon_monitor_groups"][0]
        replay_indices = group["replay_source_gate_indices"]
        self.assertEqual(len(replay_indices), len(set(replay_indices)))
        self.assertEqual(replay_indices, flattened[0])
        self.assertEqual(group["qubits"], (2, 3))
        fidelity, violations = self.full_fidelity(circuit, emitted)
        self.assertAlmostEqual(fidelity, 1.0, places=12)
        self.assertEqual(violations, [])

    def test_later_groups_use_disjoint_fresh_ancilla_slices(self):
        circuit = QuantumCircuit(4)
        circuit.x(1)
        circuit.cx(1, 0)
        circuit.x(3)
        circuit.cx(3, 2)
        operation_list = {"two-groups": []}
        for gate, qubit in ((1, 0), (3, 2)):
            operation_list["two-groups"].append([
                gate, "cx", qubit, 0.0, 0.0,
                dfs_quantum_circuit(circuit, qubit, gate, set(), []),
            ])
        with contextlib.redirect_stdout(io.StringIO()):
            emitted = generate_monitoring_circuit(
                circuit, "two-groups", operation_list, 10 ** 7,
                {"picked_nodes": {1, 3}},
            )
        groups = emitted.metadata["qmon_monitor_groups"]
        self.assertEqual([group["gate_index"] for group in groups], [1, 3])
        self.assertEqual(groups[0]["ancilla_slice"], (0,))
        self.assertEqual(groups[1]["ancilla_slice"], (1,))
        self.assertTrue(
            set(groups[0]["ancilla_qubits"]).isdisjoint(
                groups[1]["ancilla_qubits"]))
        prior_ancillas = set(groups[0]["ancilla_qubits"])
        for data_index in groups[1]["replay_data_indices"]:
            touched = {
                emitted.find_bit(bit).index
                for bit in emitted.data[data_index].qubits
            }
            self.assertTrue(prior_ancillas.isdisjoint(touched))
        fidelity, violations = self.full_fidelity(circuit, emitted)
        self.assertAlmostEqual(fidelity, 1.0, places=12)
        self.assertEqual(violations, [])

    def test_forced_bell_and_ghz_blocks_match_tr_rho_cubed(self):
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        emitted_bell, _ = self.forced_group(
            bell, 1, (0, 1), key="bell-block")
        bell_fidelity, bell_violations = self.full_fidelity(
            bell, emitted_bell)
        self.assertAlmostEqual(bell_fidelity, 1.0, places=12)
        self.assertTrue(bell_violations)
        bell_state = Statevector.from_instruction(bell)
        for qubit in (0, 1):
            marginal = partial_trace(
                bell_state, [other for other in range(2) if other != qubit]
            ).data
            self.assertAlmostEqual(float(np.linalg.det(marginal).real),
                                   0.25, places=12)

        ghz = QuantumCircuit(3)
        ghz.h(0)
        ghz.cx(0, 2)
        ghz.cx(0, 1)
        emitted_ghz, _ = self.forced_group(
            ghz, 2, (0, 1), key="ghz-partial-block")
        ghz_fidelity, ghz_violations = self.full_fidelity(
            ghz, emitted_ghz)
        self.assertTrue(ghz_violations)
        ghz_state = Statevector.from_instruction(ghz)
        rho_m = partial_trace(ghz_state, [2]).data
        eigenvalues = np.linalg.eigvalsh(rho_m)
        tr_rho_cubed = float(np.sum(eigenvalues ** 3).real)
        self.assertAlmostEqual(tr_rho_cubed, 0.25, places=12)
        self.assertAlmostEqual(ghz_fidelity, tr_rho_cubed, delta=1e-8)
        for qubit in (0, 1):
            marginal = partial_trace(
                ghz_state, [other for other in range(3) if other != qubit]
            ).data
            self.assertAlmostEqual(float(np.linalg.det(marginal).real),
                                   0.25, places=12)
        _, _, paths = analyze_per_gate_nolock(ghz)
        selected = {(index, qubit) for qubit, indices in paths.items()
                    for index in indices}
        self.assertNotIn((2, 0), selected)
        self.assertNotIn((2, 1), selected)

    def test_unsupported_replay_gate_fails_closed(self):
        circuit = QuantumCircuit()
        circuit.add_register(QuantumRegister(1, "q"))
        circuit.add_register(QuantumRegister(1, "eq"))
        replay = [[0, "not_a_qiskit_gate", [], [0]]]
        with self.assertRaisesRegex(
                ReplayConstructionError, "unsupported replay gate"):
            generate_gate(replay, circuit, [0], {}, [0], 0, {})


def main():
    check_reconverging_path_deduplicates()
    check_batched_instrumentation_partitions()
    shots = 32768   # mid-measure bits are now retained per-event, growing the
    # joint support; more shots keep two-sample TVD noise well under threshold
    n_validated = 0
    for key in ['ghz_indep_qiskit_3.qasm', 'ghz_indep_qiskit_4.qasm',
                'ghz_indep_qiskit_5.qasm', 'dj_indep_qiskit_3.qasm',
                'dj_indep_qiskit_4.qasm', 'wstate_indep_qiskit_3.qasm',
                'wstate_indep_qiskit_4.qasm', 'graphstate_indep_qiskit_3.qasm',
                'qft_indep_qiskit_3.qasm', 'twolocalrandom_indep_qiskit_3.qasm',
                'grover-v-chain_indep_qiskit_3.qasm',
                'qaoa_indep_qiskit_6.qasm']:
        qc, new_qc = build_instrumented_circuit(key)
        print(f'{key}: reconstructed {len(new_qc.data)} gates, '
              f'{new_qc.num_qubits} qubits', flush=True)
        if new_qc.num_qubits > 14:
            print('  SKIP Aer reference (instrumented circuit too large for '
                  'trajectory simulation in reasonable time)', flush=True)
            continue

        t0 = time.perf_counter()
        aer_counts = execute(new_qc, backend=Aer.get_backend('qasm_simulator'),
                             shots=shots, seed_transpiler=2025,
                             seed_simulator=2025).result().get_counts()
        t_aer = time.perf_counter() - t0

        t0 = time.perf_counter()
        exact_counts, mids, violations = run_reconstructed_exact(
            new_qc, shots=shots, seed=2025)
        t_exact = time.perf_counter() - t0

        tvd = tvd_counts(aer_counts, exact_counts, shots)
        # expected TVD between two independent `shots`-samples grows with
        # the support size K: E[TVD] <= 0.5*sqrt(K/shots) (Cauchy-Schwarz)
        support = len(set(aer_counts) | set(exact_counts))
        thresh = max(0.05, 0.6 * (support / shots) ** 0.5)
        print(f'  aer={t_aer:.1f}s exact={t_exact:.1f}s '
              f'speedup={t_aer / t_exact:.0f}x', flush=True)
        print(f'  TVD(aer, exact)={tvd:.4f} (threshold {thresh:.3f}, '
              f'support {support}), mid measurements={len(mids)}, '
              f'separability violations={len(violations)}', flush=True)
        assert tvd < thresh, f'TVD too high for {key}'
        assert not violations, f'unexpected separability violation in {key}'
        n_validated += 1
    print(f'\nAll {n_validated} exact-execution validations passed.')


if __name__ == "__main__":
    main()
