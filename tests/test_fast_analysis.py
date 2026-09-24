"""
Equivalence and timing test: analyze_per_gate (original, O(m^2 * 2^n))
vs analyze_per_gate_fast (incremental evolution, O(m * 2^n)).

Run from the project root:
    PYTHONPATH=src python tests/test_fast_analysis.py
"""
import time
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

import circuits_selection
from circuits_selection import (
    _single_vs_rest_entangled,
    analyze_per_gate,
    bulid_partial_circuit,
    dfs_quantum_circuit,
    filter_circuits_with_output_num,
    standard_final_read_nodes,
)
from fast_analysis import analyze_per_gate_fast, analyze_per_gate_nolock
from utilities import quantum_path

ATOL = 1e-8
# Concurrence involves matrix square roots, which amplify ~1e-16 statevector
# noise to ~1e-8; it is log-only and does not drive lock/unlock decisions.
ATOL_CONCURRENCE = 1e-6

TEST_CIRCUITS = [
    "ghz_indep_qiskit_5.qasm",
    "dj_indep_qiskit_5.qasm",
    "qft_indep_qiskit_6.qasm",
    "qftentangled_indep_qiskit_6.qasm",
    "wstate_indep_qiskit_6.qasm",
    "qaoa_indep_qiskit_6.qasm",
    "graphstate_indep_qiskit_5.qasm",
    "su2random_indep_qiskit_4.qasm",
    "qpeexact_indep_qiskit_5.qasm",
    "twolocalrandom_indep_qiskit_3.qasm",
    "grover-v-chain_indep_qiskit_3.qasm",
    "qwalk-v-chain_indep_qiskit_3.qasm",
]


class SelectorBoundaryTests(unittest.TestCase):
    def test_declared_tolerance_is_not_overridden_by_library_cutoff(self):
        a2 = 5e-9
        state = Statevector([
            np.sqrt(1.0 - a2 * a2), 0.0, 0.0, a2,
        ])

        entangled, observed, amplitudes = _single_vs_rest_entangled(
            state, 0, tol=1e-12
        )

        self.assertTrue(entangled)
        self.assertAlmostEqual(observed, a2, delta=1e-15)
        self.assertEqual(len(amplitudes), 2)
        self.assertFalse(
            _single_vs_rest_entangled(state, 0, tol=1e-8)[0]
        )

    def test_product_state_has_zero_second_amplitude(self):
        entangled, observed, _ = _single_vs_rest_entangled(
            Statevector.from_label("00"), 1, tol=1e-12
        )
        self.assertFalse(entangled)
        self.assertEqual(observed, 0.0)


class MpsSnapshotSelectionTests(unittest.TestCase):
    def test_saved_density_matrices_preserve_selected_locations(self):
        from mps_analysis import monitorable_nodes_mps

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        original = circuit.copy()

        self.assertEqual(
            monitorable_nodes_mps(circuit), {0: [0], 2: [0, 1]}
        )
        self.assertEqual(circuit, original)


class DeploymentPointClassificationTests(unittest.TestCase):
    def test_only_a_terminally_measured_qubit_gets_a_free_final_read(self):
        circuit = QuantumCircuit(2, 1)
        circuit.x(0)
        circuit.h(1)
        circuit.measure(1, 0)
        selected = {(0, 0), (1, 1)}

        self.assertEqual(
            standard_final_read_nodes(circuit, selected),
            {(1, 1)},
        )

        with tempfile.TemporaryDirectory() as temporary:
            key = "partial-output.qasm"
            qasm2.dump(circuit, Path(temporary) / key)
            res, _, paths = analyze_per_gate_nolock(circuit)
            with mock.patch.object(
                circuits_selection, "quantum_path", temporary
            ), contextlib.redirect_stdout(io.StringIO()):
                filtered = filter_circuits_with_output_num(
                    {key: res}, {key: paths}
                )

        identities = {(row[0], row[2]) for row in filtered[key]}
        self.assertIn((0, 0), identities)
        self.assertNotIn((1, 1), identities)

    def test_legacy_vqe_names_do_not_suppress_valid_points(self):
        fixtures = (
            ("vqe_indep_qiskit_8.qasm", 13, 1, 2),
            ("vqe_indep_qiskit_9.qasm", 12, 4, 5),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for key, gate_index, qubit, width in fixtures:
                circuit = QuantumCircuit(width)
                for index in range(gate_index + 2):
                    circuit.h(qubit if index == gate_index else 0)
                qasm2.dump(circuit, Path(temporary) / key)
                path = dfs_quantum_circuit(
                    circuit, qubit, gate_index, set(), []
                )
                selected = {
                    key: [[gate_index, "h", qubit, 0.5, 0.5, path]]
                }
                with mock.patch.object(
                    circuits_selection, "quantum_path", temporary
                ), contextlib.redirect_stdout(io.StringIO()):
                    operation_list = bulid_partial_circuit(selected)
                identities = {
                    (row[0], row[2]) for row in operation_list[key]
                }
                self.assertIn((gate_index, qubit), identities)


def compare_res(res_a, res_b):
    assert res_a.keys() == res_b.keys(), \
        f"res keys differ: {sorted(set(res_a) ^ set(res_b))}"
    for k in res_a:
        assert len(res_a[k]) == len(res_b[k]), f"res[{k}] length differs"
        for ent_a, ent_b in zip(res_a[k], res_b[k]):
            assert ent_a[0] == ent_b[0] and ent_a[1] == ent_b[1] and ent_a[2] == ent_b[2], \
                f"res[{k}] meta differs: {ent_a[:3]} vs {ent_b[:3]}"
            assert abs(ent_a[3] - ent_b[3]) < ATOL and abs(ent_a[4] - ent_b[4]) < ATOL, \
                f"res[{k}] probs differ: {ent_a[3:]} vs {ent_b[3:]}"


def compare_events(ev_a, ev_b):
    assert len(ev_a) == len(ev_b), f"events length differs: {len(ev_a)} vs {len(ev_b)}"
    for e_a, e_b in zip(ev_a, ev_b):
        for key in ('index', 'gate', 'a', 'b'):
            assert e_a[key] == e_b[key], f"event {key} differs: {e_a[key]} vs {e_b[key]}"
        assert abs(e_a['pairwise_C'] - e_b['pairwise_C']) < ATOL_CONCURRENCE
        assert abs(e_a['vs_rest_s2'] - e_b['vs_rest_s2']) < ATOL
        for side in ('a', 'b'):
            assert e_a['single'][side]['ent_vs_rest'] == e_b['single'][side]['ent_vs_rest'], \
                f"event {e_a['index']} side {side}: entanglement decision differs"


def main():
    speedups = []
    for name in TEST_CIRCUITS:
        path = Path(quantum_path) / name
        if not path.exists():
            print(f"SKIP (missing): {name}")
            continue

        qc = QuantumCircuit.from_qasm_file(str(path))

        t0 = time.perf_counter()
        res_o, ev_o, paths_o = analyze_per_gate(qc)
        t1 = time.perf_counter()
        res_f, ev_f, paths_f = analyze_per_gate_fast(qc)
        t2 = time.perf_counter()

        compare_res(res_o, res_f)
        compare_events(ev_o, ev_f)
        assert paths_o == paths_f, f"paths differ for {name}"

        t_orig, t_fast = t1 - t0, t2 - t1
        speedups.append(t_orig / t_fast)
        print(f"OK  {name}: qubits={qc.num_qubits} gates={len(qc.data)} "
              f"orig={t_orig:.2f}s fast={t_fast:.2f}s speedup={t_orig / t_fast:.1f}x")

    print(f"\nAll {len(speedups)} circuits identical "
          f"(res/events/paths). Median speedup: {sorted(speedups)[len(speedups) // 2]:.1f}x")


if __name__ == "__main__":
    main()
