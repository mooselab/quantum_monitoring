"""
RQ3 operational evaluation under each method's declared execution semantics
and decision procedure, at a common per-run shot budget.

This replaces the exact-limit "decision quantity" scoring of rq3_eval.py.
Affected baseline checkpoints execute their published mechanisms; checkpoints
provably outside the fault's backward cone use an explicit analytical null
shortcut.  The artifact records both counts and never describes shortcut
checkpoints as physically executed.

  QMon          The instrumented monitoring circuit (generate_monitoring_
                circuit, batched no-lock plan under Qmax=24) is executed on
                the MUTANT: per batch, a numerically converged branch-mixture
                simulation of
                the generated monitoring circuit (branching when the computed
                determinant residual exceeds the declared branch-convergence
                tolerance, and otherwise merging the reset branches) yields
                each monitor's mid-measurement p1; aligned per-shot reads are
                then drawn at S shots and tested: a deterministic expected
                value flags on ANY flipped read; otherwise an exact
                two-sided binomial test at alpha/K (Bonferroni over the K
                checkpoints) flags the node. Batches whose nodes all remain
                separable under the mutation provably read the raw marginal
                (Theorem 1), so only entanglement-broken batches need the
                monitoring-circuit simulation.
  QMon(node)    A finite-shot counterfactual: at each deployed checkpoint it
                samples the exact mutant Z marginal independently. It does
                not generate or execute K_qmon separately instrumented
                circuits.
  statistical   Huang & Martonosi (ISCA'19): the mutant prefix up to the
                checkpoint is REALLY RUN on Aer at S shots with a Z-basis
                measurement of the monitored qubit; their chi-square
                goodness-of-fit test against the expected distribution
                decides (alpha/K). A deterministic expected value is their
                classical assertion: any wrong outcome flags. Chi-square is
                replaced by the exact binomial test when an expected cell
                count is < 5 (the test's own validity condition).
  proq          Li et al. (OOPSLA'20) through the OFFICIAL pytket runtime
                (ProjectorAssertionBox, the authors' method as adopted into
                t|ket>): the rank-1 projector onto the expected (pure,
                separable-by-construction) node state is asserted on the
                mutant prefix and the compiled circuit is run at S shots;
                ANY assertion failure among the shots flags (their noise-free
                best case: tr(P rho)=1 exactly on a correct run).
  liu           Liu, Byrd & Zhou (ASPLOS'20), the authors' assertion circuits
                (faithful port in baseline_liu.py): classical assertion for a
                deterministic expected state, the arbitrary-superposition
                H/controlled-U/H ancilla sandwich (flag=1) otherwise; the
                instrumented prefix is run on Aer at S shots and ANY nonzero
                ancilla reading flags. (The sandwich's reflection U has the
                expected state as its +1 eigenvector, so p(ancilla=1) =
                1 - <psi|rho|psi> exactly: same signal as proq, by
                construction, but paid for with their own circuit.)
  mqt           Rovara, Burgholzer & Wille through the OFFICIAL mqt.debugger
                DD backend: `assert-eq <sim>, q[k] {a0, a1}` is appended to
                the mutant prefix and the tool decides (simulation-only
                method: exact, no shots). A checkpoint is usable for MQT only
                if the tool can evaluate (and pass) the assertion on the
                ORIGINAL circuit; on the mutant, both a failed assertion and
                the tool's "entangled sub-state" refusal count as detections.

Shared frame (identical to rq3_eval.py so both tables are comparable):
same RQ3 circuit set, same seeds/mutants (the mutation stream is replicated
verbatim), same checkpoint set (all no-lock monitorable nodes), oracle =
original circuit, detection scanned in ascending gate order with early exit
(the earliest flagged node also defines the localization cone), same
equivalent-mutant filter and in-scope notion. A separate false-alarm
experiment runs every method on the CORRECT circuits.
"""
import warnings
import io
import math
import sys
import time
import signal
import pickle
import contextlib
import hashlib
import os
import tempfile
from pathlib import Path
import numpy as np

sys.setrecursionlimit(1_000_000)

from qiskit import QuantumCircuit
from scipy.stats import chisquare, binomtest

from fast_analysis import analyze_per_gate_nolock, SKIP_OPS
from mps_analysis import causal_cone
from mutation import mutate_circuit
from utilities import (quantum_path, quantum_gates, filter_operation_list,
                       generate_monitoring_circuit)
from circuits_selection import (filter_circuits_with_output_num,
                                dfs_quantum_circuit, bulid_partial_circuit,
                                standard_final_read_nodes)
from batch_partition import partition_nodes
from rq3_eval import (monitored_nodes, marginals, final_probs,
                      unitary_gate_indices)
from rq3_manifest import (apply_manifest_slot, circuit_qasm_sha256,
                          load_manifest, slots_for_circuit, stable_hash,
                          runtime_fingerprints)

METHODS = ["qmon", "qmon_pernode", "statistical", "proq", "liu", "mqt"]
METHOD_DISPLAY_NAMES = {
    "qmon": "Batched QMon",
    "qmon_pernode": "Analytical per-checkpoint QMon",
    "statistical": "Statistical assertion",
    "proq": "Projection assertion",
    "liu": "Dynamic ancilla",
    "mqt": "MQT combined flag",
}
METHOD_ASSERTED_SUBSYSTEM = {
    "qmon": "one-qubit node Z read",
    "qmon_pernode": "one-qubit node Z read",
    "statistical": "one-qubit node Z marginal from full-prefix execution",
    "proq": "one-qubit node rank-1 projector",
    "liu": "one-qubit node assertion ancilla",
    "mqt": "one-qubit node assert-eq q[k]",
}
PRUNING_SCOPE_RULE = (
    "execute a checkpoint only when a fault gate belongs to that node "
    "qubit's original backward causal cone"
)
EXACT_BINOMIAL_CONVENTION = {
    "implementation": "scipy.stats.binomtest",
    "alternative": "two-sided",
    "decision": "flag iff pvalue < alpha_node",
    "equality": "pvalue == alpha_node does not flag",
    "deterministic_expected_state": "flag iff at least one wrong outcome",
}
BONFERRONI_FAMILY_RULE = {
    "qmon": "alpha/K_qmon over deployed QMon checkpoints",
    "qmon_pernode": "alpha/K_qmon over deployed QMon checkpoints",
    "statistical": "alpha/K over all monitorable baseline checkpoints",
    "proq": "no Bonferroni p-value; any projector failure flags",
    "liu": "no Bonferroni p-value; any assertion-ancilla failure flags",
    "mqt": "exact simulation assertion; no finite-shot p-value",
}
EXECUTION_SHORTCUT_POLICY = {
    "qmon": (
        "every emitted Qmax=24 batch is evaluated with aligned joint shots; "
        "no null shortcut"
    ),
    "qmon_pernode": (
        "analytical one-node marginal with finite-shot draw; counterfactual "
        "independent-run arm, not an emitted-circuit execution"
    ),
    "statistical": (
        "affected checkpoints execute Aer prefixes; unaffected checkpoints "
        "draw the exact analytical null distribution"
    ),
    "proq": (
        "affected checkpoints execute pytket ProjectorAssertionBox; unaffected "
        "checkpoints use the exact zero-failure null"
    ),
    "liu": (
        "affected checkpoints execute the assertion circuit; unaffected "
        "checkpoints use the exact zero-failure null"
    ),
    "mqt": (
        "affected applicable checkpoints invoke mqt.debugger; unaffected "
        "checkpoints use exact causal-cone pruning"
    ),
}
MUTANT_EXECUTION_MODE = {
    "qmon": "emitted-batch-joint-trajectory-simulation",
    "qmon_pernode": "finite-shot-counterfactual-exact-mutant-z-marginal",
    "statistical": "prefix-circuit-execution-with-causal-exact-null-pruning",
    "proq": "projector-circuit-execution-with-causal-exact-null-pruning",
    "liu": "assertion-circuit-execution-with-causal-exact-null-pruning",
    "mqt": "exact-debugger-with-causal-exact-null-pruning",
}
SCOPE_CLASSIFICATION = {
    "in-scope-qmon": (
        "the mutated gate is in at least one deployed QMon checkpoint cone"
    ),
    "deployment-gap": (
        "the mutated gate is in the full numerical candidate scope but not "
        "in any deployed QMon checkpoint cone"
    ),
    "outside-full-scope": (
        "the mutated gate is outside every full numerical candidate cone"
    ),
}
FALSE_ALARM_EXECUTION_CLASSES = (
    "physical",
    "counterfactual",
    "exact-null",
    "definition-shortcut",
)
GROUPED_MONITOR_CHANNEL = {
    "emission": (
        "all selected operands at one source gate are measured and reset as "
        "one program group, followed by one shared replay"
    ),
    "replay": (
        "union of every member's full backward cone; each source gate emitted "
        "once in source program order"
    ),
    "ancillas": "later groups receive disjoint fresh ancilla slices",
    "certificate_scope": (
        "the implemented selector requires each member qubit pure; this stronger "
        "condition implies the grouped block is product with the remainder"
    ),
    "proof_scope": "grouped block channel; not a composition claim over separate runs",
}

FINAL_READ_POLICY = {
    "source": "standard terminal measurements already present in the circuit",
    "assignment": "deterministically ordered deployment batch 0",
    "multiplicity": "each eligible final checkpoint is tested exactly once",
    "reason": (
        "avoid counting the same standard output once per deployment batch"
    ),
}


class BudgetExceeded(BaseException):
    """Per-circuit wall-budget signal. BaseException so that library-level
    `except Exception` fallbacks (e.g. the Aer transpile retry) cannot
    swallow it."""


SHOTS = 8192
ALPHA = 1e-2          # familywise false-alarm budget per circuit-run
SEP_TOL = 1e-9        # determinant cutoff for branch convergence, not selection
MQT_SIMILARITY = 0.99
BRANCH_CAP = 64
CANONICAL_QMAX = 24
DETERMINISTIC_EPS = 1e-12
BRANCH_STATE_NORM_EPS = 1e-15
BRANCH_WEIGHT_EPS = 1e-12
CHI_SQUARE_MIN_EXPECTED_COUNT = 5.0
INSTRUMENTATION_INTERNAL_QUBIT_LIMIT = 10 ** 7
FALSE_ALARM_REAL_RUN_K_CAP = 200
FALSE_ALARM_SHOT_SEED = 1
MQT_APPLICABLE_TIMEOUT = 120
MQT_RUN_TIMEOUT = 120
MQT_TIMEOUT_BUDGET = 10
MQT_WALL_BUDGET = 600.0
MUTANT_CIRCUIT_BUDGET_SECONDS = 21600
FALSE_ALARM_CIRCUIT_BUDGET_SECONDS = 2700
RQ3_ARTIFACT_SCHEMA = "qmon-rq3-realexec-canonical-joint-shots-v4"
TERMINAL_CIRCUIT_STATUSES = frozenset({"complete", "terminal-incomplete"})
RETRYABLE_CIRCUIT_STATUSES = frozenset({"budget-timeout", "error"})

_AER = None
_TK_BACKEND = None
_CANONICAL_MANIFEST = None
_CANONICAL_SLOT_BY_ID = None
_CANONICAL_SOURCE_BY_CIRCUIT = None
_CANONICAL_ORACLE_SIGNATURES = {}


class JointTrajectoryIncomplete(RuntimeError):
    """A generated monitoring batch could not produce a complete aligned
    shot table."""


def _canonical_manifest():
    global _CANONICAL_MANIFEST
    if _CANONICAL_MANIFEST is None:
        _CANONICAL_MANIFEST = load_manifest()
    return _CANONICAL_MANIFEST


def canonical_mutation_slots(key, manifest=None):
    if manifest is None:
        manifest = _canonical_manifest()
    return slots_for_circuit(manifest, key)


def _aer():
    global _AER
    if _AER is None:
        from qiskit_aer import AerSimulator
        _AER = AerSimulator()
    return _AER


def _aer_run(circ, shots, seed):
    """Run on Aer; a gate outside Aer's native set (e.g. ch in the
    randomcircuit family) triggers an EXACT u/cx transpile retry."""
    try:
        return _aer().run(circ, shots=shots,
                          seed_simulator=seed).result().get_counts()
    except Exception:
        from qiskit import transpile
        t = transpile(circ, basis_gates=["u", "cx", "measure"],
                      optimization_level=0)
        return _aer().run(t, shots=shots,
                          seed_simulator=seed).result().get_counts()


def _tk():
    global _TK_BACKEND
    if _TK_BACKEND is None:
        from pytket.extensions.qiskit import AerBackend
        _TK_BACKEND = AerBackend()
    return _TK_BACKEND


def _seed32(*parts):
    """63-bit derived seed. A 31-bit crc32 space produced one collision over
    the roughly 2M planned draws; blake2b-64 makes collisions negligible.
    Aer accepts ints up to 2^63-1."""
    import hashlib
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") >> 1


def _rq3_artifact_protocol(methods, arm, circuit_budget_seconds):
    manifest = load_manifest()
    thresholds_and_caps = {
        "qmax": CANONICAL_QMAX,
        "shots": SHOTS,
        "alpha_familywise": ALPHA,
        "deterministic_probability_epsilon": DETERMINISTIC_EPS,
        "branch_convergence_determinant_tolerance": SEP_TOL,
        "branch_cap": BRANCH_CAP,
        "branch_state_norm_epsilon": BRANCH_STATE_NORM_EPS,
        "branch_probability_weight_epsilon": BRANCH_WEIGHT_EPS,
        "chi_square_min_expected_count": CHI_SQUARE_MIN_EXPECTED_COUNT,
        "instrumentation_internal_qubit_limit": (
            INSTRUMENTATION_INTERNAL_QUBIT_LIMIT
        ),
        "mqt_assert_eq_similarity": MQT_SIMILARITY,
        "mqt_applicability_timeout_seconds": MQT_APPLICABLE_TIMEOUT,
        "mqt_run_timeout_seconds": MQT_RUN_TIMEOUT,
        "mqt_timeout_budget": MQT_TIMEOUT_BUDGET,
        "mqt_wall_budget_seconds": MQT_WALL_BUDGET,
        "false_alarm_real_run_k_cap": FALSE_ALARM_REAL_RUN_K_CAP,
        "false_alarm_shot_seed": FALSE_ALARM_SHOT_SEED,
        "circuit_budget_seconds": circuit_budget_seconds,
    }
    core = {
        "schema": RQ3_ARTIFACT_SCHEMA,
        "arm": arm,
        "qmax": CANONICAL_QMAX,
        "shots": SHOTS,
        "alpha_familywise": ALPHA,
        "branch_cap": BRANCH_CAP,
        "circuit_budget_seconds": circuit_budget_seconds,
        "canonical_manifest_sha256": manifest["manifest_sha256"],
        "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
        "pruning_scope_rule": PRUNING_SCOPE_RULE,
        "exact_binomial_convention": dict(EXACT_BINOMIAL_CONVENTION),
        "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
        "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
        "mutant_execution_mode": dict(MUTANT_EXECUTION_MODE),
        "scope_classification": dict(SCOPE_CLASSIFICATION),
        "paired_in_scope_denominator": "in_scope_full",
        "false_alarm_execution_classes": list(
            FALSE_ALARM_EXECUTION_CLASSES),
        "mqt_false_alarm_semantics": {
            "reference_applicability": (
                "conditional-on-original-assertion-pass"
            ),
            "status": "definition-shortcut-not-empirical",
            "empirical_false_alarm": None,
        },
        "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
        "final_read_policy": dict(FINAL_READ_POLICY),
        "joint_shots": True,
        "thresholds_and_caps": thresholds_and_caps,
    }
    protocol = dict(core)
    protocol["methods"] = list(methods)
    fingerprints = runtime_fingerprints(
        core,
        source_paths=(
            Path(__file__),
            Path(__file__).with_name("rq3_manifest.py"),
            Path(__file__).with_name("rq3_eval.py"),
            Path(__file__).with_name("mutation.py"),
            Path(__file__).with_name("batch_partition.py"),
            Path(__file__).with_name("utilities.py"),
            Path(__file__).with_name("circuits_selection.py"),
            Path(__file__).with_name("fast_analysis.py"),
            Path(__file__).with_name("mps_analysis.py"),
            Path(__file__).with_name("baseline_statistical.py"),
            Path(__file__).with_name("baseline_proq.py"),
            Path(__file__).with_name("baseline_liu.py"),
            Path(__file__).with_name("baseline_mqtdebugger.py"),
        ),
    )
    budget = dict(thresholds_and_caps)
    fingerprints["circuit_budget"] = {
        "value": budget,
        "sha256": stable_hash(budget),
    }
    return protocol, fingerprints


def new_rq3_artifact(methods=METHODS, *, arm="mutant",
                     circuit_budget_seconds=None):
    protocol, fingerprints = _rq3_artifact_protocol(
        methods, arm, circuit_budget_seconds
    )
    return {
        "schema": RQ3_ARTIFACT_SCHEMA,
        "protocol": protocol,
        "fingerprints": fingerprints,
        "records": [],
        "circuit_status": {},
    }


def _validate_qmax_plan(plan, location):
    if not isinstance(plan, dict) or plan.get("qmax") != CANONICAL_QMAX:
        raise ValueError(f"{location}: missing Qmax={CANONICAL_QMAX} plan")
    original = plan.get("original_qubits")
    if not isinstance(original, int):
        raise ValueError(f"{location}: missing original_qubits")
    for batch in plan.get("batches", ()):
        if original + int(batch.get("cost", 10**9)) > CANONICAL_QMAX:
            raise ValueError(f"{location}: infeasible Qmax batch {batch}")
    count_fields = (
        "all_monitorable_checkpoint_count",
        "prebudget_instrumentable_checkpoint_count",
        "deployed_checkpoint_count",
        "deployed_mid_checkpoint_count",
        "deployed_final_read_count",
    )
    if any(not isinstance(plan.get(field), int) or plan[field] < 0
           for field in count_fields):
        raise ValueError(f"{location}: missing checkpoint-budget counts")
    if (plan["deployed_checkpoint_count"]
            != plan["deployed_mid_checkpoint_count"]
            + plan["deployed_final_read_count"]):
        raise ValueError(f"{location}: deployed checkpoint counts disagree")
    if plan.get("final_read_policy") != FINAL_READ_POLICY:
        raise ValueError(f"{location}: final-read policy mismatch")
    rejected = plan.get("individually_infeasible_checkpoint_nodes")
    if not isinstance(rejected, list) or any(
            not isinstance(node, list) or len(node) != 2 for node in rejected):
        raise ValueError(f"{location}: malformed individually infeasible nodes")
    groups = plan.get("monitor_groups")
    if not isinstance(groups, list):
        raise ValueError(f"{location}: missing grouped-monitor membership")
    group_gates = []
    for group in groups:
        if (not isinstance(group, dict)
                or not isinstance(group.get("gate_index"), int)
                or not isinstance(group.get("qubits"), list)
                or not group["qubits"]
                or len(group["qubits"]) != len(set(group["qubits"]))):
            raise ValueError(f"{location}: malformed monitor group {group!r}")
        group_gates.append(group["gate_index"])
    if len(group_gates) != len(set(group_gates)):
        raise ValueError(f"{location}: duplicate monitor group gate")
    covered = {
        int(gate) for batch in plan.get("batches", ())
        for gate in batch.get("nodes", ())
    }
    if set(group_gates) != covered:
        raise ValueError(f"{location}: monitor groups disagree with batches")
    plan_value = dict(plan)
    plan_sha256 = plan_value.pop("plan_sha256", None)
    if not isinstance(plan_sha256, str) or plan_sha256 != stable_hash(plan_value):
        raise ValueError(f"{location}: QMon plan hash mismatch")


def _validate_qmon_batch_diagnostics(result, plan, location):
    diag = result.get("diag")
    if not isinstance(diag, dict) or type(diag.get("complete")) is not bool:
        raise ValueError(f"{location}: malformed QMon diagnostics")
    if diag["complete"] is not result.get("complete"):
        raise ValueError(
            f"{location}: QMon diagnostics completion mismatch"
        )
    diagnostics = diag.get("batches", ())
    if not isinstance(diagnostics, (list, tuple)):
        raise ValueError(f"{location}: malformed QMon batch diagnostics")
    expected_batches = plan.get("batches", ())
    if (len(diagnostics) != len(expected_batches)
            or sorted(batch.get("batch") for batch in diagnostics)
            != list(range(len(expected_batches)))):
        raise ValueError(
            f"{location}: diagnostics do not cover every planned batch")
    expected_groups = {
        int(group["gate_index"]): tuple(int(q) for q in group["qubits"])
        for group in plan["monitor_groups"]
    }
    for batch in diagnostics:
        if (
            not isinstance(batch, dict)
            or type(batch.get("complete")) is not bool
            or type(batch.get("replay_fail_closed")) is not bool
        ):
            raise ValueError(
                f"{location}: malformed batch completion metadata"
            )
        expected_batch = expected_batches[int(batch["batch"])]
        if (tuple(batch.get("nodes", ()))
                != tuple(int(node) for node in expected_batch["nodes"])
                or batch.get("cost") != int(expected_batch["cost"])):
            raise ValueError(f"{location}: batch diagnostics disagree with plan")
        complete = batch["complete"]
        emitted = batch.get("emitted_qubits")
        if emitted is not None and not _plain_nonnegative_int(emitted):
            raise ValueError(f"{location}: malformed instrumented qubit count")
        if emitted is not None and emitted > CANONICAL_QMAX:
            raise ValueError(f"{location}: instrumented circuit uses {emitted}>24 qubits")
        if complete and (
            emitted is None or batch["replay_fail_closed"] is not True
        ):
            raise ValueError(
                f"{location}: complete batch lacks a generated circuit with "
                "replay checks")
        if emitted is None:
            continue
        actual_groups = {
            int(group["gate_index"]): tuple(int(q) for q in group["qubits"])
            for group in batch.get("monitor_groups", ())
        }
        expected = {
            int(gate): expected_groups[int(gate)]
            for gate in batch.get("nodes", ())
        }
        if actual_groups != expected:
            raise ValueError(
                f"{location}: actual grouped-monitor membership mismatch")
    if result.get("complete") is True and any(
            batch.get("complete") is not True for batch in diagnostics):
        raise ValueError(f"{location}: complete method contains incomplete batch")


def _canonical_indexes():
    global _CANONICAL_SLOT_BY_ID, _CANONICAL_SOURCE_BY_CIRCUIT
    if _CANONICAL_SLOT_BY_ID is None:
        manifest = _canonical_manifest()
        _CANONICAL_SLOT_BY_ID = {
            slot["slot_id"]: slot for slot in manifest["slots"]
        }
        _CANONICAL_SOURCE_BY_CIRCUIT = {
            entry["circuit"]: entry["source_qasm_sha256"]
            for entry in manifest["circuits"]
        }
    return _CANONICAL_SLOT_BY_ID, _CANONICAL_SOURCE_BY_CIRCUIT


def _canonical_oracle_signature(key, cache=None):
    """Recompute the oracle and complete Qmax plan from the source QASM.

    The cache stores only fixed validation data, not the statevector-heavy
    Oracle object. Final validation supplies a fresh cache so every circuit is
    recomputed from its source QASM in that validation pass.
    """
    if cache is None:
        cache = _CANONICAL_ORACLE_SIGNATURES
    if key in cache:
        return cache[key]
    _, sources = _canonical_indexes()
    if key not in sources:
        raise ValueError(f"unknown RQ3 circuit {key!r}")
    oracle = Oracle(key, CANONICAL_QMAX)
    if not oracle.eligible():
        raise ValueError(
            f"{key}: source QASM does not produce an eligible complete oracle"
        )
    if oracle.qmon_plan.get("source_qasm_sha256") != sources[key]:
        raise ValueError(f"{key}: source QASM hash drift")
    signature = {
        "K": int(oracle.K),
        "K_qmon": int(oracle.K_qmon),
        "B": int(oracle.B),
        "qmon_plan": pickle.loads(pickle.dumps(oracle.qmon_plan)),
        "nodes": tuple(oracle.nodes),
        "qmon_nodes": tuple(oracle.qmon_nodes),
        "cone": {
            tuple(node): frozenset(int(gate) for gate in oracle.cone[node])
            for node in oracle.nodes
        },
    }
    cache[key] = signature
    return signature


def _scope_from_oracle_signature(signature, gate):
    gate = int(gate)
    in_scope_full = any(
        gate in signature["cone"][node] for node in signature["nodes"]
    )
    in_scope_qmon = any(
        gate in signature["cone"][node] for node in signature["qmon_nodes"]
    )
    if in_scope_qmon:
        scope_class = "in-scope-qmon"
    elif in_scope_full:
        scope_class = "deployment-gap"
    else:
        scope_class = "outside-full-scope"
    return {
        "in_scope_full": bool(in_scope_full),
        "in_scope_qmon": bool(in_scope_qmon),
        "deployment_gap": scope_class == "deployment-gap",
        "scope_class": scope_class,
    }


def _validate_stored_oracle(record, signature, location, *, gate=None):
    for field in ("K", "K_qmon", "B"):
        if record.get(field) != signature[field]:
            raise ValueError(
                f"{location}: stored {field} disagrees with the recomputed oracle"
            )
    plan = record.get("qmon_plan")
    _validate_qmax_plan(plan, location)
    expected_plan = signature["qmon_plan"]
    if plan != expected_plan:
        _check_plan_equivalent(plan, expected_plan, location)
    if gate is not None:
        expected_scope = _scope_from_oracle_signature(signature, gate)
        for field, value in expected_scope.items():
            if record.get(field) != value:
                raise ValueError(
                    f"{location}: {field} disagrees with the recomputed oracle"
                )


def _check_plan_equivalent(plan, expected_plan, location):
    """Accept a stored plan that is an alternative optimum of the batch MILP.

    The batch-partition MILP minimises the number of batches B; several
    assignments of checkpoints to batches can attain the same B, and the one a
    solver returns depends on the machine it runs on. Everything that fixes the
    experiment (the deployed checkpoints, the monitor groups, the budget counts,
    B itself) must agree exactly; only the grouping into batches may differ,
    and that difference is reported as a warning.
    """
    ignore = {"batches", "plan_sha256"}
    for field in sorted(set(plan) | set(expected_plan)):
        if field in ignore:
            continue
        if plan.get(field) != expected_plan.get(field):
            raise ValueError(
                f"{location}: stored plan field {field!r} disagrees with the "
                f"recomputed Qmax={CANONICAL_QMAX} plan"
            )
    stored_batches = plan.get("batches", ())
    expected_batches = expected_plan.get("batches", ())
    if len(stored_batches) != len(expected_batches):
        raise ValueError(
            f"{location}: stored plan uses {len(stored_batches)} batches but the "
            f"recomputed Qmax={CANONICAL_QMAX} plan uses {len(expected_batches)}"
        )
    stored_nodes = sorted(int(g) for b in stored_batches for g in b.get("nodes", ()))
    expected_nodes = sorted(int(g) for b in expected_batches for g in b.get("nodes", ()))
    if stored_nodes != expected_nodes:
        raise ValueError(
            f"{location}: stored plan deploys a different checkpoint set than the "
            f"recomputed Qmax={CANONICAL_QMAX} plan"
        )
    if plan.get("plan_sha256") == expected_plan.get("plan_sha256"):
        raise ValueError(f"{location}: stored plan differs but carries the recomputed plan hash")
    warnings.warn(
        f"{location}: the stored Qmax={CANONICAL_QMAX} plan groups the same "
        f"{len(stored_batches)} batches over the same checkpoints differently from "
        "the plan recomputed on this machine (alternative MILP optimum; solver "
        "tie-breaking is hardware dependent); results are unaffected",
        RuntimeWarning, stacklevel=3,
    )

def _plain_nonnegative_int(value):
    return type(value) is int and value >= 0


def _validate_mutant_stats(method, result, record, location):
    stats = result.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"{location}: {method} missing execution stats")
    if type(result.get("complete")) is not bool:
        raise ValueError(f"{location}: {method} complete must be a bool")
    if type(stats.get("complete")) is not bool:
        raise ValueError(f"{location}: {method} stats.complete must be a bool")
    if stats["complete"] is not result["complete"]:
        raise ValueError(
            f"{location}: {method} complete disagrees with stats.complete"
        )

    reasons = stats.get("incomplete_reasons")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise ValueError(f"{location}: {method} malformed incomplete reasons")
    if result["complete"] is True and reasons:
        raise ValueError(
            f"{location}: {method} complete result has incomplete reasons"
        )
    if result["complete"] is False and not reasons:
        raise ValueError(
            f"{location}: {method} incomplete result lacks a reason"
        )

    base_counts = (
        "n_exec", "n_null_shortcuts", "n_analytical_marginals"
    )
    if any(not _plain_nonnegative_int(stats.get(field))
           for field in base_counts):
        raise ValueError(f"{location}: {method} malformed execution counters")
    dynamic_counts = {
        "n_affected", "n_applicable", "budget_exhausted", "timeout", "error"
    }
    for field, value in stats.items():
        if (
            field in dynamic_counts
            or field.startswith("applicability_")
            or field.startswith("run_")
        ) and not _plain_nonnegative_int(value):
            raise ValueError(
                f"{location}: {method} malformed counter {field}"
            )

    node_count = (
        record["K_qmon"] if method in ("qmon", "qmon_pernode")
        else record["K"]
    )
    if not _plain_nonnegative_int(node_count):
        raise ValueError(f"{location}: malformed checkpoint count")
    expected_family_size = (
        node_count if method in ("qmon", "qmon_pernode", "statistical")
        else None
    )
    if stats.get("family_size") != expected_family_size:
        raise ValueError(f"{location}: {method} family-size mismatch")
    expected_alpha = (
        ALPHA / max(node_count, 1) if expected_family_size is not None else None
    )
    alpha_node = stats.get("alpha_node")
    if expected_alpha is None:
        if alpha_node is not None:
            raise ValueError(f"{location}: {method} unexpected alpha_node")
    elif (
        isinstance(alpha_node, bool)
        or not isinstance(alpha_node, (int, float, np.integer, np.floating))
        or not np.isfinite(float(alpha_node))
        or not np.isclose(float(alpha_node), expected_alpha, rtol=0.0, atol=1e-18)
    ):
        raise ValueError(f"{location}: {method} alpha_node mismatch")
    expected_stats = {
        "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[method],
        "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[method],
        "execution_mode": MUTANT_EXECUTION_MODE[method],
    }
    for field, expected in expected_stats.items():
        if stats.get(field) != expected:
            raise ValueError(f"{location}: {method} stats.{field} mismatch")

    n_exec = stats["n_exec"]
    n_null = stats["n_null_shortcuts"]
    n_analytical = stats["n_analytical_marginals"]
    if n_null > node_count or n_analytical > node_count:
        raise ValueError(f"{location}: {method} checkpoint counter exceeds K")
    if method == "qmon":
        if n_analytical != 0 or n_null != 0 or n_exec not in (0, record["B"]):
            raise ValueError(f"{location}: malformed physical QMon accounting")
        if result["complete"] is True and n_exec != record["B"]:
            raise ValueError(f"{location}: complete QMon did not execute every batch")
    elif method == "qmon_pernode":
        if n_exec != 0 or n_null > n_analytical:
            raise ValueError(
                f"{location}: malformed QMon(node) counterfactual accounting"
            )
    else:
        if n_analytical != 0 or n_exec + n_null > node_count:
            raise ValueError(f"{location}: malformed baseline execution accounting")

    if method == "mqt" and "budget" in stats:
        budget = stats["budget"]
        if not isinstance(budget, dict):
            raise ValueError(f"{location}: malformed MQT budget")
        for field in ("timeout_limit", "timeouts"):
            if not _plain_nonnegative_int(budget.get(field)):
                raise ValueError(f"{location}: malformed MQT budget {field}")
        for field in ("wall_limit", "wall_seconds"):
            value = budget.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{location}: malformed MQT budget {field}")
        if type(budget.get("remaining")) is not bool:
            raise ValueError(f"{location}: malformed MQT budget remaining")
        expected_remaining = (
            budget["timeouts"] < budget["timeout_limit"]
            and float(budget["wall_seconds"]) < float(budget["wall_limit"])
        )
        if budget["remaining"] is not expected_remaining:
            raise ValueError(f"{location}: inconsistent MQT remaining budget")
    return stats


def _validate_mutant_localization(method, result, record, signature, location):
    if type(result.get("detected")) is not bool:
        raise ValueError(f"{location}: {method} detected must be a bool")
    if type(result.get("loc_hit")) is not bool:
        raise ValueError(f"{location}: {method} loc_hit must be a bool")
    precision = result.get("loc_precision")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, (int, float, np.integer, np.floating))
        or not np.isfinite(float(precision))
        or not 0.0 <= float(precision) <= 1.0
    ):
        raise ValueError(f"{location}: {method} invalid localization precision")

    first = result.get("first_node")
    if result["detected"] is False:
        if first is not None:
            raise ValueError(
                f"{location}: {method} undetected result has a first node"
            )
        expected_hit = False
        expected_precision = 0.0
    else:
        if (
            not isinstance(first, (tuple, list))
            or len(first) != 2
            or any(type(value) is not int for value in first)
        ):
            raise ValueError(f"{location}: {method} malformed first node")
        node = tuple(first)
        allowed_nodes = (
            signature["qmon_nodes"]
            if method in ("qmon", "qmon_pernode")
            else signature["nodes"]
        )
        if node not in allowed_nodes:
            raise ValueError(
                f"{location}: {method} first node is outside its scan set"
            )
        cone = signature["cone"].get(node)
        if not isinstance(cone, frozenset):
            raise ValueError(f"{location}: missing recomputed cone for {node}")
        expected_hit = int(record["mut_gate"]) in cone
        expected_precision = (
            1.0 / len(cone) if expected_hit and cone else 0.0
        )
    if result["loc_hit"] is not expected_hit:
        raise ValueError(
            f"{location}: {method} localization hit disagrees with recomputed cone"
        )
    if not np.isclose(
        float(precision), expected_precision, rtol=0.0, atol=1e-15
    ):
        raise ValueError(
            f"{location}: {method} localization precision disagrees with "
            "recomputed cone"
        )


def _validate_mutant_method_result(
        method, result, record_or_location, signature=None, location=None):
    # Retain the old three-argument accounting check for callers that never
    # handled whole artifacts.  validate_rq3_artifact always supplies the
    # record and recomputed oracle signature and therefore always takes the
    # strict branch below.
    if location is None:
        location = record_or_location
        if not isinstance(result, dict):
            raise ValueError(f"{location}: malformed {method} result")
        if result.get("execution_mode") != MUTANT_EXECUTION_MODE[method]:
            raise ValueError(f"{location}: {method} execution_mode mismatch")
        if "executions" in result:
            raise ValueError(f"{location}: ambiguous legacy executions field")
        stats = result.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"{location}: {method} missing execution stats")
        n_exec = stats.get("n_exec")
        n_analytical = stats.get("n_analytical_marginals", 0)
        if (
            not _plain_nonnegative_int(n_exec)
            or not _plain_nonnegative_int(n_analytical)
            or result.get("executed_program_count") != n_exec
        ):
            raise ValueError(
                f"{location}: {method} executed-program count mismatch"
            )
        expected_counterfactuals = (
            n_analytical if method == "qmon_pernode" else 0
        )
        if (
            result.get("counterfactual_marginal_count")
            != expected_counterfactuals
        ):
            raise ValueError(
                f"{location}: {method} counterfactual-marginal count mismatch"
            )
        if method == "qmon_pernode" and result["executed_program_count"] != 0:
            raise ValueError(
                f"{location}: QMon(node) must not claim instrumented executions"
            )
        return

    record = record_or_location
    if not isinstance(signature, dict):
        raise ValueError(f"{location}: missing recomputed oracle signature")
    if not isinstance(result, dict):
        raise ValueError(f"{location}: malformed {method} result")
    expected_result_fields = {
        "asserted_subsystem": METHOD_ASSERTED_SUBSYSTEM[method],
        "pruning_scope": PRUNING_SCOPE_RULE,
        "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[method],
        "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[method],
        "execution_mode": MUTANT_EXECUTION_MODE[method],
    }
    for field, expected in expected_result_fields.items():
        if result.get(field) != expected:
            raise ValueError(f"{location}: {method} {field} mismatch")
    if "executions" in result:
        raise ValueError(f"{location}: ambiguous legacy executions field")
    stats = _validate_mutant_stats(method, result, record, location)
    if (
        not _plain_nonnegative_int(result.get("executed_program_count"))
        or result["executed_program_count"] != stats["n_exec"]
    ):
        raise ValueError(f"{location}: {method} executed-program count mismatch")
    expected_counterfactuals = (
        stats["n_analytical_marginals"]
        if method == "qmon_pernode" else 0
    )
    if (
        not _plain_nonnegative_int(result.get("counterfactual_marginal_count"))
        or result["counterfactual_marginal_count"] != expected_counterfactuals
    ):
        raise ValueError(
            f"{location}: {method} counterfactual-marginal count mismatch"
        )
    if method == "qmon_pernode" and result["executed_program_count"] != 0:
        raise ValueError(
            f"{location}: QMon(node) must not claim instrumented executions"
        )
    _validate_mutant_localization(
        method, result, record, signature, location
    )


def _validate_false_alarm_method_result(method, result, record, location):
    if not isinstance(result, dict):
        raise ValueError(f"{location}: malformed {method} false-alarm result")
    if type(result.get("complete")) is not bool:
        raise ValueError(f"{location}: {method} complete must be a bool")
    expected_class = (
        "physical" if method == "qmon"
        else "counterfactual" if method == "qmon_pernode"
        else "definition-shortcut" if method == "mqt"
        else "physical" if record.get("real_run")
        else "exact-null"
    )
    if result.get("execution_class") != expected_class:
        raise ValueError(f"{location}: {method} execution class mismatch")
    if result.get("execution_class") not in FALSE_ALARM_EXECUTION_CLASSES:
        raise ValueError(f"{location}: unknown false-alarm execution class")
    expected_mode = (
        MUTANT_EXECUTION_MODE["qmon"] if method == "qmon"
        else "finite-shot-counterfactual-exact-reference-z-marginal"
        if method == "qmon_pernode"
        else "conditional-reference-definition-shortcut"
        if method == "mqt"
        else MUTANT_EXECUTION_MODE[method]
        if expected_class == "physical"
        else "analytical-exact-null-shortcut"
    )
    if result.get("execution_mode") != expected_mode:
        raise ValueError(f"{location}: {method} execution_mode mismatch")
    count_fields = (
        "n_executed_programs",
        "n_counterfactual_marginals",
        "n_exact_null_shortcuts",
        "n_definition_shortcuts",
    )
    if any(
        not _plain_nonnegative_int(result.get(field))
        for field in count_fields
    ):
        raise ValueError(f"{location}: malformed false-alarm accounting")
    expected_counts = {
        "physical": (record["B"] if method == "qmon" else record["K"], 0, 0, 0),
        "counterfactual": (0, record["K_qmon"], 0, 0),
        "exact-null": (0, 0, record["K"], 0),
        "definition-shortcut": (0, 0, 0, record["K"]),
    }[expected_class]
    if tuple(result[field] for field in count_fields) != expected_counts:
        raise ValueError(f"{location}: false-alarm accounting mismatch")
    if method == "mqt":
        if (
            result.get("complete") is not True
            or result.get("reference_applicability")
            != "conditional-on-original-assertion-pass"
            or result.get("reference_applicability_status") != "not-evaluated"
            or result.get("status") != "definition-shortcut-not-empirical"
            or result.get("n_flagged") is not None
            or result.get("false_alarm") is not None
            or result.get("conditional_false_alarm") is not False
        ):
            raise ValueError(
                f"{location}: MQT false alarm masquerades as empirical"
            )
    else:
        expected_status = (
            "complete" if result["complete"] is True else "incomplete"
        )
        expected_reference_status = (
            "analytical-counterfactual" if method == "qmon_pernode"
            else "proved-null" if expected_class == "exact-null"
            else "executed"
        )
        if (
            result.get("reference_applicability") != "unconditional"
            or result.get("reference_applicability_status")
            != expected_reference_status
            or result.get("status") != expected_status
            or not isinstance(result.get("false_alarm"), (bool, type(None)))
            or (
                result["complete"] is True
                and not isinstance(result.get("false_alarm"), bool)
            )
        ):
            raise ValueError(f"{location}: malformed false-alarm status")


def _validate_canonical_mutant_record(record, location, oracle_cache=None):
    slots, sources = _canonical_indexes()
    slot_id = record.get("slot_id")
    expected = slots.get(slot_id)
    if expected is None:
        raise ValueError(f"{location}: unknown mutation-list slot {slot_id!r}")
    expected_values = {
        "circuit": expected["circuit"],
        "seed": int(expected["seed"]),
        "slot_ordinal": int(expected["ordinal"]),
        "mut_gate": int(expected["gate"]),
        "original": expected["original"],
        "replacement": expected["replacement"],
        "mutant_qasm_sha256": expected["mutant_qasm_sha256"],
        "equivalent": bool(expected["output_equivalent"]),
        "output_l1": expected["output_l1"],
        "manifest_sha256": _canonical_manifest()["manifest_sha256"],
    }
    for field, value in expected_values.items():
        if record.get(field) != value:
            raise ValueError(f"{location}: {field} mismatch with the mutation list")
    signature = _canonical_oracle_signature(
        expected["circuit"], cache=oracle_cache
    )
    _validate_stored_oracle(
        record, signature, location, gate=int(expected["gate"])
    )
    if record["qmon_plan"].get("source_qasm_sha256") != sources[
        expected["circuit"]
    ]:
        raise ValueError(f"{location}: original QASM/plan mismatch")
    return expected


def validate_rq3_artifact(
        artifact, source="artifact", *, force_oracle_recompute=False):
    if (not isinstance(artifact, dict)
            or artifact.get("schema") != RQ3_ARTIFACT_SCHEMA):
        raise ValueError(
            f"{source}: legacy/missing schema; expected {RQ3_ARTIFACT_SCHEMA}. "
            "Old list artifacts must be rerun")
    protocol = artifact.get("protocol", {})
    if protocol.get("arm") not in ("mutant", "false-alarm"):
        raise ValueError(f"{source}: invalid RQ3 arm")
    if (
        not isinstance(protocol.get("methods"), list)
        or len(protocol["methods"]) != len(set(protocol["methods"]))
        or not set(protocol["methods"]).issubset(METHODS)
    ):
        raise ValueError(f"{source}: invalid method list")
    if protocol.get("qmax") != CANONICAL_QMAX:
        raise ValueError(
            f"{source}: invalid/missing qmax={protocol.get('qmax')!r}; "
            f"Qmax={CANONICAL_QMAX} is required")
    expected_protocol, current = _rq3_artifact_protocol(
        protocol.get("methods", ()),
        protocol.get("arm"),
        protocol.get("circuit_budget_seconds"),
    )
    if protocol != expected_protocol:
        raise ValueError(f"{source}: protocol drift")
    stored = artifact.get("fingerprints")
    if not isinstance(stored, dict):
        raise ValueError(f"{source}: missing checksums")
    # The QASM corpus, protocol parameters, and circuit budget define the
    # experiment and must match exactly. Source-file and dependency checksums
    # record the environment that produced the artifact; they differ after
    # documentation edits or package upgrades without changing any result, so
    # a mismatch there is reported as a warning rather than rejected.
    for field in ("qasm_corpus", "protocol", "circuit_budget"):
        if stored.get(field, {}).get("sha256") != current[field]["sha256"]:
            raise ValueError(f"{source}: {field} checksum mismatch")
    for field in ("sources", "dependencies"):
        if stored.get(field, {}).get("sha256") != current[field]["sha256"]:
            warnings.warn(
                f"{source}: {field} checksum differs from the recorded "
                "value (documentation or environment change); results are "
                "unaffected",
                RuntimeWarning,
                stacklevel=2,
            )
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{source}: records must be a list")
    statuses = artifact.get("circuit_status")
    if not isinstance(statuses, dict):
        raise ValueError(f"{source}: circuit_status must be a dict")
    _, sources = _canonical_indexes()
    oracle_cache = {} if force_oracle_recompute else _CANONICAL_ORACLE_SIGNATURES
    extra_statuses = set(statuses) - set(sources)
    if extra_statuses:
        raise ValueError(
            f"{source}: circuit status outside the RQ3 circuit set "
            f"{sorted(extra_statuses)[:5]}")
    if protocol.get("arm") == "mutant":
        identities = []
        by_circuit = {}
        for index, record in enumerate(records):
            location = f"{source}: records[{index}]"
            _validate_canonical_mutant_record(
                record, location, oracle_cache=oracle_cache
            )
            signature = _canonical_oracle_signature(
                record["circuit"], cache=oracle_cache
            )
            if record["circuit"] not in statuses:
                raise ValueError(f"{location}: missing circuit status")
            identities.append((
                record.get("circuit"), record.get("seed"),
                record.get("slot_ordinal"),
            ))
            expected_methods = (
                set() if record.get("equivalent")
                else set(protocol.get("methods", ()))
            )
            if set(record.get("methods", ())) != expected_methods:
                raise ValueError(f"{location}: method coverage mismatch")
            for method, result in record.get("methods", {}).items():
                _validate_mutant_method_result(
                    method, result, record, signature,
                    f"{location}/{method}"
                )
            qmon = record.get("methods", {}).get("qmon")
            if qmon:
                _validate_qmon_batch_diagnostics(
                    qmon, record["qmon_plan"],
                    location)
            by_circuit.setdefault(record["circuit"], []).append(record)
        if len(identities) != len(set(identities)):
            raise ValueError(f"{source}: duplicate mutant slots")
        for key, circuit_records in by_circuit.items():
            expected_ids = {
                slot["slot_id"]
                for slot in canonical_mutation_slots(key)
            }
            if ({record["slot_id"] for record in circuit_records}
                    != expected_ids
                    or len(circuit_records) != len(expected_ids)):
                raise ValueError(f"{source}: {key} lacks exactly the mutation-list slots")
    else:
        identities = [record.get("circuit") for record in records]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{source}: duplicate false-alarm circuits")
        for index, record in enumerate(records):
            location = f"{source}: records[{index}]"
            key = record.get("circuit")
            if key not in sources:
                raise ValueError(f"{location}: circuit outside the RQ3 circuit set")
            if key not in statuses:
                raise ValueError(f"{location}: missing circuit status")
            signature = _canonical_oracle_signature(
                key, cache=oracle_cache
            )
            _validate_stored_oracle(record, signature, location)
            if (record["qmon_plan"].get("source_qasm_sha256")
                    != sources[key]):
                raise ValueError(f"{location}: original QASM/plan mismatch")
            if (
                record.get("real_run_k_cap") != FALSE_ALARM_REAL_RUN_K_CAP
                or record.get("shot_seed") != FALSE_ALARM_SHOT_SEED
                or record.get("real_run")
                != (record["K"] <= FALSE_ALARM_REAL_RUN_K_CAP)
            ):
                raise ValueError(f"{location}: false-alarm protocol drift")
            if set(record.get("methods", ())) != set(
                    protocol.get("methods", ())):
                raise ValueError(f"{location}: method coverage mismatch")
            for method, result in record.get("methods", {}).items():
                _validate_false_alarm_method_result(
                    method, result, record, f"{location}/{method}"
                )
            qmon = record.get("methods", {}).get("qmon")
            if qmon:
                _validate_qmon_batch_diagnostics(
                    qmon, record["qmon_plan"],
                    location)
    allowed_statuses = (
        TERMINAL_CIRCUIT_STATUSES | RETRYABLE_CIRCUIT_STATUSES
    )
    by_circuit = {
        key: [record for record in records if record.get("circuit") == key]
        for key in statuses
    }
    for key, status in statuses.items():
        if (not isinstance(status, dict)
                or status.get("status") not in allowed_statuses):
            raise ValueError(f"{source}: invalid circuit status for {key}")
        circuit_records = by_circuit[key]
        status_name = status["status"]
        if status_name in TERMINAL_CIRCUIT_STATUSES:
            expected_status = _terminal_circuit_status(
                circuit_records,
                protocol["methods"],
                arm=protocol["arm"],
            )
            for field, expected_value in expected_status.items():
                if status.get(field) != expected_value:
                    raise ValueError(
                        f"{source}: false {field} for terminal circuit {key}"
                    )
        elif circuit_records:
            raise ValueError(
                f"{source}: retryable circuit {key} contains terminal records"
            )
    return artifact


def validate_final_rq3_artifact(artifact, source="RQ3 result"):
    """Validate a complete inventory of the RQ3 circuit set with terminal
    method outcomes.

    A final artifact may contain method-level timeouts or other explicit
    ``complete=False`` outcomes.  Those records remain available for inspection
    but are excluded from the common-method-complete statistical intersection.
    """
    validate_rq3_artifact(
        artifact, source, force_oracle_recompute=True
    )
    protocol = artifact["protocol"]
    _, sources = _canonical_indexes()
    expected = set(sources)
    statuses = artifact["circuit_status"]
    record_keys = {record["circuit"] for record in artifact["records"]}
    if set(protocol.get("methods", ())) != set(METHODS):
        raise ValueError(f"{source}: final result requires all RQ3 methods")
    if (set(statuses) != expected or record_keys != expected
            or any(status.get("status") not in TERMINAL_CIRCUIT_STATUSES
                   for status in statuses.values())):
        raise ValueError(
            f"{source}: final result requires a complete terminal "
            f"238-circuit inventory")
    expected_records = (
        15 * len(expected)
        if protocol.get("arm") == "mutant"
        else len(expected)
    )
    if len(artifact["records"]) != expected_records:
        raise ValueError(
            f"{source}: expected {expected_records} records, got "
            f"{len(artifact['records'])}")
    return artifact


def validate_complete_rq3_artifact(artifact, source="RQ3 result"):
    """Compatibility alias for the terminal-inventory final validator."""
    return validate_final_rq3_artifact(artifact, source)


def load_rq3_artifact(path):
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    return validate_rq3_artifact(artifact, str(path))


def save_rq3_artifact(path, artifact):
    validate_rq3_artifact(artifact, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _rq3_circuit_inventory_complete(records, methods, *, arm="mutant"):
    """Whether every requested slot and method outcome has been recorded.

    Method-level ``complete=False`` is still a recorded terminal outcome.  It
    must not turn a deterministic tool timeout into a missing slot that resume
    executes forever.
    """
    methods = set(methods)
    if arm == "false-alarm":
        return (
            len(records) == 1
            and set(records[0].get("methods", ())) == methods
        )
    if arm != "mutant" or len(records) != 15:
        return False
    circuits = {record.get("circuit") for record in records}
    if len(circuits) != 1:
        return False
    key = next(iter(circuits))
    expected_slots = {slot["slot_id"] for slot in canonical_mutation_slots(key)}
    if {record.get("slot_id") for record in records} != expected_slots:
        return False
    for record in records:
        expected_methods = set() if record.get("equivalent") else methods
        if set(record.get("methods", ())) != expected_methods:
            return False
    return True


def _rq3_circuit_complete(records, methods, *, arm="mutant"):
    if not _rq3_circuit_inventory_complete(records, methods, arm=arm):
        return False
    for record in records:
        if record.get("equivalent"):
            continue
        if not all(
            method in record.get("methods", {})
            and record["methods"][method].get("complete") is True
            for method in methods
        ):
            return False
    return True


def _terminal_circuit_status(records, methods, *, arm):
    """Describe a full-inventory circuit without scoring incomplete methods."""
    if not _rq3_circuit_inventory_complete(records, methods, arm=arm):
        raise ValueError(f"{arm}: circuit lacks a complete recorded inventory")
    incomplete_methods = sorted({
        method
        for record in records
        for method in methods
        if method in record.get("methods", {})
        and record["methods"][method].get("complete") is not True
    })
    all_complete = _rq3_circuit_complete(records, methods, arm=arm)
    return {
        "status": "complete" if all_complete else "terminal-incomplete",
        "records": len(records),
        "methods": list(methods),
        "incomplete_methods": incomplete_methods,
        "terminal": True,
    }


@contextlib.contextmanager
def _mute():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


# ==========================================================================
# Oracle per circuit: expected node states, Liu/Proq parameters, cones,
# QMon batch structure, MQT per-node applicability.
# ==========================================================================
def _phase_fixed(amps):
    k = int(np.argmax(np.abs(amps)))
    return amps * np.exp(-1j * np.angle(amps[k]))


def _prefix(circ, upto_idx, keep_clbits=False):
    """Copy of `circ` truncated to data index `upto_idx` inclusive, dropping
    non-unitary ops, with fresh (classical-register-free) framing."""
    pre = QuantumCircuit(circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            if j == upto_idx:
                break
            continue
        pre.append(ci.operation,
                   [circ.find_bit(b).index for b in ci.qubits])
        if j == upto_idx:
            break
    return pre


_GATE_EXP_CACHE = {}


def _ucx_expansion(op, nq):
    """u/cx expansion of ONE gate, cached by (name, params, arity). At
    optimization_level=0 the BasisTranslator maps gates independently, so
    concatenating per-gate translations equals translating the whole prefix,
    while a deep prefix (e.g. no-ancilla multi-controlled gates) is paid for
    once per distinct gate instead of once per (node, mutant) call."""
    key = (op.name, tuple(float(p) for p in op.params), nq)
    if key not in _GATE_EXP_CACHE:
        from qiskit import transpile
        c = QuantumCircuit(nq)
        c.append(op, range(nq))
        t = transpile(c, basis_gates=["u", "cx"], optimization_level=0)
        _GATE_EXP_CACHE[key] = [
            (ci.operation, [t.find_bit(b).index for b in ci.qubits])
            for ci in t.data]
    return _GATE_EXP_CACHE[key]


def _prefix_ucx(circ, upto_idx):
    """u/cx-basis prefix via the per-gate cache (exact, level-0 semantics)."""
    pre = QuantumCircuit(circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            if j == upto_idx:
                break
            continue
        qmap = [circ.find_bit(b).index for b in ci.qubits]
        for op, local in _ucx_expansion(ci.operation, len(qmap)):
            pre.append(op, [qmap[k] for k in local])
        if j == upto_idx:
            break
    return pre


class Oracle:
    """Everything derived from the ORIGINAL circuit, computed once."""

    def __init__(self, key, qmax=24):
        if int(qmax) != CANONICAL_QMAX:
            raise ValueError(
                f"RQ3 operational protocol requires Qmax={CANONICAL_QMAX}; "
                f"got {qmax}")
        self.key = key
        self.qmax = int(qmax)
        qc = QuantumCircuit.from_qasm_file(quantum_path + key)
        self.qc = qc
        self.nodes = sorted(monitored_nodes(qc))          # ascending gate idx
        self.K = len(self.nodes)
        if not self.nodes:
            return
        exp_rho = marginals(qc, self.nodes)
        self.p1e = {}
        self.amps = {}
        for nd in self.nodes:
            rho = exp_rho[nd]
            self.p1e[nd] = float(rho[1, 1].real)
            w, v = np.linalg.eigh(rho)
            self.amps[nd] = _phase_fixed(v[:, int(np.argmax(w))])
        self.cone = {nd: causal_cone(qc, nd[0], nd[1])[0]
                     for nd in self.nodes}
        self.measured = [qc.find_bit(ci.qubits[0]).index
                         for ci in qc.data if ci.operation.name == "measure"]
        expected_final_reads = standard_final_read_nodes(qc, self.nodes)
        self.base_probs = final_probs(qc, self.measured)

        # QMon batch structure (identical to rq3_eval.batch_count)
        res, _, paths = analyze_per_gate_nolock(qc)
        dnd = filter_circuits_with_output_num({key: res}, {key: paths})
        self.op_list = None
        self.batches = None
        self.infeasible = ()
        if key in dnd:
            final = {key: []}
            for index, gname, qi, p0, p1 in dnd[key]:
                final[key].append([index, gname, qi, p0, p1,
                                   dfs_quantum_circuit(qc, qi, index,
                                                       set(), [])])
            op_list = bulid_partial_circuit(final)
            if key in op_list:
                with _mute():
                    _, eqc = filter_operation_list(
                        key, op_list, INSTRUMENTATION_INTERNAL_QUBIT_LIMIT
                    )
                constraint = {"num_oq": qc.num_qubits}
                for nd_i, eq in eqc.items():
                    constraint[nd_i] = {"num_eq": len(eq), "qubits": []}
                    for g in op_list[key]:
                        if g[0] == nd_i:
                            constraint[nd_i]["qubits"].append(g[2])
                batches, infeasible = partition_nodes(constraint, qmax)
                self.op_list = op_list
                self.batches = batches                     # gate-index groups
                self.infeasible = tuple(infeasible)
                self.batch_of = {}
                for b_id, b in enumerate(batches):
                    for gi in b["nodes"]:
                        self.batch_of[gi] = b_id
        self.B = len(self.batches) if self.batches else None
        # QMon is scored ONLY on checkpoints its deployed instrumentation
        # actually observes: the instrumented mid-circuit monitors, PLUS each
        # measured qubit's final pre-measurement node (dropped from op_list
        # as redundant because the run's standard final measurement reads it
        # for free). Nodes on never-measured qubits and infeasible-batch
        # nodes have no physical readout and are excluded (the baselines can
        # still assert there, so their scan keeps the full set).
        self.emitted, self.lastgate = [], []
        self.prebudget_instrumentable = []
        self.infeasible_checkpoint_nodes = []
        if self.batches is not None:
            covered = {gi for b in self.batches for gi in b["nodes"]}
            op_pairs = {(g[0], g[2]) for g in self.op_list[key]}
            expected_mid_points = set(self.nodes) - expected_final_reads
            if op_pairs != expected_mid_points:
                missing = sorted(expected_mid_points - op_pairs)
                extra = sorted(op_pairs - expected_mid_points)
                raise ValueError(
                    f"RQ3 deployment plan disagrees with the final-read "
                    f"convention for {key}: missing={missing}, extra={extra}"
                )
            self.prebudget_instrumentable = sorted(op_pairs)
            infeasible_gates = set(self.infeasible)
            self.infeasible_checkpoint_nodes = sorted(
                node for node in op_pairs if node[0] in infeasible_gates)
            self.emitted = sorted({(g[0], g[2]) for g in self.op_list[key]
                                   if g[0] in covered})
            self.lastgate = sorted(expected_final_reads)
        self.qmon_nodes = sorted(set(self.emitted) | set(self.lastgate))
        self.K_qmon = len(self.qmon_nodes)
        grouped_members = {}
        for gate, qubit in self.emitted:
            grouped_members.setdefault(gate, []).append(qubit)
        plan_value = {
            "circuit": key,
            "source_qasm_sha256": circuit_qasm_sha256(qc),
            "qmax": self.qmax,
            "original_qubits": qc.num_qubits,
            "batches": [
                {"batch": index, "nodes": list(batch["nodes"]),
                 "cost": int(batch["cost"])}
                for index, batch in enumerate(self.batches or [])
            ],
            "infeasible_nodes": list(self.infeasible),
            "individually_infeasible_gate_nodes": list(self.infeasible),
            "individually_infeasible_checkpoint_nodes": [
                list(node) for node in self.infeasible_checkpoint_nodes],
            "all_monitorable_checkpoint_count": self.K,
            "prebudget_instrumentable_checkpoint_count": len(
                self.prebudget_instrumentable),
            "deployed_checkpoint_count": self.K_qmon,
            "deployed_mid_checkpoint_count": len(self.emitted),
            "deployed_final_read_count": len(self.lastgate),
            "coverage_scope": (
                "every individually feasible instrumentable gate node plus "
                "eligible standard final reads; not all pre-budget candidates"
            ),
            "monitor_groups": [
                {"gate_index": int(gate),
                 "qubits": sorted(int(qubit) for qubit in qubits)}
                for gate, qubits in sorted(grouped_members.items())
            ],
            "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
            "final_read_policy": dict(FINAL_READ_POLICY),
        }
        self.qmon_plan = dict(plan_value)
        self.qmon_plan["plan_sha256"] = stable_hash(plan_value)
        # Only completed semantic applicability is cacheable.  A timeout or
        # tool error is transient and must be retried under the next mutant's
        # fresh budget.
        self._mqt_semantic_applicability = {}

    def eligible(self):
        if not self.nodes or self.B is None:
            return False
        last_node = max(i for i, _ in self.nodes)
        return last_node > self.qc.num_qubits - 1

    def deterministic(self, nd, eps=DETERMINISTIC_EPS):
        p = self.p1e[nd]
        return p < eps or p > 1 - eps

    def liu_params(self, nd):
        """Bloch parametrization used by the authors' phaseDict: the asserted
        state is cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> (their
        cu3(2*theta) reflection has the HALF-angle state as its +1
        eigenvector), so theta = 2*atan2(|a1|, |a0|)."""
        a0, a1 = self.amps[nd]
        theta = 2.0 * float(np.arctan2(abs(a1), abs(a0)))
        phi = float(np.angle(a1) - np.angle(a0))
        return theta, phi

    def mqt_applicability_status(self, nd, budget):
        """Return original-circuit applicability under one mutant's budget.

        ``pass`` and completed semantic rejections (``fail`` or
        ``unevaluable``) are stable and cached.  ``timeout``/``error`` and an
        exhausted wall budget are returned to the record as incomplete and
        are deliberately not cached.
        """
        if nd in self._mqt_semantic_applicability:
            return self._mqt_semantic_applicability[nd]
        if not budget.can_call():
            return "budget-exhausted"
        started = time.perf_counter()
        status = _mqt_call(
            _mqt_qasm(self.qc, nd, self.amps[nd]),
            timeout=MQT_APPLICABLE_TIMEOUT,
        )
        budget.record(status, time.perf_counter() - started)
        if status in ("pass", "fail", "unevaluable"):
            self._mqt_semantic_applicability[nd] = status
        return status

    def mqt_applicable(self, nd, budget=None):
        budget = budget or MQTMutantBudget()
        return self.mqt_applicability_status(nd, budget) == "pass"


# ==========================================================================
# Per-checkpoint REAL runs, one per method. Each returns True iff flagged.
# ==========================================================================
def run_statistical(mqc, nd, oracle, shots=SHOTS, seed=0, alpha_node=ALPHA):
    """Huang-Martonosi: REALLY run the prefix + Z-measure of the monitored
    qubit on Aer; chi-square GOF (their published test) vs the expected
    distribution at nominal (asymptotic) level alpha/K, with the exact
    binomial test substituted where the chi-square validity condition fails
    (an expected cell < 5); classical assertion (any wrong outcome) when the
    expected value is deterministic. The familywise false-alarm control is
    therefore nominal for the chi-square branch and exact for the fallback
    and deterministic branches (measured false-alarm rates are reported)."""
    i, q = nd
    pre = _prefix(mqc, i)
    pre.measure_all()                       # creg 'meas', all qubits
    counts = _aer_run(pre, shots, seed)
    # the 'meas' register is the last added -> FIRST space-separated token
    c1 = sum(v for k, v in counts.items()
             if k.split(" ")[0][::-1][q] == "1")
    p1 = oracle.p1e[nd]
    if oracle.deterministic(nd):
        wrong = c1 if p1 < 0.5 else shots - c1
        return wrong > 0
    f_exp = np.array([(1 - p1) * shots, p1 * shots])
    if f_exp.min() >= CHI_SQUARE_MIN_EXPECTED_COUNT:
        _, pval = chisquare([shots - c1, c1], f_exp)
    else:
        pval = binomtest(c1, shots, p1, alternative="two-sided").pvalue
    return bool(pval < alpha_node)


def run_proq(mqc, nd, oracle, shots=SHOTS, seed=0):
    """Proq through the OFFICIAL pytket ProjectorAssertionBox runtime: any
    assertion failure among the shots flags the checkpoint."""
    from pytket.circuit import ProjectorAssertionBox
    from pytket.extensions.qiskit import qiskit_to_tk
    i, q = nd
    amps = oracle.amps[nd]
    P = np.outer(amps, amps.conj())
    # u/cx basis via the per-gate cache: pytket cannot ingest some mutation
    # replacements (e.g. rzx), and re-transpiling a deep prefix per call
    # dominated the cost on no-ancilla multi-controlled circuits
    pre = _prefix_ucx(mqc, i)
    tkc = qiskit_to_tk(pre)
    tkc.add_assertion(ProjectorAssertionBox(P), [q], name=f"n{i}_q{q}")
    backend = _tk()
    # optimisation_level=0: gate-set rebase only. Level 2's rewrite passes
    # were observed to enter a pathological loop on some assertion circuits
    # (hours on a 3-qubit prefix); the rebase is linear-time and the
    # assertion semantics and sampling distribution are unchanged.
    compiled = backend.get_compiled_circuit(tkc, optimisation_level=0)
    result = backend.run_circuit(compiled, n_shots=shots, seed=seed)
    success = float(next(iter(result.get_debug_info().values())))
    failures = int(round((1.0 - success) * shots))
    return failures >= 1


def run_liu(mqc, nd, oracle, shots=SHOTS, seed=0):
    """Liu-Byrd-Zhou: the authors' assertion circuit (classical or
    arbitrary-superposition ancilla sandwich) appended to the prefix and
    REALLY run on Aer; any nonzero ancilla reading flags."""
    from baseline_liu import classical_assertion, superposition_assertion
    i, q = nd
    pre = _prefix(mqc, i)
    if oracle.deterministic(nd):
        value = 0 if oracle.p1e[nd] < 0.5 else 1
        ok = classical_assertion(pre, [pre.qubits[q]], value)
    else:
        theta, phi = oracle.liu_params(nd)
        ok = superposition_assertion(pre, [pre.qubits[q]],
                                     {pre.qubits[q]: (theta, phi)}, 1)
    if not ok:
        return False
    counts = _aer_run(pre, shots, seed)
    # the assertion creg is the last register added -> FIRST token in the key
    failures = sum(v for k, v in counts.items()
                   if set(k.split(" ")[0]) != {"0"})
    return failures >= 1


MQT_STATS = {"timeout": 0, "error": 0, "fail": 0, "unevaluable": 0,
             "pass": 0}


class MQTMutantBudget:
    """Fresh wall/timeout allowance for exactly one mutant/mode/method run."""

    def __init__(self, timeout_limit=MQT_TIMEOUT_BUDGET,
                 wall_limit=MQT_WALL_BUDGET):
        self.timeout_limit = int(timeout_limit)
        self.wall_limit = float(wall_limit)
        self.timeouts = 0
        self.wall_seconds = 0.0

    def can_call(self):
        return (self.timeouts < self.timeout_limit
                and self.wall_seconds < self.wall_limit)

    def record(self, status, elapsed):
        self.wall_seconds += float(elapsed)
        if status == "timeout":
            self.timeouts += 1

    def snapshot(self):
        return {
            "timeout_limit": self.timeout_limit,
            "wall_limit": self.wall_limit,
            "timeouts": self.timeouts,
            "wall_seconds": self.wall_seconds,
            "remaining": self.can_call(),
        }


def _mqt_qasm(circ, nd, amps):
    """Extended-OpenQASM for the debugger, rewritten into the tool's
    EMPIRICALLY SUPPORTED basis. The DD executor cannot run y/rx/rz/u/u2/u3/
    sx/rzz/rxx: such gates raise the same generic RuntimeError as a genuine
    sub-state refusal, which would hand MQT spurious 'detections' on the
    mutant and spurious inapplicability on the original. We transpile to
    u/cx exactly, then expand each u(theta,phi,lam) as p(lam);ry(theta);p(phi)
    (equal up to global phase, which the tool's similarity check ignores),
    leaving only p/ry/cx, all supported."""
    from qiskit import qasm2, QuantumCircuit as _QC
    from baseline_mqtdebugger import _fmt
    i, q = nd
    pre = _prefix_ucx(circ, i)
    out = _QC(pre.num_qubits)
    for ci in pre.data:
        qs = [pre.find_bit(b).index for b in ci.qubits]
        if ci.operation.name == "u":
            th, ph, lam = [float(x) for x in ci.operation.params]
            out.p(lam, qs[0])
            out.ry(th, qs[0])
            out.p(ph, qs[0])
        elif ci.operation.name == "cx":
            out.cx(qs[0], qs[1])
        else:                        # barriers etc. never reach here
            out.append(ci.operation, qs)
    body = qasm2.dumps(out)
    line = (f"assert-eq {MQT_SIMILARITY}, q[{q}] "
            f"{{ {_fmt(amps[0])}, {_fmt(amps[1])} }}")
    return body + "\n" + line + "\n"


def _mqt_call(qasm_str, timeout):
    """Run one debugger evaluation in a subprocess (the DD core holds the GIL,
    so only a process boundary gives a hard timeout). Returns a status in
    {pass, fail, unevaluable, timeout, error} and tallies MQT_STATS."""
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, __file__, "--mqt-worker"],
            input=qasm_str.encode(), capture_output=True, timeout=timeout)
        status = r.stdout.decode().strip().splitlines()[-1] \
            if r.stdout.strip() else "error"
        if status not in MQT_STATS:
            status = "error"
        # the DD core RAISES a generic message but PRINTS the informative
        # refusal to stderr, in two guises: 'entangled sub-state is not
        # allowed' and 'No valid index found' (the sub-state extraction
        # failing on a deviated program). With the prefix rewritten to the
        # supported basis, both are genuine can't-assert-this-state refusals.
        if status == "error":
            se = r.stderr.decode()
            if "entangled sub-state" in se or "No valid index" in se:
                status = "unevaluable"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception:
        status = "error"
    MQT_STATS[status] += 1
    return status


def run_mqt_status(mqc, nd, oracle, timeout=MQT_RUN_TIMEOUT):
    return _mqt_call(_mqt_qasm(mqc, nd, oracle.amps[nd]), timeout=timeout)


def run_mqt(mqc, nd, oracle, timeout=MQT_RUN_TIMEOUT):
    """MQT debugger (official DD backend), exact: assert-eq on the mutant
    prefix; a failed assertion OR the tool's genuine 'entangled sub-state'
    refusal both flag (at an applicable node the original was evaluable, so a
    mutant-side refusal means the monitored sub-state no longer has the
    separable form the oracle asserts: a state deviation). Tool errors and
    timeouts are scored as not flagged and tallied for disclosure."""
    status = run_mqt_status(mqc, nd, oracle, timeout=timeout)
    return status in ("fail", "unevaluable")


# ==========================================================================
# QMon: exact branch-mixture execution of the GENERATED monitoring circuit.
# ==========================================================================
_SEG_SIM = None


def _aer_evolve(vec, seg):
    """Evolve a statevector through a unitary segment with Aer's C++ core
    (set_statevector avoids state-prep synthesis; one round trip per segment;
    ~20x faster than per-gate numpy evolution on 24-qubit monitoring circuits).
    Falls back to a u/cx transpile if a segment gate is not Aer-native."""
    global _SEG_SIM
    if _SEG_SIM is None:
        from qiskit_aer import AerSimulator
        _SEG_SIM = AerSimulator(method="statevector",
                                max_parallel_threads=1)
    n = seg.num_qubits
    c = QuantumCircuit(n)
    c.set_statevector(vec)
    c.compose(seg, inplace=True)
    c.save_statevector()
    try:
        res = _SEG_SIM.run(c, shots=1).result()
    except Exception:
        from qiskit import transpile
        c = QuantumCircuit(n)
        c.set_statevector(vec)
        c.compose(transpile(seg, basis_gates=["u", "cx"],
                            optimization_level=0), inplace=True)
        c.save_statevector()
        res = _SEG_SIM.run(c, shots=1).result()
    return np.asarray(res.get_statevector()).ravel()


def _joint_measurement_plan(mon):
    """Split a generated dynamic circuit into unitary segments and mid reads."""
    mid = {}
    consumed_resets = set()
    final_measure_indices = []
    for index, ci in enumerate(mon.data):
        if ci.operation.name != "measure":
            continue
        qubit = mon.find_bit(ci.qubits[0]).index
        next_index = None
        next_name = None
        for later in range(index + 1, len(mon.data)):
            candidate = mon.data[later]
            if candidate.operation.name in ("barrier", "delay", "snapshot"):
                continue
            if any(mon.find_bit(bit).index == qubit
                   for bit in candidate.qubits):
                next_index = later
                next_name = candidate.operation.name
                break
        if next_name == "reset":
            mid[index] = qubit
            consumed_resets.add(next_index)
        else:
            final_measure_indices.append(index)

    reasons = []
    if final_measure_indices:
        first_final = min(final_measure_indices)
        for index in range(first_final + 1, len(mon.data)):
            ci = mon.data[index]
            if ci.operation.name not in (
                "measure", "barrier", "delay", "snapshot"
            ):
                reasons.append("nonterminal-final-measurement")
                break

    plan = []
    segment = QuantumCircuit(mon.num_qubits)
    for index, ci in enumerate(mon.data):
        name = ci.operation.name
        if name in ("barrier", "delay", "snapshot"):
            continue
        if name == "measure":
            if index in mid:
                if segment.data:
                    plan.append(("segment", segment))
                    segment = QuantumCircuit(mon.num_qubits)
                plan.append(("mid", mid[index]))
            continue
        if name == "reset":
            if index not in consumed_resets:
                reasons.append(f"unsupported-reset:{index}")
            continue
        if getattr(ci.operation, "condition", None) is not None:
            reasons.append(f"conditional-operation:{index}")
            continue
        segment.append(
            ci.operation,
            [mon.find_bit(bit).index for bit in ci.qubits],
        )
    if segment.data:
        plan.append(("segment", segment))
    return plan, [mid[index] for index in sorted(mid)], sorted(set(reasons))


def _reset_embedding(substate, n, axis):
    norm = float(np.linalg.norm(substate))
    if norm < BRANCH_STATE_NORM_EPS:
        return None
    output = np.zeros((2,) * n, dtype=complex)
    target = [slice(None)] * n
    target[axis] = 0
    output[tuple(target)] = (substate / norm).reshape((2,) * (n - 1))
    return output.reshape(-1)


def _joint_qubit_probabilities(state, n, qubits):
    """Joint Z distribution in the exact order supplied by ``qubits``."""
    qubits = list(qubits)
    if not qubits:
        return np.array([1.0])
    axes = [n - 1 - int(qubit) for qubit in qubits]
    if len(axes) != len(set(axes)):
        raise ValueError(f"final joint sample contains duplicate qubits: {qubits}")
    probabilities = np.abs(state) ** 2
    array = probabilities.reshape((2,) * n)
    keep = sorted(axes)
    drop = tuple(axis for axis in range(n) if axis not in set(axes))
    marginal = array.sum(axis=drop) if drop else array
    permutation = [keep.index(axis) for axis in axes]
    if len(permutation) > 1:
        marginal = np.transpose(marginal, axes=permutation)
    flat = np.clip(marginal.reshape(-1), 0.0, None)
    total = float(flat.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid final joint probability distribution")
    return flat / total


def _node_label(node):
    return f"{int(node[0])}:{int(node[1])}"


def sample_joint_trajectory(mon, *, mid_nodes=(), final_nodes=(),
                            shots=SHOTS, seed=0, cap=BRANCH_CAP):
    """Sample one instrumented QMon batch as aligned physical shot trajectories.

    Every returned column has ``shots`` entries and columns share row identity:
    row ``s`` is the complete mid-measurement path (plus jointly sampled final
    measurements) of physical shot ``s``.  Entangled mid reads split trajectory
    groups; separable measure/reset paths converge and continue as one state.

    The return value always carries ``complete``.  A branch-cap, read mapping,
    unsupported dynamic operation, or final-sampling failure returns
    ``complete=False`` and must be excluded from rate denominators.
    """
    shots = int(shots)
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    if cap is not None and int(cap) <= 0:
        raise ValueError(f"branch cap must be positive, got {cap}")
    cap = None if cap is None else int(cap)
    mid_nodes = [tuple(node) for node in mid_nodes]
    final_nodes = [tuple(node) for node in final_nodes]
    plan, event_qubits, reasons = _joint_measurement_plan(mon)

    queues = {}
    for node in sorted(mid_nodes):
        queues.setdefault(int(node[1]), []).append(node)
    event_nodes = []
    for event_index, qubit in enumerate(event_qubits):
        queue = queues.get(qubit, [])
        if not queue:
            reasons.append(f"unmapped-mid-read:{event_index}:q{qubit}")
            event_nodes.append(None)
        else:
            event_nodes.append(queue.pop(0))
    for qubit, queue in queues.items():
        for node in queue:
            reasons.append(f"missing-mid-read:{_node_label(node)}:q{qubit}")

    columns = {node: np.zeros(shots, dtype=np.uint8)
               for node in mid_nodes + final_nodes}
    residuals = {}
    diagnostics = {
        "cap": cap,
        "cap_hits": 0,
        "max_live_branches": 1,
        "mid_events": len(event_qubits),
        "final_qubits": len({node[1] for node in final_nodes}),
        "incomplete_reasons": list(dict.fromkeys(reasons)),
    }
    if reasons:
        return {
            "complete": False,
            "shots": shots,
            "shot_table": columns,
            "counts": {},
            "event_order": [node for node in event_nodes if node is not None]
            + final_nodes,
            "residuals": residuals,
            "diagnostics": diagnostics,
            "shot_table_sha256": None,
        }

    n = mon.num_qubits
    initial = np.zeros(2 ** n, dtype=complex)
    initial[0] = 1.0
    groups = [(np.arange(shots, dtype=np.int32), initial)]
    rng = np.random.default_rng(seed)
    event_offset = 0
    complete = True

    for kind, payload in plan:
        if kind == "segment":
            try:
                groups = [(indices, _aer_evolve(state, payload))
                          for indices, state in groups]
            except Exception as exc:
                diagnostics["incomplete_reasons"].append(
                    f"segment-error:{type(exc).__name__}:{str(exc)[:160]}"
                )
                complete = False
                break
            continue

        qubit = int(payload)
        node = event_nodes[event_offset]
        event_offset += 1
        axis = n - 1 - qubit
        next_groups = []
        max_residual = 0.0
        for indices, state in groups:
            array = state.reshape((2,) * n)
            state0 = np.take(array, 0, axis=axis).reshape(-1)
            state1 = np.take(array, 1, axis=axis).reshape(-1)
            p0 = float(np.vdot(state0, state0).real)
            p1 = min(max(float(np.vdot(state1, state1).real), 0.0), 1.0)
            residual = p0 * p1 - abs(np.vdot(state0, state1)) ** 2
            max_residual = max(max_residual, float(residual))
            outcomes = (rng.random(len(indices)) < p1).astype(np.uint8)
            columns[node][indices] = outcomes
            if residual <= SEP_TOL:
                representative = state0 if p0 >= p1 else state1
                embedded = _reset_embedding(representative, n, axis)
                if embedded is None:
                    diagnostics["incomplete_reasons"].append(
                        f"reset-embedding-failure:{_node_label(node)}"
                    )
                    complete = False
                    break
                next_groups.append((indices, embedded))
                continue
            for outcome, substate in ((0, state0), (1, state1)):
                selected = indices[outcomes == outcome]
                if not len(selected):
                    continue
                embedded = _reset_embedding(substate, n, axis)
                if embedded is None:
                    diagnostics["incomplete_reasons"].append(
                        f"branch-embedding-failure:{_node_label(node)}:{outcome}"
                    )
                    complete = False
                    break
                next_groups.append((selected, embedded))
            if not complete:
                break
        residuals[node] = max_residual
        if not complete:
            break
        if cap is not None and len(next_groups) > cap:
            diagnostics["cap_hits"] += 1
            diagnostics["incomplete_reasons"].append(
                f"branch-cap:{len(next_groups)}>{cap}@{_node_label(node)}"
            )
            complete = False
            break
        groups = next_groups
        diagnostics["max_live_branches"] = max(
            diagnostics["max_live_branches"], len(groups)
        )

    if complete and final_nodes:
        final_qubits = []
        for node in final_nodes:
            if int(node[1]) not in final_qubits:
                final_qubits.append(int(node[1]))
        by_qubit = {qubit: np.zeros(shots, dtype=np.uint8)
                    for qubit in final_qubits}
        try:
            for indices, state in groups:
                probabilities = _joint_qubit_probabilities(
                    state, n, final_qubits
                )
                outcomes = rng.choice(
                    len(probabilities), size=len(indices), p=probabilities
                )
                width = len(final_qubits)
                for position, qubit in enumerate(final_qubits):
                    by_qubit[qubit][indices] = (
                        (outcomes >> (width - 1 - position)) & 1
                    ).astype(np.uint8)
            for node in final_nodes:
                columns[node] = by_qubit[int(node[1])]
        except Exception as exc:
            diagnostics["incomplete_reasons"].append(
                f"final-sample-error:{type(exc).__name__}:{str(exc)[:160]}"
            )
            complete = False

    diagnostics["incomplete_reasons"] = list(
        dict.fromkeys(diagnostics["incomplete_reasons"])
    )
    event_order = [node for node in event_nodes if node is not None] + final_nodes
    table_hash = None
    counts = {}
    if complete:
        digest = hashlib.sha256()
        for node in event_order:
            digest.update(_node_label(node).encode("ascii"))
            digest.update(np.packbits(columns[node]).tobytes())
            counts[node] = int(columns[node].sum())
        table_hash = digest.hexdigest()
    return {
        "complete": complete,
        "shots": shots,
        "shot_table": columns,
        "counts": counts,
        "event_order": event_order,
        "residuals": residuals,
        "diagnostics": diagnostics,
        "shot_table_sha256": table_hash,
    }


def _mid_reads_branch_exact(mon, sep_tol=SEP_TOL, cap=BRANCH_CAP,
                            diag=None, final_qubits=None):
    """Exact mixture semantics of the monitoring circuit's mid-measurements:
    evolve a weighted set of statevector branches; a mid-measure (measure
    followed by reset on the same qubit) records p1 = sum_b w_b p1_b and
    SPLITS the branch only when the computed determinant residual exceeds
    ``sep_tol``; branches at or below that numerical convergence cutoff are
    merged after reset. This implementation cutoff is distinct from both the
    selector's Schmidt-amplitude tolerance and the theorem's exact-zero
    premise. Returns
    [(qubit, p1, max_residual), ...] in program order. `diag` (dict)
    accumulates cap-truncation events and the total discarded weight."""
    n = mon.num_qubits
    if diag is not None:
        diag.setdefault("complete", True)
        diag.setdefault("incomplete_reasons", [])
    final_qubits = final_qubits or set()
    # mid-measure = measure whose next op on the qubit is reset
    is_mid = {}
    for i, ci in enumerate(mon.data):
        if ci.operation.name != "measure":
            continue
        q = mon.find_bit(ci.qubits[0]).index
        nxt = None
        for j in range(i + 1, len(mon.data)):
            cj = mon.data[j]
            if cj.operation.name in ("barrier", "delay", "snapshot"):
                continue
            if any(mon.find_bit(qb).index == q for qb in cj.qubits):
                nxt = cj.operation.name
                break
        is_mid[i] = (nxt == "reset")
    last_mid = max((i for i, v in is_mid.items() if v), default=-1)
    if final_qubits:
        # evolve past the last mid-measure through the remaining unitaries so
        # the final-measurement marginals of the requested qubits can be read
        # (projective Z-measurements of other qubits do not change them)
        last_mid = len(mon.data) - 1

    state0 = np.zeros(2 ** n, dtype=complex)
    state0[0] = 1.0
    branches = [(1.0, state0)]
    reads = []
    # unitary stretches between mid-measures are evolved as WHOLE SEGMENTS by
    # Aer's C++ statevector core (one round trip per segment, ~20x faster
    # than per-gate numpy evolution on 24-qubit monitoring circuits); the
    # measure/collapse/reset bookkeeping stays in numpy and is unchanged
    plan = []
    seg = QuantumCircuit(n)
    for i, ci in enumerate(mon.data):
        if i > last_mid:
            break
        name = ci.operation.name
        if name in ("barrier", "delay", "snapshot"):
            continue
        if name == "measure":
            if not is_mid[i]:
                continue    # final measures do not change any Z-marginal
            if len(seg.data):
                plan.append(("seg", seg))
                seg = QuantumCircuit(n)
            plan.append(("mid", mon.find_bit(ci.qubits[0]).index))
            continue
        if name == "reset":
            continue        # consumed jointly with its mid-measure below
        seg.append(ci.operation,
                   [mon.find_bit(qb).index for qb in ci.qubits])
    if len(seg.data):
        plan.append(("seg", seg))

    for kind, payload in plan:
        if kind == "seg":
            branches = [(w, _aer_evolve(vec, payload))
                        for w, vec in branches]
            continue
        q = payload
        axis = n - 1 - q
        p1_mix, res_max = 0.0, 0.0
        new_branches = []
        for w, vec in branches:
            arr = vec.reshape((2,) * n)
            s0 = np.take(arr, 0, axis=axis).reshape(-1)
            s1 = np.take(arr, 1, axis=axis).reshape(-1)
            p0 = float(np.vdot(s0, s0).real)
            p1 = float(np.vdot(s1, s1).real)
            res = p0 * p1 - abs(np.vdot(s0, s1)) ** 2
            p1_mix += w * p1
            res_max = max(res_max, res)
            # measure + the immediately following reset, jointly:
            def _embed(sub):
                nrm = np.linalg.norm(sub)
                if nrm < BRANCH_STATE_NORM_EPS:
                    return None
                out = np.zeros((2,) * n, dtype=complex)
                idx = [slice(None)] * n
                idx[axis] = 0
                out[tuple(idx)] = (sub / nrm).reshape((2,) * (n - 1))
                return out.reshape(-1)
            if res <= sep_tol:
                keep = _embed(s0 if p0 >= p1 else s1)
                new_branches.append((w, keep))
            else:
                for pb, sb in ((p0, s0), (p1, s1)):
                    emb = _embed(sb)
                    if emb is not None and w * pb > BRANCH_WEIGHT_EPS:
                        new_branches.append((w * pb, emb))
        new_branches.sort(key=lambda t: -t[0])
        if len(new_branches) > cap:
            dropped = sum(w for w, _ in new_branches[cap:])
            if diag is not None:
                diag["cap_hits"] = diag.get("cap_hits", 0) + 1
                diag["dropped_w"] = max(diag.get("dropped_w", 0.0), dropped)
                diag["complete"] = False
                diag["incomplete_reasons"].append(
                    f"branch-cap:{len(new_branches)}>{cap}:q{q}")
            new_branches = new_branches[:cap]
            tot = sum(w for w, _ in new_branches)
            new_branches = [(w / tot, v) for w, v in new_branches]
        branches = new_branches
        reads.append((q, min(max(p1_mix, 0.0), 1.0), res_max))
    finals = {}
    for q in final_qubits:
        axis = n - 1 - q
        p1 = 0.0
        for w, vec in branches:
            arr = vec.reshape((2,) * n)
            s1 = np.take(arr, 1, axis=axis).reshape(-1)
            p1 += w * float(np.vdot(s1, s1).real)
        finals[q] = min(max(p1, 0.0), 1.0)
    return reads, finals


def qmon_reads_for_batch(mqc, key, oracle, batch_nodes, raw_p1, raw_res,
                         mut_gate, diag=None):
    """Per-node mid-read p1 for one QMon batch executed on the mutant.

    The monitoring circuit must be branch-exactly simulated whenever an
    earlier node A of the batch can change what a later node B reads. Two
    mechanisms:
      (a) entanglement broken at A: the measure+reset collapse back-acts on
          the rest of the register;
      (b) HEALING: A's reset+replay re-prepares its qubit from the ORIGINAL
          circuit's plan, so if the mutated gate lies in A's backward causal
          cone (A's state deviates), the re-preparation replaces the deviated
          state with the oracle state and B's read differs from the raw
          mutant marginal even with zero entanglement residual.
    Only when neither mechanism can act (no such A precedes any B) does
    Theorem 1 give read = raw marginal."""
    nds = sorted(nd for nd in oracle.emitted if nd[0] in set(batch_nodes))
    disturbing = [nd for nd in nds
                  if raw_res[nd] > SEP_TOL or mut_gate in oracle.cone[nd]]
    # a later node B can only read something other than its raw marginal if
    # it is itself cone-affected: for g outside cone[B], upstream collapse
    # cannot shift B's marginal (no-signaling) and upstream healing cannot
    # reach B (cone transitivity: if A's qubit influenced B, cone[A] would be
    # inside cone[B]); so simulate only when an affected node FOLLOWS a
    # disturbing one
    need_sim = bool(disturbing) and any(
        (mut_gate in oracle.cone[nd])
        and (nd[0] > min(d[0] for d in disturbing)
             or (nd[0] == min(d[0] for d in disturbing)
                 and nd not in disturbing))
        for nd in nds)
    if not need_sim:
        return {nd: raw_p1[nd] for nd in nds}, {nd: raw_res[nd] for nd in nds}
    if diag is not None:
        diag["sim_batches"] = diag.get("sim_batches", 0) + 1
    ans = {"picked_nodes": set(batch_nodes)}
    with _mute():
        mon = generate_monitoring_circuit(
            mqc, key, oracle.op_list,
            INSTRUMENTATION_INTERNAL_QUBIT_LIMIT, ans
        )
    reads, _ = _mid_reads_branch_exact(mon, diag=diag)
    by_q = {}
    for nd in nds:
        by_q.setdefault(nd[1], []).append(nd)
    got_p1, got_res = {}, {}
    seen = {}
    for q, p1, res in reads:
        k = seen.get(q, 0)
        seen[q] = k + 1
        if q in by_q and k < len(by_q[q]):
            nd = by_q[q][k]
            got_p1[nd], got_res[nd] = p1, res
    for nd in nds:
        if nd not in got_p1:            # mapping failure: log, then fall back
            if diag is not None:
                diag["read_fallbacks"] = diag.get("read_fallbacks", 0) + 1
                diag["complete"] = False
                diag.setdefault("incomplete_reasons", []).append(
                    f"read-fallback:{nd}")
            got_p1[nd], got_res[nd] = raw_p1[nd], raw_res[nd]
    return got_p1, got_res


def qmon_node_flag(nd, p1_read, oracle, shots=SHOTS, seed=0,
                   alpha_node=ALPHA):
    """Draw the node's S-shot count from its actual mid-read distribution and
    apply QMon's decision: any-flip at a deterministic expected value, exact
    binomial test otherwise."""
    rng = np.random.default_rng(seed)
    k = int(rng.binomial(shots, min(max(p1_read, 0.0), 1.0)))
    return qmon_count_flag(nd, k, oracle, shots=shots,
                           alpha_node=alpha_node)


def qmon_count_flag(nd, count1, oracle, shots=SHOTS, alpha_node=ALPHA):
    """Apply QMon's node decision to a count from an aligned batch table."""
    k = int(count1)
    if not 0 <= k <= int(shots):
        raise ValueError(f"invalid |1> count {k} for {shots} shots")
    p1 = oracle.p1e[nd]
    if oracle.deterministic(nd):
        wrong = k if p1 < 0.5 else shots - k
        return wrong > 0
    pvalue = binomtest(
        k, shots, p1, alternative="two-sided"
    ).pvalue
    return bool(pvalue < alpha_node)


# ==========================================================================
# Raw mutant marginals + residuals at all checkpoints (one pass).
# ==========================================================================
def raw_reads(mqc, nodes, n):
    by_gate = {}
    for (i, q) in nodes:
        by_gate.setdefault(i, set()).add(q)
    from qiskit.quantum_info import Statevector
    state = Statevector.from_int(0, dims=2 ** n)
    p1_out, res_out = {}, {}
    for j, ci in enumerate(mqc.data):
        if ci.operation.name in SKIP_OPS:
            continue
        state = state.evolve(ci.operation,
                             qargs=[mqc.find_bit(b).index for b in ci.qubits])
        if j in by_gate:
            arr = state.data.reshape((2,) * n)
            for q in by_gate[j]:
                axis = n - 1 - q
                s0 = np.take(arr, 0, axis=axis).reshape(-1)
                s1 = np.take(arr, 1, axis=axis).reshape(-1)
                p0 = float(np.vdot(s0, s0).real)
                p1 = float(np.vdot(s1, s1).real)
                p1_out[(j, q)] = p1
                res_out[(j, q)] = p0 * p1 - abs(np.vdot(s0, s1)) ** 2
    return p1_out, res_out


# ==========================================================================
# Scan: ascending checkpoints, early exit at the first flag (which is also
# the localization cone). Returns (detected, first_node or None, n_checked).
# ==========================================================================
def _null_statistical_flag(nd, oracle, seed, alpha_node):
    """At a node whose backward cone excludes the mutated gate, the reduced
    state (hence the Z-count distribution) is EXACTLY the original's
    (Prop. 2), so Aer counts are Binomial(S, p1e): drawing from that
    distribution and applying the same test is distribution-identical to
    running the circuit."""
    p1 = oracle.p1e[nd]
    if oracle.deterministic(nd):
        return False                     # draws land in the expected bin
    rng = np.random.default_rng(seed)
    c1 = int(rng.binomial(SHOTS, p1))
    f_exp = np.array([(1 - p1) * SHOTS, p1 * SHOTS])
    if f_exp.min() >= CHI_SQUARE_MIN_EXPECTED_COUNT:
        _, pval = chisquare([SHOTS - c1, c1], f_exp)
    else:
        pval = binomtest(
            c1, SHOTS, p1, alternative="two-sided"
        ).pvalue
    return bool(pval < alpha_node)


def scan_method(method, mqc, oracle, mut_tag, g, qmon_ctx=None):
    """Ascending-index scan with early exit; returns (detected, first_node,
    stats). QMon scans its DEPLOYED checkpoints (Bonferroni over K_qmon), the
    baselines scan the full monitorable set (Bonferroni over K where they
    need one). Nodes whose cone excludes the mutated gate g have EXACTLY the
    original reduced state there (Prop. 2): their per-shot outcome
    distributions are identical to the correct run's, so the sampling-based
    decisions are drawn from that exact distribution (statistical/qmon read
    the plan value) and the zero-background decisions (proq/liu: failure
    probability exactly 0; mqt: exact tool, assertion passes) never fire;
    only cone-affected nodes require executing the instrumented circuit."""
    family_size = (
        oracle.K_qmon if method in ("qmon", "qmon_pernode")
        else oracle.K if method == "statistical" else None
    )
    stats = {
        "n_exec": 0,
        "n_null_shortcuts": 0,
        "n_analytical_marginals": 0,
        "complete": True,
        "incomplete_reasons": [],
        "family_size": family_size,
        "alpha_node": (
            ALPHA / max(family_size, 1) if family_size is not None else None),
        "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[method],
        "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[method],
        "execution_mode": MUTANT_EXECUTION_MODE[method],
    }
    node_set = (oracle.qmon_nodes if method in ("qmon", "qmon_pernode")
                else oracle.nodes)
    mqt_budget = MQTMutantBudget() if method == "mqt" else None
    if method == "qmon":
        if qmon_ctx is None or not qmon_ctx["materialize"]():
            stats["complete"] = False
            stats["incomplete_reasons"] = list(
                (qmon_ctx or {"diag": {}})["diag"].get(
                    "incomplete_reasons", ["missing-qmon-context"]
                )
            )
            return None, None, stats
        stats["n_exec"] = int(oracle.B or 0)
    for nd in node_set:
        seed = _seed32(oracle.key, mut_tag, nd[0], nd[1], method)
        affected = g in oracle.cone[nd]
        if method == "qmon":
            try:
                count1 = qmon_ctx["counts"](nd)
            except JointTrajectoryIncomplete as exc:
                stats["complete"] = False
                stats["incomplete_reasons"].append(str(exc))
                return None, None, stats
            f = qmon_count_flag(
                nd, count1, oracle, shots=SHOTS,
                alpha_node=ALPHA / max(oracle.K_qmon, 1)
            )
        elif method == "qmon_pernode":
            # one monitor per run (cost K_qmon, same as the baselines'
            # per-checkpoint model): no earlier same-run monitor exists, so
            # the mid-read is the raw mutant marginal (its statistics equal
            # the marginal even at an entangled node; nothing follows the
            # read in this run); this isolates the batching/healing trade-off
            p1_read = (qmon_ctx["raw_p1"][nd] if affected
                       else oracle.p1e[nd])
            stats["n_analytical_marginals"] += 1
            if not affected:
                stats["n_null_shortcuts"] += 1
            f = qmon_node_flag(nd, p1_read, oracle, seed=seed,
                               alpha_node=ALPHA / max(oracle.K_qmon, 1))
        elif method == "statistical":
            if affected:
                stats["n_exec"] += 1
                f = run_statistical(mqc, nd, oracle, seed=seed,
                                    alpha_node=ALPHA / oracle.K)
            else:
                stats["n_null_shortcuts"] += 1
                f = _null_statistical_flag(nd, oracle, seed,
                                           ALPHA / oracle.K)
        elif method == "proq":
            if affected:
                stats["n_exec"] += 1
                f = run_proq(mqc, nd, oracle, seed=seed)
            else:
                stats["n_null_shortcuts"] += 1
                f = False
        elif method == "liu":
            if affected:
                stats["n_exec"] += 1
                f = run_liu(mqc, nd, oracle, seed=seed)
            else:
                stats["n_null_shortcuts"] += 1
                f = False
        elif method == "mqt":
            if not affected:
                stats["n_null_shortcuts"] += 1
                continue    # exact tool, unchanged state: assertion passes
            stats["n_affected"] = stats.get("n_affected", 0) + 1
            applicability = oracle.mqt_applicability_status(nd, mqt_budget)
            stats[f"applicability_{applicability}"] = (
                stats.get(f"applicability_{applicability}", 0) + 1
            )
            if applicability in ("timeout", "error", "budget-exhausted"):
                stats["complete"] = False
                stats["incomplete_reasons"].append(
                    f"applicability-{applicability}:{nd}"
                )
                continue
            if applicability != "pass":
                continue
            stats["n_applicable"] = stats.get("n_applicable", 0) + 1
            if not mqt_budget.can_call():
                stats["complete"] = False
                stats["budget_exhausted"] = stats.get("budget_exhausted", 0) + 1
                stats["incomplete_reasons"].append(f"run-budget-exhausted:{nd}")
                continue
            stats["n_exec"] += 1
            started = time.perf_counter()
            status = run_mqt_status(mqc, nd, oracle)
            mqt_budget.record(status, time.perf_counter() - started)
            stats[f"run_{status}"] = stats.get(f"run_{status}", 0) + 1
            if status in ("timeout", "error"):
                stats["complete"] = False
                stats[status] = stats.get(status, 0) + 1
                stats["incomplete_reasons"].append(f"run-{status}:{nd}")
                f = False
            else:
                f = status in ("fail", "unevaluable")
        else:
            raise ValueError(method)
        if f:
            if mqt_budget is not None:
                stats["budget"] = mqt_budget.snapshot()
            stats["incomplete_reasons"] = list(
                dict.fromkeys(stats["incomplete_reasons"])
            )
            return True, nd, stats
    if mqt_budget is not None:
        stats["budget"] = mqt_budget.snapshot()
    stats["incomplete_reasons"] = list(
        dict.fromkeys(stats["incomplete_reasons"])
    )
    return False, None, stats


def make_qmon_joint_ctx(mqc, key, oracle, sample_tag, *, shots=SHOTS,
                        include_raw=True):
    """Create one aligned-shot provider for every physical QMon batch.

    Batch 0 also supplies all final-read nodes in one joint final sample.  A
    batch is seeded as a whole, so checkpoints in the batch share trajectories;
    distinct batch IDs receive independent RNG streams.
    """
    raw_p1, raw_res = ({}, {})
    if include_raw:
        raw_p1, raw_res = raw_reads(
            mqc, oracle.nodes, oracle.qc.num_qubits
        )
    cache = {}
    diag = {
        "complete": True,
        "incomplete_reasons": [],
        "plan_sha256": oracle.qmon_plan["plan_sha256"],
        "batches": [],
    }
    final_set = set(oracle.lastgate)

    def batch_result(batch_id):
        if batch_id in cache:
            return cache[batch_id]
        batch = oracle.batches[batch_id]
        picked = set(batch["nodes"])
        nodes = sorted(
            node for node in oracle.emitted if node[0] in picked
        )
        finals = list(oracle.lastgate) if batch_id == 0 else []
        answer = {"picked_nodes": picked}
        try:
            with _mute():
                monitored = generate_monitoring_circuit(
                    mqc, key, oracle.op_list,
                    INSTRUMENTATION_INTERNAL_QUBIT_LIMIT, answer
                )
            if monitored.num_qubits > oracle.qmax:
                result = {
                    "complete": False,
                    "counts": {},
                    "shot_table": {},
                    "shot_table_sha256": None,
                    "diagnostics": {
                        "incomplete_reasons": [
                            f"qmax:{monitored.num_qubits}>{oracle.qmax}"
                        ],
                        "cap_hits": 0,
                    },
                }
            else:
                result = sample_joint_trajectory(
                    monitored,
                    mid_nodes=nodes,
                    final_nodes=finals,
                    shots=shots,
                    seed=_seed32(
                        oracle.key, sample_tag, "qmon-batch", batch_id
                    ),
                )
        except Exception as exc:
            result = {
                "complete": False,
                "counts": {},
                "shot_table": {},
                "shot_table_sha256": None,
                "diagnostics": {
                    "incomplete_reasons": [
                        f"batch-error:{type(exc).__name__}:{str(exc)[:200]}"
                    ],
                    "cap_hits": 0,
                },
            }
            monitored = None
        entry = {
            "batch": int(batch_id),
            "nodes": tuple(int(node) for node in batch["nodes"]),
            "cost": int(batch["cost"]),
            "emitted_qubits": (
                int(monitored.num_qubits) if monitored is not None else None
            ),
            "monitor_groups": (
                tuple(monitored.metadata.get("qmon_monitor_groups", ()))
                if monitored is not None else ()
            ),
            "replay_fail_closed": (
                bool(monitored.metadata.get("qmon_replay_fail_closed", False))
                if monitored is not None else False
            ),
            "complete": bool(result["complete"]),
            "shot_table_sha256": result.get("shot_table_sha256"),
            "incomplete_reasons": tuple(
                result.get("diagnostics", {}).get("incomplete_reasons", ())
            ),
        }
        diag["batches"].append(entry)
        diag["cap_hits"] = diag.get("cap_hits", 0) + int(
            result.get("diagnostics", {}).get("cap_hits", 0)
        )
        if not result["complete"]:
            diag["complete"] = False
            diag["incomplete_reasons"].extend(entry["incomplete_reasons"])
        cache[batch_id] = result
        return result

    def materialize():
        for batch_id in range(len(oracle.batches or [])):
            batch_result(batch_id)
        diag["batches"].sort(key=lambda item: item["batch"])
        diag["incomplete_reasons"] = list(
            dict.fromkeys(diag["incomplete_reasons"])
        )
        return bool(diag["complete"])

    def counts(node):
        node = tuple(node)
        batch_id = 0 if node in final_set else oracle.batch_of[node[0]]
        result = batch_result(batch_id)
        if not result["complete"] or node not in result["counts"]:
            diag["complete"] = False
            reason = f"count-unavailable:{node}:batch{batch_id}"
            if reason not in diag["incomplete_reasons"]:
                diag["incomplete_reasons"].append(reason)
            raise JointTrajectoryIncomplete(reason)
        return int(result["counts"][node])

    def shot_column(node):
        node = tuple(node)
        batch_id = 0 if node in final_set else oracle.batch_of[node[0]]
        result = batch_result(batch_id)
        if not result["complete"] or node not in result["shot_table"]:
            raise JointTrajectoryIncomplete(
                f"shot-column-unavailable:{node}:batch{batch_id}"
            )
        return result["shot_table"][node]

    return {
        "counts": counts,
        "shot_column": shot_column,
        "materialize": materialize,
        "is_complete": lambda: bool(diag["complete"]),
        "raw_p1": raw_p1,
        "raw_res": raw_res,
        "diag": diag,
        "plan": oracle.qmon_plan,
    }


def make_qmon_ctx(mqc, key, oracle, mutation_identity):
    """Build a QMon context seeded by the unique mutation identity."""
    return make_qmon_joint_ctx(
        mqc,
        key,
        oracle,
        sample_tag=f"mutant-{mutation_identity}",
        include_raw=True,
    )


# ==========================================================================
# Mutant loop.  Slots and replacements are replayed from the fixed mutation
# list; no consumer redraws the mutation stream.
# ==========================================================================
def evaluate_circuit(key, seeds=(1, 2, 3, 4, 5), muts_per_seed=3, qmax=24,
                     methods=METHODS, manifest=None, slot_ids=None):
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(f"RQ3 requires Qmax={CANONICAL_QMAX}, got {qmax}")
    if manifest is None:
        canonical_mutation_slots(key)
        manifest = _CANONICAL_MANIFEST
    oracle = Oracle(key, qmax)
    if not oracle.eligible():
        return []
    qc = oracle.qc
    scope_signature = {
        "nodes": tuple(oracle.nodes),
        "qmon_nodes": tuple(oracle.qmon_nodes),
        "cone": oracle.cone,
    }
    records = []
    seed_set = {int(seed) for seed in seeds}
    slots = [
        slot for slot in canonical_mutation_slots(key, manifest)
        if int(slot["seed"]) in seed_set
        and int(slot["ordinal"]) < int(muts_per_seed)
    ]
    if slot_ids is None:
        expected_slots = len(seed_set) * min(int(muts_per_seed), 3)
        if len(slots) != expected_slots:
            raise ValueError(
                f"mutation list has {len(slots)} requested slots for "
                f"{key}; expected {expected_slots}"
            )
    else:
        requested = tuple(str(slot_id) for slot_id in slot_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("slot_ids must be nonempty and unique")
        slots = [slot for slot in slots if slot["slot_id"] in requested]
        actual = {slot["slot_id"] for slot in slots}
        if actual != set(requested):
            missing = sorted(set(requested) - actual)
            unexpected = sorted(actual - set(requested))
            raise ValueError(
                f"mutation list slot selection mismatch for {key}: "
                f"missing={missing}, unexpected={unexpected}"
            )
    for slot in slots:
            seed = int(slot["seed"])
            g = int(slot["gate"])
            mqc = apply_manifest_slot(qc, slot)
            equivalent = bool(slot["output_equivalent"])
            scope = _scope_from_oracle_signature(scope_signature, g)
            rec = {"circuit": key, "seed": seed,
                   "slot_id": slot["slot_id"],
                   "slot_ordinal": int(slot["ordinal"]),
                   "manifest_sha256": manifest["manifest_sha256"],
                   "mut_gate": g,
                   "original": slot["original"],
                   "replacement": slot["replacement"],
                   "mutant_qasm_sha256": slot["mutant_qasm_sha256"],
                   "K": oracle.K, "K_qmon": oracle.K_qmon, "B": oracle.B,
                   "equivalent": equivalent,
                   "output_l1": slot["output_l1"],
                   **scope,
                   "qmon_plan": oracle.qmon_plan,
                   "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
                   "pruning_scope_rule": PRUNING_SCOPE_RULE,
                   "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
                   "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
                   "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
                   "exact_binomial_convention": dict(
                       EXACT_BINOMIAL_CONVENTION),
                   "methods": {}}
            if not equivalent:
                mut_tag = slot["slot_id"]
                qmon_ctx = (make_qmon_ctx(
                    mqc, key, oracle, slot["slot_id"]
                )
                            if any(m in methods for m in
                                   ("qmon", "qmon_pernode")) else None)
                for m in methods:
                    detected, first, stats = scan_method(
                        m, mqc, oracle, mut_tag, g, qmon_ctx)
                    detected = bool(detected)
                    if not detected:
                        first = None
                    suspect = oracle.cone[first] if first else set()
                    loc_hit = g in suspect
                    rec["methods"][m] = {
                        "complete": bool(stats.get("complete", True)),
                        "asserted_subsystem": METHOD_ASSERTED_SUBSYSTEM[m],
                        "pruning_scope": PRUNING_SCOPE_RULE,
                        "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[m],
                        "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[m],
                        "execution_mode": MUTANT_EXECUTION_MODE[m],
                        "detected": detected,
                        "first_node": first,
                        "loc_hit": loc_hit,
                        "loc_precision": (1.0 / len(suspect)
                                          if (loc_hit and suspect) else 0.0),
                        "executed_program_count": int(stats.get("n_exec", 0)),
                        "counterfactual_marginal_count": (
                            int(stats.get("n_analytical_marginals", 0))
                            if m == "qmon_pernode" else 0
                        ),
                        "stats": stats,
                    }
                if qmon_ctx is not None and "qmon" in rec["methods"]:
                    rec["methods"]["qmon"]["diag"] = dict(qmon_ctx["diag"])
                    rec["methods"]["qmon"]["complete"] = bool(
                        rec["methods"]["qmon"]["complete"]
                        and qmon_ctx["diag"].get("complete", False)
                    )
            records.append(rec)
    return records


# ==========================================================================
# False-alarm experiment: every method, full scan, CORRECT circuit.
# ==========================================================================
def false_alarm_circuit(
        key, qmax=24, methods=METHODS,
        shot_seed=FALSE_ALARM_SHOT_SEED,
        real_run_k_cap=FALSE_ALARM_REAL_RUN_K_CAP):
    """Full correct-circuit scan with explicit physical/shortcut accounting.

    QMon evaluates every instrumented Qmax=24 batch with aligned shots.
    Baselines above ``real_run_k_cap`` use their exact null shortcuts; this is
    recorded per method and is never described as physical execution.
    """
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(f"RQ3 requires Qmax={CANONICAL_QMAX}, got {qmax}")
    if (
        int(shot_seed) != FALSE_ALARM_SHOT_SEED
        or int(real_run_k_cap) != FALSE_ALARM_REAL_RUN_K_CAP
    ):
        raise ValueError(
            "false-alarm shot seed and K cap are fixed protocol parameters"
        )
    oracle = Oracle(key, qmax)
    if not oracle.eligible():
        return None
    qc = oracle.qc
    real = oracle.K <= real_run_k_cap
    out = {"circuit": key, "K": oracle.K, "K_qmon": oracle.K_qmon,
           "B": oracle.B, "real_run": real,
           "real_run_k_cap": int(real_run_k_cap),
           "shot_seed": int(shot_seed),
           "qmon_plan": oracle.qmon_plan,
           "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
           "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
           "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
           "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
           "exact_binomial_convention": dict(EXACT_BINOMIAL_CONVENTION),
           "methods": {}}
    qmon_ctx = None
    if "qmon" in methods:
        qmon_ctx = make_qmon_joint_ctx(
            qc, key, oracle, sample_tag=f"false-alarm-{shot_seed}",
            include_raw=False,
        )
        qmon_ctx["materialize"]()
    for m in methods:
        n_flag = 0
        complete = True
        node_set = (oracle.qmon_nodes if m in ("qmon", "qmon_pernode")
                    else oracle.nodes)
        for nd in node_set:
            seed = _seed32(key, f"fa{shot_seed}", nd[0], nd[1], m)
            if m == "qmon":
                if not qmon_ctx["is_complete"]():
                    complete = False
                    break
                try:
                    count1 = qmon_ctx["counts"](nd)
                except JointTrajectoryIncomplete:
                    complete = False
                    break
                f = qmon_count_flag(
                    nd, count1, oracle, shots=SHOTS,
                    alpha_node=ALPHA / max(oracle.K_qmon, 1)
                )
            elif m == "qmon_pernode":
                f = qmon_node_flag(nd, oracle.p1e[nd], oracle, seed=seed,
                                   alpha_node=ALPHA / max(oracle.K_qmon, 1))
            elif m == "statistical":
                f = (run_statistical(qc, nd, oracle, seed=seed,
                                     alpha_node=ALPHA / oracle.K) if real
                     else _null_statistical_flag(nd, oracle, seed,
                                                 ALPHA / oracle.K))
            elif m == "proq":
                f = run_proq(qc, nd, oracle, seed=seed) if real else False
            elif m == "liu":
                f = run_liu(qc, nd, oracle, seed=seed) if real else False
            elif m == "mqt":
                # Applicability is defined by a pass on this same reference.
                # This is a conditional definition shortcut, not an executed
                # empirical false-alarm observation.
                continue
            if f:
                n_flag += 1
        execution_class = (
            "physical" if m == "qmon"
            else "counterfactual" if m == "qmon_pernode"
            else "definition-shortcut" if m == "mqt"
            else "physical" if real else "exact-null"
        )
        execution_mode = (
            MUTANT_EXECUTION_MODE["qmon"] if m == "qmon"
            else "finite-shot-counterfactual-exact-reference-z-marginal"
            if m == "qmon_pernode"
            else "conditional-reference-definition-shortcut"
            if m == "mqt"
            else MUTANT_EXECUTION_MODE[m] if real
            else "analytical-exact-null-shortcut"
        )
        reference_status = (
            "not-evaluated" if m == "mqt"
            else "analytical-counterfactual" if m == "qmon_pernode"
            else "proved-null" if execution_class == "exact-null"
            else "executed"
        )
        out["methods"][m] = {
            "complete": complete,
            "status": (
                "definition-shortcut-not-empirical" if m == "mqt"
                else "complete" if complete else "incomplete"
            ),
            "execution_class": execution_class,
            "execution_mode": execution_mode,
            "reference_applicability": (
                "conditional-on-original-assertion-pass"
                if m == "mqt" else "unconditional"
            ),
            "reference_applicability_status": reference_status,
            "n_flagged": (
                None if m == "mqt" or not complete else n_flag
            ),
            "false_alarm": (
                None if m == "mqt" or not complete else n_flag > 0
            ),
            "conditional_false_alarm": False if m == "mqt" else None,
            "family_size": (
                oracle.K_qmon if m in ("qmon", "qmon_pernode")
                else oracle.K if m == "statistical" else None),
            "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[m],
            "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[m],
            "n_executed_programs": (
                int(oracle.B or 0) if m == "qmon"
                else oracle.K if execution_class == "physical" else 0
            ),
            "n_counterfactual_marginals": (
                oracle.K_qmon if m == "qmon_pernode" else 0
            ),
            "n_exact_null_shortcuts": (
                oracle.K if execution_class == "exact-null" else 0
            ),
            "n_definition_shortcuts": (
                oracle.K if m == "mqt" else 0
            ),
        }
        if m == "qmon":
            out["methods"][m]["diag"] = dict(qmon_ctx["diag"])
    return out


def run_rq3_keys(keys, out, *, methods=METHODS, arm="mutant",
                 circuit_budget_seconds=None, resume=True):
    """Schema-checked per-circuit checkpointing for either experiment
    (mutant or false-alarm)."""
    methods = tuple(methods)
    if arm not in ("mutant", "false-alarm"):
        raise ValueError(f"unknown RQ3 arm {arm!r}")
    path = Path(out)
    if path.exists() and resume:
        artifact = load_rq3_artifact(path)
        expected = new_rq3_artifact(
            methods, arm=arm,
            circuit_budget_seconds=circuit_budget_seconds,
        )
        if artifact["protocol"] != expected["protocol"]:
            raise ValueError(
                f"resume protocol differs for {path}; old/mixed artifacts "
                "must be rerun")
    elif path.exists():
        raise FileExistsError(f"refusing to replace {path}; choose a new output")
    else:
        artifact = new_rq3_artifact(
            methods, arm=arm,
            circuit_budget_seconds=circuit_budget_seconds,
        )
    normalized = [key if key.endswith(".qasm") else key + ".qasm"
                  for key in keys]
    previous_handler = None
    if circuit_budget_seconds:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(
            signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(BudgetExceeded()),
        )
    try:
        for index, key in enumerate(normalized, start=1):
            existing = [
                record for record in artifact["records"]
                if record.get("circuit") == key
            ]
            status = artifact["circuit_status"].get(key, {})
            inventory_complete = _rq3_circuit_inventory_complete(
                existing, methods, arm=arm
            )
            if (
                status.get("status") in TERMINAL_CIRCUIT_STATUSES
                and inventory_complete
            ):
                continue
            artifact["records"] = [
                record for record in artifact["records"]
                if record.get("circuit") != key
            ]
            if circuit_budget_seconds:
                signal.alarm(int(circuit_budget_seconds))
            try:
                if arm == "mutant":
                    records = evaluate_circuit(key, methods=methods)
                    artifact["records"].extend(records)
                else:
                    record = false_alarm_circuit(key, methods=methods)
                    records = [record] if record else []
                    artifact["records"].extend(records)
                artifact["circuit_status"][key] = _terminal_circuit_status(
                    records, methods, arm=arm
                )
            except BudgetExceeded:
                artifact["circuit_status"][key] = {
                    "status": "budget-timeout",
                    "records": 0,
                    "budget_seconds": circuit_budget_seconds,
                }
            except Exception as exc:
                artifact["circuit_status"][key] = {
                    "status": "error",
                    "records": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            finally:
                if circuit_budget_seconds:
                    signal.alarm(0)
            save_rq3_artifact(path, artifact)
            print(
                f"[{index}/{len(normalized)}] {key}: "
                f"{artifact['circuit_status'][key]['status']}", flush=True
            )
    finally:
        if circuit_budget_seconds:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    return artifact


def merge_rq3_artifacts(paths, *, strict=True):
    artifacts = [load_rq3_artifact(path) for path in paths]
    if not artifacts:
        raise ValueError("merge requires at least one current-schema artifact")
    first = artifacts[0]
    arm = first["protocol"]["arm"]
    budget = first["protocol"]["circuit_budget_seconds"]
    for artifact in artifacts[1:]:
        for field in (
                "arm", "qmax", "shots", "alpha_familywise", "branch_cap",
                "circuit_budget_seconds", "canonical_manifest_sha256"):
            if artifact["protocol"].get(field) != first["protocol"].get(field):
                raise ValueError(f"merge protocol mismatch at {field}")
        for field in ("sources", "qasm_corpus", "dependencies", "protocol",
                      "circuit_budget"):
            if (artifact["fingerprints"][field]["sha256"]
                    != first["fingerprints"][field]["sha256"]):
                raise ValueError(f"merge checksum mismatch at {field}")
    methods = []
    for artifact in artifacts:
        for method in artifact["protocol"]["methods"]:
            if method not in methods:
                methods.append(method)
    merged = new_rq3_artifact(
        methods, arm=arm, circuit_budget_seconds=budget
    )
    by_identity = {}
    for artifact in artifacts:
        for record in artifact["records"]:
            identity = (
                (record["circuit"], record.get("seed"),
                 record.get("slot_ordinal"))
                if arm == "mutant" else (record["circuit"],)
            )
            if identity not in by_identity:
                by_identity[identity] = pickle.loads(pickle.dumps(record))
                continue
            existing = by_identity[identity]
            for field, value in record.items():
                if field == "methods":
                    continue
                if existing.get(field) != value:
                    raise ValueError(
                        f"merge metadata conflict at {identity}/{field}")
            for method, result in record.get("methods", {}).items():
                if (method in existing["methods"]
                        and existing["methods"][method] != result):
                    raise ValueError(
                        f"merge method conflict at {identity}/{method}")
                existing["methods"][method] = result
    merged["records"] = sorted(by_identity.values(), key=lambda record: (
        record["circuit"], int(record.get("seed", 0)),
        int(record.get("slot_ordinal", 0))))
    for key in sorted({record["circuit"] for record in merged["records"]}):
        records = [r for r in merged["records"] if r["circuit"] == key]
        merged["circuit_status"][key] = _terminal_circuit_status(
            records, methods, arm=arm
        )
        merged["circuit_status"][key]["merged"] = True
    validate_rq3_artifact(merged, "merged artifact")
    if strict:
        validate_final_rq3_artifact(merged, "merged artifact")
    return merged


# ==========================================================================
# Aggregation
# ==========================================================================
def _method_complete(record, method):
    """Treat a method as incomplete when its result or completion flag is
    absent."""
    return (
        record.get("methods", {}).get(method, {}).get("complete") is True
    )


def _common_complete_records(records, methods=METHODS):
    """Return the paired intersection completed by every requested method."""
    return [
        record for record in records
        if all(_method_complete(record, method) for method in methods)
    ]


def _method_completion_report(records, methods=METHODS):
    """Report terminal method outcomes without interpreting them as misses."""
    records = list(records)
    by_method = {}
    for method in methods:
        complete = sum(_method_complete(record, method) for record in records)
        reasons = {}
        for record in records:
            result = record.get("methods", {}).get(method)
            if not isinstance(result, dict) or result.get("complete") is True:
                continue
            candidates = list(
                result.get("stats", {}).get("incomplete_reasons", ())
            )
            candidates.extend(
                result.get("diag", {}).get("incomplete_reasons", ())
            )
            if not candidates:
                candidates = [result.get("status", "unspecified-incomplete")]
            for reason in dict.fromkeys(str(value) for value in candidates):
                reasons[reason] = reasons.get(reason, 0) + 1
        by_method[method] = {
            "complete": complete,
            "terminal_incomplete": len(records) - complete,
            "incomplete_reasons": dict(sorted(reasons.items())),
        }
    return {
        "attempted": len(records),
        "common_method_complete": len(_common_complete_records(records, methods)),
        "excluded_from_paired_statistics": (
            len(records) - len(_common_complete_records(records, methods))
        ),
        "by_method": by_method,
    }


def _unique_circuit_plan_costs(records):
    """Summarize plan costs once per circuit, never once per mutant."""
    plans = {}
    for index, record in enumerate(records):
        key = record.get("circuit")
        plan = record.get("qmon_plan", {})
        value = {
            "K": record.get("K"),
            "K_qmon": record.get("K_qmon"),
            "B": record.get("B"),
            "plan_sha256": plan.get("plan_sha256"),
        }
        if (
            not isinstance(key, str)
            or any(not isinstance(value[field], int) for field in ("K", "K_qmon", "B"))
            or not isinstance(value["plan_sha256"], str)
        ):
            raise ValueError(f"records[{index}]: missing circuit-plan cost data")
        if key in plans and plans[key] != value:
            raise ValueError(f"{key}: inconsistent circuit-plan cost data")
        plans[key] = value
    if not plans:
        raise ValueError("RQ3 cost aggregation has no circuit plans")

    def summarize(field):
        values = sorted(plan[field] for plan in plans.values())
        p75_index = math.ceil(0.75 * len(values)) - 1
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "percentile_75_nearest_rank": int(values[p75_index]),
            "maximum": int(values[-1]),
        }

    return {
        "unique_circuits": len(plans),
        "B": summarize("B"),
        "K_qmon": summarize("K_qmon"),
        "K": summarize("K"),
    }


def aggregate(records, fa_records=None):
    import statistics
    records = list(records)
    buggy_attempted = [r for r in records if not r["equivalent"]]
    common_records = _common_complete_records(records)
    buggy = [r for r in common_records if not r["equivalent"]]
    seeds = sorted({r["seed"] for r in records})

    def ms(vals, d=1):
        if not vals:
            return "n/a"
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{np.mean(vals):.{d}f}+/-{sd:.{d}f}"

    plan_costs = _unique_circuit_plan_costs(records)
    scoped = [r for r in buggy if r["in_scope_full"]]
    qmon_scoped = [r for r in buggy if r["in_scope_qmon"]]
    deployment_gap = [r for r in buggy if r["deployment_gap"]]
    print(f"\nRQ3 SUMMARY (S={SHOTS} shots/run, alpha={ALPHA} "
          f"familywise): {len(records)} attempted records, "
          f"{len(common_records)} complete for every method; "
          f"{len(buggy)}/{len(buggy_attempted)} non-output-equivalent "
          f"records are complete for every method, {len(scoped)} in full "
          f"numerical scope, {len(qmon_scoped)} in deployed-QMon scope, "
          f"{len(deployment_gap)} deployment gaps")
    fa_common = []
    if fa_records:
        fa_records = [record for record in fa_records if record]
        fa_common = _common_complete_records(fa_records)
        print(f"FALSE ALARMS: {len(fa_records)} attempted circuits, "
              f"{len(fa_common)} complete for every method")
    mutant_completion = _method_completion_report(buggy_attempted)
    false_alarm_completion = _method_completion_report(fa_records or ())
    print(
        "COMPLETION: mutant paired intersection "
        f"{mutant_completion['common_method_complete']}/"
        f"{mutant_completion['attempted']}; false-alarm paired intersection "
        f"{false_alarm_completion['common_method_complete']}/"
        f"{false_alarm_completion['attempted']}. Terminal-incomplete outcomes "
        "are excluded, never scored as misses."
    )
    print(
        "PLANNED FULL-SCAN COUNTS PER CIRCUIT "
        "(mean/median/nearest-rank P75/maximum): "
        f"batched QMon={plan_costs['B']['mean']:.1f}/"
        f"{plan_costs['B']['median']:.1f}/"
        f"{plan_costs['B']['percentile_75_nearest_rank']}/"
        f"{plan_costs['B']['maximum']}, independent checkpoint QMon="
        f"{plan_costs['K_qmon']['mean']:.1f}/"
        f"{plan_costs['K_qmon']['median']:.1f}/"
        f"{plan_costs['K_qmon']['percentile_75_nearest_rank']}/"
        f"{plan_costs['K_qmon']['maximum']}, separate-run assertions="
        f"{plan_costs['K']['mean']:.1f}/{plan_costs['K']['median']:.1f}/"
        f"{plan_costs['K']['percentile_75_nearest_rank']}/"
        f"{plan_costs['K']['maximum']}"
    )
    print(f"{'method':32} {'detect%(all)':>13} {'detect%|candidate-covered':>26} "
          f"{'loc-precision':>13} {'false-alarm%':>13} {'planned count':>22}")
    for m in METHODS:
        det, dets, prec = [], [], []
        method_complete = sum(
            _method_complete(record, m) for record in buggy_attempted
        )
        eligible = buggy
        eligible_scope = scoped
        for s in seeds:
            b = [r for r in eligible if r["seed"] == s]
            bs = [r for r in eligible_scope if r["seed"] == s]
            if b:
                det.append(100 * np.mean([r["methods"][m]["detected"]
                                          for r in b]))
            if bs:
                dets.append(100 * np.mean([r["methods"][m]["detected"]
                                           for r in bs]))
            pv = [r["methods"][m]["loc_precision"] for r in b
                  if r["methods"][m]["detected"]]
            if pv:
                prec.append(np.mean(pv))
        fa = "n/a"
        if fa_common:
            fam = [
                fr["methods"][m].get("false_alarm") for fr in fa_common
                if isinstance(fr["methods"][m].get("false_alarm"), bool)
            ]
            if fam:
                fa = f"{100 * np.mean(fam):.1f}"
            elif m == "mqt" and fa_common:
                fa = "conditional n/a"
        ex = (
            f"batches {plan_costs['B']['mean']:.1f}/"
            f"{plan_costs['B']['median']:.1f}"
            if m == "qmon"
            else f"reads {plan_costs['K_qmon']['mean']:.1f}/"
                 f"{plan_costs['K_qmon']['median']:.1f}"
            if m == "qmon_pernode"
            else "simulation only"
            if m == "mqt"
            else f"runs {plan_costs['K']['mean']:.1f}/"
                 f"{plan_costs['K']['median']:.1f}"
        )
        print(f"{METHOD_DISPLAY_NAMES[m]:32} {ms(det):>13} "
              f"{ms(dets):>26} {ms(prec, 3):>13} "
              f"{fa:>13} {ex:>22}  paired={len(eligible)}/"
              f"{len(buggy_attempted)}, complete={method_complete}/"
              f"{len(buggy_attempted)}")

    miss = [r for r in buggy
            if not r["methods"]["qmon"]["detected"]]
    outside_full = sum(1 for r in miss if not r["in_scope_full"])
    deploy_gap = sum(1 for r in miss if r["deployment_gap"])
    subthr = sum(1 for r in miss if r["in_scope_full"]
                 and not r["deployment_gap"]
                 and not any(r["methods"][m]["detected"] for m in METHODS))
    other = len(miss) - outside_full - deploy_gap - subthr
    print(f"\nQMon miss breakdown ({len(miss)}/{len(buggy)} "
          f"records complete for every method): outside-full-scope {outside_full}, "
          f"deployment-gap {deploy_gap}, missed-by-every-method {subthr}, "
          f"caught-by-a-baseline {other}")

    # Additional diagnostics: realized early-exit executions, MQT
    # applicability, tool errors/timeouts, and QMon batch simulation.
    for m in METHODS:
        ex = [r["methods"][m]["stats"].get("n_exec", 0) for r in buggy
              if "stats" in r["methods"][m]]
        extra = ""
        if m == "mqt":
            aff = sum(r["methods"][m]["stats"].get("n_affected", 0)
                      for r in buggy if "stats" in r["methods"][m])
            app = sum(r["methods"][m]["stats"].get("n_applicable", 0)
                      for r in buggy if "stats" in r["methods"][m])
            tmo = sum(r["methods"][m]["stats"].get("timeout", 0)
                      for r in buggy if "stats" in r["methods"][m])
            err = sum(r["methods"][m]["stats"].get("error", 0)
                      for r in buggy if "stats" in r["methods"][m])
            extra = (f"  applicable/affected={app}/{aff} "
                     f"({100*app/max(aff,1):.0f}%), timeouts={tmo}, "
                     f"errors={err}")
        if m == "qmon":
            sims = sum(r["methods"][m].get("diag", {}).get("sim_batches", 0)
                       for r in buggy)
            caps = sum(r["methods"][m].get("diag", {}).get("cap_hits", 0)
                       for r in buggy)
            fbs = sum(r["methods"][m].get("diag", {}).get("read_fallbacks", 0)
                      for r in buggy)
            maxw = max([r["methods"][m].get("diag", {}).get("dropped_w", 0.0)
                        for r in buggy], default=0.0)
            extra = (f"  monitoring-circuit sims={sims}, cap-hits={caps} "
                     f"(max dropped weight {maxw:.1e}), read-fallbacks={fbs}")
        if ex and m != "qmon_pernode":
            unit = (
                "realized batch executions per mutant"
                if m == "qmon"
                else "evaluated debugger checkpoints per mutant"
                if m == "mqt"
                else "realized circuit executions per mutant"
            )
            print(
                f"  {METHOD_DISPLAY_NAMES[m]}: {unit} "
                f"mean={np.mean(ex):.1f}" + extra
            )

    return {
        "attempted_records": len(records),
        "common_complete_records": len(common_records),
        "attempted_non_output_equivalent": len(buggy_attempted),
        "common_complete_non_output_equivalent": len(buggy),
        "common_complete_in_scope_full": len(scoped),
        "common_complete_in_scope_qmon": len(qmon_scoped),
        "common_complete_deployment_gap": len(deployment_gap),
        "unique_circuit_plan_costs": plan_costs,
        "method_complete_non_output_equivalent": {
            method: sum(
                _method_complete(record, method)
                for record in buggy_attempted
            )
            for method in METHODS
        },
        "false_alarm_attempted": len(fa_records or ()),
        "false_alarm_common_complete": len(fa_common),
        "false_alarm_method_complete": {
            method: sum(
                _method_complete(record, method)
                for record in (fa_records or ())
            )
            for method in METHODS
        },
        "false_alarm_method_empirical": {
            method: sum(
                isinstance(
                    record.get("methods", {}).get(method, {}).get(
                        "false_alarm"),
                    bool,
                )
                for record in (fa_records or ())
                if _method_complete(record, method)
            )
            for method in METHODS
        },
        "completion": {
            "mutant_non_output_equivalent": mutant_completion,
            "false_alarm": false_alarm_completion,
            "denominator_rule": (
                "paired statistics use only records complete for every method; "
                "terminal-incomplete outcomes are excluded and are not misses"
            ),
        },
    }


# ==========================================================================
# CLI
# ==========================================================================
if __name__ == "__main__":
    if sys.argv[1:2] == ["--mqt-worker"]:
        # one debugger evaluation per process (hard-timeout boundary for the
        # GIL-holding DD core); prints the status as the last stdout line.
        # Classification is precise: only the genuine entangled-
        # sub-state refusal is 'unevaluable'; every other tool failure is
        # 'error' and is never credited as a detection.
        qasm = sys.stdin.read()
        try:
            import mqt.debugger as mqtd
            state = mqtd.create_ddsim_simulation_state()
            state.load_code(qasm)
            n_failed = state.run_all()
            status = "fail" if n_failed >= 1 else "pass"
        except RuntimeError as e:
            msg = str(e).lower()
            status = ("unevaluable" if "entangled sub-state" in msg
                      else "error")
        except Exception:
            status = "error"
        print(status)
        sys.exit()
    # Current-schema operational artifact commands. A list-shaped or
    # missing-qmax legacy pickle cannot be resumed, merged, or aggregated.
    if sys.argv[1:2] == ["--chunk"]:
        keys = [line.strip() for line in open(sys.argv[2]) if line.strip()]
        fast = [method for method in METHODS if method != "mqt"]
        artifact = run_rq3_keys(
            keys, sys.argv[3], methods=fast, arm="mutant",
            circuit_budget_seconds=MUTANT_CIRCUIT_BUDGET_SECONDS,
        )
        print(f"chunk done: {len(artifact['records'])} records", flush=True)
        sys.exit()
    if sys.argv[1:2] == ["--remethod"]:
        method = sys.argv[2]
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}")
        keys = [line.strip() for line in open(sys.argv[3]) if line.strip()]
        artifact = run_rq3_keys(
            keys, sys.argv[4], methods=[method], arm="mutant",
            circuit_budget_seconds=MUTANT_CIRCUIT_BUDGET_SECONDS,
        )
        print(f"remethod {method}: {len(artifact['records'])} records")
        sys.exit()
    if sys.argv[1:2] == ["--falsealarm"]:
        keys = [line.strip() for line in open(sys.argv[2]) if line.strip()]
        artifact = run_rq3_keys(
            keys, sys.argv[3], methods=METHODS, arm="false-alarm",
            circuit_budget_seconds=FALSE_ALARM_CIRCUIT_BUDGET_SECONDS,
        )
        print(f"false-alarm done: {len(artifact['records'])} records")
        sys.exit()
    if sys.argv[1:2] == ["--merge"]:
        merged = merge_rq3_artifacts(sys.argv[3:])
        save_rq3_artifact(sys.argv[2], merged)
        print(f"merged {len(merged['records'])} -> {sys.argv[2]}")
        sys.exit()
    if sys.argv[1:2] == ["--aggregate"]:
        artifact = load_rq3_artifact(sys.argv[2])
        validate_complete_rq3_artifact(artifact, sys.argv[2])
        fa_records = None
        if len(sys.argv) > 3:
            false_alarm_artifact = load_rq3_artifact(sys.argv[3])
            validate_complete_rq3_artifact(
                false_alarm_artifact, sys.argv[3])
            if false_alarm_artifact["protocol"].get("arm") != "false-alarm":
                raise ValueError("false-alarm aggregate input has wrong arm")
            fa_records = false_alarm_artifact["records"]
        if artifact["protocol"].get("arm") != "mutant":
            raise ValueError("RQ3 aggregate input has wrong arm")
        aggregate(artifact["records"], fa_records)
        sys.exit()
    raise SystemExit(
        "usage: rq3_realexec_eval.py "
        "--chunk KEYS OUT | --remethod METHOD KEYS OUT | "
        "--falsealarm KEYS OUT | --merge OUT INPUT... | "
        "--aggregate MUTANT [FALSE_ALARM]"
    )
