"""Finite-shot, real-execution evaluation for simultaneous gate mutations.

Protocol
========
For every eligible circuit in the RQ3 circuit set and seeds 1 to 5, this
module reproduces the deterministic mutation stream from
``multi_mutation_eval.py`` for four modes: one, two, and three independently
selected gates (``m1``, ``m2``, ``m3``), and ``clus``, a two-qubit gate plus
up to two nearby downstream gates touching the same qubits.  Replacement
gate draws are seeded independently with ``crc32(f"{key}|{seed}|{gate}")``.
The exact selected gate order, seed, original operation, and replacement
operation are stored in every record, and each single replacement used by a
masking check is regenerated with that same seed.

The oracle, monitor plan, expected node states, original backward cones, and
QMon batches are built from the ORIGINAL intended circuit by
``rq3_realexec_eval.Oracle``.  The jointly mutated circuit is what all methods
execute.  A checkpoint is affected exactly when at least one mutated gate is
in its original backward cone.  The evaluator imports the current RQ3 shot
budget (8192), familywise alpha (0.01), method list, statistical/Proq/Liu/MQT
runners, final-output probabilities, and branch-mixture QMon machinery.
Sampling methods retain RQ3's per-node deterministic seed formula and its
Bonferroni denominators (K_qmon for QMon, K for statistical assertions).
MQT remains simulation-only and retains RQ3's applicability and timeout
rules.

Concretely, QMon samples each branch-mixture read of the generated
monitoring circuit with an exact
binomial test (or any-wrong-outcome rule for deterministic plans);
``statistical`` runs each affected mutant prefix on Aer and applies the
published chi-square test with RQ3's exact-binomial low-count fallback;
``proq`` executes pytket's ProjectorAssertionBox; ``liu`` executes the
authors' classical/superposition assertion circuit on Aer; and ``mqt`` calls
the official debugger DD backend without claiming hardware executions.

Unlike RQ3's early-exit scan, this module scans every checkpoint.  The first
flag is still the detection/localization result that early exit would have
returned, while the union of ALL flagged nodes' original cones supplies the
multi-fault ``loc_all`` metric.  ``loc_any`` is only a structural cone-hit
diagnostic for the first flag, not a separate empirical signal.

Differences from ``multi_mutation_eval.py``
===========================================
The old experiment scored exact reduced-density-matrix surrogates at the
shot-to-infinity limit and did not execute QMon's monitoring circuit.  This
extension uses the actual finite-shot mechanisms in ``rq3_realexec_eval``.
For a true multi-gate mutant, every affected/healing decision uses the FULL
set of fault gates.  If an upstream monitor can collapse an entangled qubit
or heal a state affected by any fault, the generated joint-mutant monitoring
circuit is branch-simulated before later reads.  Final reads are taken from
the first batch's monitoring circuit under the same rule.  Singleton contexts delegate to
the current RQ3 implementation verbatim, preserving the ``m1`` QMon path.

Assumptions and limitations
===========================
``clus`` is a synthetic compiled-footprint proxy; it is not evidence from a
specific compiler's physical error model.  ``output_equivalent`` means only
that the final measured-output distributions have L1 distance below 1e-9.
It is NOT state equivalence or circuit equivalence, and internally visible
differences may remain.  Every valid mutant is therefore scanned; headline
detection denominators follow RQ3's non-output-equivalent filter, while flags
on output-equivalent mutants are reported separately as internal-deviation
diagnostics.  Unchanged-cone shortcuts are exact under the original-circuit
causal-cone model.  The branch engine inherits RQ3's finite BRANCH_CAP; cap
hits and discarded weight are recorded, so a capped batch must not be
described as strictly exact.  This module does not alter main RQ3's source or
protocol.  Its cwd-independent Oracle loader temporarily supplies an absolute
circuit path to two existing module globals and restores both immediately.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _datetime
import math
import numbers
import os
import pickle
import random
import signal
import statistics
import sys
import time
import zlib
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CIRCUIT_DIR = PROJECT_ROOT / "quantum_circuits"
if str(SCRIPT_DIR) not in sys.path:
    # Existing source modules use sibling imports and are also
    # directly executable.  Adding their directory supports both invocation
    # styles without modifying those modules.
    sys.path.insert(0, str(SCRIPT_DIR))

from qiskit import QuantumCircuit

import rq3_realexec_eval as rq3
from rq3_manifest import (
    DEFAULT_MANIFEST,
    EXPECTED_CIRCUITS,
    apply_manifest_slots,
    circuit_qasm_sha256,
    load_manifest,
    operation_signature as canonical_operation_signature,
    runtime_fingerprints,
    slots_for_circuit,
    stable_hash,
)
from multi_mutation_eval import (
    WINDOW,
    _cluster_gates as _legacy_cluster_gates,
    _mutate_many as _legacy_mutate_many,
)


# Explicit aliases make shared protocol calls clear and keep the main RQ3 module
# untouched.  Private helpers are imported because this is an extension of
# that experiment, not a replacement implementation of its decision rules.
Oracle = rq3.Oracle
SHOTS = rq3.SHOTS
ALPHA = rq3.ALPHA
METHODS = tuple(rq3.METHODS)
METHOD_ASSERTED_SUBSYSTEM = dict(rq3.METHOD_ASSERTED_SUBSYSTEM)
PRUNING_SCOPE_RULE = rq3.PRUNING_SCOPE_RULE
EXACT_BINOMIAL_CONVENTION = dict(rq3.EXACT_BINOMIAL_CONVENTION)
BONFERRONI_FAMILY_RULE = dict(rq3.BONFERRONI_FAMILY_RULE)
EXECUTION_SHORTCUT_POLICY = dict(rq3.EXECUTION_SHORTCUT_POLICY)
GROUPED_MONITOR_CHANNEL = dict(rq3.GROUPED_MONITOR_CHANNEL)
final_probs = rq3.final_probs
unitary_gate_indices = rq3.unitary_gate_indices
quantum_gates = rq3.quantum_gates
run_statistical = rq3.run_statistical
run_proq = rq3.run_proq
run_liu = rq3.run_liu
run_mqt = rq3.run_mqt

if SHOTS != 8192 or ALPHA != 0.01:
    raise RuntimeError(
        "multi-mutation protocol requires current RQ3 SHOTS=8192 and "
        "ALPHA=0.01"
    )


SCHEMA = "qmon-multimut-realexec-v3-explicit-applicability-joint-shots"
MODES = ("m1", "m2", "m3", "clus")
SEEDS = (1, 2, 3, 4, 5)
OUTPUT_EQ_L1 = 1e-9
DEFAULT_OUTPUT = SCRIPT_DIR / "multimut_realexec_records.pkl"
LEGACY_OUTPUT_NAME = "multimut_records.pkl"
CANONICAL_KEYFILE = PROJECT_ROOT / "inputs" / "rq3_circuits.txt"
SMOKE_KEYS = ("dj_indep_qiskit_3.qasm",)
CANONICAL_QMAX = 24
_CANONICAL_MANIFEST = None


def _normal_key(key):
    key = str(key).strip()
    return key if key.endswith(".qasm") else key + ".qasm"


def _plain_param(value):
    """Return a pickle-stable exact value for the numeric QASM parameters."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return repr(value)


def operation_signature(qc, gate):
    """Serializable operation identity at a circuit-data index."""
    return canonical_operation_signature(qc, gate)


def replacement_seed(key, seed, gate):
    """The exact replacement seed used by the legacy multi-mutant stream."""
    return zlib.crc32(f"{key}|{seed}|{gate}".encode())


def mutate_many_with_manifest(qc, gates, key="", seed=0):
    """Run the legacy mutator and return its exact replacement provenance.

    Gate order is intentionally not sorted: ``random.sample`` order and the
    cluster's anchor-first order are part of the old deterministic semantics.
    """
    gates = [int(g) for g in gates]
    if len(gates) != len(set(gates)):
        raise ValueError(f"mutation gate indices must be distinct: {gates}")
    mutated = _legacy_mutate_many(qc, gates, key=key, seed=seed)
    manifest = []
    for gate in gates:
        original = operation_signature(qc, gate)
        replacement = operation_signature(mutated, gate)
        manifest.append(
            {
                "gate": gate,
                "replacement_seed": replacement_seed(key, seed, gate),
                "original": original,
                "replacement": replacement,
                "changed": replacement != original,
            }
        )
    return mutated, manifest


def select_variants(qc, candidates, last_node, seed):
    """Reproduce the old selection stream and declare every protocol mode.

    Clus is returned with an empty gate list when the synthetic footprint has
    no eligible anchor/follower. The producer turns that explicit identity into
    an inapplicable terminal record instead of silently omitting it.
    """
    candidates = list(candidates)
    if len(candidates) < 3:
        return []
    rng = random.Random(seed)
    random.seed(seed)  # retained verbatim; the legacy mutator uses global RNG
    variants = [
        ("m1", rng.sample(candidates, 1)),
        ("m2", rng.sample(candidates, 2)),
        ("m3", rng.sample(candidates, 3)),
    ]
    cluster = _legacy_cluster_gates(qc, list(candidates), last_node, rng)
    variants.append(("clus", cluster if len(cluster) >= 2 else []))
    return variants


def any_fault_in_cone(fault_gates, cone):
    """True iff at least one fault index belongs to this original cone."""
    return not set(fault_gates).isdisjoint(cone)


def node_is_affected(oracle, node, fault_gates):
    return any_fault_in_cone(fault_gates, oracle.cone[node])


def localization_metrics(flagged_nodes, cones, fault_gates):
    """Structural localization metrics for a completed full checkpoint scan."""
    faults = set(fault_gates)
    flagged = list(flagged_nodes)
    first = flagged[0] if flagged else None
    first_cone = set(cones[first]) if first is not None else set()
    union_cone = set()
    for node in flagged:
        union_cone.update(cones[node])
    return {
        "first_node": first,
        "loc_any": bool(flagged and not faults.isdisjoint(first_cone)),
        "loc_all": bool(flagged and faults.issubset(union_cone)),
        "faults_in_first_cone": sorted(faults & first_cone),
        "faults_in_flagged_union": sorted(faults & union_cone),
        "first_cone_size": len(first_cone),
        "flagged_union_cone_size": len(union_cone),
    }


@contextlib.contextmanager
def _oracle_circuit_path():
    """Temporarily make RQ3's legacy relative circuit path cwd-independent."""
    # Oracle.__init__ loads through rq3_realexec_eval and then through the
    # separately imported circuits_selection helper.  Each module captured
    # its own ``quantum_path`` global, so both must point at the same absolute
    # circuit directory during construction.
    global_dicts = [
        rq3.__dict__,
        rq3.filter_circuits_with_output_num.__globals__,
    ]
    unique = []
    for values in global_dicts:
        if all(values is not existing for existing in unique):
            unique.append(values)
    previous = [(values, values.get("quantum_path")) for values in unique]
    for values, _ in previous:
        values["quantum_path"] = str(CIRCUIT_DIR) + os.sep
    try:
        yield
    finally:
        for values, old_path in previous:
            values["quantum_path"] = old_path


def build_oracle(key, qmax=24):
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(
            f"multi-error operational protocol requires Qmax="
            f"{CANONICAL_QMAX}; got {qmax}")
    with _oracle_circuit_path():
        return Oracle(key, qmax)


# ===========================================================================
# Set-aware QMon monitoring-circuit reads.
# ===========================================================================
def qmon_reads_for_batch_many(
    mqc,
    key,
    oracle,
    batch_nodes,
    raw_p1,
    raw_res,
    fault_gates,
    diag=None,
):
    """Return QMon reads for one batch of a joint mutant.

    A singleton delegates to main RQ3.  For multiple faults, a disturbance is
    caused by either entanglement at a monitor or ANY fault in that monitor's
    original cone (intended-plan healing).  Once a batch contains such a
    disturbance and a later monitor exists, simulating the full generated
    joint-mutant circuit is the conservative exact choice; no representative
    ``mut_gate`` is selected.
    """
    faults = frozenset(int(g) for g in fault_gates)
    if not faults:
        raise ValueError("QMon context requires at least one fault gate")
    if len(faults) == 1:
        return rq3.qmon_reads_for_batch(
            mqc,
            key,
            oracle,
            batch_nodes,
            raw_p1,
            raw_res,
            next(iter(faults)),
            diag,
        )

    picked = set(batch_nodes)
    nodes = sorted(nd for nd in oracle.emitted if nd[0] in picked)
    disturbing = [
        nd
        for nd in nodes
        if raw_res[nd] > rq3.SEP_TOL
        or any_fault_in_cone(faults, oracle.cone[nd])
    ]
    # A lone monitor's read precedes its own collapse/reset, and no disturbance
    # means Theorem 1 applies to every read.  Every other disturbing batch is
    # simulated so same-index monitor ordering cannot be approximated away.
    if not disturbing or len(nodes) <= 1:
        return (
            {nd: raw_p1[nd] for nd in nodes},
            {nd: raw_res[nd] for nd in nodes},
        )

    if diag is not None:
        diag["sim_batches"] = diag.get("sim_batches", 0) + 1
        if any(raw_res[nd] > rq3.SEP_TOL for nd in disturbing):
            diag["sim_batches_entanglement"] = (
                diag.get("sim_batches_entanglement", 0) + 1
            )
        if any(any_fault_in_cone(faults, oracle.cone[nd]) for nd in disturbing):
            diag["sim_batches_healing"] = diag.get("sim_batches_healing", 0) + 1

    answer = {"picked_nodes": picked}
    with rq3._mute():
        monitored = rq3.generate_monitoring_circuit(
            mqc, key, oracle.op_list, 10**7, answer
        )
    reads, _ = rq3._mid_reads_branch_exact(monitored, diag=diag)

    nodes_by_qubit = {}
    for node in nodes:
        nodes_by_qubit.setdefault(node[1], []).append(node)
    got_p1, got_res, seen = {}, {}, {}
    for qubit, p1, residual in reads:
        offset = seen.get(qubit, 0)
        seen[qubit] = offset + 1
        if qubit in nodes_by_qubit and offset < len(nodes_by_qubit[qubit]):
            node = nodes_by_qubit[qubit][offset]
            got_p1[node] = p1
            got_res[node] = residual

    for node in nodes:
        if node not in got_p1:
            if diag is not None:
                diag["read_fallbacks"] = diag.get("read_fallbacks", 0) + 1
            got_p1[node] = raw_p1[node]
            got_res[node] = raw_res[node]
    return got_p1, got_res


def make_qmon_ctx_many(mqc, key, oracle, fault_gates, sample_tag=None):
    """Lazy QMon read provider with full-set healing and final-read semantics."""
    faults = frozenset(int(g) for g in fault_gates)
    if not faults:
        raise ValueError("QMon context requires at least one fault gate")
    if len(faults) == 1:
        identity = sample_tag if sample_tag is not None else next(iter(faults))
        return rq3.make_qmon_ctx(mqc, key, oracle, identity)
    identity = (
        sample_tag
        if sample_tag is not None
        else "multi-" + ",".join(str(gate) for gate in sorted(faults))
    )
    context = rq3.make_qmon_joint_ctx(
        mqc,
        key,
        oracle,
        sample_tag=identity,
        include_raw=True,
    )
    context["diag"]["fault_count"] = len(faults)
    context["fault_gates"] = sorted(faults)
    return context


# ===========================================================================
# Full scans using the current RQ3 per-node mechanisms.
# ===========================================================================
def _add_mqt_deltas(stats, before, prefix):
    for status in rq3.MQT_STATS:
        delta = rq3.MQT_STATS[status] - before[status]
        if delta:
            key = f"{prefix}_{status}"
            stats[key] = stats.get(key, 0) + delta


def _scan_error(stats, node, exc):
    stats.setdefault("errors", []).append(
        {
            "node": tuple(node),
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }
    )
    stats["n_errors"] = len(stats["errors"])


def scan_method_full(method, mqc, oracle, mut_tag, fault_gates, qmon_ctx=None):
    """Scan all checkpoints and retain both the first and every later flag.

    The node order, seed derivation, affected-node shortcuts, method runners,
    and test thresholds match ``rq3_realexec_eval.scan_method``.  Continuing
    after a flag is the only intentional scan change and is required solely
    to compute the union-of-flagged-cones ``loc_all`` metric.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if method in ("qmon", "qmon_pernode") and qmon_ctx is None:
        raise ValueError(f"{method} requires a QMon context")

    faults = frozenset(int(g) for g in fault_gates)
    node_set = (
        oracle.qmon_nodes
        if method in ("qmon", "qmon_pernode")
        else oracle.nodes
    )
    stats = {
        "n_exec": 0,
        "n_checked": 0,
        "n_decisions": 0,
        "n_affected": 0,
        "n_null_shortcuts": 0,
        "n_errors": 0,
        "shots": None if method == "mqt" else SHOTS,
        "alpha_node": (
            ALPHA / max(oracle.K_qmon, 1)
            if method in ("qmon", "qmon_pernode")
            else (ALPHA / oracle.K if method == "statistical" else None)
        ),
        "complete": True,
        "incomplete_reasons": [],
    }
    flagged = []
    mqt_budget = rq3.MQTMutantBudget() if method == "mqt" else None

    if method == "qmon" and "materialize" in qmon_ctx:
        if not qmon_ctx["materialize"]():
            stats["complete"] = False
            stats["incomplete_reasons"] = list(
                qmon_ctx["diag"].get("incomplete_reasons", ())
            )
            return False, None, [], stats

    for node in node_set:
        stats["n_checked"] += 1
        seed = rq3._seed32(
            oracle.key, mut_tag, node[0], node[1], method
        )
        affected = node_is_affected(oracle, node, faults)
        if affected:
            stats["n_affected"] += 1
        try:
            if method == "qmon":
                if "counts" in qmon_ctx:
                    count1 = qmon_ctx["counts"](node)
                    flag = rq3.qmon_count_flag(
                        node,
                        count1,
                        oracle,
                        shots=SHOTS,
                        alpha_node=ALPHA / max(oracle.K_qmon, 1),
                    )
                else:  # compatibility for focused mocked-context tests
                    p1_read = qmon_ctx["reads"](node)
                    flag = rq3.qmon_node_flag(
                        node,
                        p1_read,
                        oracle,
                        shots=SHOTS,
                        seed=seed,
                        alpha_node=ALPHA / max(oracle.K_qmon, 1),
                    )
            elif method == "qmon_pernode":
                p1_read = (
                    qmon_ctx["raw_p1"][node]
                    if affected
                    else oracle.p1e[node]
                )
                flag = rq3.qmon_node_flag(
                    node,
                    p1_read,
                    oracle,
                    shots=SHOTS,
                    seed=seed,
                    alpha_node=ALPHA / max(oracle.K_qmon, 1),
                )
            elif method == "statistical":
                if affected:
                    stats["n_exec"] += 1
                    flag = run_statistical(
                        mqc,
                        node,
                        oracle,
                        shots=SHOTS,
                        seed=seed,
                        alpha_node=ALPHA / oracle.K,
                    )
                else:
                    stats["n_null_shortcuts"] += 1
                    flag = rq3._null_statistical_flag(
                        node, oracle, seed, ALPHA / oracle.K
                    )
            elif method == "proq":
                if affected:
                    stats["n_exec"] += 1
                    flag = run_proq(
                        mqc, node, oracle, shots=SHOTS, seed=seed
                    )
                else:
                    stats["n_null_shortcuts"] += 1
                    flag = False
            elif method == "liu":
                if affected:
                    stats["n_exec"] += 1
                    flag = run_liu(
                        mqc, node, oracle, shots=SHOTS, seed=seed
                    )
                else:
                    stats["n_null_shortcuts"] += 1
                    flag = False
            else:  # mqt
                if not affected:
                    stats["n_null_shortcuts"] += 1
                    continue
                applicability = oracle.mqt_applicability_status(
                    node, mqt_budget
                )
                stats[f"applicability_{applicability}"] = (
                    stats.get(f"applicability_{applicability}", 0) + 1
                )
                if applicability in ("timeout", "error", "budget-exhausted"):
                    stats["complete"] = False
                    stats["incomplete_reasons"].append(
                        f"applicability-{applicability}:{node}"
                    )
                    continue
                if applicability != "pass":
                    continue
                stats["n_applicable"] = stats.get("n_applicable", 0) + 1
                if not mqt_budget.can_call():
                    stats["complete"] = False
                    stats["budget_exhausted"] = (
                        stats.get("budget_exhausted", 0) + 1
                    )
                    stats["incomplete_reasons"].append(
                        f"run-budget-exhausted:{node}"
                    )
                    continue
                stats["n_exec"] += 1
                started = time.perf_counter()
                status = rq3.run_mqt_status(mqc, node, oracle)
                mqt_budget.record(status, time.perf_counter() - started)
                stats[f"run_{status}"] = stats.get(f"run_{status}", 0) + 1
                if status in ("timeout", "error"):
                    stats["complete"] = False
                    stats[status] = stats.get(status, 0) + 1
                    stats["incomplete_reasons"].append(
                        f"run-{status}:{node}"
                    )
                    flag = False
                else:
                    flag = status in ("fail", "unevaluable")
        except Exception as exc:
            _scan_error(stats, node, exc)
            stats["complete"] = False
            stats["incomplete_reasons"].append(
                f"exception:{node}:{type(exc).__name__}"
            )
            continue

        stats["n_decisions"] += 1
        if flag:
            flagged.append(tuple(node))

    stats["n_flagged"] = len(flagged)
    if mqt_budget is not None:
        stats["budget"] = mqt_budget.snapshot()
    stats["incomplete_reasons"] = list(
        dict.fromkeys(stats["incomplete_reasons"])
    )
    first = flagged[0] if flagged else None
    return bool(flagged), first, flagged, stats


def mutation_tag(seed, mode, gates, canonical_slot_ids=()):
    """Return a stable, unique sampling identity for a mutation variant."""
    gates = list(gates)
    slot_ids = list(canonical_slot_ids)
    if mode == "m1" and len(gates) == 1 and len(slot_ids) == 1:
        return str(slot_ids[0])
    return f"s{seed}{mode}g{','.join(str(g) for g in gates)}"


def _planned_executions(method, oracle):
    if method == "qmon":
        return oracle.B
    if method == "qmon_pernode":
        return oracle.K_qmon
    if method == "mqt":
        return "sim-only"
    return oracle.K


def _execution_model(method):
    if method == "qmon":
        return "batched finite-shot hardware runs"
    if method == "qmon_pernode":
        return "analytical per-node marginal plus finite-shot draw"
    if method == "mqt":
        return "simulation-only with applicability/timeouts"
    return "affected checkpoints physically execute; unaffected exact null shortcut"


def _method_skipped(method, oracle, reason, fault_gates=()):
    faults = set(int(gate) for gate in fault_gates)
    if faults:
        nodes = (
            oracle.qmon_nodes
            if method in ("qmon", "qmon_pernode")
            else oracle.nodes
        )
        scoped = sorted(
            gate for gate in faults
            if any(gate in oracle.cone[node] for node in nodes)
        )
    else:
        scoped = []
    return {
        "applicable": True,
        "evaluated": False,
        "complete": False,
        "terminal": True,
        "asserted_subsystem": METHOD_ASSERTED_SUBSYSTEM[method],
        "pruning_scope": PRUNING_SCOPE_RULE,
        "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[method],
        "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[method],
        "reason": reason,
        "detected": None,
        "first_node": None,
        "flagged_nodes": [],
        "loc_any": None,
        "loc_all": None,
        "any_fault_in_scope": bool(scoped),
        "all_faults_in_scope": bool(faults)
        and len(scoped) == len(faults),
        "in_scope_faults": scoped,
        "n_flagged": 0,
        "executions": _planned_executions(method, oracle),
        "execution_model": _execution_model(method),
        "stats": {},
    }


def _method_inapplicable(method, oracle, reason):
    result = _method_skipped(method, oracle, reason)
    result.update({
        "applicable": False,
        "complete": True,
        "terminal": True,
    })
    return result


def _method_context_failure(method, oracle, exc, fault_gates):
    result = _method_skipped(
        method, oracle, "qmon-context-error", fault_gates)
    result["evaluated"] = True
    result["detected"] = False
    result["loc_any"] = False
    result["loc_all"] = False
    result["complete"] = False
    result["stats"] = {
        "n_exec": 0,
        "n_checked": 0,
        "n_affected": 0,
        "n_errors": 1,
        "errors": [
            {
                "node": None,
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        ],
    }
    return result


def _output_comparison(expected, observed):
    if len(expected) != len(observed):
        return False, None
    l1 = float(np.abs(expected - observed).sum())
    return bool(l1 < OUTPUT_EQ_L1), l1


def _single_check(qc, base_probs, measured, key, seed, gate):
    try:
        single, manifest = mutate_many_with_manifest(
            qc, [gate], key=key, seed=seed
        )
        probabilities = final_probs(single, measured)
        equivalent, l1 = _output_comparison(base_probs, probabilities)
        return {
            "gate": int(gate),
            "status": "ok",
            "replacement_seed": replacement_seed(key, seed, gate),
            "replacement": manifest[0]["replacement"],
            "output_equivalent": equivalent,
            "output_l1": l1,
        }
    except Exception as exc:
        return {
            "gate": int(gate),
            "status": "error",
            "replacement_seed": replacement_seed(key, seed, gate),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            },
        }


def _canonical_manifest(manifest=None):
    global _CANONICAL_MANIFEST
    if manifest is not None:
        return manifest
    if _CANONICAL_MANIFEST is None:
        _CANONICAL_MANIFEST = load_manifest(DEFAULT_MANIFEST)
    return _CANONICAL_MANIFEST


def canonical_nested_variants(manifest, key, seed):
    slots = [
        slot for slot in slots_for_circuit(manifest, key)
        if int(slot["seed"]) == int(seed)
    ]
    if len(slots) != 3 or [int(slot["ordinal"]) for slot in slots] != [0, 1, 2]:
        raise ValueError(f"mutation list seed has malformed slots: {key}/s{seed}")
    return [(f"m{width}", slots[:width]) for width in (1, 2, 3)]


def _canonical_replacement(slot):
    return {
        "gate": int(slot["gate"]),
        "slot_id": slot["slot_id"],
        "original": copy.deepcopy(slot["original"]),
        "replacement": copy.deepcopy(slot["replacement"]),
        "changed": slot["replacement"] != slot["original"],
    }


def _protocol_inapplicable_records(
    key, seeds, modes, methods, oracle, manifest, candidates, reason
):
    records = []
    for seed in seeds:
        nested = dict(canonical_nested_variants(manifest, key, int(seed)))
        for mode in modes:
            canonical_slots = nested[mode] if mode in nested else []
            gates = [int(slot["gate"]) for slot in canonical_slots]
            replacements = [
                _canonical_replacement(slot) for slot in canonical_slots
            ]
            variant_source = (
                "canonical-nested"
                if canonical_slots else "synthetic-footprint-proxy"
            )
            record = {
                "circuit": key,
                "seed": int(seed),
                "mode": mode,
                "tag": mode,
                "variant_source": variant_source,
                "synthetic_footprint_proxy": mode == "clus",
                "attempted": True,
                "applicable": False,
                "executed": False,
                "terminal": True,
                "fault_count": len(gates),
                "gates": gates,
                "fault_gates": gates,
                "canonical_slot_ids": [
                    slot["slot_id"] for slot in canonical_slots
                ],
                "canonical_manifest_sha256": manifest["manifest_sha256"],
                "mutation_tag": mutation_tag(
                    seed,
                    mode,
                    gates,
                    [slot["slot_id"] for slot in canonical_slots],
                ),
                "replacement_seed_rule": (
                    "canonical-main-rq3-global-stream"
                    if canonical_slots else "not-applicable"
                ),
                "application_order": gates,
                "selection_semantics": (
                    "nested prefixes of the three canonical main-RQ3 slots"
                    if canonical_slots
                    else f"synthetic clustered footprint; WINDOW={WINDOW}"
                ),
                "candidate_count": len(candidates),
                "oracle_source": "original intended circuit",
                "executed_source": "not executed: protocol inapplicable",
                "K": oracle.K,
                "K_qmon": oracle.K_qmon,
                "B": oracle.B,
                "shots": SHOTS,
                "alpha_familywise": ALPHA,
                "output_equivalent": None,
                "output_l1": None,
                "masked": None,
                "masking_checked": False,
                "in_scope": False,
                "any_fault_in_scope": False,
                "all_faults_in_scope": False,
                "in_scope_faults": [],
                "qmon_plan": copy.deepcopy(oracle.qmon_plan),
                "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
                "pruning_scope_rule": PRUNING_SCOPE_RULE,
                "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
                "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
                "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
                "exact_binomial_convention": dict(
                    EXACT_BINOMIAL_CONVENTION),
                "replacements": replacements,
                "single_replacements": [],
                "status": "inapplicable",
                "inapplicable_reason": reason,
                "methods": {
                    method: _method_inapplicable(method, oracle, reason)
                    for method in methods
                },
                "complete": True,
            }
            records.append(record)
    return records


def evaluate_circuit(
    key,
    seeds=SEEDS,
    qmax=24,
    methods=METHODS,
    modes=MODES,
    manifest=None,
):
    """Evaluate all requested variants for one RQ3 circuit."""
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(
            f"multi-error protocol requires Qmax={CANONICAL_QMAX}, got {qmax}")
    key = _normal_key(key)
    methods = tuple(methods)
    modes = tuple(modes)
    unknown_methods = [method for method in methods if method not in METHODS]
    unknown_modes = [mode for mode in modes if mode not in MODES]
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}")
    manifest = _canonical_manifest(manifest)

    oracle = build_oracle(key, qmax=qmax)
    qc = oracle.qc
    last_node = max((i for i, _ in oracle.nodes), default=-1)
    candidates = [
        gate
        for gate in unitary_gate_indices(qc)
        if gate <= last_node
    ]
    if not oracle.eligible():
        return _protocol_inapplicable_records(
            key, seeds, modes, methods, oracle, manifest, candidates,
            "canonical circuit has no operational monitorable checkpoint",
        )
    if len(candidates) < 3:
        return _protocol_inapplicable_records(
            key, seeds, modes, methods, oracle, manifest, candidates,
            "fewer than three pre-output unitary mutation candidates",
        )

    records = []
    for seed in seeds:
        canonical_variants = canonical_nested_variants(
            manifest, key, int(seed)
        )
        legacy_variants = select_variants(
            qc, candidates, last_node, int(seed)
        )
        cluster = next(
            list(gates) for mode, gates in legacy_variants if mode == "clus"
        )
        variants = [
            (mode, slots, "canonical-nested")
            for mode, slots in canonical_variants
        ]
        variants.append(("clus", cluster, "synthetic-footprint-proxy"))

        for mode, variant, variant_source in variants:
            if mode not in modes:
                continue
            canonical_slots = variant if variant_source == "canonical-nested" else []
            gates = (
                [int(slot["gate"]) for slot in canonical_slots]
                if canonical_slots else [int(gate) for gate in variant]
            )
            scoped_faults = sorted(
                gate for gate in gates
                if any(gate in oracle.cone[node] for node in oracle.nodes)
            )
            applicable = bool(mode != "clus" or len(gates) >= 2)
            record = {
                "circuit": key,
                "seed": int(seed),
                "mode": mode,
                "tag": mode,
                "variant_source": variant_source,
                "synthetic_footprint_proxy": mode == "clus",
                "attempted": True,
                "applicable": applicable,
                "executed": False,
                "terminal": False,
                "fault_count": len(gates),
                "gates": gates,
                "fault_gates": gates,
                "canonical_slot_ids": [
                    slot["slot_id"] for slot in canonical_slots
                ],
                "canonical_manifest_sha256": manifest["manifest_sha256"],
                "mutation_tag": mutation_tag(
                    seed,
                    mode,
                    gates,
                    [slot["slot_id"] for slot in canonical_slots],
                ),
                "replacement_seed_rule": (
                    "canonical-main-rq3-global-stream"
                    if canonical_slots
                    else "crc32(f'{key}|{seed}|{gate}')"
                ),
                "application_order": gates,
                "selection_semantics": (
                    "nested prefixes of the three canonical main-RQ3 slots"
                    if canonical_slots
                    else f"synthetic clustered footprint; WINDOW={WINDOW}"
                ),
                "candidate_count": len(candidates),
                "oracle_source": "original intended circuit",
                "executed_source": "jointly mutated circuit",
                "K": oracle.K,
                "K_qmon": oracle.K_qmon,
                "B": oracle.B,
                "shots": SHOTS,
                "alpha_familywise": ALPHA,
                "output_equivalent": None,
                "output_l1": None,
                "masked": None,
                "masking_checked": False,
                "in_scope": bool(scoped_faults),
                "any_fault_in_scope": bool(scoped_faults),
                "all_faults_in_scope": (
                    bool(gates) and len(scoped_faults) == len(gates)
                ),
                "in_scope_faults": scoped_faults,
                "qmon_plan": copy.deepcopy(oracle.qmon_plan),
                "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
                "pruning_scope_rule": PRUNING_SCOPE_RULE,
                "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
                "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
                "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
                "exact_binomial_convention": dict(
                    EXACT_BINOMIAL_CONVENTION),
                "replacements": [],
                "single_replacements": [],
                "status": "pending",
                "methods": {},
            }

            if not applicable:
                reason = (
                    "no eligible two-qubit anchor with a downstream touching "
                    f"unitary gate within WINDOW={WINDOW}"
                )
                record["status"] = "inapplicable"
                record["inapplicable_reason"] = reason
                record["methods"] = {
                    method: _method_inapplicable(method, oracle, reason)
                    for method in methods
                }
                record["complete"] = True
                record["terminal"] = True
                records.append(record)
                continue

            try:
                if canonical_slots:
                    mutant = apply_manifest_slots(qc, canonical_slots)
                    replacement_manifest = [
                        _canonical_replacement(slot) for slot in canonical_slots
                    ]
                    if len(canonical_slots) == 1:
                        expected_hash = canonical_slots[0]["mutant_qasm_sha256"]
                        if circuit_qasm_sha256(mutant) != expected_hash:
                            raise ValueError(
                                f"m1 replay hash drift against the mutation list for {key}/s{seed}"
                            )
                else:
                    mutant, replacement_manifest = mutate_many_with_manifest(
                        qc, gates, key=key, seed=seed
                    )
                mutant_probs = final_probs(mutant, oracle.measured)
                output_equivalent, output_l1 = _output_comparison(
                    oracle.base_probs, mutant_probs
                )
                if canonical_slots and len(canonical_slots) == 1:
                    if output_equivalent != bool(canonical_slots[0]["output_equivalent"]):
                        raise AssertionError(
                            f"m1 equivalence drift against the mutation list for {key}/s{seed}"
                        )
                record["replacements"] = replacement_manifest
                record["output_equivalent"] = output_equivalent
                record["output_l1"] = output_l1
                record["status"] = "ok"
                record["executed"] = True
            except Exception as exc:
                record["status"] = "error"
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
                for method in methods:
                    record["methods"][method] = _method_skipped(
                        method, oracle, record["status"], gates
                    )
                record["complete"] = False
                record["terminal"] = True
                records.append(record)
                continue

            replacement_by_gate = {
                int(item["gate"]): item for item in replacement_manifest
            }
            if canonical_slots:
                for slot in canonical_slots:
                    gate = int(slot["gate"])
                    record["single_replacements"].append({
                        "gate": gate,
                        "status": "ok",
                        "slot_id": slot["slot_id"],
                        "replacement": copy.deepcopy(slot["replacement"]),
                        "output_equivalent": bool(slot["output_equivalent"]),
                        "output_l1": slot["output_l1"],
                        "matches_combined_replacement": (
                            replacement_by_gate[gate]["replacement"]
                            == slot["replacement"]
                        ),
                    })
            else:
                for gate in gates:
                    check = _single_check(
                        qc, oracle.base_probs, oracle.measured, key, seed, gate
                    )
                    check["matches_combined_replacement"] = (
                        check.get("status") == "ok"
                        and check.get("replacement")
                        == replacement_by_gate[gate]["replacement"]
                    )
                    record["single_replacements"].append(check)

            if len(gates) >= 2:
                record["masking_checked"] = True
                checks_ok = all(
                    item["status"] == "ok"
                    and item["matches_combined_replacement"]
                    for item in record["single_replacements"]
                )
                if checks_ok:
                    record["masked"] = bool(
                        output_equivalent
                        and all(
                            not item["output_equivalent"]
                            for item in record["single_replacements"]
                        )
                    )
                else:
                    record["masked"] = None
                    record["masking_error"] = (
                        "one or more exact-single checks failed or did not "
                        "reproduce the combined replacement"
                    )
            else:
                record["masked"] = False

            qmon_ctx = None
            qmon_ctx_error = None
            if any(m in methods for m in ("qmon", "qmon_pernode")):
                try:
                    qmon_ctx = make_qmon_ctx_many(
                        mutant,
                        key,
                        oracle,
                        gates,
                        sample_tag=record["mutation_tag"],
                    )
                except Exception as exc:
                    qmon_ctx_error = exc

            for method in methods:
                if method in ("qmon", "qmon_pernode") and qmon_ctx is None:
                    record["methods"][method] = _method_context_failure(
                        method, oracle, qmon_ctx_error, gates
                    )
                    continue
                detected, first, flagged, stats = scan_method_full(
                    method,
                    mutant,
                    oracle,
                    record["mutation_tag"],
                    gates,
                    qmon_ctx=qmon_ctx,
                )
                localization = localization_metrics(
                    flagged, oracle.cone, gates
                )
                method_nodes = (
                    oracle.qmon_nodes
                    if method in ("qmon", "qmon_pernode")
                    else oracle.nodes
                )
                method_scope_faults = sorted(
                    gate for gate in gates
                    if any(gate in oracle.cone[node] for node in method_nodes)
                )
                complete = bool(
                    stats.get("complete", True)
                    and stats.get("n_errors", 0) == 0
                )
                if method == "qmon":
                    complete = bool(
                        complete and qmon_ctx["diag"].get("complete", False)
                    )
                result = {
                    "applicable": True,
                    "evaluated": True,
                    "complete": complete,
                    "terminal": True,
                    "asserted_subsystem": METHOD_ASSERTED_SUBSYSTEM[method],
                    "pruning_scope": PRUNING_SCOPE_RULE,
                    "bonferroni_family_rule": BONFERRONI_FAMILY_RULE[method],
                    "execution_shortcut_policy": EXECUTION_SHORTCUT_POLICY[method],
                    "detected": detected,
                    "first_node": first,
                    "flagged_nodes": flagged,
                    "n_flagged": len(flagged),
                    "loc_any": localization["loc_any"],
                    "loc_all": localization["loc_all"],
                    "any_fault_in_scope": bool(method_scope_faults),
                    "all_faults_in_scope": (
                        len(method_scope_faults) == len(gates)
                    ),
                    "in_scope_faults": method_scope_faults,
                    "faults_in_first_cone": localization[
                        "faults_in_first_cone"
                    ],
                    "faults_in_flagged_union": localization[
                        "faults_in_flagged_union"
                    ],
                    "first_cone_size": localization["first_cone_size"],
                    "flagged_union_cone_size": localization[
                        "flagged_union_cone_size"
                    ],
                    "localization_basis": (
                        "structural containment in original backward cones"
                    ),
                    "executions": _planned_executions(method, oracle),
                    "execution_model": _execution_model(method),
                    "stats": stats,
                }
                if method == "qmon":
                    result["diag"] = dict(qmon_ctx["diag"])
                record["methods"][method] = result
            record["complete"] = bool(
                record["status"] == "ok"
                and all(
                    method in record["methods"]
                    and record["methods"][method].get("complete", False)
                    for method in methods
                )
            )
            record["terminal"] = True
            records.append(record)
    return records


# ===========================================================================
# Checkpoint artifacts, resume, merge, and aggregation.
# ===========================================================================
def _utc_now():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _artifact_protocol(methods, modes, seeds, qmax, circuit_budget=None):
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(
            f"artifact protocol requires Qmax={CANONICAL_QMAX}, got {qmax}")
    manifest = _canonical_manifest()
    core = {
        "schema": SCHEMA,
        "shots": SHOTS,
        "alpha_familywise": ALPHA,
        "output_equivalence_l1": OUTPUT_EQ_L1,
        "modes": list(modes),
        "seeds": [int(seed) for seed in seeds],
        "qmax": int(qmax),
        "branch_cap": int(rq3.BRANCH_CAP),
        "circuit_budget_seconds": circuit_budget,
        "canonical_manifest_sha256": manifest["manifest_sha256"],
        "joint_shots": True,
        "nested_modes": ["m1", "m2", "m3"],
        "cluster_semantics": "synthetic footprint proxy",
        "identity_policy": (
            "every declared circuit/seed/mode identity emits exactly one "
            "terminal ok, inapplicable, or error record"
        ),
        "terminal_completion_semantics": (
            "a complete circuit frame retains terminal method timeouts and "
            "errors; rates use the common method-complete intersection"
        ),
        "cluster_inapplicable": (
            "no eligible two-qubit anchor with a downstream touching unitary "
            f"gate within WINDOW={WINDOW}"
        ),
        "denominators": [
            "attempted", "applicable", "executed",
            "measured-output-classified", "non-output-equivalent",
            "common-method-complete",
        ],
        "output_equivalence_scope": (
            "final measured-output distribution L1 only; never state or "
            "circuit equivalence"
        ),
        "asserted_subsystems": dict(METHOD_ASSERTED_SUBSYSTEM),
        "pruning_scope_rule": PRUNING_SCOPE_RULE,
        "exact_binomial_convention": dict(EXACT_BINOMIAL_CONVENTION),
        "bonferroni_family_rule": dict(BONFERRONI_FAMILY_RULE),
        "execution_shortcut_policy": dict(EXECUTION_SHORTCUT_POLICY),
        "grouped_monitor_channel": dict(GROUPED_MONITOR_CHANNEL),
        "final_read_policy": dict(rq3.FINAL_READ_POLICY),
    }
    protocol = dict(core)
    protocol.update({
        "methods": list(methods),
        "canonical_manifest": str(DEFAULT_MANIFEST),
        "canonical_keyfile": str(CANONICAL_KEYFILE),
        "oracle": "original intended circuit",
        "execution": "jointly mutated circuit",
        "affected": "any fault gate in original backward cone",
        "localization": (
            "first flag for loc_any; all-flag cone union for loc_all; "
            "loc_all headline requires all_faults_in_scope"
        ),
        "cluster_limitation": "synthetic compiled-footprint proxy",
    })
    source_paths = (
        Path(__file__),
        SCRIPT_DIR / "multi_mutation_eval.py",
        SCRIPT_DIR / "rq3_realexec_eval.py",
        SCRIPT_DIR / "rq3_manifest.py",
        SCRIPT_DIR / "mutation.py",
        SCRIPT_DIR / "batch_partition.py",
        SCRIPT_DIR / "utilities.py",
        SCRIPT_DIR / "circuits_selection.py",
        SCRIPT_DIR / "fast_analysis.py",
        SCRIPT_DIR / "mps_analysis.py",
        SCRIPT_DIR / "baseline_statistical.py",
        SCRIPT_DIR / "baseline_proq.py",
        SCRIPT_DIR / "baseline_liu.py",
        SCRIPT_DIR / "baseline_mqtdebugger.py",
    )
    fingerprints = runtime_fingerprints(core, source_paths=source_paths)
    fingerprints["circuit_budget"] = {
        "value": {
            "qmax": int(qmax),
            "branch_cap": int(rq3.BRANCH_CAP),
            "circuit_budget_seconds": circuit_budget,
            "mqt_timeout_budget": int(rq3.MQT_TIMEOUT_BUDGET),
            "mqt_wall_budget": float(rq3.MQT_WALL_BUDGET),
        }
    }
    fingerprints["circuit_budget"]["sha256"] = stable_hash(
        fingerprints["circuit_budget"]["value"]
    )
    return protocol, fingerprints


def new_artifact(methods=METHODS, modes=MODES, seeds=SEEDS, qmax=24,
                 circuit_budget=None):
    now = _utc_now()
    protocol, fingerprints = _artifact_protocol(
        methods, modes, seeds, qmax, circuit_budget
    )
    return {
        "schema": SCHEMA,
        "created_at": now,
        "updated_at": now,
        "protocol": protocol,
        "fingerprints": fingerprints,
        "records": [],
        "circuit_status": {},
    }


def _refuse_legacy_output(path):
    if Path(path).name == LEGACY_OUTPUT_NAME:
        raise ValueError(
            f"refusing to overwrite legacy surrogate artifact {path}; "
            f"use {DEFAULT_OUTPUT.name} or another distinct name"
        )


def atomic_pickle_dump(value, path):
    """Checkpoint via fsync + atomic replace, never via an in-place pickle."""
    path = Path(path)
    _refuse_legacy_output(path)
    validate_artifact(value, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_nonnegative_int(value):
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
    )


def _is_finite_number(value):
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _require_bool(mapping, field, location):
    value = mapping.get(field)
    if type(value) is not bool:
        raise ValueError(f"{location}: {field} must be a strict bool")
    return value


def _validate_accounting_numbers(value, location):
    """Reject impossible accounting values, including unrecognized extras.

    The artifact is pickle-based and therefore permits callers to append
    fields that the producer never wrote.  Walking dictionaries prevents a
    forged negative ``execution_cost`` or ``loc_precision`` from escaping
    validation merely because it is not currently consumed by aggregation.
    """
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            child_location = f"{location}.{key}"
            numeric_accounting_field = (
                key == "executions"
                or key in {
                    "shots", "fault_count", "candidate_count",
                    "first_cone_size", "flagged_union_cone_size",
                }
                or key.startswith("n_")
                or key == "count"
                or key.endswith("_count")
                or key.endswith("_executions")
                or key == "cost"
                or key.endswith("_cost")
                or key == "precision"
                or key.endswith("_precision")
                or key.endswith("_seconds")
            )
            if (
                nested is None
                and numeric_accounting_field
                and key not in {"shots", "executions"}
            ):
                raise ValueError(f"{child_location} must be numeric, not None")
            if nested is not None:
                if key == "executions":
                    if nested != "sim-only" and not _is_nonnegative_int(nested):
                        raise ValueError(
                            f"{child_location} must be a non-negative integer "
                            "or 'sim-only'"
                        )
                elif (
                    key in {
                        "shots", "fault_count", "candidate_count",
                        "first_cone_size", "flagged_union_cone_size",
                    }
                    or key.startswith("n_")
                    or key == "count"
                    or key.endswith("_count")
                    or key.endswith("_executions")
                ):
                    if not _is_nonnegative_int(nested):
                        raise ValueError(
                            f"{child_location} must be a non-negative integer"
                        )
                elif key == "cost" or key.endswith("_cost"):
                    if not _is_finite_number(nested) or float(nested) < 0.0:
                        raise ValueError(
                            f"{child_location} must be finite and non-negative"
                        )
                elif key == "precision" or key.endswith("_precision"):
                    if (
                        not _is_finite_number(nested)
                        or not 0.0 <= float(nested) <= 1.0
                    ):
                        raise ValueError(
                            f"{child_location} must be finite and in [0, 1]"
                        )
                elif key.endswith("_seconds"):
                    if not _is_finite_number(nested) or float(nested) < 0.0:
                        raise ValueError(
                            f"{child_location} must be finite and non-negative"
                        )
            if isinstance(nested, (dict, list, tuple)):
                _validate_accounting_numbers(nested, child_location)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list, tuple)):
                _validate_accounting_numbers(
                    nested, f"{location}[{index}]"
                )


def _validate_gate_list(value, location):
    if (
        not isinstance(value, list)
        or any(not _is_nonnegative_int(gate) for gate in value)
        or len(value) != len(set(int(gate) for gate in value))
    ):
        raise ValueError(
            f"{location} must be a list of distinct non-negative gate indices"
        )
    return [int(gate) for gate in value]


def _validate_scope_metadata(mapping, fault_gates, location, *, top_level=False):
    scoped = _validate_gate_list(
        mapping.get("in_scope_faults"), f"{location}.in_scope_faults"
    )
    if scoped != sorted(scoped) or not set(scoped).issubset(fault_gates):
        raise ValueError(f"{location}: malformed in-scope fault set")
    any_scope = _require_bool(mapping, "any_fault_in_scope", location)
    all_scope = _require_bool(mapping, "all_faults_in_scope", location)
    expected_any = bool(scoped)
    expected_all = bool(fault_gates) and set(scoped) == set(fault_gates)
    if any_scope != expected_any or all_scope != expected_all:
        raise ValueError(f"{location}: scope booleans disagree with fault set")
    if top_level and _require_bool(mapping, "in_scope", location) != expected_any:
        raise ValueError(f"{location}: in_scope disagrees with fault set")
    return scoped


def _validate_node(node, location):
    if (
        not isinstance(node, (list, tuple))
        or len(node) != 2
        or any(not _is_nonnegative_int(value) for value in node)
    ):
        raise ValueError(f"{location} must be a non-negative (gate, qubit) node")
    return tuple(int(value) for value in node)


def _validate_error_payload(value, location):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("type"), str)
        or not value["type"].strip()
        or not isinstance(value.get("message"), str)
    ):
        raise ValueError(f"{location}: malformed error payload")


def _validate_method_stats(stats, complete, method, location):
    if not isinstance(stats, dict):
        raise ValueError(f"{location}: stats must be a dictionary")
    _validate_accounting_numbers(stats, f"{location}.stats")
    for field in (
        "n_exec", "n_checked", "n_decisions", "n_affected",
        "n_null_shortcuts", "n_errors", "n_flagged",
    ):
        if field in stats and not _is_nonnegative_int(stats[field]):
            raise ValueError(f"{location}: {field} must be non-negative")
    for field, value in stats.items():
        if (
            field.startswith("applicability_")
            or field.startswith("run_")
            or field in {"timeout", "error", "budget_exhausted"}
        ) and not _is_nonnegative_int(value):
            raise ValueError(f"{location}: {field} must be non-negative")
    if "complete" in stats and type(stats["complete"]) is not bool:
        raise ValueError(f"{location}: stats.complete must be a strict bool")
    if "shots" in stats and stats["shots"] is not None:
        if not _is_nonnegative_int(stats["shots"]):
            raise ValueError(f"{location}: stats.shots must be non-negative")
    alpha_node = stats.get("alpha_node")
    if alpha_node is not None and (
        not _is_finite_number(alpha_node)
        or not 0.0 <= float(alpha_node) <= 1.0
    ):
        raise ValueError(f"{location}: alpha_node must be finite and in [0, 1]")
    errors = stats.get("errors")
    if errors is not None:
        if not isinstance(errors, list):
            raise ValueError(f"{location}: errors must be a list")
        for index, error in enumerate(errors):
            _validate_error_payload(error, f"{location}.errors[{index}]")
        if stats.get("n_errors") != len(errors):
            raise ValueError(f"{location}: n_errors disagrees with errors")
    reasons = stats.get("incomplete_reasons")
    if reasons is not None and (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise ValueError(f"{location}: malformed incomplete reasons")
    n_errors = int(stats.get("n_errors", 0))
    stats_complete = stats.get("complete")
    if complete and (n_errors != 0 or stats_complete is False):
        raise ValueError(f"{location}: complete result has incomplete/error stats")
    if method != "qmon" and stats_complete is not None:
        expected = bool(stats_complete and n_errors == 0)
        if complete != expected:
            raise ValueError(f"{location}: result completeness disagrees with stats")
    budget = stats.get("budget")
    if budget is not None:
        if not isinstance(budget, dict):
            raise ValueError(f"{location}: malformed MQT budget")
        for field in ("timeout_limit", "timeouts"):
            if not _is_nonnegative_int(budget.get(field)):
                raise ValueError(f"{location}: malformed MQT budget {field}")
        for field in ("wall_limit", "wall_seconds"):
            if (
                not _is_finite_number(budget.get(field))
                or float(budget[field]) < 0.0
            ):
                raise ValueError(f"{location}: malformed MQT budget {field}")
        if type(budget.get("remaining")) is not bool:
            raise ValueError(f"{location}: malformed MQT budget remaining")
        if int(budget["timeouts"]) > int(budget["timeout_limit"]):
            raise ValueError(f"{location}: MQT timeouts exceed their limit")


def _validate_method_result(method, result, record, location):
    if not isinstance(result, dict):
        raise ValueError(f"{location}: method result must be a dictionary")
    _validate_accounting_numbers(result, location)
    status = record["status"]
    fault_gates = set(record["gates"])
    applicable = _require_bool(result, "applicable", location)
    evaluated = _require_bool(result, "evaluated", location)
    complete = _require_bool(result, "complete", location)
    if result.get("terminal") is not True:
        raise ValueError(f"{location}: method result is not terminal")
    _validate_scope_metadata(result, fault_gates, location)

    flagged = result.get("flagged_nodes")
    if not isinstance(flagged, list):
        raise ValueError(f"{location}: flagged_nodes must be a list")
    normalized_flagged = [
        _validate_node(node, f"{location}.flagged_nodes[{index}]")
        for index, node in enumerate(flagged)
    ]
    if len(normalized_flagged) != len(set(normalized_flagged)):
        raise ValueError(f"{location}: duplicate flagged nodes")
    if not _is_nonnegative_int(result.get("n_flagged")):
        raise ValueError(f"{location}: n_flagged must be non-negative")
    if int(result["n_flagged"]) != len(flagged):
        raise ValueError(f"{location}: n_flagged disagrees with flagged_nodes")

    executions = result.get("executions")
    if method == "mqt":
        if executions != "sim-only":
            raise ValueError(f"{location}: MQT executions must be 'sim-only'")
    else:
        expected_field = (
            "B" if method == "qmon"
            else "K_qmon" if method == "qmon_pernode"
            else "K"
        )
        expected_executions = record.get(expected_field)
        if executions is None:
            if not (
                status == "inapplicable"
                and method == "qmon"
                and expected_executions is None
            ):
                raise ValueError(f"{location}: missing execution count")
        elif (
            not _is_nonnegative_int(executions)
            or (
                expected_executions is not None
                and int(executions) != int(expected_executions)
            )
        ):
            raise ValueError(f"{location}: invalid planned execution count")

    detected = result.get("detected")
    loc_any = result.get("loc_any")
    loc_all = result.get("loc_all")
    first_node = result.get("first_node")
    stats = result.get("stats")

    if status == "inapplicable":
        reason = record["inapplicable_reason"]
        if (
            applicable is not False
            or evaluated is not False
            or complete is not True
            or detected is not None
            or loc_any is not None
            or loc_all is not None
            or first_node is not None
            or flagged
            or stats != {}
            or result.get("reason") != reason
        ):
            raise ValueError(f"{location}: malformed inapplicable method branch")
        return

    if status == "error":
        if (
            applicable is not True
            or evaluated is not False
            or complete is not False
            or detected is not None
            or loc_any is not None
            or loc_all is not None
            or first_node is not None
            or flagged
            or stats != {}
            or not isinstance(result.get("reason"), str)
            or not result["reason"]
        ):
            raise ValueError(f"{location}: malformed error method branch")
        return

    if (
        applicable is not True
        or evaluated is not True
        or type(detected) is not bool
        or type(loc_any) is not bool
        or type(loc_all) is not bool
    ):
        raise ValueError(f"{location}: malformed evaluated method branch")
    if detected != bool(flagged):
        raise ValueError(f"{location}: detected disagrees with flagged nodes")
    expected_first = normalized_flagged[0] if normalized_flagged else None
    normalized_first = (
        _validate_node(first_node, f"{location}.first_node")
        if first_node is not None else None
    )
    if normalized_first != expected_first:
        raise ValueError(f"{location}: first_node is not the first flagged node")
    if (not detected and (loc_any or loc_all)):
        raise ValueError(f"{location}: localization cannot succeed without detection")
    if loc_any and not result["any_fault_in_scope"]:
        raise ValueError(f"{location}: loc_any contradicts method scope")
    if loc_all and not result["all_faults_in_scope"]:
        raise ValueError(f"{location}: loc_all contradicts method scope")

    _validate_method_stats(stats, complete, method, location)
    if "n_flagged" in stats and int(stats["n_flagged"]) != len(flagged):
        raise ValueError(f"{location}: stats.n_flagged disagrees with flags")
    for smaller, larger in (
        ("n_affected", "n_checked"),
        ("n_decisions", "n_checked"),
        ("n_flagged", "n_decisions"),
    ):
        if smaller in stats and larger in stats and stats[smaller] > stats[larger]:
            raise ValueError(f"{location}: impossible {smaller}/{larger} counts")

    localization_fields = (
        "faults_in_first_cone", "faults_in_flagged_union",
        "first_cone_size", "flagged_union_cone_size",
    )
    present = [field in result for field in localization_fields]
    if any(present) and not all(present):
        raise ValueError(f"{location}: partial localization accounting")
    if all(present):
        first_faults = _validate_gate_list(
            result["faults_in_first_cone"],
            f"{location}.faults_in_first_cone",
        )
        union_faults = _validate_gate_list(
            result["faults_in_flagged_union"],
            f"{location}.faults_in_flagged_union",
        )
        if (
            first_faults != sorted(first_faults)
            or union_faults != sorted(union_faults)
            or not set(first_faults).issubset(fault_gates)
            or not set(union_faults).issubset(fault_gates)
            or not set(first_faults).issubset(union_faults)
        ):
            raise ValueError(f"{location}: malformed localization fault sets")
        if not _is_nonnegative_int(result["first_cone_size"]):
            raise ValueError(f"{location}: invalid first cone size")
        if not _is_nonnegative_int(result["flagged_union_cone_size"]):
            raise ValueError(f"{location}: invalid flagged-union cone size")
        if (
            len(first_faults) > int(result["first_cone_size"])
            or len(union_faults) > int(result["flagged_union_cone_size"])
            or int(result["first_cone_size"])
            > int(result["flagged_union_cone_size"])
            or loc_any != bool(first_faults)
            or loc_all != (
                bool(fault_gates) and set(union_faults) == fault_gates
            )
        ):
            raise ValueError(f"{location}: localization metrics are inconsistent")
    elif complete or detected:
        raise ValueError(f"{location}: completed/detected result lacks localization accounting")


def _validate_record_semantics(record, allowed_methods, location):
    if not isinstance(record, dict):
        raise ValueError(f"{location}: record must be a dictionary")
    _validate_accounting_numbers(record, location)
    if not _is_nonnegative_int(record.get("seed")):
        raise ValueError(f"{location}: seed must be a non-negative integer")
    if not isinstance(record.get("circuit"), str) or not record["circuit"]:
        raise ValueError(f"{location}: circuit must be a non-empty string")
    gates = _validate_gate_list(record.get("gates"), f"{location}.gates")
    if not _is_nonnegative_int(record.get("fault_count")):
        raise ValueError(f"{location}: fault_count must be non-negative")
    if int(record["fault_count"]) != len(gates):
        raise ValueError(f"{location}: fault_count disagrees with gates")
    if record.get("fault_gates") != gates:
        raise ValueError(f"{location}: fault_gates disagree with gates")
    if record.get("application_order") != gates:
        raise ValueError(f"{location}: application_order disagrees with gates")
    for field in ("K", "K_qmon"):
        if not _is_nonnegative_int(record.get(field)):
            raise ValueError(f"{location}: {field} must be non-negative")
    if record.get("B") is not None and not _is_nonnegative_int(record["B"]):
        raise ValueError(f"{location}: B must be non-negative or None")
    if not _is_nonnegative_int(record.get("shots")) or record["shots"] == 0:
        raise ValueError(f"{location}: shots must be a positive integer")
    alpha = record.get("alpha_familywise")
    if (
        not _is_finite_number(alpha)
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise ValueError(f"{location}: invalid familywise alpha")
    _validate_scope_metadata(record, set(gates), location, top_level=True)

    if record.get("attempted") is not True or record.get("terminal") is not True:
        raise ValueError(f"{location}: record is not an explicit terminal attempt")
    applicable = _require_bool(record, "applicable", location)
    executed = _require_bool(record, "executed", location)
    complete = _require_bool(record, "complete", location)
    status = record.get("status")
    if status not in {"ok", "inapplicable", "error"}:
        raise ValueError(f"{location}: invalid terminal status")

    output_equivalent = record.get("output_equivalent")
    output_l1 = record.get("output_l1")
    if status == "ok":
        if applicable is not True or executed is not True:
            raise ValueError(f"{location}: malformed ok status")
        if type(output_equivalent) is not bool:
            raise ValueError(f"{location}: ok record lacks output classification")
        if not _is_finite_number(output_l1) or float(output_l1) < 0.0:
            raise ValueError(f"{location}: invalid output L1 distance")
        if output_equivalent != (float(output_l1) < OUTPUT_EQ_L1):
            raise ValueError(f"{location}: output equivalence contradicts L1 distance")
        if "error" in record or "inapplicable_reason" in record:
            raise ValueError(f"{location}: ok branch contains error metadata")
    elif status == "inapplicable":
        if (
            applicable is not False
            or executed is not False
            or output_equivalent is not None
            or output_l1 is not None
            or not isinstance(record.get("inapplicable_reason"), str)
            or not record["inapplicable_reason"]
            or "error" in record
        ):
            raise ValueError(f"{location}: malformed inapplicable status")
    else:
        if (
            applicable is not True
            or executed is not False
            or complete is not False
            or output_equivalent is not None
            or output_l1 is not None
            or "inapplicable_reason" in record
        ):
            raise ValueError(f"{location}: malformed error status")
        _validate_error_payload(record.get("error"), location)

    if "masking_checked" in record:
        masking_checked = _require_bool(record, "masking_checked", location)
        masked = record.get("masked")
        if masked is not None and type(masked) is not bool:
            raise ValueError(f"{location}: masked must be bool or None")
        if status != "ok" and (masking_checked or masked is not None):
            raise ValueError(f"{location}: non-executed record has masking result")

    methods = record.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(allowed_methods):
        raise ValueError(f"{location}: method coverage mismatch")
    for method, result in methods.items():
        _validate_method_result(method, result, record, f"{location}/{method}")
    expected_complete = bool(
        status in {"ok", "inapplicable"}
        and all(result["complete"] for result in methods.values())
    )
    if complete != expected_complete:
        raise ValueError(f"{location}: record completeness mismatch")


def validate_artifact(artifact, path="artifact"):
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA:
        raise ValueError(
            f"{path} is not a {SCHEMA} artifact (the old exact-RDM pickle "
            "cannot be mixed with real-execution records)"
        )
    if not isinstance(artifact.get("records"), list):
        raise ValueError(f"{path} has no valid records list")
    protocol = artifact.get("protocol", {})
    if protocol.get("qmax") != CANONICAL_QMAX:
        raise ValueError(
            f"{path} has invalid/missing qmax={protocol.get('qmax')!r}; "
            f"Qmax={CANONICAL_QMAX} is required")
    stored_fingerprints = artifact.get("fingerprints")
    if not isinstance(stored_fingerprints, dict):
        raise ValueError(f"{path} has no protocol checksums")
    expected_protocol, current_fingerprints = _artifact_protocol(
        protocol.get("methods", ()),
        protocol.get("modes", ()),
        protocol.get("seeds", ()),
        protocol.get("qmax"),
        protocol.get("circuit_budget_seconds"),
    )
    if protocol != expected_protocol:
        raise ValueError(f"{path} protocol drift")
    drift = {}
    for field in (
        "sources", "qasm_corpus", "dependencies", "protocol",
        "circuit_budget",
    ):
        stored_hash = stored_fingerprints.get(field, {}).get("sha256")
        current_hash = current_fingerprints[field]["sha256"]
        if stored_hash != current_hash:
            drift[field] = (stored_hash, current_hash)
    if drift:
        raise ValueError(f"{path} checksum mismatch: {drift}")
    identities = [_record_id(record) for record in artifact["records"]]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path} contains duplicate circuit/seed/mode records")
    allowed_methods = set(protocol.get("methods", ()))
    for identity, record in zip(identities, artifact["records"]):
        if (
            record.get("mode") not in protocol.get("modes", ())
            or record.get("seed") not in protocol.get("seeds", ())
        ):
            raise ValueError(f"{path}: {identity} is outside declared slots")
        actual_methods = set(record.get("methods", ()))
        if actual_methods != allowed_methods:
            raise ValueError(
                f"{path}: {identity} method coverage mismatch "
                f"{actual_methods} != {allowed_methods}")
        _validate_record_semantics(
            record, allowed_methods, f"{path}: {identity}"
        )
        plan = record.get("qmon_plan")
        rq3._validate_qmax_plan(plan, f"{path}: {identity}")
        if "qmon" in record.get("methods", {}):
            qmon = record["methods"]["qmon"]
            if (record.get("status") != "inapplicable"
                    and (qmon.get("complete") or "diag" in qmon)):
                rq3._validate_qmon_batch_diagnostics(
                    qmon, plan, f"{path}: {identity}")
    return artifact


def load_artifact(path):
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    return validate_artifact(artifact, path)


def _validate_resume(artifact, methods, modes, seeds, qmax,
                     circuit_budget=None):
    protocol = artifact["protocol"]
    expected = {
        "shots": SHOTS,
        "alpha_familywise": ALPHA,
        "methods": list(methods),
        "modes": list(modes),
        "seeds": [int(seed) for seed in seeds],
        "qmax": int(qmax),
        "circuit_budget_seconds": circuit_budget,
    }
    mismatch = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatch:
        raise ValueError(
            "resume configuration differs from checkpoint; use a separate "
            f"output file. mismatches={mismatch}"
        )


@contextlib.contextmanager
def _circuit_budget(seconds):
    if not seconds:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("--circuit-budget requires SIGALRM on this platform")
    previous = signal.getsignal(signal.SIGALRM)

    def exceeded(*_):
        raise rq3.BudgetExceeded()

    signal.signal(signal.SIGALRM, exceeded)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _circuit_records_complete(records, methods, modes, seeds):
    """Whether every requested identity has a validated terminal record.

    Method-level ``complete=False`` is a reportable terminal outcome (for
    example, a debugger timeout), not a reason to delete and repeat an entire
    circuit forever. Detection rates separately use the common method-complete
    intersection.
    """
    expected = {
        (int(seed), mode) for seed in seeds for mode in modes
    }
    by_slot = {(int(record["seed"]), record["mode"]): record
               for record in records}
    if set(by_slot) != expected or len(by_slot) != len(records):
        return False
    return all(
        record.get("status") in {"ok", "inapplicable", "error"}
        and record.get("terminal") is True
        and all(
            method in record.get("methods", {})
            and record["methods"][method].get("terminal") is True
            and type(record["methods"][method].get("complete")) is bool
            for method in methods
        )
        for record in records
    )


def _circuit_methods_complete(records, methods, modes, seeds):
    return bool(
        _circuit_records_complete(records, methods, modes, seeds)
        and all(
            record.get("complete") is True
            and all(
                record["methods"][method].get("complete") is True
                for method in methods
            )
            for record in records
        )
    )


def run_keys(
    keys,
    out=DEFAULT_OUTPUT,
    methods=METHODS,
    modes=MODES,
    seeds=SEEDS,
    qmax=24,
    resume=True,
    circuit_budget=None,
):
    """Run sequential circuits with an atomic checkpoint after each one."""
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(
            f"multi-error protocol requires Qmax={CANONICAL_QMAX}, got {qmax}")
    methods = tuple(methods)
    modes = tuple(modes)
    seeds = tuple(int(seed) for seed in seeds)
    out = Path(out) if out is not None else None
    if out is not None:
        _refuse_legacy_output(out)

    if out is not None and out.exists() and not resume:
        raise FileExistsError(
            f"refusing to replace existing checkpoint {out} with "
            "--no-resume; choose a new output path"
        )

    if out is not None and resume and out.exists():
        artifact = load_artifact(out)
        _validate_resume(
            artifact, methods, modes, seeds, qmax, circuit_budget
        )
        print(
            f"resume: {len(artifact['circuit_status'])} circuit statuses, "
            f"{len(artifact['records'])} records from {out}",
            flush=True,
        )
    else:
        artifact = new_artifact(
            methods, modes, seeds, qmax, circuit_budget
        )

    normalized = [_normal_key(key) for key in keys]
    started = time.perf_counter()
    for index, key in enumerate(normalized, start=1):
        prior = artifact["circuit_status"].get(key, {})
        existing = [
            record for record in artifact["records"]
            if record.get("circuit") == key
        ]
        if (resume and prior.get("status") == "complete"
                and _circuit_records_complete(
                    existing, methods, modes, seeds
                )):
            print(f"[{index}/{len(normalized)}] skip completed {key}", flush=True)
            continue
        one_started = time.perf_counter()
        # Incomplete records are never append-resumed: remove the circuit and
        # regenerate all requested slots/methods under fresh budgets.
        artifact["records"] = [
            record for record in artifact["records"]
            if record.get("circuit") != key
        ]
        before = len(artifact["records"])
        try:
            with _circuit_budget(circuit_budget):
                records = evaluate_circuit(
                    key,
                    seeds=seeds,
                    qmax=qmax,
                    methods=methods,
                    modes=modes,
                )
            artifact["records"].extend(records)
            status = (
                "complete"
                if _circuit_records_complete(records, methods, modes, seeds)
                else ("no-records" if not records else "incomplete")
            )
            artifact["circuit_status"][key] = {
                "status": status,
                "records": len(records),
                "methods": list(methods),
                "all_methods_complete": _circuit_methods_complete(
                    records, methods, modes, seeds
                ),
                "elapsed_seconds": time.perf_counter() - one_started,
            }
        except rq3.BudgetExceeded:
            artifact["circuit_status"][key] = {
                "status": "budget-timeout",
                "records": 0,
                "methods": list(methods),
                "budget_seconds": circuit_budget,
            }
        except Exception as exc:
            artifact["circuit_status"][key] = {
                "status": "error",
                "records": 0,
                "methods": list(methods),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                },
            }
        artifact["updated_at"] = _utc_now()
        if out is not None:
            atomic_pickle_dump(artifact, out)
        current = artifact["circuit_status"][key]
        detail = ""
        if "error" in current:
            error = current["error"]
            detail = f", {error['type']}: {error['message']}"
        print(
            f"[{index}/{len(normalized)}] {key}: {current['status']}, "
            f"+{len(artifact['records']) - before} records, "
            f"{time.perf_counter() - one_started:.1f}s "
            f"(total {time.perf_counter() - started:.1f}s){detail}",
            flush=True,
        )
    return artifact


def _record_id(record):
    return (
        record["circuit"],
        int(record["seed"]),
        record["mode"],
    )


def _merge_record(existing, incoming):
    for key, value in incoming.items():
        if key in {"methods", "complete"}:
            continue
        if key not in existing:
            existing[key] = copy.deepcopy(value)
        elif existing[key] != value:
            raise ValueError(
                f"record metadata conflict for {_record_id(existing)} at {key}"
            )
    for method, result in incoming["methods"].items():
        if method in existing["methods"] and existing["methods"][method] != result:
            raise ValueError(
                f"duplicate method conflict for {_record_id(existing)} / {method}"
            )
        existing["methods"][method] = copy.deepcopy(result)
    existing["complete"] = bool(
        existing.get("status") in {"ok", "inapplicable"}
        and all(result.get("complete") is True
                for result in existing["methods"].values())
    )


def _validate_identity_coverage(records, keys, seeds, modes, location):
    expected = {
        (key, int(seed), mode)
        for key in keys for seed in seeds for mode in modes
    }
    identities = [_record_id(record) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{location}: duplicate circuit/seed/mode identities")
    by_identity = {
        identity: record for identity, record in zip(identities, records)
    }
    if set(by_identity) != expected:
        raise ValueError(
            f"{location}: slot coverage failure: "
            f"missing={sorted(expected - set(by_identity))[:5]}, "
            f"extra={sorted(set(by_identity) - expected)[:5]}"
        )
    return by_identity


def _strict_validate_merged(merged):
    # Strict merging adds the circuit-set and slot checks below, but the
    # merged artifact must first pass the same full record validator used for
    # every input artifact, so a merge is never validated less strictly.
    validate_artifact(merged, "strict merge")
    protocol = merged["protocol"]
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError(
            f"strict merge requires all methods {METHODS}, got "
            f"{tuple(protocol['methods'])}"
        )
    if tuple(protocol["modes"]) != MODES or tuple(protocol["seeds"]) != SEEDS:
        raise ValueError("strict merge requires the fixed modes and seeds")
    with CANONICAL_KEYFILE.open() as handle:
        keys = [
            _normal_key(line.split("#", 1)[0].strip())
            for line in handle
            if line.split("#", 1)[0].strip()
        ]
    if len(keys) != EXPECTED_CIRCUITS:
        raise ValueError(
            f"circuit keyfile must contain {EXPECTED_CIRCUITS} circuits"
        )
    manifest = _canonical_manifest()
    manifest_keys = [entry["circuit"] for entry in manifest["circuits"]]
    if keys != manifest_keys:
        raise ValueError(
            "circuit keyfile and mutation list circuit order disagree"
        )
    by_identity = _validate_identity_coverage(
        merged["records"], keys, SEEDS, MODES, "strict merge")
    for identity, record in by_identity.items():
        key, seed, mode = identity
        status = record.get("status")
        if (record.get("canonical_manifest_sha256")
                != manifest["manifest_sha256"]):
            raise ValueError(f"mutation list drift at {identity}")
        if (record.get("fault_count") != len(record.get("gates", ()))
                or record.get("fault_gates") != record.get("gates")
                or record.get("application_order") != record.get("gates")):
            raise ValueError(f"fault identity mismatch at {identity}")
        rq3._validate_qmax_plan(
            record.get("qmon_plan"), f"strict merge {identity}")
        if mode in ("m1", "m2", "m3"):
            nested = dict(canonical_nested_variants(manifest, key, seed))[mode]
            expected_gates = [int(slot["gate"]) for slot in nested]
            expected_replacements = [
                _canonical_replacement(slot) for slot in nested
            ]
            if (record.get("gates") != expected_gates
                    or record.get("canonical_slot_ids")
                    != [slot["slot_id"] for slot in nested]
                    or record.get("variant_source") != "canonical-nested"
                    or (status == "inapplicable"
                        and record.get("applicable") is not False)
                    or (status != "inapplicable"
                        and record.get("applicable") is not True)
                    or (status in {"ok", "inapplicable"}
                        and record.get("replacements")
                        != expected_replacements)):
                raise ValueError(
                    f"strict merge replacement conflict with the mutation list at {identity}"
                )
        elif record.get("variant_source") != "synthetic-footprint-proxy":
            raise ValueError(f"invalid clustered proxy source at {identity}")
        elif status == "inapplicable":
            if (record.get("applicable") is not False
                    or record.get("executed") is not False
                    or record.get("gates") != []
                    or record.get("replacements") != []
                    or record.get("canonical_slot_ids") != []
                    or record.get("output_equivalent") is not None):
                raise ValueError(
                    f"malformed inapplicable clustered slot at {identity}")
        elif (record.get("applicable") is not True
              or len(record.get("gates", ())) not in (2, 3)):
            raise ValueError(f"invalid applicable clustered proxy at {identity}")
    merged["circuit_status"] = {
        key: {
            "status": (
                "complete"
                if _circuit_records_complete(
                    [by_identity[(key, seed, mode)]
                     for seed in SEEDS for mode in MODES],
                    METHODS, MODES, SEEDS)
                else "incomplete"
            ),
            "records": len(SEEDS) * len(MODES),
            "methods": list(METHODS),
            "all_methods_complete": _circuit_methods_complete(
                [by_identity[(key, seed, mode)]
                 for seed in SEEDS for mode in MODES],
                METHODS, MODES, SEEDS),
            "merged": True,
        }
        for key in keys
    }


def merge_artifacts(paths, *, strict=True):
    """Merge chunks/method shards; strict mode requires the full protocol."""
    artifacts = [load_artifact(path) for path in paths]
    if not artifacts:
        raise ValueError("at least one input artifact is required")
    first_protocol = artifacts[0]["protocol"]
    for artifact in artifacts[1:]:
        protocol = artifact["protocol"]
        for field in (
            "shots",
            "alpha_familywise",
            "output_equivalence_l1",
            "modes",
            "seeds",
            "qmax",
            "branch_cap",
            "circuit_budget_seconds",
            "canonical_manifest_sha256",
        ):
            if protocol.get(field) != first_protocol.get(field):
                raise ValueError(
                    f"cannot merge artifacts with different {field}: "
                    f"{protocol.get(field)} != {first_protocol.get(field)}"
                )
        for field in (
            "sources", "qasm_corpus", "dependencies", "protocol",
            "circuit_budget",
        ):
            if (artifact["fingerprints"][field]["sha256"]
                    != artifacts[0]["fingerprints"][field]["sha256"]):
                raise ValueError(
                    f"cannot merge artifacts with different {field} checksum"
                )

    merged = new_artifact(
        methods=(),
        modes=first_protocol["modes"],
        seeds=first_protocol["seeds"],
        qmax=first_protocol["qmax"],
        circuit_budget=first_protocol.get("circuit_budget_seconds"),
    )
    merged["created_at"] = min(a["created_at"] for a in artifacts)
    merged["merged_from"] = [str(Path(path)) for path in paths]
    by_id = {}
    method_union = []
    for artifact in artifacts:
        for method in artifact["protocol"].get("methods", []):
            if method not in method_union:
                method_union.append(method)
        for record in artifact["records"]:
            identity = _record_id(record)
            if identity not in by_id:
                by_id[identity] = copy.deepcopy(record)
            else:
                _merge_record(by_id[identity], record)

    mode_order = {mode: index for index, mode in enumerate(MODES)}
    merged["records"] = sorted(
        by_id.values(),
        key=lambda record: (
            record["circuit"],
            int(record["seed"]),
            mode_order.get(record["mode"], len(mode_order)),
        ),
    )
    merged_methods = [
        method for method in METHODS if method in method_union
    ] + [method for method in method_union if method not in METHODS]
    protocol, fingerprints = _artifact_protocol(
        merged_methods,
        first_protocol["modes"],
        first_protocol["seeds"],
        first_protocol["qmax"],
        first_protocol.get("circuit_budget_seconds"),
    )
    merged["protocol"] = protocol
    merged["fingerprints"] = fingerprints
    if strict:
        _strict_validate_merged(merged)
    else:
        circuits = sorted({record["circuit"] for record in merged["records"]})
        for key in circuits:
            records = [
                record for record in merged["records"]
                if record["circuit"] == key
            ]
            complete = _circuit_records_complete(
                records,
                merged_methods,
                first_protocol["modes"],
                first_protocol["seeds"],
            )
            merged["circuit_status"][key] = {
                "status": "complete" if complete else "incomplete",
                "records": len(records),
                "methods": list(merged_methods),
                "all_methods_complete": _circuit_methods_complete(
                    records,
                    merged_methods,
                    first_protocol["modes"],
                    first_protocol["seeds"],
                ),
                "merged": True,
            }
    merged["updated_at"] = _utc_now()
    return merged


def _seed_rate(records, seeds, predicate):
    values = []
    for seed in seeds:
        subset = [record for record in records if record["seed"] == seed]
        if subset:
            values.append(100.0 * np.mean([bool(predicate(r)) for r in subset]))
    return values


def _mean_sd(values):
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    if len(values) < 2:
        return f"{mean:.1f}+/-n/a [seed n={len(values)}]"
    return (
        f"{mean:.1f}+/-{statistics.stdev(values):.1f} "
        f"[seed n={len(values)}]"
    )


def _pooled(records, predicate):
    numerator = sum(bool(predicate(record)) for record in records)
    return numerator, len(records)


def aggregate(artifact_or_records, methods=None):
    """Print per-mode rates, denominators, and execution failures."""
    if not isinstance(artifact_or_records, dict):
        raise ValueError(
            "aggregate requires one current-schema strict merged artifact; "
            "record lists and legacy exact-RDM artifacts are rejected"
        )
    artifact = copy.deepcopy(
        validate_artifact(artifact_or_records, "aggregate input")
    )
    _strict_validate_merged(artifact)
    records = artifact["records"]

    available = []
    for method in METHODS:
        if any(method in record.get("methods", {}) for record in records):
            available.append(method)
    if methods is not None:
        requested = set(methods)
        available = [method for method in available if method in requested]
    seeds = sorted({int(record["seed"]) for record in records})

    print(
        f"\nMULTI-ERROR REAL EXECUTION: records={len(records)}, "
        f"circuits={len({r['circuit'] for r in records})}, "
        f"S={SHOTS}, familywise alpha={ALPHA}, seeds={seeds}"
    )
    print(
        "Output-equivalent means final measured-output L1 < 1e-9; it does "
        "not mean state or circuit equivalence."
    )
    print(
        "loc_any/loc_all are structural original-cone containment metrics "
        "conditional on detection, not independent empirical signals."
    )

    statuses = artifact.get("circuit_status", {})
    status_counts = {}
    for value in statuses.values():
        status = value.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Circuit checkpoint statuses: {status_counts}")
    for circuit, value in statuses.items():
        if value.get("status") == "complete":
            continue
        print(f"  circuit-status {circuit}: {value.get('status')}")

    print("\nProtocol denominators (output equivalence is measured-output only):")
    for mode in MODES:
        attempted = [
            record for record in records
            if record.get("mode") == mode and record.get("attempted") is True
        ]
        applicable = [
            record for record in attempted if record.get("applicable") is True
        ]
        executed = [
            record for record in applicable if record.get("executed") is True
        ]
        classified = [
            record for record in executed
            if isinstance(record.get("output_equivalent"), bool)
        ]
        non_equivalent = [
            record for record in classified
            if not record["output_equivalent"]
        ]
        print(
            f"  {mode}: attempted={len(attempted)}, "
            f"applicable={len(applicable)}, executed={len(executed)}, "
            f"measured-output-classified={len(classified)}, "
            f"non-output-equivalent={len(non_equivalent)}"
        )

    strata = []
    for mode in MODES:
        selected = [record for record in records if record["mode"] == mode]
        if mode == "clus":
            for fault_count in (2, 3):
                subset = [
                    record for record in selected
                    if len(record.get("gates", ())) == fault_count
                ]
                if subset:
                    strata.append((
                        f"clus-{fault_count}fault",
                        subset,
                        "synthetic footprint proxy; paired eligible subset",
                    ))
        elif selected:
            strata.append((mode, selected, "nested paired variant from the mutation list"))

    for label, mode_records, stratum_note in strata:
        attempted = [
            record for record in mode_records
            if record.get("attempted") is True
        ]
        applicable = [
            record for record in attempted if record.get("applicable") is True
        ]
        executed = [
            record for record in applicable if record.get("executed") is True
        ]
        valid = [
            record
            for record in executed
            if isinstance(record.get("output_equivalent"), bool)
        ]
        non_equivalent = [
            record for record in valid if not record["output_equivalent"]
        ]
        masked = [record for record in valid if record.get("masked") is True]
        common_valid = [
            record for record in valid
            if all(
                record.get("methods", {}).get(method, {}).get("evaluated")
                and record["methods"][method].get("complete", False)
                for method in available
            )
        ]
        common_non_equivalent = [
            record for record in common_valid
            if not record["output_equivalent"]
        ]
        record_errors = len(applicable) - len(executed)
        masking_errors = sum(
            record.get("masking_checked") and record.get("masked") is None
            for record in mode_records
        )
        print(
            f"\n[{label}] {stratum_note}; records={len(mode_records)}; "
            f"attempted={len(attempted)}; applicable={len(applicable)}; "
            f"executed={len(executed)}; output-classified={len(valid)}; "
            f"non-output-equivalent={len(non_equivalent)}; masked={len(masked)}; "
            f"common-complete={len(common_valid)}/{len(valid)}; "
            f"common-complete non-output-equivalent="
            f"{len(common_non_equivalent)}/{len(non_equivalent)}; "
            f"record-errors={record_errors}; masking-errors={masking_errors}"
        )

        for method in available:
            all_evaluated = common_valid
            evaluated = common_non_equivalent
            evaluated_scope = [
                record for record in evaluated
                if record["methods"][method]["any_fault_in_scope"]
            ]
            detected = [
                record
                for record in evaluated
                if record["methods"][method].get("detected")
            ]
            detected_any_scope = [
                record for record in detected
                if record["methods"][method]["any_fault_in_scope"]
            ]
            detected_all_scope = [
                record for record in detected
                if record["methods"][method]["all_faults_in_scope"]
            ]
            equivalent_evaluated = [
                record for record in all_evaluated if record["output_equivalent"]
            ]
            equivalent_detected = sum(
                bool(record["methods"][method].get("detected"))
                for record in equivalent_evaluated
            )
            det_all_seed = _seed_rate(
                evaluated,
                seeds,
                lambda record: record["methods"][method]["detected"],
            )
            det_scope_seed = _seed_rate(
                evaluated_scope,
                seeds,
                lambda record: record["methods"][method]["detected"],
            )
            loc_any_seed = _seed_rate(
                detected_any_scope,
                seeds,
                lambda record: record["methods"][method]["loc_any"],
            )
            loc_all_seed = _seed_rate(
                detected_all_scope,
                seeds,
                lambda record: record["methods"][method]["loc_all"],
            )
            da_num, da_den = _pooled(
                evaluated,
                lambda record: record["methods"][method]["detected"],
            )
            ds_num, ds_den = _pooled(
                evaluated_scope,
                lambda record: record["methods"][method]["detected"],
            )
            la_num, la_den = _pooled(
                detected_any_scope,
                lambda record: record["methods"][method]["loc_any"],
            )
            ll_num, ll_den = _pooled(
                detected_all_scope,
                lambda record: record["methods"][method]["loc_all"],
            )

            stats = [
                record["methods"][method].get("stats", {})
                for record in all_evaluated
            ]
            realized = [entry.get("n_exec", 0) for entry in stats]
            planned_numeric = [
                record["methods"][method]["executions"]
                for record in all_evaluated
                if isinstance(record["methods"][method]["executions"], int)
            ]
            all_method_results = [
                record["methods"][method]
                for record in valid
                if method in record.get("methods", {})
            ]
            incomplete_records = sum(
                not result.get("complete", False)
                for result in all_method_results
            )
            coverage_failures = sum(
                not record["methods"][method]["all_faults_in_scope"]
                for record in evaluated
            )
            if method == "mqt":
                execution_text = "sim-only"
            elif method == "qmon":
                execution_text = (
                    f"protocol batch-runs mean={np.mean(planned_numeric):.1f}"
                    if planned_numeric
                    else "n/a"
                )
            elif method == "qmon_pernode":
                execution_text = (
                    f"protocol per-node runs mean={np.mean(planned_numeric):.1f}"
                    if planned_numeric
                    else "n/a"
                )
            else:
                execution_text = (
                    f"realized affected-runs mean={np.mean(realized):.1f}; "
                    f"protocol planned mean={np.mean(planned_numeric):.1f}"
                    if realized and planned_numeric
                    else "n/a"
                )
            print(
                f"  {method}: common complete intersection="
                f"{len(evaluated)}/{len(non_equivalent)} non-output-equivalent; "
                f"detect all {_mean_sd(det_all_seed)}, pooled={da_num}/{da_den}; "
                f"detect in-scope {_mean_sd(det_scope_seed)}, pooled={ds_num}/{ds_den}"
            )
            print(
                f"    loc_any {_mean_sd(loc_any_seed)}, pooled={la_num}/{la_den} "
                f"complete+detected+any-in-scope; loc_all "
                f"{_mean_sd(loc_all_seed)}, pooled={ll_num}/{ll_den} "
                f"complete+detected+all-faults-in-scope; "
                f"coverage failures(all-faults)={coverage_failures}/"
                f"{len(evaluated)}; {execution_text}; "
                f"incomplete records excluded={incomplete_records}; "
                f"output-equivalent internal flags={equivalent_detected}/"
                f"{len(equivalent_evaluated)}"
            )
            if method == "qmon":
                diagnostics = [
                    record["methods"][method].get("diag", {})
                    for record in all_evaluated
                ]
                print(
                    "    QMon aligned-trajectory diagnostics: "
                    f"physical-batches={sum(len(d.get('batches', ())) for d in diagnostics)}, "
                    f"cap-hits={sum(d.get('cap_hits', 0) for d in diagnostics)}, "
                    f"incomplete={sum(not d.get('complete', False) for d in diagnostics)}"
                )
            if method == "mqt":
                affected_count = sum(entry.get("n_affected", 0) for entry in stats)
                applicable_count = sum(entry.get("n_applicable", 0) for entry in stats)
                print(
                    f"    MQT applicability={applicable_count}/{affected_count} "
                    "affected checkpoints; execution model is simulation-only"
                )


# ===========================================================================
# Built-in fast invariants; no dataset sweep.
# ===========================================================================
def self_check():
    # Set-valued affected logic.
    assert any_fault_in_cone({2, 9}, {1, 2, 3})
    assert not any_fault_in_cone({8, 9}, {1, 2, 3})

    # Deterministic selection and exact replacements, including exact singles.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.25, 1)
    candidates = [0, 1, 2]
    assert select_variants(qc, candidates, 2, 7) == select_variants(
        qc, candidates, 2, 7
    )
    mutant_a, manifest_a = mutate_many_with_manifest(
        qc, [0, 1], key="self.qasm", seed=7
    )
    mutant_b, manifest_b = mutate_many_with_manifest(
        qc, [0, 1], key="self.qasm", seed=7
    )
    assert manifest_a == manifest_b
    assert [operation_signature(mutant_a, i) for i in range(3)] == [
        operation_signature(mutant_b, i) for i in range(3)
    ]
    for item in manifest_a:
        single, single_manifest = mutate_many_with_manifest(
            qc, [item["gate"]], key="self.qasm", seed=7
        )
        assert single_manifest[0]["replacement"] == item["replacement"]
        assert operation_signature(single, item["gate"]) == item["replacement"]

    # Singleton QMon wrapper delegates verbatim to the current RQ3 path.
    sentinel = object()
    called = []
    original_ctx = rq3.make_qmon_ctx

    def fake_ctx(mqc, key, oracle, gate):
        called.append((mqc, key, oracle, gate))
        return sentinel

    rq3.make_qmon_ctx = fake_ctx
    try:
        fake_mqc, fake_oracle = object(), object()
        assert make_qmon_ctx_many(fake_mqc, "x.qasm", fake_oracle, {11}) is sentinel
        assert called == [(fake_mqc, "x.qasm", fake_oracle, 11)]
    finally:
        rq3.make_qmon_ctx = original_ctx

    # Full scan must retain first flag while collecting a later cone for all.
    nodes = [(1, 0), (2, 0), (3, 0)]

    class FakeOracle:
        key = "self"
        qmon_nodes = nodes
        K_qmon = 3
        K = 3
        p1e = {node: 0.0 for node in nodes}
        cone = {nodes[0]: {10}, nodes[1]: {20}, nodes[2]: {10, 20}}

    original_flag = rq3.qmon_node_flag
    seen = []

    def fake_flag(node, p1_read, oracle, **_):
        seen.append(node)
        return p1_read > 0.5

    rq3.qmon_node_flag = fake_flag
    try:
        ctx = {
            "reads": lambda node: {nodes[0]: 1.0, nodes[1]: 1.0, nodes[2]: 0.0}[node],
            "raw_p1": {node: 0.0 for node in nodes},
        }
        detected, first, flagged, stats = scan_method_full(
            "qmon", None, FakeOracle(), "tag", {10, 20}, ctx
        )
    finally:
        rq3.qmon_node_flag = original_flag
    localization = localization_metrics(flagged, FakeOracle.cone, {10, 20})
    assert detected and first == nodes[0]
    assert seen == nodes and stats["n_checked"] == 3
    assert localization["loc_any"] and localization["loc_all"]
    assert localization["faults_in_flagged_union"] == [10, 20]
    print(
        "self-check passed: set affected logic, deterministic mutation, "
        "m1 QMon delegation, and full-scan localization union"
    )


def _csv_values(raw, cast=str):
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return tuple(values)


def _parse_methods(raw):
    methods = _csv_values(raw)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; choose from {','.join(METHODS)}"
        )
    return methods


def _parse_modes(raw):
    modes = _csv_values(raw)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown modes {unknown}; choose from {','.join(MODES)}"
        )
    return modes


def _parse_seeds(raw):
    seeds = _csv_values(raw, int)
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


def _read_keys(path):
    keys = []
    with Path(path).open() as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                keys.append(_normal_key(value))
    return keys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Finite-shot real-execution evaluation of multi-gate mutants"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--chunk",
        nargs=2,
        metavar=("KEYFILE", "OUT"),
        help="run/resume a chunk of circuit keys and checkpoint to OUT",
    )
    action.add_argument(
        "--merge",
        nargs="+",
        metavar="PATH",
        help="merge as: --merge OUT INPUT [INPUT ...]",
    )
    action.add_argument(
        "--aggregate", metavar="FILE",
        help="aggregate one strict full-corpus merged artifact"
    )
    action.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=METHODS,
        help=f"comma list; default is all current methods: {','.join(METHODS)}",
    )
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default=MODES,
        help=f"comma list from {','.join(MODES)}",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=SEEDS,
        help="comma list; default is 1,2,3,4,5",
    )
    parser.add_argument("--qmax", type=int, default=24)
    parser.add_argument(
        "--circuit-budget",
        type=int,
        default=None,
        help="optional per-circuit SIGALRM budget in seconds",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUTPUT), help="local smoke checkpoint path"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not write the local smoke run"
    )
    parser.add_argument("keys", nargs="*", help="local smoke circuit keys")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.qmax != CANONICAL_QMAX:
        raise SystemExit(
            f"Qmax is fixed at {CANONICAL_QMAX}; old mixed-budget artifacts "
            "must be rerun")
    if args.self_check:
        self_check()
        return 0
    if args.merge:
        if len(args.merge) < 2:
            raise SystemExit("--merge requires OUT and at least one INPUT")
        output, inputs = args.merge[0], args.merge[1:]
        merged = merge_artifacts(inputs)
        atomic_pickle_dump(merged, output)
        print(
            f"merged {len(inputs)} artifacts -> {output}: "
            f"{len(merged['records'])} records, "
            f"methods={merged['protocol']['methods']}"
        )
        return 0
    if args.aggregate:
        aggregate(load_artifact(args.aggregate), methods=args.methods)
        return 0
    if args.chunk:
        keyfile, output = args.chunk
        keys = _read_keys(keyfile)
        if args.limit is not None:
            keys = keys[: args.limit]
        artifact = run_keys(
            keys,
            out=output,
            methods=args.methods,
            modes=args.modes,
            seeds=args.seeds,
            qmax=args.qmax,
            resume=not args.no_resume,
            circuit_budget=args.circuit_budget,
        )
        print(
            f"chunk complete: {len(artifact['records'])} records in {output}"
        )
        return 0

    keys = [_normal_key(key) for key in (args.keys or SMOKE_KEYS)]
    if args.limit is not None:
        keys = keys[: args.limit]
    artifact = run_keys(
        keys,
        out=None if args.no_save else args.out,
        methods=args.methods,
        modes=args.modes,
        seeds=args.seeds,
        qmax=args.qmax,
        resume=not args.no_resume,
        circuit_budget=args.circuit_budget,
    )
    print(
        f"smoke complete: {len(artifact['records'])} terminal records across "
        f"{len(artifact['circuit_status'])} circuits; no partial rates printed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
