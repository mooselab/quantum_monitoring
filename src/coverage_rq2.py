"""RQ2 coverage experiment over the fixed 310-circuit corpus.

Any inconsistency stops the run with an error.  Two scopes are reported and
never conflated:

``certificate_*``
    Every ``(gate i, acted qubit q)`` selected by the reference second-Schmidt-
    amplitude rule ``a2(i,q) <= 1e-12``.

``numerical_candidate_*``
    An explicit alias for the full certificate-selected gate--qubit event
    universe.  It includes terminally readable events and must not be
    conflated with a monitor insertion.

``emitted_mid_*``
    The subset actually instrumented under ``Qmax=24`` after same-gate
    grouping, union-cone ancilla accounting, batch partitioning, and circuit
    construction for every batch. Standard
    final measurements are reported separately as ``final_read_points``.

``deployed_observable_*``
    The reported scope under ``Qmax=24``: every instrumented mid-circuit point
    plus a certificate-selected last source-gate point whose qubit is read by
    a standard terminal measurement.  RQ2's gate-node, qubit, and normalized
    depth metrics must use this scope; ``certificate_*`` describes potential
    coverage before the deployment budget is applied.

Every acted-on point contributes to ``E_mean`` and ``E_peak``.  No circuit or
zero-coverage row may be omitted, and any source drift, failed circuit
generation, or row drop makes the final artifact invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import spearmanr

sys.setrecursionlimit(1_000_000)

from batch_partition import (
    DEFAULT_REFINEMENT_TIME_LIMIT_SECONDS,
    partition_nodes,
    solver_provenance,
    validate_partition_certificate,
)
from circuit_families import canonical_family_of
from circuits_selection import dfs_quantum_circuit, transformation
from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    EXPECTED_CIRCUITS as FIXED_CORPUS_CIRCUITS,
    EXPECTED_CORPUS_SHA256 as FIXED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256 as FIXED_MANIFEST_SHA256,
    ProtocolError,
    SCRIPT_DIR,
    atomic_write_csv,
    atomic_write_json,
    resolve_circuit_corpus,
    stable_hash,
)
from fast_analysis import SKIP_OPS
from rq1_certificate_check import (
    EXPECTED_SELECTED_IDENTITY_SHA256,
    EXPECTED_SELECTED_POINT_COUNT,
    EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256,
    SELECTOR_A2_TOL,
    crosscheck_selector,
)
from utilities import filter_operation_list, generate_monitoring_circuit


SCHEMA = "qmon-rq2-coverage-v3"
QMAX = 24
REFINEMENT_TIME_LIMIT_SECONDS = DEFAULT_REFINEMENT_TIME_LIMIT_SECONDS
NONEXECUTABLE_DEPTH_OPS = frozenset({"barrier", "delay", "snapshot"})


def protocol_descriptor() -> dict:
    """Return the exact protocol descriptor, including reporting scope."""

    return {
        "selector": {
            "quantity": "second Schmidt amplitude a2(i,q)",
            "operator": "<=",
            "threshold": SELECTOR_A2_TOL,
            "cross_check": "implemented selector plus independent direct SVD",
        },
        "point_universe": "every gate x qubit acted on by that gate",
        "numerical_candidate_events": (
            "every certificate-selected gate x qubit event before terminal-read "
            "or Qmax=24 deployment classification"
        ),
        "entanglement_summaries": {
            "E_mean": "arithmetic mean of det(rho_q) over every acted point",
            "E_peak": "maximum det(rho_q) over every acted point",
            "thresholding": "none; only negative floating roundoff is clipped",
        },
        "qmax": QMAX,
        "partition_refinement_time_limit_seconds": (
            REFINEMENT_TIME_LIMIT_SECONDS
        ),
        "grouping": "same source gate, exact member set",
        "ancilla_demand": "union of member backward cones minus group members",
        "partitioner": "partition_nodes",
        "partition_solver": solver_provenance(),
        "emission_validation": "every batch is built by generate_monitoring_circuit",
        "instrumentation_accounting": {
            "replay_gates": (
                "counted from emitted replay_data_indices after verifying a "
                "one-to-one mapping to deduplicated source-gate indices"
            ),
            "work": (
                "B*m + G_replay, where B is the number of deployment runs "
                "and m excludes source measurements, resets, and directives"
            ),
            "depth": (
                "Qiskit circuit depth including measurement and reset while "
                "excluding barriers, delays, and snapshots; inflation is the "
                "emitted-run depth divided by the corresponding source depth"
            ),
        },
        "final_read_convention": (
            "a certificate-selected point at the last source gate acting on a "
            "terminally measured qubit is observable through that standard "
            "final read and requires no inserted mid-circuit monitor"
        ),
        "publication_reporting_scope": {
            "scope": "deployed_observable_points = emitted_mid_points union final_read_points",
            "table_and_figure_metrics": [
                "deployed_observable_gate_coverage",
                "deployed_observable_qubit_coverage",
                "deployed_observable_depth_reach",
            ],
            "certificate_metrics_role": (
                "pre-budget potential only; do not report as Qmax=24 deployed coverage"
            ),
        },
        "coverage_denominators": {
            "gate_node": "non-skipped source gate count",
            "gate_qubit_event": "non-skipped acted gate x qubit event count",
            "qubit": "source circuit qubit count",
            "normalized_depth": "ASAP non-skipped source depth",
        },
        "silent_skips": "forbidden",
        "expected_selector_result": {
            "selected_point_count": EXPECTED_SELECTED_POINT_COUNT,
            "selected_identity_sha256": EXPECTED_SELECTED_IDENTITY_SHA256,
            "source_bound_selected_identity_sha256": (
                EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256
            ),
        },
    }


def family_of(key: str) -> str:
    return canonical_family_of(key)


def gate_layers(qc: QuantumCircuit) -> tuple[dict[int, int], int]:
    """ASAP layers over non-skipped source operations."""

    frontier = {}
    layers = {}
    depth = 0
    for gate_index, instruction in enumerate(qc.data):
        if instruction.operation.name in SKIP_OPS:
            continue
        acted = [qc.find_bit(bit).index for bit in instruction.qubits]
        layer = 1 + max((frontier.get(qubit, 0) for qubit in acted), default=0)
        for qubit in acted:
            frontier[qubit] = layer
        layers[gate_index] = layer
        depth = max(depth, layer)
    return layers, depth


def executable_depth(qc: QuantumCircuit) -> int:
    """Circuit depth including measurements/resets but excluding directives."""

    return int(qc.depth(filter_function=lambda item: (
        item.operation.name not in NONEXECUTABLE_DEPTH_OPS
    )))


def executable_operation_count(qc: QuantumCircuit) -> int:
    return sum(
        item.operation.name not in NONEXECUTABLE_DEPTH_OPS for item in qc.data
    )


def _last_source_gate_and_measured(qc: QuantumCircuit):
    last_gate = [None] * qc.num_qubits
    measured = set()
    measurement_started = False
    for gate_index, instruction in enumerate(qc.data):
        name = instruction.operation.name
        if name == "measure":
            measurement_started = True
            measured.update(qc.find_bit(bit).index for bit in instruction.qubits)
            continue
        if name == "reset":
            raise ProtocolError("RQ2 input circuit contains reset")
        if name in SKIP_OPS:
            continue
        if measurement_started:
            raise ProtocolError(
                f"unitary operation {name} follows a measurement at index "
                f"{gate_index}"
            )
        for bit in instruction.qubits:
            last_gate[qc.find_bit(bit).index] = gate_index
    return last_gate, measured


def _as_point_list(points) -> list[list[int]]:
    ordered = sorted(points)
    if any(
        type(gate) is not int or type(qubit) is not int
        for gate, qubit in ordered
    ):
        raise ProtocolError("gate/qubit point identities must be integers")
    return [[gate, qubit] for gate, qubit in ordered]


def _build_operation_list(
    qc: QuantumCircuit, key: str, selected_rows: list[dict]
) -> tuple[dict, set[tuple[int, int]], set[tuple[int, int]]]:
    """Build every non-final replay operation without legacy hard-coded skips."""

    last_gate, measured = _last_source_gate_and_measured(qc)
    final_reads = {
        (row["gate"], row["qubit"])
        for row in selected_rows
        if row["qubit"] in measured
        and row["gate"] == last_gate[row["qubit"]]
    }
    prebudget = {
        (row["gate"], row["qubit"]) for row in selected_rows
    } - final_reads

    operation_rows = []
    by_identity = {
        (row["gate"], row["qubit"]): row for row in selected_rows
    }
    for gate_index, qubit in sorted(prebudget):
        row = by_identity[(gate_index, qubit)]
        cone = dfs_quantum_circuit(qc, qubit, gate_index, set(), [])
        replay = transformation(qc, qubit, cone, [])
        if not replay:
            raise ProtocolError(
                f"empty replay for selected point {(gate_index, qubit)} in {key}"
            )
        operation_rows.append([
            gate_index,
            row["gate_name"],
            qubit,
            row["p0"],
            row["p1"],
            replay,
        ])
    operation_list = {key: operation_rows} if operation_rows else {}
    return operation_list, prebudget, final_reads


def _group_demands(
    qc: QuantumCircuit,
    key: str,
    operation_list: dict,
    prebudget: set[tuple[int, int]],
    *,
    partition_override: Mapping | None = None,
) -> tuple[list[dict], list[dict], list[int], dict]:
    if not prebudget:
        if partition_override is None:
            batches, infeasible, provenance = partition_nodes(
                {"num_oq": qc.num_qubits},
                QMAX,
                return_metadata=True,
                refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
            )
        else:
            batches = []
            infeasible = list(
                partition_override.get("infeasible_gate_groups", ())
            )
            provenance = partition_override.get("partition_provenance")
            if not isinstance(provenance, Mapping):
                raise ProtocolError(
                    f"stored empty partition lacks provenance in {key}"
                )
        if batches or infeasible:
            raise ProtocolError(f"empty monitor plan was not empty in {key}")
        try:
            validate_partition_certificate(
                {"num_oq": qc.num_qubits},
                QMAX,
                batches,
                infeasible,
                dict(provenance),
                refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
            )
        except ValueError as exc:
            raise ProtocolError(
                f"invalid empty partition certificate in {key}: {exc}"
            ) from exc
        return [], [], [], dict(provenance)
    with contextlib.redirect_stdout(io.StringIO()):
        _, extra_qubits = filter_operation_list(key, operation_list, 10 ** 7)

    expected_gates = {gate for gate, _ in prebudget}
    if set(extra_qubits) != expected_gates:
        raise ProtocolError(
            f"same-gate union-cone grouping dropped gates in {key}: "
            f"expected={sorted(expected_gates)}, actual={sorted(extra_qubits)}"
        )

    group_demands = []
    constraint = {"num_oq": qc.num_qubits}
    for gate in sorted(expected_gates):
        members = sorted(qubit for source_gate, qubit in prebudget
                         if source_gate == gate)
        demand = len(extra_qubits[gate])
        group_demands.append({
            "gate": gate,
            "members": list(members),
            "ancilla_demand": demand,
        })
        constraint[gate] = {
            "num_eq": demand,
            "qubits": list(members),
        }

    if partition_override is None:
        batches, infeasible, provenance = partition_nodes(
            constraint,
            QMAX,
            return_metadata=True,
            refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
        )
    else:
        raw_batches = partition_override.get("batches")
        raw_infeasible = partition_override.get("infeasible_gate_groups")
        provenance = partition_override.get("partition_provenance")
        if (
            not isinstance(raw_batches, list)
            or not isinstance(raw_infeasible, list)
            or not isinstance(provenance, Mapping)
        ):
            raise ProtocolError(f"stored partition evidence is malformed in {key}")
        batches = []
        for raw in raw_batches:
            if not isinstance(raw, Mapping):
                raise ProtocolError(f"stored partition batch is malformed in {key}")
            batches.append({
                "nodes": list(raw.get("gate_groups", ())),
                "cost": raw.get("ancilla_cost"),
            })
        infeasible = list(raw_infeasible)
    try:
        validate_partition_certificate(
            constraint,
            QMAX,
            batches,
            infeasible,
            dict(provenance),
            refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
        )
    except ValueError as exc:
        raise ProtocolError(f"invalid partition certificate in {key}: {exc}") from exc
    batches = sorted(
        ({"nodes": sorted(batch["nodes"]), "cost": batch["cost"]}
         for batch in batches),
        key=lambda batch: tuple(batch["nodes"]),
    )
    feasible = expected_gates - set(infeasible)
    covered = [gate for batch in batches for gate in batch["nodes"]]
    if len(covered) != len(set(covered)) or set(covered) != feasible:
        raise ProtocolError(
            f"partitioner did not cover every feasible group in {key}"
        )
    return group_demands, batches, sorted(infeasible), dict(provenance)


def _emit_batches(
    qc: QuantumCircuit,
    key: str,
    operation_list: dict,
    group_demands: list[dict],
    batches: list[dict],
) -> tuple[set[tuple[int, int]], list[dict]]:
    members_by_gate = {
        group["gate"]: tuple(group["members"]) for group in group_demands
    }
    emitted_points = set()
    emitted_batches = []

    for batch_index, batch in enumerate(batches):
        picked_nodes = list(batch["nodes"])
        picked_qubits = sorted({
            qubit for gate in picked_nodes for qubit in members_by_gate[gate]
        })
        answer = {
            "picked_nodes": picked_nodes,
            "picked_qubits": picked_qubits,
            "obj1": len(picked_nodes),
            "obj2": len(picked_qubits),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            emitted = generate_monitoring_circuit(
                qc, key, operation_list, 10 ** 7, answer
            )

        expected_extra = batch["cost"]
        actual_extra = emitted.num_qubits - qc.num_qubits
        if actual_extra != expected_extra:
            raise ProtocolError(
                f"union-cone demand/instrumentation mismatch in {key} batch "
                f"{batch_index}: expected {expected_extra}, got {actual_extra}"
            )
        if emitted.num_qubits > QMAX:
            raise ProtocolError(
                f"generated batch exceeds Qmax={QMAX} in {key}: "
                f"{emitted.num_qubits}"
            )

        metadata_groups = tuple(
            (emitted.metadata or {}).get("qmon_monitor_groups", ())
        )
        actual_members = {}
        used_ancilla_offsets = set()
        normalized_groups = []
        for group in metadata_groups:
            gate = group["gate_index"]
            if type(gate) is not int:
                raise ProtocolError(f"non-integer generated gate identity in {key}")
            if gate in actual_members:
                raise ProtocolError(f"duplicate generated gate group {gate} in {key}")
            members = tuple(group["qubits"])
            if any(type(qubit) is not int for qubit in members):
                raise ProtocolError(
                    f"non-integer generated qubit identity for gate {gate} in {key}"
                )
            actual_members[gate] = members
            replay_indices = tuple(group["replay_source_gate_indices"])
            if any(type(value) is not int for value in replay_indices):
                raise ProtocolError(
                    f"non-integer replay source identity for gate {gate} in {key}"
                )
            if replay_indices != tuple(sorted(set(replay_indices))):
                raise ProtocolError(
                    f"non-deduplicated replay union for gate {gate} in {key}"
                )
            replay_data_indices = tuple(group["replay_data_indices"])
            if any(type(value) is not int for value in replay_data_indices):
                raise ProtocolError(
                    f"non-integer replay data identity for gate {gate} in {key}"
                )
            if replay_data_indices != tuple(sorted(set(replay_data_indices))):
                raise ProtocolError(
                    f"unsorted or duplicated replay indices for gate {gate} in {key}"
                )
            if len(replay_data_indices) != len(replay_indices):
                raise ProtocolError(
                    f"source/generated replay gate-count mismatch for gate {gate} "
                    f"in {key}"
                )
            raw_ancilla_slice = tuple(group["ancilla_slice"])
            if any(type(value) is not int for value in raw_ancilla_slice):
                raise ProtocolError(
                    f"non-integer ancilla identity for gate {gate} in {key}"
                )
            ancilla_slice = set(raw_ancilla_slice)
            if used_ancilla_offsets & ancilla_slice:
                raise ProtocolError(
                    f"overlapping fresh-ancilla slices in {key} batch {batch_index}"
                )
            used_ancilla_offsets.update(ancilla_slice)
            normalized_groups.append({
                "gate": gate,
                "qubits": list(members),
                "replay_source_gate_indices": list(replay_indices),
                "replay_data_indices": list(replay_data_indices),
                "ancilla_slice": sorted(ancilla_slice),
            })

        expected_members = {
            gate: members_by_gate[gate] for gate in picked_nodes
        }
        if actual_members != expected_members:
            raise ProtocolError(
                f"instrumentation group membership mismatch in {key} batch {batch_index}: "
                f"expected={expected_members}, actual={actual_members}"
            )
        for gate, members in actual_members.items():
            emitted_points.update((gate, qubit) for qubit in members)
        monitor_point_count = sum(len(members) for members in actual_members.values())
        replay_gate_count = sum(
            len(group["replay_data_indices"]) for group in normalized_groups
        )
        source_depth = executable_depth(qc)
        emitted_depth = executable_depth(emitted)
        source_operations = executable_operation_count(qc)
        emitted_operations = executable_operation_count(emitted)
        expected_emitted_operations = (
            source_operations + replay_gate_count + 2 * monitor_point_count
        )
        if emitted_operations != expected_emitted_operations:
            raise ProtocolError(
                f"generated-circuit operation accounting mismatch in {key} batch "
                f"{batch_index}: expected {expected_emitted_operations}, "
                f"got {emitted_operations}"
            )
        emitted_batches.append({
            "batch": batch_index,
            "gate_groups": picked_nodes,
            "ancilla_cost": expected_extra,
            "emitted_total_qubits": emitted.num_qubits,
            "monitor_groups": normalized_groups,
            "monitor_group_count": len(normalized_groups),
            "monitor_point_count": monitor_point_count,
            "replay_gate_count": replay_gate_count,
            "monitor_measurement_count": monitor_point_count,
            "monitor_reset_count": monitor_point_count,
            "source_executable_operation_count": source_operations,
            "emitted_executable_operation_count": emitted_operations,
            "source_executable_depth": source_depth,
            "emitted_executable_depth": emitted_depth,
            "depth_increase": emitted_depth - source_depth,
            "depth_inflation_ratio": (
                emitted_depth / source_depth if source_depth else None
            ),
            "metadata_sha256": stable_hash(normalized_groups),
        })
    return emitted_points, emitted_batches


def _coverage_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ProtocolError("coverage denominator must be positive")
    return float(numerator / denominator)


def _group_records_by_feasibility(
    group_demands: list[dict], num_qubits: int
) -> tuple[list[dict], list[dict]]:
    """Split same-gate union-cone groups by their individual Qmax feasibility."""

    feasible = []
    infeasible = []
    for group in group_demands:
        target = (
            feasible
            if num_qubits + group["ancilla_demand"] <= QMAX
            else infeasible
        )
        target.append(group.copy())
    return feasible, infeasible


def evaluate_circuit(
    entry: dict,
    qasm_dir: Path,
    *,
    partition_override: Mapping | None = None,
) -> tuple[dict, dict]:
    key = entry["circuit"]
    qc = QuantumCircuit.from_qasm_file(str(qasm_dir / key))
    validation = crosscheck_selector(qc)
    all_points = validation["all_points"]
    selected_rows = validation["selected_points"]

    gate_count = sum(
        instruction.operation.name not in SKIP_OPS for instruction in qc.data
    )
    if gate_count <= 0 or not all_points:
        raise ProtocolError(f"empty RQ2 point universe for {key}")
    acted_points = {(row["gate"], row["qubit"]) for row in all_points}
    if len(acted_points) != len(all_points):
        raise ProtocolError(f"duplicate acted point in {key}")
    certificate_points = {
        (row["gate"], row["qubit"]) for row in selected_rows
    }
    if not certificate_points.issubset(acted_points):
        raise ProtocolError(f"selector returned a non-acted point in {key}")

    determinants = np.asarray(
        [row["det_rho_svd"] for row in all_points], dtype=float
    )
    if not np.all(np.isfinite(determinants)):
        raise ProtocolError(f"non-finite RQ2 entanglement score in {key}")
    if np.any(determinants < 0.0) or np.any(determinants > 0.25 + 1e-12):
        raise ProtocolError(f"RQ2 entanglement score outside [0,1/4] in {key}")

    operation_list, prebudget, final_reads = _build_operation_list(
        qc, key, selected_rows
    )
    if certificate_points != prebudget | final_reads or prebudget & final_reads:
        raise ProtocolError(f"selected-point accounting failure in {key}")
    group_demands, batches, infeasible_gates, partition_provenance = _group_demands(
        qc,
        key,
        operation_list,
        prebudget,
        partition_override=partition_override,
    )
    feasible_groups, infeasible_groups = _group_records_by_feasibility(
        group_demands, qc.num_qubits
    )
    if [group["gate"] for group in infeasible_groups] != infeasible_gates:
        raise ProtocolError(f"individual infeasible-group classification drift in {key}")
    emitted_points, emitted_batches = _emit_batches(
        qc, key, operation_list, group_demands, batches
    )
    infeasible_points = {
        point for point in prebudget if point[0] in set(infeasible_gates)
    }
    if emitted_points | infeasible_points != prebudget:
        raise ProtocolError(f"Qmax point accounting failure in {key}")
    if emitted_points & infeasible_points:
        raise ProtocolError(f"point is both instrumented and infeasible in {key}")
    deployed_points = emitted_points | final_reads

    layers, depth = gate_layers(qc)
    certificate_gates = {gate for gate, _ in certificate_points}
    emitted_gates = {gate for gate, _ in emitted_points}
    deployed_gates = {gate for gate, _ in deployed_points}
    certificate_qubits = {qubit for _, qubit in certificate_points}
    emitted_qubits = {qubit for _, qubit in emitted_points}
    deployed_qubits = {qubit for _, qubit in deployed_points}
    two_qubit_gates = sum(
        instruction.operation.name not in SKIP_OPS
        and instruction.operation.num_qubits == 2
        for instruction in qc.data
    )

    record = {
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "circuit": key,
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "family": family_of(key),
        "qmax": QMAX,
        "num_qubits": qc.num_qubits,
        "depth": depth,
        "gate_count": gate_count,
        "acted_point_count": len(acted_points),
        "two_qubit_density": two_qubit_gates / gate_count,
        "E_mean": float(np.mean(determinants)),
        "E_peak": float(np.max(determinants)),
        "coverage_denominators": {
            "gate_node_count": gate_count,
            "gate_qubit_event_count": len(acted_points),
            "qubit_count": qc.num_qubits,
            "normalized_depth_layer_count": depth,
        },
        "gate_node_denominator": gate_count,
        "gate_qubit_event_denominator": len(acted_points),
        "qubit_denominator": qc.num_qubits,
        "normalized_depth_denominator": depth,
        "acted_points": _as_point_list(acted_points),
        "certificate_points": _as_point_list(certificate_points),
        "numerical_candidate_points": _as_point_list(certificate_points),
        "prebudget_mid_points": _as_point_list(prebudget),
        "final_read_points": _as_point_list(final_reads),
        "emitted_mid_points": _as_point_list(emitted_points),
        "infeasible_mid_points": _as_point_list(infeasible_points),
        "deployed_observable_points": _as_point_list(deployed_points),
        "group_demands": group_demands,
        "partition_provenance": partition_provenance,
        "individually_feasible_gate_groups": feasible_groups,
        "individually_infeasible_gate_groups": infeasible_groups,
        "batches": emitted_batches,
        "infeasible_gate_groups": infeasible_gates,
        "certificate_selected_point_count": len(certificate_points),
        "certificate_selected_gate_count": len(certificate_gates),
        "certificate_selected_qubit_count": len(certificate_qubits),
        "certificate_acted_point_coverage": _coverage_ratio(
            len(certificate_points), len(acted_points)
        ),
        "certificate_gate_coverage": _coverage_ratio(
            len(certificate_gates), gate_count
        ),
        "certificate_qubit_coverage": _coverage_ratio(
            len(certificate_qubits), qc.num_qubits
        ),
        "certificate_depth_reach": (
            max((layers[gate] for gate in certificate_gates), default=0)
            / depth if depth else 0.0
        ),
        "emitted_mid_point_count": len(emitted_points),
        "emitted_mid_gate_count": len(emitted_gates),
        "emitted_mid_qubit_count": len(emitted_qubits),
        "emitted_mid_acted_point_coverage": _coverage_ratio(
            len(emitted_points), len(acted_points)
        ),
        "emitted_mid_gate_coverage": _coverage_ratio(
            len(emitted_gates), gate_count
        ),
        "emitted_mid_qubit_coverage": _coverage_ratio(
            len(emitted_qubits), qc.num_qubits
        ),
        "emitted_mid_depth_reach": (
            max((layers[gate] for gate in emitted_gates), default=0)
            / depth if depth else 0.0
        ),
        "final_read_point_count": len(final_reads),
        "deployed_observable_point_count": len(deployed_points),
        "deployed_observable_gate_count": len(deployed_gates),
        "deployed_observable_qubit_count": len(deployed_qubits),
        "deployed_observable_acted_point_coverage": _coverage_ratio(
            len(deployed_points), len(acted_points)
        ),
        "deployed_observable_gate_coverage": _coverage_ratio(
            len(deployed_gates), gate_count
        ),
        "deployed_observable_qubit_coverage": _coverage_ratio(
            len(deployed_qubits), qc.num_qubits
        ),
        "deployed_observable_depth_reach": (
            max((layers[gate] for gate in deployed_gates), default=0)
            / depth if depth else 0.0
        ),
        "infeasible_mid_point_count": len(infeasible_points),
        "infeasible_gate_group_count": len(infeasible_gates),
        "batch_count": len(emitted_batches),
        "source_executable_operation_count": executable_operation_count(qc),
        "source_executable_depth": executable_depth(qc),
        "selector_max_a2": max(
            (row["selector_a2"] for row in selected_rows), default=0.0
        ),
        "selector_max_independent_svd_a2": max(
            (row["independent_svd_a2"] for row in selected_rows), default=0.0
        ),
        "selected_identity_sha256": stable_hash(_as_point_list(certificate_points)),
        "emitted_identity_sha256": stable_hash(_as_point_list(emitted_points)),
        "deployed_observable_identity_sha256": stable_hash(
            _as_point_list(deployed_points)
        ),
    }
    status = {
        "status": "complete",
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "record_sha256": stable_hash(record),
    }
    return status, record


def _error_status(entry: dict, error: Exception) -> dict:
    return {
        "status": "error",
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _safe_spearman(xs, ys):
    value = float(spearmanr(xs, ys).correlation)
    return value if math.isfinite(value) else None


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"circuit_count": 0, "averages": {}, "spearman": {}}
    average_fields = (
        "certificate_gate_coverage",
        "certificate_acted_point_coverage",
        "certificate_qubit_coverage",
        "certificate_depth_reach",
        "emitted_mid_gate_coverage",
        "emitted_mid_acted_point_coverage",
        "emitted_mid_qubit_coverage",
        "emitted_mid_depth_reach",
        "deployed_observable_gate_coverage",
        "deployed_observable_acted_point_coverage",
        "deployed_observable_qubit_coverage",
        "deployed_observable_depth_reach",
        "E_mean",
        "E_peak",
    )
    targets = (
        "certificate_gate_coverage",
        "certificate_depth_reach",
        "certificate_qubit_coverage",
        "emitted_mid_gate_coverage",
        "emitted_mid_qubit_coverage",
        "emitted_mid_depth_reach",
        "deployed_observable_gate_coverage",
        "deployed_observable_qubit_coverage",
        "deployed_observable_depth_reach",
    )
    predictors = ("E_mean", "E_peak", "two_qubit_density", "depth", "num_qubits")
    return {
        "circuit_count": len(records),
        "family_counts": dict(sorted(Counter(
            record["family"] for record in records
        ).items())),
        "averages": {
            field: float(np.mean([record[field] for record in records]))
            for field in average_fields
        },
        "ranges": {
            field: [
                float(min(record[field] for record in records)),
                float(max(record[field] for record in records)),
            ]
            for field in average_fields
        },
        "spearman": {
            predictor: {
                target: _safe_spearman(
                    [record[predictor] for record in records],
                    [record[target] for record in records],
                )
                for target in targets
            }
            for predictor in predictors
        },
        "certificate_selected_point_count": sum(
            record["certificate_selected_point_count"] for record in records
        ),
        "certificate_selected_identity_sha256": stable_hash([
            [record["circuit"], point[0], point[1]]
            for record in records for point in record["certificate_points"]
        ]),
        "source_bound_certificate_selected_identity_sha256": stable_hash([
            [record["source_qasm_sha256"], point[0], point[1]]
            for record in records for point in record["certificate_points"]
        ]),
    }


def build_artifact(frame: dict) -> dict:
    statuses = {}
    records = []
    qasm_dir = Path(frame["qasm_dir"])
    for position, entry in enumerate(frame["circuits"], start=1):
        key = entry["circuit"]
        try:
            status, record = evaluate_circuit(entry, qasm_dir)
        except Exception as error:
            statuses[key] = _error_status(entry, error)
        else:
            statuses[key] = status
            records.append(record)
        if position % 10 == 0 or position == frame["circuit_count"]:
            print(
                f"RQ2 emission check: {position}/{frame['circuit_count']}",
                flush=True,
            )
    records.sort(key=lambda record: record["attempt_index"])
    artifact = {
        "schema": SCHEMA,
        "protocol": protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "circuit_count": frame["circuit_count"],
        },
        "circuit_status": statuses,
        "record_count": len(records),
        "records_sha256": stable_hash(records),
        "summary": summarize(records),
        "records": records,
    }
    validate_artifact(artifact, frame, require_complete=False)
    return artifact


def _point_set(record: dict, field: str) -> set[tuple[int, int]]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ProtocolError(f"{field} must be a list in {record.get('circuit')}")
    points = set()
    for point in value:
        if (not isinstance(point, list) or len(point) != 2
                or not all(type(item) is int for item in point)):
            raise ProtocolError(f"malformed {field} entry in {record.get('circuit')}")
        points.add((point[0], point[1]))
    if len(points) != len(value):
        raise ProtocolError(f"duplicate point in {field} for {record.get('circuit')}")
    if value != _as_point_list(points):
        raise ProtocolError(f"unsorted ordering in {field} for {record.get('circuit')}")
    return points


def _validate_record(record: dict, entry: dict, qasm_dir: Path) -> None:
    key = entry["circuit"]
    for field in ("attempt_index", "attempt_id", "circuit", "source_qasm_sha256"):
        if field == "attempt_index" and type(record.get(field)) is not int:
            raise ProtocolError(f"{field} drift in RQ2 record for {key}")
        if record.get(field) != entry[field]:
            raise ProtocolError(f"{field} drift in RQ2 record for {key}")
    if type(record.get("qmax")) is not int or record.get("qmax") != QMAX:
        raise ProtocolError(f"unexpected Qmax in RQ2 record for {key}")

    qc = QuantumCircuit.from_qasm_file(str(qasm_dir / key))
    layers, depth = gate_layers(qc)
    expected_acted = {
        (gate, qc.find_bit(bit).index)
        for gate, instruction in enumerate(qc.data)
        if instruction.operation.name not in SKIP_OPS
        for bit in instruction.qubits
    }
    expected_gate_count = sum(
        instruction.operation.name not in SKIP_OPS for instruction in qc.data
    )
    if (
        type(record.get("num_qubits")) is not int
        or record.get("num_qubits") != qc.num_qubits
    ):
        raise ProtocolError(f"source qubit count mismatch for {key}")
    if (
        type(record.get("gate_count")) is not int
        or record.get("gate_count") != expected_gate_count
    ):
        raise ProtocolError(f"source gate count mismatch for {key}")
    if type(record.get("depth")) is not int or record.get("depth") != depth:
        raise ProtocolError(f"source depth mismatch for {key}")

    acted = _point_set(record, "acted_points")
    certificate = _point_set(record, "certificate_points")
    numerical_candidates = _point_set(record, "numerical_candidate_points")
    prebudget = _point_set(record, "prebudget_mid_points")
    final_reads = _point_set(record, "final_read_points")
    emitted = _point_set(record, "emitted_mid_points")
    infeasible = _point_set(record, "infeasible_mid_points")
    deployed = _point_set(record, "deployed_observable_points")
    if acted != expected_acted:
        raise ProtocolError(f"acted-point universe mismatch for {key}")
    if not certificate.issubset(acted):
        raise ProtocolError(f"certificate point outside universe for {key}")
    if numerical_candidates != certificate:
        raise ProtocolError(f"numerical candidate/selector mismatch for {key}")
    if certificate != prebudget | final_reads or prebudget & final_reads:
        raise ProtocolError(f"certificate scope accounting mismatch for {key}")
    if prebudget != emitted | infeasible or emitted & infeasible:
        raise ProtocolError(f"Qmax scope accounting mismatch for {key}")
    if deployed != emitted | final_reads:
        raise ProtocolError(f"deployed scope accounting mismatch for {key}")
    last_gate, measured = _last_source_gate_and_measured(qc)
    expected_final_reads = {
        (gate, qubit) for gate, qubit in certificate
        if qubit in measured and gate == last_gate[qubit]
    }
    if final_reads != expected_final_reads:
        raise ProtocolError(f"terminal-read convention mismatch for {key}")

    certificate_gates = {gate for gate, _ in certificate}
    emitted_gates = {gate for gate, _ in emitted}
    deployed_gates = {gate for gate, _ in deployed}
    certificate_qubits = {qubit for _, qubit in certificate}
    emitted_qubits = {qubit for _, qubit in emitted}
    deployed_qubits = {qubit for _, qubit in deployed}

    batches = record.get("batches")
    if not isinstance(batches, list):
        raise ProtocolError(f"missing group/batch plan for {key}")
    count_checks = {
        "acted_point_count": len(acted),
        "certificate_selected_point_count": len(certificate),
        "certificate_selected_gate_count": len(certificate_gates),
        "certificate_selected_qubit_count": len(certificate_qubits),
        "emitted_mid_point_count": len(emitted),
        "emitted_mid_gate_count": len(emitted_gates),
        "emitted_mid_qubit_count": len(emitted_qubits),
        "final_read_point_count": len(final_reads),
        "deployed_observable_point_count": len(deployed),
        "deployed_observable_gate_count": len(deployed_gates),
        "deployed_observable_qubit_count": len(deployed_qubits),
        "infeasible_mid_point_count": len(infeasible),
        "infeasible_gate_group_count": len({gate for gate, _ in infeasible}),
        "batch_count": len(batches),
        "source_executable_operation_count": executable_operation_count(qc),
        "source_executable_depth": executable_depth(qc),
    }
    for field, expected in count_checks.items():
        actual = record.get(field)
        if type(actual) is not int or actual != expected:
            raise ProtocolError(f"{field} mismatch for {key}")

    expected_denominators = {
        "gate_node_count": expected_gate_count,
        "gate_qubit_event_count": len(expected_acted),
        "qubit_count": qc.num_qubits,
        "normalized_depth_layer_count": depth,
    }
    denominators = record.get("coverage_denominators")
    if (
        not isinstance(denominators, dict)
        or any(type(denominators.get(field)) is not int
               for field in expected_denominators)
        or denominators != expected_denominators
    ):
        raise ProtocolError(f"coverage denominator mismatch for {key}")
    flat_denominators = {
        "gate_node_denominator": expected_denominators["gate_node_count"],
        "gate_qubit_event_denominator": expected_denominators[
            "gate_qubit_event_count"
        ],
        "qubit_denominator": expected_denominators["qubit_count"],
        "normalized_depth_denominator": expected_denominators[
            "normalized_depth_layer_count"
        ],
    }
    for field, expected in flat_denominators.items():
        actual = record.get(field)
        if type(actual) is not int or actual != expected:
            raise ProtocolError(f"coverage denominator mismatch for {key}: {field}")

    demands = record.get("group_demands")
    if not isinstance(demands, list):
        raise ProtocolError(f"missing group/batch plan for {key}")
    demand_by_gate = {}
    for group in demands:
        gate = group.get("gate")
        members = group.get("members")
        demand = group.get("ancilla_demand")
        if (type(gate) is not int or gate in demand_by_gate
                or not isinstance(members, list)
                or not all(type(qubit) is int for qubit in members)
                or members != sorted(set(members))
                or type(demand) is not int or demand < 0):
            raise ProtocolError(f"malformed group demand for {key}")
        expected_members = sorted(qubit for source_gate, qubit in prebudget
                                  if source_gate == gate)
        if members != expected_members:
            raise ProtocolError(f"same-gate member mismatch for {key} gate {gate}")
        demand_by_gate[gate] = demand
    if set(demand_by_gate) != {gate for gate, _ in prebudget}:
        raise ProtocolError(f"group demand coverage mismatch for {key}")

    raw_infeasible = record.get("infeasible_gate_groups")
    if (
        not isinstance(raw_infeasible, list)
        or any(type(gate) is not int for gate in raw_infeasible)
        or raw_infeasible != sorted(set(raw_infeasible))
    ):
        raise ProtocolError(f"infeasible group identities are malformed for {key}")
    expected_infeasible = {
        gate for gate, demand in demand_by_gate.items()
        if record["num_qubits"] + demand > QMAX
    }
    if set(raw_infeasible) != expected_infeasible:
        raise ProtocolError(f"infeasible group mismatch for {key}")
    feasible_groups = record.get("individually_feasible_gate_groups")
    infeasible_groups = record.get("individually_infeasible_gate_groups")
    if not isinstance(feasible_groups, list) or not isinstance(infeasible_groups, list):
        raise ProtocolError(f"missing individual-feasibility groups for {key}")
    expected_feasible_groups = [
        group for group in demands if group["gate"] not in expected_infeasible
    ]
    expected_infeasible_groups = [
        group for group in demands if group["gate"] in expected_infeasible
    ]
    for scope, groups in (
        ("feasible", feasible_groups),
        ("infeasible", infeasible_groups),
    ):
        for group in groups:
            if not isinstance(group, dict):
                raise ProtocolError(
                    f"malformed individually {scope} group for {key}"
                )
            members = group.get("members")
            if (
                type(group.get("gate")) is not int
                or not isinstance(members, list)
                or any(type(qubit) is not int for qubit in members)
                or type(group.get("ancilla_demand")) is not int
            ):
                raise ProtocolError(
                    f"malformed individually {scope} group for {key}"
                )
    if feasible_groups != expected_feasible_groups:
        raise ProtocolError(f"individually feasible group record mismatch for {key}")
    if infeasible_groups != expected_infeasible_groups:
        raise ProtocolError(f"individually infeasible group record mismatch for {key}")
    seen_gates = set()
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise ProtocolError(f"malformed generated batch for {key}")
        gates = batch.get("gate_groups")
        cost = batch.get("ancilla_cost")
        if (type(batch.get("batch")) is not int
                or batch.get("batch") != batch_index
                or not isinstance(gates, list)
                or any(type(gate) is not int for gate in gates)
                or gates != sorted(set(gates)) or type(cost) is not int):
            raise ProtocolError(f"malformed generated batch for {key}")
        if seen_gates & set(gates):
            raise ProtocolError(f"gate group repeated across batches for {key}")
        seen_gates.update(gates)
        if cost != sum(demand_by_gate[gate] for gate in gates):
            raise ProtocolError(f"batch cost mismatch for {key}")
        if record["num_qubits"] + cost > QMAX:
            raise ProtocolError(f"batch exceeds Qmax for {key}")
        emitted_total_qubits = batch.get("emitted_total_qubits")
        if (
            type(emitted_total_qubits) is not int
            or emitted_total_qubits != record["num_qubits"] + cost
        ):
            raise ProtocolError(f"generated-circuit qubit count mismatch for {key}")
        batch_count_fields = (
            "monitor_group_count",
            "monitor_point_count",
            "replay_gate_count",
            "monitor_measurement_count",
            "monitor_reset_count",
            "source_executable_operation_count",
            "emitted_executable_operation_count",
            "source_executable_depth",
            "emitted_executable_depth",
            "depth_increase",
        )
        if any(type(batch.get(field)) is not int for field in batch_count_fields):
            raise ProtocolError(f"malformed generated batch count for {key}")
        monitor_groups = batch.get("monitor_groups")
        if not isinstance(monitor_groups, list):
            raise ProtocolError(f"malformed generated monitor groups for {key}")
        for group in monitor_groups:
            if not isinstance(group, dict):
                raise ProtocolError(f"malformed generated monitor group for {key}")
            if type(group.get("gate")) is not int:
                raise ProtocolError(f"malformed generated monitor group for {key}")
            for field in (
                "qubits",
                "replay_source_gate_indices",
                "replay_data_indices",
                "ancilla_slice",
            ):
                identities = group.get(field)
                if (
                    not isinstance(identities, list)
                    or any(type(identity) is not int for identity in identities)
                ):
                    raise ProtocolError(
                        f"malformed generated monitor group for {key}"
                    )
    if seen_gates != set(demand_by_gate) - expected_infeasible:
        raise ProtocolError(f"feasible gate groups were silently dropped for {key}")

    ratio_checks = {
        "certificate_acted_point_coverage": len(certificate) / len(acted),
        "certificate_gate_coverage": len(certificate_gates) / record["gate_count"],
        "certificate_qubit_coverage": len(certificate_qubits) / record["num_qubits"],
        "emitted_mid_acted_point_coverage": len(emitted) / len(acted),
        "emitted_mid_gate_coverage": len(emitted_gates) / record["gate_count"],
        "emitted_mid_qubit_coverage": len(emitted_qubits) / record["num_qubits"],
        "deployed_observable_acted_point_coverage": len(deployed) / len(acted),
        "deployed_observable_gate_coverage": len(deployed_gates) / record["gate_count"],
        "deployed_observable_qubit_coverage": len(deployed_qubits) / record["num_qubits"],
    }
    for field, expected in ratio_checks.items():
        if not math.isclose(record.get(field, math.nan), expected,
                            rel_tol=0.0, abs_tol=1e-15):
            raise ProtocolError(f"derived coverage mismatch for {key}: {field}")
    depth_checks = {
        "certificate_depth_reach": (
            max((layers[gate] for gate in certificate_gates), default=0) / depth
            if depth else 0.0
        ),
        "emitted_mid_depth_reach": (
            max((layers[gate] for gate in emitted_gates), default=0) / depth
            if depth else 0.0
        ),
        "deployed_observable_depth_reach": (
            max((layers[gate] for gate in deployed_gates), default=0) / depth
            if depth else 0.0
        ),
    }
    for field, expected in depth_checks.items():
        if not math.isclose(record.get(field, math.nan), expected,
                            rel_tol=0.0, abs_tol=1e-15):
            raise ProtocolError(f"derived depth reach mismatch for {key}: {field}")
    if record.get("selected_identity_sha256") != stable_hash(_as_point_list(certificate)):
        raise ProtocolError(f"certificate identity checksum mismatch for {key}")
    if record.get("emitted_identity_sha256") != stable_hash(_as_point_list(emitted)):
        raise ProtocolError(f"instrumented identity checksum mismatch for {key}")
    if record.get("deployed_observable_identity_sha256") != stable_hash(
        _as_point_list(deployed)
    ):
        raise ProtocolError(f"deployed identity checksum mismatch for {key}")

    # Recompute every source-derived selector, cone, and generated-circuit field,
    # but replay the stored feasible partition.  The optional refinement is
    # wall-clock bounded, so asking a validator to solve it again can produce a
    # different yet equally valid packing near the limit.
    _, expected_record = evaluate_circuit(
        entry,
        qasm_dir,
        partition_override=record,
    )
    for field, expected in expected_record.items():
        actual = record.get(field)
        if field == "batches" and isinstance(actual, list):
            actual = [
                {name: batch.get(name) for name in expected_batch}
                for batch, expected_batch in zip(actual, expected)
            ] if len(actual) == len(expected) else actual
        if actual != expected:
            raise ProtocolError(
                f"source-recomputed scientific field mismatch for {key}: {field}"
            )


def validate_artifact(
    artifact: dict, frame: dict, *, require_complete: bool = True
) -> dict:
    if artifact.get("schema") != SCHEMA:
        raise ProtocolError("wrong RQ2 artifact schema")
    expected_protocol = protocol_descriptor()
    protocol = artifact.get("protocol")
    partition_solver = (
        protocol.get("partition_solver") if isinstance(protocol, dict) else None
    )
    if (
        not isinstance(partition_solver, dict)
        or type(partition_solver.get("threads")) is not int
        or (
            expected_protocol["partition_solver"]["seed"] is not None
            and type(partition_solver.get("seed")) is not int
        )
        or (
            expected_protocol["partition_solver"]["relative_mip_gap"] is not None
            and type(partition_solver.get("relative_mip_gap")) is not float
        )
        or protocol != expected_protocol
    ):
        raise ProtocolError("RQ2 protocol drift")
    statuses = artifact.get("circuit_status")
    records = artifact.get("records")
    if not isinstance(statuses, dict) or not isinstance(records, list):
        raise ProtocolError("RQ2 artifact lacks statuses or records")
    expected_corpus = {
        "manifest_sha256": frame.get("manifest_sha256"),
        "corpus_sha256": frame.get("corpus_sha256"),
        "circuit_count": frame.get("circuit_count", len(frame["circuits"])),
    }
    corpus = artifact.get("corpus")
    if (
        not isinstance(corpus, dict)
        or type(corpus.get("circuit_count")) is not int
        or corpus != expected_corpus
    ):
        raise ProtocolError(
            f"RQ2 corpus provenance mismatch: "
            f"{artifact.get('corpus')} != {expected_corpus}"
        )
    entries = {entry["circuit"]: entry for entry in frame["circuits"]}
    if set(statuses) != set(entries):
        raise ProtocolError("RQ2 terminal accounting is not exact")
    if (
        type(artifact.get("record_count")) is not int
        or artifact.get("record_count") != len(records)
    ):
        raise ProtocolError("RQ2 record_count mismatch")
    if artifact.get("records_sha256") != stable_hash(records):
        raise ProtocolError("RQ2 records checksum mismatch")

    seen = set()
    for record in records:
        key = record.get("circuit")
        if key not in entries or key in seen:
            raise ProtocolError(f"duplicate or unexpected RQ2 row: {key}")
        seen.add(key)
        _validate_record(record, entries[key], Path(frame["qasm_dir"]))
        if statuses[key].get("record_sha256") != stable_hash(record):
            raise ProtocolError(f"RQ2 row checksum mismatch for {key}")

    errors = []
    for key, entry in entries.items():
        status = statuses[key]
        for field in ("attempt_index", "attempt_id", "source_qasm_sha256"):
            if field == "attempt_index" and type(status.get(field)) is not int:
                raise ProtocolError(f"{field} drift in RQ2 status for {key}")
            if status.get(field) != entry[field]:
                raise ProtocolError(f"{field} drift in RQ2 status for {key}")
        status_kind = status.get("status")
        if status_kind == "complete" and key not in seen:
            raise ProtocolError(f"silent RQ2 circuit-row drop for {key}")
        if status_kind == "error":
            errors.append(key)
            if key in seen:
                raise ProtocolError(f"error circuit contributes RQ2 row: {key}")
            if not isinstance(status.get("error"), dict):
                raise ProtocolError(f"missing structured RQ2 error for {key}")
        elif status_kind != "complete":
            raise ProtocolError(f"invalid RQ2 terminal status for {key}")
    summary = artifact.get("summary")
    if not isinstance(summary, dict) or type(summary.get("circuit_count")) is not int:
        raise ProtocolError("RQ2 summary is not derived from records")
    family_counts = summary.get("family_counts")
    if family_counts is not None and (
        not isinstance(family_counts, dict)
        or any(type(count) is not int for count in family_counts.values())
    ):
        raise ProtocolError("RQ2 summary is not derived from records")
    if "certificate_selected_point_count" in summary and type(
        summary["certificate_selected_point_count"]
    ) is not int:
        raise ProtocolError("RQ2 summary is not derived from records")
    if summary != summarize(records):
        raise ProtocolError("RQ2 summary is not derived from records")
    if require_complete and errors:
        raise ProtocolError(
            f"RQ2 artifact is not complete; errors={errors}"
        )
    if require_complete and len(records) != frame["circuit_count"]:
        raise ProtocolError(
            f"RQ2 requires exactly {frame['circuit_count']} circuit rows"
        )
    reference_frame = (
        frame.get("circuit_count") == FIXED_CORPUS_CIRCUITS
        and frame.get("manifest_sha256") == FIXED_MANIFEST_SHA256
        and frame.get("corpus_sha256") == FIXED_CORPUS_SHA256
    )
    if require_complete and reference_frame:
        selector_actual = {
            "selected_point_count": artifact["summary"][
                "certificate_selected_point_count"
            ],
            "selected_identity_sha256": artifact["summary"][
                "certificate_selected_identity_sha256"
            ],
            "source_bound_selected_identity_sha256": artifact["summary"][
                "source_bound_certificate_selected_identity_sha256"
            ],
        }
        selector_expected = {
            "selected_point_count": EXPECTED_SELECTED_POINT_COUNT,
            "selected_identity_sha256": EXPECTED_SELECTED_IDENTITY_SHA256,
            "source_bound_selected_identity_sha256": (
                EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256
            ),
        }
        if selector_actual != selector_expected:
            raise ProtocolError(
                "RQ2 certificate selector drift: "
                f"{selector_actual} != {selector_expected}"
            )
    return artifact


CSV_FIELDS = [
    "attempt_index", "attempt_id", "circuit", "source_qasm_sha256", "family",
    "qmax", "num_qubits", "depth", "gate_count", "acted_point_count",
    "gate_node_denominator", "gate_qubit_event_denominator",
    "qubit_denominator", "normalized_depth_denominator",
    "two_qubit_density", "E_mean", "E_peak",
    "certificate_selected_point_count", "certificate_selected_gate_count",
    "certificate_selected_qubit_count", "certificate_acted_point_coverage",
    "certificate_gate_coverage", "certificate_qubit_coverage",
    "certificate_depth_reach", "emitted_mid_point_count",
    "emitted_mid_gate_count", "emitted_mid_qubit_count",
    "emitted_mid_acted_point_coverage", "emitted_mid_gate_coverage",
    "emitted_mid_qubit_coverage", "emitted_mid_depth_reach",
    "final_read_point_count", "deployed_observable_point_count",
    "deployed_observable_gate_count", "deployed_observable_qubit_count",
    "deployed_observable_acted_point_coverage",
    "deployed_observable_gate_coverage",
    "deployed_observable_qubit_coverage", "deployed_observable_depth_reach",
    "infeasible_mid_point_count",
    "infeasible_gate_group_count", "batch_count", "selector_max_a2",
    "selector_max_independent_svd_a2", "mean_ent", "peak_ent", "node_cov",
    "emitted_node_cov", "certificate_node_cov", "deployed_node_cov",
]


def write_csv(path: Path | str, artifact: dict) -> None:
    rows = []
    for record in artifact["records"]:
        row = {field: record.get(field) for field in CSV_FIELDS}
        # The ambiguous historical node_cov alias now follows the reported
        # Qmax=24 scope.  Explicit scope-named columns remain authoritative.
        row.update({
            "mean_ent": record["E_mean"],
            "peak_ent": record["E_peak"],
            "node_cov": record["deployed_observable_gate_coverage"],
            "emitted_node_cov": record["emitted_mid_gate_coverage"],
            "certificate_node_cov": record["certificate_gate_coverage"],
            "deployed_node_cov": record["deployed_observable_gate_coverage"],
        })
        rows.append(row)
    atomic_write_csv(path, CSV_FIELDS, rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed full-corpus RQ2 certificate/instrumented coverage"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--out-json", type=Path, default=SCRIPT_DIR / "coverage_rq2.json"
    )
    parser.add_argument(
        "--out-csv", type=Path, default=SCRIPT_DIR / "coverage.csv"
    )
    args = parser.parse_args(argv)

    frame = resolve_circuit_corpus(args.manifest, args.qasm_dir)
    artifact = build_artifact(frame)
    atomic_write_json(args.out_json, artifact)
    try:
        validate_artifact(artifact, frame, require_complete=True)
    except ProtocolError as error:
        print(f"artifact rejected: {error}", file=sys.stderr)
        return 2
    write_csv(args.out_csv, artifact)
    print(
        f"wrote {args.out_json} and {args.out_csv}: "
        f"{len(artifact['records'])} circuits, "
        f"{artifact['summary']['certificate_selected_point_count']} "
        "a2-selected points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
