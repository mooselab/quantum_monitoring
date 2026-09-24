"""Evaluate RQ2 coverage over seven Qmax values.

The main RQ1 to RQ3 analyses use Qmax=24. This module reuses the RQ2
candidate selector, grouping rule, batch planner, circuit construction, and
validation at each sensitivity-analysis budget. Summary statistics are
computed only when all 310 circuits have a successful result at every budget.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import signal
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import batch_partition
import coverage_rq2 as rq2
from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    EXPECTED_CIRCUITS,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256,
    ProtocolError,
    atomic_write_json,
    file_sha256,
    resolve_circuit_corpus,
    stable_hash,
)


SCHEMA = "qmon.qmax-sensitivity.v5"
RECORD_SET_SCHEMA = "qmon.qmax-sensitivity.v5.record-set"
AGGREGATE_SCHEMA = "qmon.qmax-sensitivity.v5.aggregate"
DEFAULT_RUN_ID = "0" * 64
HEADLINE_QMAX = 24
QMAX_VALUES = (16, 20, 24, 28, 32, 36, 40)
EXPECTED_GRID_RECORDS = EXPECTED_CIRCUITS * len(QMAX_VALUES)
TERMINAL_STATUSES = frozenset({"complete", "error"})
RECORD_FIELDS = frozenset({
    "schema",
    "attempt_index",
    "attempt_id",
    "circuit_index",
    "budget_index",
    "circuit",
    "qmax",
    "source_attempt_id",
    "source_qasm_sha256",
    "status",
    "record_complete",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "error",
    "result",
    "fingerprints",
    "run",
    "record_sha256",
})
ATTEMPT_TIMEOUT_SECONDS = 7200
REFINEMENT_TIME_LIMIT_SECONDS = rq2.REFINEMENT_TIME_LIMIT_SECONDS

PROTOCOL_PATH = SCRIPT_DIR.parent / "protocols" / "QMAX_SENSITIVITY_PROTOCOL.md"
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(rq2.__file__).resolve(),
    SCRIPT_DIR / "batch_partition.py",
    SCRIPT_DIR / "circuits_selection.py",
    SCRIPT_DIR / "entanglement_disturbance.py",
    SCRIPT_DIR / "fast_analysis.py",
    SCRIPT_DIR / "rq1_certificate_check.py",
    SCRIPT_DIR / "utilities.py",
    PROTOCOL_PATH,
)


@dataclass(frozen=True)
class BudgetAttempt:
    """One fixed circuit-budget identity in the sweep."""

    attempt_index: int
    attempt_id: str
    circuit_index: int
    budget_index: int
    circuit: str
    qmax: int
    source_qasm_sha256: str
    source_attempt_id: str

    def coverage_entry(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "attempt_id": self.attempt_id,
            "circuit": self.circuit,
            "source_qasm_sha256": self.source_qasm_sha256,
        }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_run_id(value: Any) -> str:
    if not _is_sha256(value):
        raise ProtocolError("run_id must be a 64-character SHA-256 value")
    normalized = str(value).lower()
    if normalized != value:
        raise ProtocolError("run_id must use lowercase hexadecimal")
    return normalized


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def protocol_descriptor() -> dict[str, Any]:
    """Return the Qmax threshold sensitivity protocol."""

    rq2_protocol = rq2.protocol_descriptor()
    return {
        "reviewer_request": "R1.5 Qmax threshold sensitivity",
        "headline_rq1_rq3_qmax": HEADLINE_QMAX,
        "headline_policy": (
            "RQ1--RQ3 remain fixed at Qmax=24; this is the only independent "
            "multi-budget sweep"
        ),
        "qmax_values": list(QMAX_VALUES),
        "canonical_circuit_count": EXPECTED_CIRCUITS,
        "canonical_grid_record_count": EXPECTED_GRID_RECORDS,
        "record_schema": SCHEMA,
        "record_run_binding": (
            "every result records one run identifier so files from different "
            "sweep executions cannot be mixed"
        ),
        "selector": rq2_protocol["selector"],
        "selector_implementation": "coverage_rq2.crosscheck_selector",
        "point_universe": rq2_protocol["point_universe"],
        "grouping": (
            "all selected non-final points at one source gate form one exact "
            "joint group; two operands of one two-qubit gate share that "
            "group's union cone and one replay"
        ),
        "ancilla_demand": rq2_protocol["ancilla_demand"],
        "partitioner": "coverage_rq2._group_demands -> partition_nodes",
        "achieved_plan": (
            "every individually feasible group is assigned to an actual "
            "partition_nodes batch and every batch is emitted and structurally "
            "validated; no capacity lower bound is reported as an achieved plan"
        ),
        "individual_feasibility": (
            "num_source_qubits + group_union_ancillas <= Qmax; a group that "
            "fails alone is infeasible and cannot be rescued by batching"
        ),
        "cross_group_batching": (
            "different individually feasible source-gate groups may be split "
            "across independently emitted runs"
        ),
        "emitter": "coverage_rq2._emit_batches -> generate_monitoring_circuit",
        "structural_validator": "coverage_rq2._validate_record",
        "partition_validation": (
            "recompute selector, cone demands, and emitted circuits from source; "
            "verify the stored achieved partition structurally without rerunning "
            "the wall-clock-bounded optional MILP refinement"
        ),
        "final_read_convention": rq2_protocol["final_read_convention"],
        "deployed_observable_scope": (
            "individually feasible emitted mid-circuit points union "
            "certificate-selected final-read points"
        ),
        "deployment_run_cost": (
            "one run per emitted mid-circuit batch; if there is no emitted "
            "batch but at least one final-read point, one source-only final-read "
            "run is counted"
        ),
        "coverage_denominators": rq2_protocol["coverage_denominators"],
        "terminal_accounting": (
            "every selected circuit-budget identity has exactly one complete "
            "terminal complete/error record; errors never leave the denominator"
        ),
        "per_attempt_timeout_seconds": ATTEMPT_TIMEOUT_SECONDS,
        "partition_refinement_time_limit_seconds": (
            REFINEMENT_TIME_LIMIT_SECONDS
        ),
        "transient_license_retry_delays_seconds": list(
            batch_partition.GUROBI_LICENSE_RETRY_DELAYS_SECONDS
        ),
        "aggregation_gate": (
            "exactly 310 circuits x seven Qmax values, no duplicates or errors"
        ),
        "batch_count_interpretation": (
            "observed partition_nodes/emitter plan only; strict monotonicity "
            "across Qmax is not assumed for the heuristic plan"
        ),
    }


def dependency_versions() -> dict[str, str]:
    packages = (
        "qiskit", "qiskit-aer", "numpy", "scipy", "pulp", "gurobipy"
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    versions["python"] = platform.python_version()
    return versions


def build_fingerprint_bundle(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a run to the circuit list, QASM corpus, dependencies, sources, and protocol."""

    dependencies = dependency_versions()
    missing_sources = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing_sources:
        raise ProtocolError(f"Qmax sensitivity source files missing: {missing_sources}")
    source_entries = [
        {"path": path.name, "sha256": file_sha256(path)}
        for path in SOURCE_FILES
    ]
    protocol = protocol_descriptor()
    manifest_sha = frame.get("manifest_sha256")
    corpus_sha = frame.get("corpus_sha256")
    circuit_count = frame.get("circuit_count")
    if not _is_sha256(manifest_sha) or not _is_sha256(corpus_sha):
        raise ProtocolError("frame lacks valid circuit-list/corpus checksums")
    if not isinstance(circuit_count, int) or circuit_count <= 0:
        raise ProtocolError("frame lacks a positive circuit count")
    return {
        "manifest": {
            "sha256": manifest_sha,
            "basename": Path(frame.get("manifest", DEFAULT_MANIFEST)).name,
        },
        "qasm_corpus": {
            "sha256": corpus_sha,
            "circuit_count": circuit_count,
        },
        "dependencies": {
            "sha256": stable_hash(dependencies),
            "versions": dependencies,
        },
        "sources": {
            "sha256": stable_hash(source_entries),
            "files": source_entries,
        },
        "protocol": {
            "sha256": stable_hash(protocol),
            "value": protocol,
        },
    }


def _record_fingerprints(
    bundle: Mapping[str, Any], attempt: BudgetAttempt
) -> dict[str, str]:
    return {
        "manifest_sha256": bundle["manifest"]["sha256"],
        "corpus_sha256": bundle["qasm_corpus"]["sha256"],
        "source_qasm_sha256": attempt.source_qasm_sha256,
        "dependencies_sha256": bundle["dependencies"]["sha256"],
        "sources_sha256": bundle["sources"]["sha256"],
        "protocol_sha256": bundle["protocol"]["sha256"],
    }


def build_attempt_grid(
    frame: Mapping[str, Any], *, require_canonical: bool = True
) -> list[BudgetAttempt]:
    """Build the budget-major circuit grid with fixed identities."""

    circuits = frame.get("circuits")
    if not isinstance(circuits, list) or not circuits:
        raise ProtocolError("frame lacks a non-empty circuit table")
    if frame.get("circuit_count") != len(circuits):
        raise ProtocolError("frame circuit_count does not match its circuit table")
    if require_canonical:
        if len(circuits) != EXPECTED_CIRCUITS:
            raise ProtocolError(
                f"Qmax frame requires exactly {EXPECTED_CIRCUITS} circuits"
            )
        if frame.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("Qmax frame circuit-list checksum is not the reference value")
        if frame.get("corpus_sha256") != EXPECTED_CORPUS_SHA256:
            raise ProtocolError("Qmax frame QASM corpus checksum is not the reference value")

    seen_circuits: set[str] = set()
    for circuit_index, entry in enumerate(circuits):
        required = (
            "attempt_index",
            "attempt_id",
            "circuit",
            "source_qasm_sha256",
        )
        if any(field not in entry for field in required):
            raise ProtocolError(f"incomplete frame circuit entry at {circuit_index}")
        if entry["attempt_index"] != circuit_index:
            raise ProtocolError("frame circuit indices are not ordered and contiguous")
        if entry["circuit"] in seen_circuits:
            raise ProtocolError(f"duplicate frame circuit {entry['circuit']}")
        if not _is_sha256(entry["source_qasm_sha256"]):
            raise ProtocolError(f"invalid QASM checksum for {entry['circuit']}")
        seen_circuits.add(entry["circuit"])

    attempts: list[BudgetAttempt] = []
    for budget_index, qmax in enumerate(QMAX_VALUES):
        for circuit_index, entry in enumerate(circuits):
            attempt_index = budget_index * len(circuits) + circuit_index
            attempts.append(
                BudgetAttempt(
                    attempt_index=attempt_index,
                    attempt_id=f"qmax:{qmax}:{entry['attempt_id']}",
                    circuit_index=circuit_index,
                    budget_index=budget_index,
                    circuit=entry["circuit"],
                    qmax=qmax,
                    source_qasm_sha256=entry["source_qasm_sha256"],
                    source_attempt_id=entry["attempt_id"],
                )
            )
    identifiers = [attempt.attempt_id for attempt in attempts]
    if len(identifiers) != len(set(identifiers)):
        raise ProtocolError("Qmax grid contains duplicate attempt identities")
    if require_canonical and len(attempts) != EXPECTED_GRID_RECORDS:
        raise ProtocolError(
            f"Qmax grid requires exactly {EXPECTED_GRID_RECORDS} identities"
        )
    return attempts


def select_attempts(
    attempts: Sequence[BudgetAttempt], *, shard_index: int, shard_count: int
) -> list[BudgetAttempt]:
    if shard_count <= 0:
        raise ProtocolError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ProtocolError(
            f"shard_index must be in [0, {shard_count}); got {shard_index}"
        )
    return [
        attempt
        for position, attempt in enumerate(attempts)
        if position % shard_count == shard_index
    ]


@contextlib.contextmanager
def _rq2_qmax(qmax: int):
    """Temporarily parameterize the sequential RQ2 implementation by budget."""

    if qmax not in QMAX_VALUES:
        raise ProtocolError(f"Qmax must be one of {list(QMAX_VALUES)}; got {qmax}")
    previous = rq2.QMAX
    previous_refinement = rq2.REFINEMENT_TIME_LIMIT_SECONDS
    rq2.QMAX = qmax
    rq2.REFINEMENT_TIME_LIMIT_SECONDS = REFINEMENT_TIME_LIMIT_SECONDS
    try:
        yield
    finally:
        rq2.QMAX = previous
        rq2.REFINEMENT_TIME_LIMIT_SECONDS = previous_refinement


def partition_group_demands(
    num_qubits: int, group_demands: Sequence[Mapping[str, Any]], qmax: int
) -> dict[str, Any]:
    """Classify groups and compute the partition used by the experiment."""

    if qmax not in QMAX_VALUES:
        raise ProtocolError(f"Qmax {qmax} is outside the sweep grid")
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ProtocolError("num_qubits must be a non-negative integer")
    canonical_groups = []
    seen_gates: set[int] = set()
    for raw in group_demands:
        gate = raw.get("gate")
        members = raw.get("members")
        demand = raw.get("ancilla_demand")
        if (
            not isinstance(gate, int)
            or gate in seen_gates
            or not isinstance(members, list)
            or not members
            or members != sorted(set(members))
            or not all(isinstance(qubit, int) and qubit >= 0 for qubit in members)
            or not isinstance(demand, int)
            or demand < 0
        ):
            raise ProtocolError("malformed same-gate group demand")
        seen_gates.add(gate)
        canonical_groups.append(
            {"gate": gate, "members": list(members), "ancilla_demand": demand}
        )
    canonical_groups.sort(key=lambda group: group["gate"])

    constraint: dict[Any, Any] = {"num_oq": num_qubits}
    for group in canonical_groups:
        constraint[group["gate"]] = {
            "num_eq": group["ancilla_demand"],
            "qubits": group["members"],
        }
    batches, infeasible, partition_provenance = rq2.partition_nodes(
        constraint,
        qmax,
        return_metadata=True,
        refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
    )
    canonical_batches = sorted(
        (
            {"nodes": sorted(batch["nodes"]), "cost": int(batch["cost"])}
            for batch in batches
        ),
        key=lambda batch: tuple(batch["nodes"]),
    )
    infeasible = sorted(int(gate) for gate in infeasible)
    infeasible_set = set(infeasible)
    feasible_groups = [
        group.copy()
        for group in canonical_groups
        if group["gate"] not in infeasible_set
    ]
    infeasible_groups = [
        group.copy()
        for group in canonical_groups
        if group["gate"] in infeasible_set
    ]
    covered = [gate for batch in canonical_batches for gate in batch["nodes"]]
    expected_feasible = {group["gate"] for group in feasible_groups}
    if len(covered) != len(set(covered)) or set(covered) != expected_feasible:
        raise ProtocolError("partition_nodes did not cover every feasible group once")
    for batch in canonical_batches:
        if num_qubits + batch["cost"] > qmax:
            raise ProtocolError("partition_nodes returned an over-budget batch")
    return {
        "batches": canonical_batches,
        "feasible_groups": feasible_groups,
        "infeasible_groups": infeasible_groups,
        "infeasible_gates": infeasible,
        "partition_provenance": partition_provenance,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _candidate_groups(
    certificate: set[tuple[int, int]],
    mid_candidates: set[tuple[int, int]],
    final_reads: set[tuple[int, int]],
    demands: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    demand_by_gate = {group["gate"]: group for group in demands}
    groups = []
    for gate in sorted({gate for gate, _ in certificate}):
        members = sorted(
            qubit for source_gate, qubit in certificate if source_gate == gate
        )
        mid_members = sorted(
            qubit for source_gate, qubit in mid_candidates if source_gate == gate
        )
        final_members = sorted(
            qubit for source_gate, qubit in final_reads if source_gate == gate
        )
        demand = demand_by_gate.get(gate)
        groups.append(
            {
                "gate": gate,
                "members": members,
                "point_count": len(members),
                "mid_members": mid_members,
                "final_read_members": final_members,
                "mid_union_ancilla_demand": (
                    demand["ancilla_demand"] if demand is not None else None
                ),
            }
        )
    return groups


def _derive_sensitivity_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the sensitivity-analysis fields from an RQ2 result."""

    qmax = result.get("qmax")
    if qmax not in QMAX_VALUES:
        raise ProtocolError(f"result has missing or out-of-grid budget {qmax!r}")
    num_qubits = result.get("num_qubits")
    if not isinstance(num_qubits, int) or num_qubits > qmax:
        raise ProtocolError(
            f"source width {num_qubits!r} is not deployable at Qmax={qmax}"
        )

    record = dict(result)
    certificate = rq2._point_set(record, "certificate_points")
    numerical = rq2._point_set(record, "numerical_candidate_points")
    mid_candidates = rq2._point_set(record, "prebudget_mid_points")
    final_reads = rq2._point_set(record, "final_read_points")
    emitted = rq2._point_set(record, "emitted_mid_points")
    infeasible = rq2._point_set(record, "infeasible_mid_points")
    deployed = rq2._point_set(record, "deployed_observable_points")
    if numerical != certificate:
        raise ProtocolError("certificate/numerical candidate identity drift")

    demands = result.get("group_demands")
    raw_batches = result.get("batches")
    if not isinstance(demands, list) or not isinstance(raw_batches, list):
        raise ProtocolError("result lacks group demands or generated batches")
    stored_plan = [
        {"nodes": batch.get("gate_groups"), "cost": batch.get("ancilla_cost")}
        for batch in raw_batches
    ]
    feasible_groups = result.get("individually_feasible_gate_groups")
    infeasible_groups = result.get("individually_infeasible_gate_groups")
    infeasible_gates = result.get("infeasible_gate_groups")
    provenance = result.get("partition_provenance")
    if not isinstance(feasible_groups, list):
        raise ProtocolError("stored individually feasible groups are malformed")
    if not isinstance(infeasible_groups, list):
        raise ProtocolError("stored individually infeasible groups are malformed")
    if not isinstance(infeasible_gates, list):
        raise ProtocolError("stored infeasible gate identities are malformed")
    if not isinstance(provenance, Mapping):
        raise ProtocolError("stored partition provenance is malformed")
    constraint = {"num_oq": num_qubits}
    for group in demands:
        constraint[group["gate"]] = {
            "num_eq": group["ancilla_demand"],
            "qubits": list(group["members"]),
        }
    try:
        batch_partition.validate_partition_certificate(
            constraint,
            qmax,
            stored_plan,
            infeasible_gates,
            dict(provenance),
            refinement_time_limit_seconds=REFINEMENT_TIME_LIMIT_SECONDS,
        )
    except ValueError as exc:
        raise ProtocolError(f"stored partition certificate is invalid: {exc}") from exc
    expected_feasible = [
        group
        for group in demands
        if num_qubits + group["ancilla_demand"] <= qmax
    ]
    expected_infeasible = [
        group
        for group in demands
        if num_qubits + group["ancilla_demand"] > qmax
    ]
    expected_infeasible_gates = [group["gate"] for group in expected_infeasible]
    if feasible_groups != expected_feasible:
        raise ProtocolError("stored individually feasible groups drifted")
    if infeasible_groups != expected_infeasible:
        raise ProtocolError("stored individually infeasible groups drifted")
    if infeasible_gates != expected_infeasible_gates:
        raise ProtocolError("stored infeasible gate identities drifted")

    feasible_gate_set = {group["gate"] for group in feasible_groups}
    seen_gates: set[int] = set()
    demand_by_gate = {group["gate"]: group["ancilla_demand"] for group in demands}
    for batch in stored_plan:
        nodes = batch["nodes"]
        cost = batch["cost"]
        if (
            not isinstance(nodes, list)
            or nodes != sorted(set(nodes))
            or not isinstance(cost, int)
        ):
            raise ProtocolError("stored partition batch is malformed")
        if seen_gates.intersection(nodes):
            raise ProtocolError("stored partition repeats a gate group")
        if set(nodes) - feasible_gate_set:
            raise ProtocolError("stored partition contains an infeasible gate group")
        if cost != sum(demand_by_gate[node] for node in nodes):
            raise ProtocolError("stored partition batch cost drifted")
        if num_qubits + cost > qmax:
            raise ProtocolError("stored partition batch exceeds Qmax")
        seen_gates.update(nodes)
    if seen_gates != feasible_gate_set:
        raise ProtocolError("stored partition silently drops feasible gate groups")

    feasible_gates = feasible_gate_set
    feasible_mid = {
        point for point in mid_candidates if point[0] in feasible_gates
    }
    if feasible_mid != emitted:
        raise ProtocolError("actual instrumented points are not all feasible mid points")
    if mid_candidates != feasible_mid | infeasible or feasible_mid & infeasible:
        raise ProtocolError("individual-feasibility point accounting failed")
    if deployed != feasible_mid | final_reads:
        raise ProtocolError("deployed observable point accounting failed")

    members_by_gate = {
        group["gate"]: tuple(group["members"]) for group in demands
    }
    batches = []
    for batch_index, raw in enumerate(raw_batches):
        gates = list(raw["gate_groups"])
        monitored_points = {
            (gate, qubit) for gate in gates for qubit in members_by_gate[gate]
        }
        batches.append(
            {
                "batch": batch_index,
                "gate_groups": gates,
                "ancilla_cost": int(raw["ancilla_cost"]),
                "emitted_total_qubits": int(raw["emitted_total_qubits"]),
                "metadata_sha256": raw["metadata_sha256"],
                "monitor_groups": raw["monitor_groups"],
                "monitor_group_count": len(gates),
                "monitor_point_count": len(monitored_points),
                "monitored_points": rq2._as_point_list(monitored_points),
                "replay_gate_count": raw["replay_gate_count"],
                "monitor_measurement_count": raw["monitor_measurement_count"],
                "monitor_reset_count": raw["monitor_reset_count"],
                "source_executable_operation_count": raw[
                    "source_executable_operation_count"
                ],
                "emitted_executable_operation_count": raw[
                    "emitted_executable_operation_count"
                ],
                "source_executable_depth": raw["source_executable_depth"],
                "emitted_executable_depth": raw["emitted_executable_depth"],
                "depth_increase": raw["depth_increase"],
                "depth_inflation_ratio": raw["depth_inflation_ratio"],
            }
        )
    flattened_points = [
        tuple(point) for batch in batches for point in batch["monitored_points"]
    ]
    if (
        set(flattened_points) != feasible_mid
        or len(flattened_points) != len(set(flattened_points))
    ):
        raise ProtocolError("per-batch monitor accounting is incomplete or duplicated")

    source_only_final_run = int(bool(final_reads) and not batches)
    deployment_run_count = len(batches) + source_only_final_run
    run_widths = [batch["emitted_total_qubits"] for batch in batches]
    run_ancillas = [batch["ancilla_cost"] for batch in batches]
    if source_only_final_run:
        run_widths.append(num_qubits)
        run_ancillas.append(0)
    source_m = result["gate_count"]
    source_executable_operations = result["source_executable_operation_count"]
    source_executable_depth = result["source_executable_depth"]
    replay_gate_executions = sum(
        batch["replay_gate_count"] for batch in batches
    )
    monitor_measurements = sum(
        batch["monitor_measurement_count"] for batch in batches
    )
    monitor_resets = sum(batch["monitor_reset_count"] for batch in batches)
    original_gate_executions = deployment_run_count * source_m
    work_bm_plus_replay = original_gate_executions + replay_gate_executions
    total_counted_operations = (
        work_bm_plus_replay + monitor_measurements + monitor_resets
    )
    deployment_depths = [batch["emitted_executable_depth"] for batch in batches]
    deployment_depth_increases = [batch["depth_increase"] for batch in batches]
    deployment_depth_ratios = [
        batch["depth_inflation_ratio"] for batch in batches
        if batch["depth_inflation_ratio"] is not None
    ]
    emitted_operation_executions = sum(
        batch["emitted_executable_operation_count"] for batch in batches
    )
    if source_only_final_run:
        deployment_depths.append(source_executable_depth)
        deployment_depth_increases.append(0)
        deployment_depth_ratios.append(1.0)
        emitted_operation_executions += source_executable_operations
    candidate_groups = _candidate_groups(
        certificate, mid_candidates, final_reads, demands
    )
    complete_denominators = {
        "circuit_count": 1,
        "gate_node_count": result["gate_node_denominator"],
        "gate_qubit_event_count": result["gate_qubit_event_denominator"],
        "qubit_count": result["qubit_denominator"],
        "normalized_depth_layer_count": result["normalized_depth_denominator"],
        "certificate_candidate_point_count": len(certificate),
        "certificate_candidate_group_count": len(candidate_groups),
        "mid_candidate_point_count": len(mid_candidates),
        "mid_candidate_group_count": len(demands),
    }
    return {
        "certificate_candidate_points": rq2._as_point_list(certificate),
        "certificate_candidate_groups": candidate_groups,
        "certificate_candidate_point_count": len(certificate),
        "certificate_candidate_group_count": len(candidate_groups),
        "mid_candidate_points": rq2._as_point_list(mid_candidates),
        "mid_candidate_groups": [dict(group) for group in demands],
        "mid_candidate_point_count": len(mid_candidates),
        "mid_candidate_group_count": len(demands),
        "individually_feasible_mid_points": rq2._as_point_list(feasible_mid),
        "individually_infeasible_mid_points": rq2._as_point_list(infeasible),
        "individually_feasible_mid_groups": feasible_groups,
        "individually_infeasible_mid_groups": infeasible_groups,
        "individually_feasible_mid_point_count": len(feasible_mid),
        "individually_infeasible_mid_point_count": len(infeasible),
        "individually_feasible_mid_group_count": len(feasible_groups),
        "individually_infeasible_mid_group_count": len(infeasible_groups),
        "deployed_observable_candidate_fraction": _ratio(
            len(deployed), len(certificate)
        ),
        "individually_feasible_mid_point_fraction": _ratio(
            len(feasible_mid), len(mid_candidates)
        ),
        "individually_feasible_mid_group_fraction": _ratio(
            len(feasible_groups), len(demands)
        ),
        "fully_deployable": deployed == certificate,
        "batches": batches,
        "instrumented_batch_count": len(batches),
        "source_only_final_read_run_count": source_only_final_run,
        "deployment_run_count": deployment_run_count,
        "batch_monitor_point_counts": [
            batch["monitor_point_count"] for batch in batches
        ],
        "batch_monitor_group_counts": [
            batch["monitor_group_count"] for batch in batches
        ],
        "peak_instrumented_width": max(
            (batch["emitted_total_qubits"] for batch in batches), default=None
        ),
        "peak_instrumented_ancilla_count": max(
            (batch["ancilla_cost"] for batch in batches), default=0
        ),
        "peak_deployment_run_width": max(run_widths, default=None),
        "peak_deployment_run_ancilla_count": max(run_ancillas, default=None),
        "cumulative_extra_ancilla_allocations": sum(run_ancillas),
        "planned_original_gate_executions": original_gate_executions,
        "planned_replay_gate_executions": replay_gate_executions,
        "planned_monitor_measurement_operations": monitor_measurements,
        "planned_monitor_reset_operations": monitor_resets,
        "planned_work_Bm_plus_replay": work_bm_plus_replay,
        "planned_total_counted_operations": total_counted_operations,
        "actual_emitted_executable_operation_executions": (
            emitted_operation_executions
        ),
        "source_executable_depth": source_executable_depth,
        "peak_deployment_run_depth": max(deployment_depths, default=None),
        "cumulative_deployment_run_depth": sum(deployment_depths),
        "peak_depth_increase": max(deployment_depth_increases, default=None),
        "peak_depth_inflation_ratio": max(deployment_depth_ratios, default=None),
        "complete_denominators": complete_denominators,
        "partition_provenance": dict(provenance),
        "plan_semantics": (
            "achieved_partition_nodes_batches_real_emitter_source_recomputed"
        ),
    }


def _validate_sensitivity_fields(result: Mapping[str, Any]) -> None:
    expected = _derive_sensitivity_fields(result)
    for field, value in expected.items():
        if result.get(field) != value:
            raise ProtocolError(f"derived Qmax sensitivity field drift: {field}")


def evaluate_budget(
    attempt: BudgetAttempt, qasm_dir: Path | str
) -> dict[str, Any]:
    """Evaluate one circuit and budget with the RQ2 implementation."""

    entry = attempt.coverage_entry()
    qasm_dir = Path(qasm_dir)
    with _rq2_qmax(attempt.qmax):
        status, result = rq2.evaluate_circuit(entry, qasm_dir)
        if status.get("status") != "complete":
            raise ProtocolError(f"RQ2 did not complete {attempt.attempt_id}")
        rq2._validate_record(result, entry, qasm_dir)
        result.update(_derive_sensitivity_fields(result))
        _validate_sensitivity_fields(result)
    return result


@contextlib.contextmanager
def _attempt_timeout(seconds: int = ATTEMPT_TIMEOUT_SECONDS):
    """Bound one circuit-budget evaluation on POSIX workers."""

    if seconds <= 0:
        raise ValueError("attempt timeout must be positive")
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"Qmax attempt exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return stable_hash(payload)


def _add_record_checksum(record: dict[str, Any]) -> dict[str, Any]:
    record["record_sha256"] = _record_sha256(record)
    return record


def _record_skeleton(
    attempt: BudgetAttempt,
    bundle: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    run_id: str,
) -> dict[str, Any]:
    run_id = _require_run_id(run_id)
    return {
        "schema": SCHEMA,
        "attempt_index": attempt.attempt_index,
        "attempt_id": attempt.attempt_id,
        "circuit_index": attempt.circuit_index,
        "budget_index": attempt.budget_index,
        "circuit": attempt.circuit,
        "qmax": attempt.qmax,
        "source_attempt_id": attempt.source_attempt_id,
        "source_qasm_sha256": attempt.source_qasm_sha256,
        "status": None,
        "record_complete": False,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "error": None,
        "result": None,
        "fingerprints": _record_fingerprints(bundle, attempt),
        "run": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "run_id": run_id,
        },
        "record_sha256": None,
    }


def _finish_record(record: dict[str, Any], started: float) -> dict[str, Any]:
    record["completed_at_utc"] = _utc_now()
    record["elapsed_seconds"] = round(time.perf_counter() - started, 9)
    record["record_complete"] = True
    return _add_record_checksum(record)


def terminal_error_record(
    attempt: BudgetAttempt,
    bundle: Mapping[str, Any],
    error: BaseException,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    run_id: str = DEFAULT_RUN_ID,
    started: float | None = None,
) -> dict[str, Any]:
    """Record an unsuccessful evaluation so it remains in the denominator."""

    started = time.perf_counter() if started is None else started
    record = _record_skeleton(
        attempt,
        bundle,
        shard_index=shard_index,
        shard_count=shard_count,
        run_id=run_id,
    )
    trace = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    record["status"] = "error"
    record["error"] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback_sha256": stable_hash(trace),
    }
    return _finish_record(record, started)


def evaluate_attempt(
    attempt: BudgetAttempt,
    qasm_dir: Path | str,
    bundle: Mapping[str, Any],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    run_id: str = DEFAULT_RUN_ID,
    attempt_timeout_seconds: int = ATTEMPT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    record = _record_skeleton(
        attempt,
        bundle,
        shard_index=shard_index,
        shard_count=shard_count,
        run_id=run_id,
    )
    try:
        with _attempt_timeout(attempt_timeout_seconds):
            record["result"] = evaluate_budget(attempt, qasm_dir)
    except Exception as error:
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        record["status"] = "error"
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_sha256": stable_hash(trace),
        }
    else:
        record["status"] = "complete"
    return _finish_record(record, started)


def validate_record(
    record: Mapping[str, Any],
    attempt: BudgetAttempt,
    frame: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    expected_run_id: str = DEFAULT_RUN_ID,
    deep: bool = True,
) -> None:
    """Validate one completed record, its checksum, and computed quantities."""

    if not isinstance(record, Mapping):
        raise ProtocolError("Qmax record is not an object")
    if set(record) != RECORD_FIELDS:
        raise ProtocolError(
            f"Qmax record field contract drift for {attempt.attempt_id}: "
            f"missing={sorted(RECORD_FIELDS-set(record))}, "
            f"extra={sorted(set(record)-RECORD_FIELDS)}"
        )
    if record.get("schema") != SCHEMA:
        raise ProtocolError(f"legacy or wrong Qmax schema: {record.get('schema')!r}")
    if record.get("record_sha256") != _record_sha256(record):
        raise ProtocolError(f"record checksum mismatch for {attempt.attempt_id}")
    if record.get("record_complete") is not True:
        raise ProtocolError(f"non-terminal partial record for {attempt.attempt_id}")

    identity = {
        "attempt_index": attempt.attempt_index,
        "attempt_id": attempt.attempt_id,
        "circuit_index": attempt.circuit_index,
        "budget_index": attempt.budget_index,
        "circuit": attempt.circuit,
        "qmax": attempt.qmax,
        "source_attempt_id": attempt.source_attempt_id,
        "source_qasm_sha256": attempt.source_qasm_sha256,
    }
    for field, expected in identity.items():
        if record.get(field) != expected:
            raise ProtocolError(
                f"missing or drifted budget identity field {field} for "
                f"{attempt.attempt_id}"
            )
    if record.get("fingerprints") != _record_fingerprints(bundle, attempt):
        raise ProtocolError(f"checksum mismatch for {attempt.attempt_id}")
    if record.get("status") not in TERMINAL_STATUSES:
        raise ProtocolError(f"invalid terminal status for {attempt.attempt_id}")
    run = record.get("run")
    if not isinstance(run, Mapping):
        raise ProtocolError(f"missing run metadata for {attempt.attempt_id}")
    if set(run) != {"shard_index", "shard_count", "run_id"}:
        raise ProtocolError(f"run metadata contract drift for {attempt.attempt_id}")
    shard_index = run.get("shard_index")
    shard_count = run.get("shard_count")
    if (
        type(shard_index) is not int
        or type(shard_count) is not int
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
    ):
        raise ProtocolError(f"invalid shard metadata for {attempt.attempt_id}")
    expected_run_id = _require_run_id(expected_run_id)
    if run.get("run_id") != expected_run_id:
        raise ProtocolError(
            f"run_id mismatch for {attempt.attempt_id}: "
            f"{run.get('run_id')!r} != {expected_run_id!r}"
        )
    for field in ("started_at_utc", "completed_at_utc"):
        value = record.get(field)
        if not isinstance(value, str):
            raise ProtocolError(f"missing UTC timestamp {field} for {attempt.attempt_id}")
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError as error:
            raise ProtocolError(
                f"invalid UTC timestamp {field} for {attempt.attempt_id}"
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
            raise ProtocolError(
                f"non-UTC timestamp {field} for {attempt.attempt_id}"
            )
    elapsed = record.get("elapsed_seconds")
    if (
        type(elapsed) not in {int, float}
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ProtocolError(f"invalid elapsed time for {attempt.attempt_id}")

    if record["status"] == "error":
        error = record.get("error")
        if record.get("result") is not None or not isinstance(error, Mapping):
            raise ProtocolError(f"malformed error record for {attempt.attempt_id}")
        required_error = {"type", "message", "traceback_sha256"}
        if set(error) != required_error:
            raise ProtocolError(f"incomplete structured error for {attempt.attempt_id}")
        if not _is_sha256(error.get("traceback_sha256")):
            raise ProtocolError(f"invalid traceback checksum for {attempt.attempt_id}")
        return

    if record.get("error") is not None or not isinstance(record.get("result"), Mapping):
        raise ProtocolError(f"malformed complete record for {attempt.attempt_id}")
    result = record["result"]
    entry = attempt.coverage_entry()
    if deep:
        with _rq2_qmax(attempt.qmax):
            rq2._validate_record(result, entry, Path(frame["qasm_dir"]))
    _validate_sensitivity_fields(result)


def _validate_identity_coverage(
    records: Sequence[Mapping[str, Any]],
    attempts: Sequence[BudgetAttempt],
    *,
    require_complete: bool,
) -> None:
    expected_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    if len(expected_by_id) != len(attempts):
        raise ProtocolError("declared attempt frame contains duplicate identities")
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for record in records:
        attempt_id = record.get("attempt_id")
        attempt_index = record.get("attempt_index")
        if attempt_id in seen_ids or attempt_index in seen_indices:
            raise ProtocolError(f"duplicate Qmax record identity: {attempt_id!r}")
        seen_ids.add(attempt_id)
        seen_indices.add(attempt_index)
    extras = sorted(seen_ids - set(expected_by_id))
    if extras:
        raise ProtocolError(f"foreign Qmax record identities: {extras}")
    if require_complete:
        missing = sorted(set(expected_by_id) - seen_ids)
        if missing:
            raise ProtocolError(
                "partial Qmax grid rejected: "
                f"missing {len(missing)} of {len(attempts)} identities"
            )
        if len(records) != len(attempts):
            raise ProtocolError("Qmax grid record count is not exact")


def validate_record_set(
    records: Sequence[Mapping[str, Any]],
    attempts: Sequence[BudgetAttempt],
    frame: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    require_complete: bool,
    require_success: bool = False,
    expected_run_id: str = DEFAULT_RUN_ID,
    deep: bool = True,
) -> dict[str, Any]:
    """Validate a shard or the full grid without survivor-only accounting."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ProtocolError("Qmax record set must be a sequence of objects")
    expected_run_id = _require_run_id(expected_run_id)
    _validate_identity_coverage(records, attempts, require_complete=require_complete)

    global_fingerprint_fields = (
        "manifest_sha256",
        "corpus_sha256",
        "dependencies_sha256",
        "sources_sha256",
        "protocol_sha256",
    )
    for field in global_fingerprint_fields:
        values = {
            record.get("fingerprints", {}).get(field)
            for record in records
            if isinstance(record.get("fingerprints"), Mapping)
        }
        if len(values) > 1:
            raise ProtocolError(f"mixed {field} checksums in Qmax records")

    attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    for record in records:
        attempt_id = record.get("attempt_id")
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None:
            raise ProtocolError(f"foreign Qmax record {attempt_id!r}")
        validate_record(
            record,
            attempt,
            frame,
            bundle,
            expected_run_id=expected_run_id,
            deep=deep,
        )

    statuses = Counter(record["status"] for record in records)
    if require_success and statuses.get("error", 0):
        raise ProtocolError(
            "Qmax aggregate requires all-success terminal grid; "
            f"errors={statuses['error']}"
        )
    return {
        "record_count": len(records),
        "complete": statuses.get("complete", 0),
        "error": statuses.get("error", 0),
        "expected_identity_count": len(attempts),
    }


def atomic_write_jsonl(
    records: Iterable[Mapping[str, Any]], path: Path | str
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_records(records: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Atomically write records as JSONL or a hash-checked JSON envelope."""

    destination = Path(path)
    if destination.suffix == ".jsonl":
        atomic_write_jsonl(records, destination)
        return
    if destination.suffix == ".json":
        run_ids = {
            record.get("run", {}).get("run_id")
            for record in records
            if isinstance(record.get("run"), Mapping)
        }
        if len(run_ids) != 1:
            raise ProtocolError("Qmax JSON envelope requires one run_id")
        run_id = _require_run_id(next(iter(run_ids)))
        value = {
            "schema": RECORD_SET_SCHEMA,
            "record_schema": SCHEMA,
            "run_id": run_id,
            "record_count": len(records),
            "records_sha256": stable_hash(list(records)),
            "records": list(records),
        }
        atomic_write_json(destination, value)
        return
    raise ProtocolError("record output must end in .jsonl or .json")


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(token):
    raise ValueError(f"non-finite JSON constant {token!r}")


def _strict_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {token!r}")
    return value


def _strict_json_loads(text: str):
    return json.loads(
        text,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_nonfinite_json_constant,
        parse_float=_strict_json_float,
    )


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """Load current-schema JSON or JSONL input."""

    source = Path(path)
    if source.name == "qmax_records.pkl" or source.suffix.lower() in {
        ".pkl",
        ".pickle",
    }:
        raise ProtocolError(
            "legacy qmax_records.pkl/list pickle is incompatible with "
            f"{SCHEMA} and must be rerun"
        )
    if source.suffix == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = _strict_json_loads(line)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ProtocolError(
                        f"{source}:{line_number}: invalid JSONL: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ProtocolError(
                        f"{source}:{line_number}: Qmax record is not an object"
                    )
                records.append(value)
        return records
    if source.suffix == ".json":
        try:
            value = _strict_json_loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ProtocolError(f"{source}: invalid strict JSON: {error}") from error
        expected_fields = {
            "schema",
            "record_schema",
            "run_id",
            "record_count",
            "records_sha256",
            "records",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ProtocolError("JSON input is not a current Qmax record-set envelope")
        if value.get("schema") != RECORD_SET_SCHEMA:
            raise ProtocolError("JSON input has a legacy Qmax record-set schema")
        records = value.get("records")
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ProtocolError("Qmax JSON envelope lacks object records")
        if value.get("record_schema") != SCHEMA:
            raise ProtocolError("Qmax JSON envelope contains a legacy record schema")
        run_ids = {
            record.get("run", {}).get("run_id")
            for record in records
            if isinstance(record.get("run"), Mapping)
        }
        if (
            len(run_ids) != 1
            or value.get("run_id") != next(iter(run_ids))
        ):
            raise ProtocolError("Qmax JSON envelope run_id mismatch")
        _require_run_id(value.get("run_id"))
        if value.get("record_count") != len(records):
            raise ProtocolError("Qmax JSON envelope record_count mismatch")
        if value.get("records_sha256") != stable_hash(records):
            raise ProtocolError("Qmax JSON envelope checksum mismatch")
        return records
    raise ProtocolError("Qmax input must be current .jsonl or .json, never pickle")


def _sorted_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda record: record["attempt_index"])


def _validate_cross_budget_invariants(
    records: Sequence[Mapping[str, Any]],
) -> None:
    static_fields = (
        "source_qasm_sha256",
        "num_qubits",
        "depth",
        "gate_count",
        "acted_points",
        "certificate_candidate_points",
        "certificate_candidate_groups",
        "mid_candidate_points",
        "mid_candidate_groups",
        "final_read_points",
        "complete_denominators",
        "source_executable_operation_count",
        "source_executable_depth",
    )
    by_circuit: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_circuit.setdefault(record["circuit"], []).append(record)
    for circuit, circuit_records in by_circuit.items():
        if sorted(record["qmax"] for record in circuit_records) != list(QMAX_VALUES):
            raise ProtocolError(f"budget grid drift for {circuit}")
        identities = {
            stable_hash(
                {field: record["result"][field] for field in static_fields}
            )
            for record in circuit_records
        }
        if len(identities) != 1:
            raise ProtocolError(
                f"selector/source/grouping invariants differ across Qmax for {circuit}"
            )


def _mean(values: Sequence[float | int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: Sequence[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _aggregate_budget(
    records: Sequence[Mapping[str, Any]], qmax: int
) -> dict[str, Any]:
    results = [record["result"] for record in records if record["qmax"] == qmax]
    if len(results) != EXPECTED_CIRCUITS:
        raise ProtocolError(
            f"Qmax={qmax} requires {EXPECTED_CIRCUITS} complete circuit rows"
        )

    denominator_fields = (
        "circuit_count",
        "gate_node_count",
        "gate_qubit_event_count",
        "qubit_count",
        "normalized_depth_layer_count",
        "certificate_candidate_point_count",
        "certificate_candidate_group_count",
        "mid_candidate_point_count",
        "mid_candidate_group_count",
    )
    denominators = {
        field: sum(result["complete_denominators"][field] for result in results)
        for field in denominator_fields
    }
    feasible_points = sum(
        result["individually_feasible_mid_point_count"] for result in results
    )
    infeasible_points = sum(
        result["individually_infeasible_mid_point_count"] for result in results
    )
    feasible_groups = sum(
        result["individually_feasible_mid_group_count"] for result in results
    )
    infeasible_groups = sum(
        result["individually_infeasible_mid_group_count"] for result in results
    )
    final_reads = sum(result["final_read_point_count"] for result in results)
    deployed = sum(result["deployed_observable_point_count"] for result in results)
    batches = [batch for result in results for batch in result["batches"]]
    batch_counts = [result["instrumented_batch_count"] for result in results]
    run_counts = [result["deployment_run_count"] for result in results]
    monitor_point_counts = [batch["monitor_point_count"] for batch in batches]
    monitor_group_counts = [batch["monitor_group_count"] for batch in batches]
    replay_gate_counts = [batch["replay_gate_count"] for batch in batches]
    depth_increases = [batch["depth_increase"] for batch in batches]
    depth_ratios = [
        batch["depth_inflation_ratio"] for batch in batches
        if batch["depth_inflation_ratio"] is not None
    ]
    fully_deployable = sum(bool(result["fully_deployable"]) for result in results)
    widths = [
        result["peak_deployment_run_width"]
        for result in results
        if result["peak_deployment_run_width"] is not None
    ]
    ancillas = [
        result["peak_deployment_run_ancilla_count"]
        for result in results
        if result["peak_deployment_run_ancilla_count"] is not None
    ]
    solver_descriptors = {
        stable_hash(result["partition_provenance"]["solver"])
        for result in results
    }
    if len(solver_descriptors) != 1:
        raise ProtocolError(f"mixed solver provenance at Qmax={qmax}")
    solver_descriptor = results[0]["partition_provenance"]["solver"]
    return {
        "qmax": qmax,
        "complete_circuit_denominator": denominators["circuit_count"],
        "denominators": denominators,
        "certificate_candidate_point_count": denominators[
            "certificate_candidate_point_count"
        ],
        "certificate_candidate_group_count": denominators[
            "certificate_candidate_group_count"
        ],
        "individually_feasible_mid_point_count": feasible_points,
        "individually_infeasible_mid_point_count": infeasible_points,
        "individually_feasible_mid_group_count": feasible_groups,
        "individually_infeasible_mid_group_count": infeasible_groups,
        "final_read_point_count": final_reads,
        "deployed_observable_point_count": deployed,
        "deployable_candidate_fraction": _ratio(
            deployed, denominators["certificate_candidate_point_count"]
        ),
        "deployed_observable_acted_point_coverage": _ratio(
            deployed, denominators["gate_qubit_event_count"]
        ),
        "individually_feasible_mid_point_fraction": _ratio(
            feasible_points, denominators["mid_candidate_point_count"]
        ),
        "individually_feasible_mid_group_fraction": _ratio(
            feasible_groups, denominators["mid_candidate_group_count"]
        ),
        "fully_deployable_circuit_count": fully_deployable,
        "fully_deployable_circuit_fraction": _ratio(
            fully_deployable, denominators["circuit_count"]
        ),
        "mean_deployed_observable_gate_coverage": _mean(
            [result["deployed_observable_gate_coverage"] for result in results]
        ),
        "mean_deployed_observable_qubit_coverage": _mean(
            [result["deployed_observable_qubit_coverage"] for result in results]
        ),
        "mean_deployed_observable_depth_reach": _mean(
            [result["deployed_observable_depth_reach"] for result in results]
        ),
        "instrumented_batch_count": sum(batch_counts),
        "deployment_run_count": sum(run_counts),
        "source_only_final_read_run_count": sum(
            result["source_only_final_read_run_count"] for result in results
        ),
        "mean_instrumented_batches_per_circuit": _mean(batch_counts),
        "median_instrumented_batches_per_circuit": _median(batch_counts),
        "max_instrumented_batches_per_circuit": max(batch_counts, default=0),
        "mean_deployment_runs_per_circuit": _mean(run_counts),
        "median_deployment_runs_per_circuit": _median(run_counts),
        "max_deployment_runs_per_circuit": max(run_counts, default=0),
        "per_batch_monitor_points": {
            "batch_denominator": len(batches),
            "total": sum(monitor_point_counts),
            "mean": _mean(monitor_point_counts),
            "median": _median(monitor_point_counts),
            "min": min(monitor_point_counts, default=None),
            "max": max(monitor_point_counts, default=None),
        },
        "per_batch_monitor_groups": {
            "batch_denominator": len(batches),
            "total": sum(monitor_group_counts),
            "mean": _mean(monitor_group_counts),
            "median": _median(monitor_group_counts),
            "min": min(monitor_group_counts, default=None),
            "max": max(monitor_group_counts, default=None),
        },
        "peak_deployment_run_width": max(widths, default=None),
        "peak_deployment_run_ancilla_count": max(ancillas, default=None),
        "cumulative_extra_ancilla_allocations": sum(
            result["cumulative_extra_ancilla_allocations"] for result in results
        ),
        "planned_original_gate_executions": sum(
            result["planned_original_gate_executions"] for result in results
        ),
        "planned_replay_gate_executions": sum(
            result["planned_replay_gate_executions"] for result in results
        ),
        "planned_monitor_measurement_operations": sum(
            result["planned_monitor_measurement_operations"] for result in results
        ),
        "planned_monitor_reset_operations": sum(
            result["planned_monitor_reset_operations"] for result in results
        ),
        "planned_work_Bm_plus_replay": sum(
            result["planned_work_Bm_plus_replay"] for result in results
        ),
        "planned_total_counted_operations": sum(
            result["planned_total_counted_operations"] for result in results
        ),
        "actual_emitted_executable_operation_executions": sum(
            result["actual_emitted_executable_operation_executions"]
            for result in results
        ),
        "per_batch_replay_gates": {
            "batch_denominator": len(batches),
            "total": sum(replay_gate_counts),
            "mean": _mean(replay_gate_counts),
            "median": _median(replay_gate_counts),
            "min": min(replay_gate_counts, default=None),
            "max": max(replay_gate_counts, default=None),
        },
        "per_batch_depth_increase": {
            "batch_denominator": len(batches),
            "mean": _mean(depth_increases),
            "median": _median(depth_increases),
            "min": min(depth_increases, default=None),
            "max": max(depth_increases, default=None),
        },
        "per_batch_depth_inflation_ratio": {
            "batch_denominator": len(depth_ratios),
            "mean": _mean(depth_ratios),
            "median": _median(depth_ratios),
            "min": min(depth_ratios, default=None),
            "max": max(depth_ratios, default=None),
        },
        "peak_deployment_run_depth": max(
            (
                result["peak_deployment_run_depth"]
                for result in results
                if result["peak_deployment_run_depth"] is not None
            ),
            default=None,
        ),
        "peak_depth_increase": max(
            (
                result["peak_depth_increase"] for result in results
                if result["peak_depth_increase"] is not None
            ),
            default=None,
        ),
        "peak_depth_inflation_ratio": max(
            (
                result["peak_depth_inflation_ratio"] for result in results
                if result["peak_depth_inflation_ratio"] is not None
            ),
            default=None,
        ),
        "solver": solver_descriptor,
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    frame: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    attempts: Sequence[BudgetAttempt] | None = None,
    expected_run_id: str = DEFAULT_RUN_ID,
    deep: bool = True,
) -> dict[str, Any]:
    """Aggregate a complete successful 310 by 7 result grid."""

    reference_attempts = build_attempt_grid(frame)
    if attempts is not None and list(attempts) != reference_attempts:
        raise ProtocolError("aggregate attempt frame is not the full 310 x 7 grid")
    attempts = reference_attempts
    validate_record_set(
        records,
        attempts,
        frame,
        bundle,
        require_complete=True,
        require_success=True,
        expected_run_id=expected_run_id,
        deep=deep,
    )
    if len(attempts) != EXPECTED_GRID_RECORDS:
        raise ProtocolError(
            f"aggregate requires the full {EXPECTED_GRID_RECORDS}-identity grid"
        )
    ordered = _sorted_records(records)
    _validate_cross_budget_invariants(ordered)
    rows = [_aggregate_budget(ordered, qmax) for qmax in QMAX_VALUES]

    by_circuit: dict[str, dict[int, int]] = {}
    for record in ordered:
        by_circuit.setdefault(record["circuit"], {})[record["qmax"]] = record[
            "result"
        ]["instrumented_batch_count"]
    nonmonotone = sum(
        any(
            counts[higher] > counts[lower]
            for lower, higher in zip(QMAX_VALUES, QMAX_VALUES[1:])
        )
        for counts in by_circuit.values()
    )

    artifact = {
        "schema": AGGREGATE_SCHEMA,
        "record_schema": SCHEMA,
        "run_id": _require_run_id(expected_run_id),
        "protocol": protocol_descriptor(),
        "fingerprints": dict(bundle),
        "grid": {
            "circuit_count": EXPECTED_CIRCUITS,
            "qmax_values": list(QMAX_VALUES),
            "record_count": EXPECTED_GRID_RECORDS,
            "terminal_complete_count": EXPECTED_GRID_RECORDS,
            "terminal_error_count": 0,
            "identity_sha256": stable_hash(
                [
                    [
                        record["attempt_index"],
                        record["attempt_id"],
                        record["source_qasm_sha256"],
                        record["qmax"],
                    ]
                    for record in ordered
                ]
            ),
        },
        "input_records_sha256": stable_hash(ordered),
        "threshold_sensitivity": {
            "question": (
                "How do achieved deployable coverage/fraction and actual "
                "batch/run cost change with Qmax?"
            ),
            "headline_qmax_reference": HEADLINE_QMAX,
            "rows": rows,
            "batch_count_monotonicity_assumed": False,
            "observed_nonmonotone_batch_count_circuit_count": nonmonotone,
            "capacity_lower_bound_used_as_plan": False,
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    return artifact


def _current_context(args: argparse.Namespace) -> tuple[
    dict[str, Any], list[BudgetAttempt], dict[str, Any]
]:
    frame = resolve_circuit_corpus(args.manifest, args.qasm_dir)
    attempts = build_attempt_grid(frame)
    bundle = build_fingerprint_bundle(frame)
    return frame, attempts, bundle


def run_protocol(args: argparse.Namespace) -> int:
    if args.attempt_timeout_seconds != ATTEMPT_TIMEOUT_SECONDS:
        raise ProtocolError(
            "the final Qmax run requires "
            f"--attempt-timeout-seconds={ATTEMPT_TIMEOUT_SECONDS}; "
            f"got {args.attempt_timeout_seconds}"
        )
    run_id = _require_run_id(args.run_id)
    frame, grid, bundle = _current_context(args)
    selected = select_attempts(
        grid, shard_index=args.shard_index, shard_count=args.shard_count
    )
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ProtocolError(f"output already exists: {output}; pass --resume")
    existing = read_records(output) if output.exists() else []
    validate_record_set(
        existing,
        selected,
        frame,
        bundle,
        require_complete=False,
        expected_run_id=run_id,
        deep=True,
    )
    by_id = {record["attempt_id"]: record for record in existing}
    if args.retry_errors:
        by_id = {
            attempt_id: record
            for attempt_id, record in by_id.items()
            if record["status"] == "complete"
        }
    pending = [attempt for attempt in selected if attempt.attempt_id not in by_id]
    print(
        f"Qmax sensitivity shard {args.shard_index}/{args.shard_count}: "
        f"selected={len(selected)} retained={len(by_id)} pending={len(pending)}",
        flush=True,
    )
    for position, attempt in enumerate(pending, start=1):
        record = evaluate_attempt(
            attempt,
            frame["qasm_dir"],
            bundle,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            run_id=run_id,
            attempt_timeout_seconds=args.attempt_timeout_seconds,
        )
        by_id[attempt.attempt_id] = record
        ordered = sorted(by_id.values(), key=lambda row: row["attempt_index"])
        write_records(ordered, output)
        print(
            f"[{position}/{len(pending)}] {attempt.attempt_id}: "
            f"{record['status']}",
            flush=True,
        )
    ordered = sorted(by_id.values(), key=lambda row: row["attempt_index"])
    summary = validate_record_set(
        ordered,
        selected,
        frame,
        bundle,
        require_complete=True,
        expected_run_id=run_id,
        deep=True,
    )
    write_records(ordered, output)
    print(f"wrote {output}: {summary}")
    return 0 if summary["error"] == 0 else 2


def validate_protocol(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args.run_id)
    frame, grid, bundle = _current_context(args)
    selected = select_attempts(
        grid, shard_index=args.shard_index, shard_count=args.shard_count
    )
    records = [record for path in args.inputs for record in read_records(path)]
    summary = validate_record_set(
        records,
        selected,
        frame,
        bundle,
        require_complete=True,
        require_success=args.require_success,
        expected_run_id=run_id,
        deep=True,
    )
    print(f"valid {SCHEMA} terminal record set: {summary}")
    return 0


def merge_protocol(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args.run_id)
    frame, grid, bundle = _current_context(args)
    records = [record for path in args.inputs for record in read_records(path)]
    summary = validate_record_set(
        records,
        grid,
        frame,
        bundle,
        require_complete=True,
        require_success=False,
        expected_run_id=run_id,
        deep=True,
    )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ProtocolError(f"merge output exists: {output}; pass --overwrite")
    ordered = _sorted_records(records)
    write_records(ordered, output)
    print(f"merged complete terminal grid into {output}: {summary}")
    return 0 if summary["error"] == 0 else 2


def aggregate_protocol(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args.run_id)
    frame, grid, bundle = _current_context(args)
    records = [record for path in args.inputs for record in read_records(path)]
    artifact = aggregate_records(
        records,
        frame,
        bundle,
        attempts=grid,
        expected_run_id=run_id,
        deep=True,
    )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ProtocolError(f"aggregate output exists: {output}; pass --overwrite")
    if output.suffix != ".json":
        raise ProtocolError("aggregate output must be a .json file")
    atomic_write_json(output, artifact)
    print(
        f"wrote {output}: {EXPECTED_GRID_RECORDS} complete records, "
        f"Qmax={list(QMAX_VALUES)}"
    )
    return 0


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "optional 64-character hexadecimal identifier used to keep "
            "independent sweep executions separate"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qmax threshold sensitivity sweep"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one deterministic shard")
    _add_frame_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument(
        "--attempt-timeout-seconds",
        type=int,
        default=ATTEMPT_TIMEOUT_SECONDS,
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(handler=run_protocol)

    validate = subparsers.add_parser(
        "validate", help="deep-validate one full grid or one exact shard"
    )
    _add_frame_arguments(validate)
    validate.add_argument("inputs", nargs="+", type=Path)
    validate.add_argument("--shard-index", type=int, default=0)
    validate.add_argument("--shard-count", type=int, default=1)
    validate.add_argument("--require-success", action="store_true")
    validate.set_defaults(handler=validate_protocol)

    merge = subparsers.add_parser(
        "merge", help="validate and atomically merge the complete 310 x 7 grid"
    )
    _add_frame_arguments(merge)
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(handler=merge_protocol)

    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate only a complete all-success 310 x 7 grid"
    )
    _add_frame_arguments(aggregate)
    aggregate.add_argument("inputs", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.set_defaults(handler=aggregate_protocol)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ProtocolError as error:
        print(f"Qmax sensitivity rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
