#!/usr/bin/env python3
"""Structural Qmax sweep for the retained 144 larger circuits.

The sweep reads precomputed MPS node-selection results. It does not rerun node
selection or execute instrumented circuits. For five records without stored
instrumentation data, it reconstructs the causal-cone group costs with a
forward bitset pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scalability_v2 as scalability
from batch_partition import partition_nodes
from fast_analysis import SKIP_OPS


SCHEMA = "qmon.larger-circuit-structural-sweep/1"
TASK_SCHEMA = "qmon.larger-circuit-structural-sweep-task/1"
SUMMARY_SCHEMA = "qmon.larger-circuit-structural-sweep-summary/1"
# Checksum and record count of the retained 144-circuit input.
SOURCE_ARTIFACT_SHA256 = (
    "11b099dbad0df637fcf7b97c1b7096bc85177ea5fde7f8702cb69e34cf47084f"
)
SOURCE_RECORD_COUNT = 144
QMAX_VALUES = (60, 70, 80, 90, 100)
SELECTOR_COMPLETE_EXTENSION_COUNT = 144
STAGE_TIMEOUT_ATTEMPT_IDS = (
    "extension:multiplier:q40",
    "extension:qwalk:q24",
    "extension:qwalk:q30",
    "extension:qwalk:q40",
    "extension:qwalk:q50",
)
EXPECTED_FAMILY_COUNTS = {
    "ae": 6,
    "bv": 7,
    "cdkm_ripple_carry_adder": 6,
    "dj": 7,
    "draper_qft_adder": 6,
    "full_adder": 6,
    "ghz": 7,
    "graphstate": 4,
    "half_adder": 1,
    "hhl": 7,
    "modular_adder": 6,
    "multiplier": 3,
    "qaoa": 2,
    "qft": 7,
    "qftentangled": 7,
    "qnn": 7,
    "qpeexact": 7,
    "qpeinexact": 7,
    "qwalk": 7,
    "randomcircuit": 1,
    "rg_qft_multiplier": 3,
    "shor": 1,
    "vbe_ripple_carry_adder": 1,
    "vqe_real_amp": 7,
    "vqe_su2": 7,
    "vqe_two_local": 7,
    "wstate": 7,
}
STRUCTURE_FIELDS = (
    "original_qubits",
    "original_unitary_gate_count_m",
    "monitorable_node_count",
    "monitorable_gate_node_count",
    "candidate_monitor_event_count",
    "candidate_gate_node_count",
    "terminally_measured_qubits",
    "free_final_read_event_count",
    "diagnostic_selected_per_event_cone_gate_occurrences",
    "diagnostic_selected_per_event_cone_qubit_occurrences",
    "diagnostic_candidate_per_event_cone_gate_occurrences",
    "diagnostic_candidate_per_event_cone_qubit_occurrences",
    "diagnostic_selected_per_event_extra_cone_qubit_occurrences",
    "gate_node_costs",
)
DYNAMIC_CONTROL_FLOW_OPS = {
    "break_loop",
    "continue_loop",
    "for_loop",
    "if_else",
    "switch_case",
    "while_loop",
}


class SweepError(RuntimeError):
    """Raised when a source, structure, or sweep contract is violated."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _family(record: Mapping[str, Any]) -> str:
    source = record.get("source") or {}
    family = source.get("family")
    if isinstance(family, str) and family:
        return family
    basename = source.get("basename")
    if isinstance(basename, str) and "_indep_qiskit_" in basename:
        return basename.rsplit("_indep_qiskit_", 1)[0]
    raise SweepError(f"record lacks a family: {record.get('attempt_id')!r}")


def _is_selector_complete(record: Mapping[str, Any]) -> bool:
    return scalability.selection_outcome(record) == "selector_complete"


def load_source_records(path: Path) -> list[dict[str, Any]]:
    """Read the retained 144-circuit input and verify its checksum."""
    path = path.expanduser().resolve(strict=True)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != SOURCE_ARTIFACT_SHA256:
        raise SweepError(
            "source file checksum mismatch: "
            f"{actual_sha256} != {SOURCE_ARTIFACT_SHA256}"
        )
    records = scalability.read_jsonl(path)
    if len(records) != SOURCE_RECORD_COUNT:
        raise SweepError(
            "source record count mismatch: "
            f"{len(records)} != {SOURCE_RECORD_COUNT}"
        )
    return records


def selector_complete_extensions(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        dict(record)
        for record in records
        if (record.get("source") or {}).get("source_kind")
        == "mqt_bench_extension"
        and _is_selector_complete(record)
    ]
    selected.sort(key=lambda record: int(record["attempt_index"]))
    if len(selected) != SELECTOR_COMPLETE_EXTENSION_COUNT:
        raise SweepError(
            "selector-complete extension count drift: "
            f"{len(selected)} != {SELECTOR_COMPLETE_EXTENSION_COUNT}"
        )
    family_counts = dict(sorted(Counter(_family(row) for row in selected).items()))
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise SweepError(
            f"selector-complete family inventory drift: {family_counts!r}"
        )
    stage_timeouts = tuple(
        row["attempt_id"]
        for row in selected
        if row.get("status") == "timeout"
        and (row.get("error") or {}).get("phase") == "instrumentation"
    )
    if stage_timeouts != STAGE_TIMEOUT_ATTEMPT_IDS:
        raise SweepError(
            f"instrumentation-timeout inventory drift: {stage_timeouts!r}"
        )
    return selected


def _instruction_rows(qc: Any) -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = []
    for instruction in qc.data:
        rows.append(
            (
                str(instruction.operation.name),
                tuple(qc.find_bit(qubit).index for qubit in instruction.qubits),
            )
        )
    return rows


def _assert_supported_static_circuit(qc: Any) -> None:
    """Reject dynamics outside the fixed structural-cone semantics."""

    measured_qubits: set[int] = set()
    for gate, instruction in enumerate(qc.data):
        operation = instruction.operation
        name = str(operation.name)
        qubits = {
            int(qc.find_bit(qubit).index) for qubit in instruction.qubits
        }
        if name == "reset":
            raise SweepError(f"reset is unsupported at gate {gate}")
        if name in DYNAMIC_CONTROL_FLOW_OPS:
            raise SweepError(f"dynamic control flow is unsupported at gate {gate}")
        if getattr(operation, "condition", None) is not None:
            raise SweepError(f"classically conditioned gate is unsupported at {gate}")
        if instruction.clbits and name != "measure":
            raise SweepError(f"classical operands are unsupported at gate {gate}")
        if name == "measure":
            measured_qubits.update(qubits)
        elif name not in SKIP_OPS and measured_qubits.intersection(qubits):
            raise SweepError(
                f"mid-circuit measurement semantics are unsupported at gate {gate}"
            )


def _validate_nodes(
    rows: Sequence[tuple[str, tuple[int, ...]]],
    num_qubits: int,
    nodes: Mapping[int, Sequence[int]],
) -> dict[int, list[int]]:
    normalized: dict[int, list[int]] = {}
    for raw_gate, raw_qubits in sorted(nodes.items()):
        gate = int(raw_gate)
        if gate < 0 or gate >= len(rows):
            raise SweepError(f"monitor gate index is outside the circuit: {gate}")
        qubits = sorted({int(qubit) for qubit in raw_qubits})
        if any(qubit < 0 or qubit >= num_qubits for qubit in qubits):
            raise SweepError(f"monitor qubit is outside the circuit at gate {gate}")
        if rows[gate][0] in SKIP_OPS:
            raise SweepError(f"monitor points at a skipped operation: gate {gate}")
        acted = set(rows[gate][1])
        if not set(qubits) <= acted:
            raise SweepError(f"monitor is not on an acted qubit at gate {gate}")
        normalized[gate] = qubits
    return normalized


def build_structure_bitset(
    qc: Any, nodes: Mapping[int, Sequence[int]]
) -> dict[str, Any]:
    """Compute the exact cone accounting in one forward pass.

    Each qubit carries a Python-integer bitset of all gate indices in its
    current backward cone and a second bitset of all qubits in that cone.
    Updating these bitsets after a gate is the forward dynamic-programming
    form of the original backward reachability calculation.
    """

    instruction_rows = _instruction_rows(qc)
    normalized_nodes = _validate_nodes(
        instruction_rows, int(qc.num_qubits), nodes
    )
    last_by_qubit: list[int | None] = [None] * int(qc.num_qubits)
    terminally_measured_qubits: set[int] = set()
    for gate, (name, qubits) in enumerate(instruction_rows):
        if name == "measure" and qubits:
            terminally_measured_qubits.update(qubits)
        if name not in SKIP_OPS:
            for qubit in qubits:
                last_by_qubit[qubit] = gate

    gate_provenance = [0] * int(qc.num_qubits)
    qubit_provenance = [1 << qubit for qubit in range(int(qc.num_qubits))]
    selected_event_count = 0
    candidate_event_count = 0
    selected_cone_gates = 0
    selected_cone_qubits = 0
    candidate_cone_gates = 0
    candidate_cone_qubits = 0
    selected_extra_cone_qubits = 0
    gate_node_costs: list[dict[str, Any]] = []

    for gate, (name, acted_qubits) in enumerate(instruction_rows):
        if name in SKIP_OPS:
            continue
        combined_gate_bits = 1 << gate
        combined_qubit_bits = 0
        for qubit in acted_qubits:
            combined_gate_bits |= gate_provenance[qubit]
            combined_qubit_bits |= qubit_provenance[qubit]
        for qubit in acted_qubits:
            gate_provenance[qubit] = combined_gate_bits
            qubit_provenance[qubit] = combined_qubit_bits

        monitored = normalized_nodes.get(gate)
        if not monitored:
            continue
        selected_event_count += len(monitored)
        candidates: list[int] = []
        group_gate_bits = 0
        group_qubit_bits = 0
        for qubit in monitored:
            event_gate_bits = gate_provenance[qubit]
            event_qubit_bits = qubit_provenance[qubit]
            event_gate_count = event_gate_bits.bit_count()
            event_qubit_count = event_qubit_bits.bit_count()
            selected_cone_gates += event_gate_count
            selected_cone_qubits += event_qubit_count
            selected_extra_cone_qubits += event_qubit_count - 1
            free_final_read = (
                qubit in terminally_measured_qubits
                and gate == last_by_qubit[qubit]
            )
            if free_final_read:
                continue
            candidates.append(qubit)
            candidate_event_count += 1
            candidate_cone_gates += event_gate_count
            candidate_cone_qubits += event_qubit_count
            group_gate_bits |= event_gate_bits
            group_qubit_bits |= event_qubit_bits
        if candidates:
            monitor_mask = sum(1 << qubit for qubit in candidates)
            gate_node_costs.append(
                {
                    "gate": gate,
                    "monitor_qubits": candidates,
                    "monitor_event_count": len(candidates),
                    "extra_qubit_demand": (
                        group_qubit_bits & ~monitor_mask
                    ).bit_count(),
                    "cone_qubit_union_count": group_qubit_bits.bit_count(),
                    "cone_gate_union_count": group_gate_bits.bit_count(),
                }
            )

    original_unitary_gate_count = sum(
        1
        for name, _ in instruction_rows
        if name not in {"measure", "barrier"}
    )
    structure = {
        "original_qubits": int(qc.num_qubits),
        "original_unitary_gate_count_m": original_unitary_gate_count,
        "monitorable_node_count": selected_event_count,
        "monitorable_gate_node_count": len(normalized_nodes),
        "candidate_monitor_event_count": candidate_event_count,
        "candidate_gate_node_count": len(gate_node_costs),
        "terminally_measured_qubits": sorted(terminally_measured_qubits),
        "free_final_read_event_count": (
            selected_event_count - candidate_event_count
        ),
        "diagnostic_selected_per_event_cone_gate_occurrences": (
            selected_cone_gates
        ),
        "diagnostic_selected_per_event_cone_qubit_occurrences": (
            selected_cone_qubits
        ),
        "diagnostic_candidate_per_event_cone_gate_occurrences": (
            candidate_cone_gates
        ),
        "diagnostic_candidate_per_event_cone_qubit_occurrences": (
            candidate_cone_qubits
        ),
        "diagnostic_selected_per_event_extra_cone_qubit_occurrences": (
            selected_extra_cone_qubits
        ),
        "gate_node_costs": gate_node_costs,
    }
    validate_structure(structure)
    return structure


def structure_from_instrumentation(
    instrumentation: Mapping[str, Any],
) -> dict[str, Any]:
    structure = {
        field: json.loads(json.dumps(instrumentation.get(field)))
        for field in STRUCTURE_FIELDS
    }
    validate_structure(structure)
    return structure


def validate_structure(structure: Mapping[str, Any]) -> None:
    for field in STRUCTURE_FIELDS[:-1]:
        if field == "terminally_measured_qubits":
            continue
        value = structure.get(field)
        if type(value) is not int or value < 0:
            raise SweepError(f"structure field {field} is invalid: {value!r}")
    terminal = structure.get("terminally_measured_qubits")
    if (
        not isinstance(terminal, list)
        or any(type(value) is not int or value < 0 for value in terminal)
        or terminal != sorted(set(terminal))
    ):
        raise SweepError("terminally measured qubit inventory is invalid")
    costs = structure.get("gate_node_costs")
    if not isinstance(costs, list):
        raise SweepError("gate-node costs are not a list")
    gates: list[int] = []
    monitor_events = 0
    for row in costs:
        if not isinstance(row, Mapping):
            raise SweepError("gate-node cost row is not an object")
        gate = row.get("gate")
        qubits = row.get("monitor_qubits")
        if type(gate) is not int or gate < 0:
            raise SweepError("gate-node cost has an invalid gate")
        if (
            not isinstance(qubits, list)
            or any(type(value) is not int or value < 0 for value in qubits)
            or qubits != sorted(set(qubits))
            or not qubits
        ):
            raise SweepError("gate-node cost has invalid monitor qubits")
        for field in (
            "monitor_event_count",
            "extra_qubit_demand",
            "cone_qubit_union_count",
            "cone_gate_union_count",
        ):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise SweepError(f"gate-node cost field {field} is invalid")
        if row["monitor_event_count"] != len(qubits):
            raise SweepError("gate-node monitor-event count drifted")
        gates.append(gate)
        monitor_events += len(qubits)
    if gates != sorted(set(gates)):
        raise SweepError("gate-node costs are not sorted and unique")
    if len(costs) != structure["candidate_gate_node_count"]:
        raise SweepError("candidate gate-node count drifted")
    if monitor_events != structure["candidate_monitor_event_count"]:
        raise SweepError("candidate monitor-event count drifted")


def structure_sha256(structure: Mapping[str, Any]) -> str:
    validate_structure(structure)
    return scalability.stable_hash(
        {field: structure[field] for field in STRUCTURE_FIELDS}
    )


def _constraint_from_structure(
    structure: Mapping[str, Any],
) -> dict[Any, Any]:
    constraint: dict[Any, Any] = {
        "num_oq": int(structure["original_qubits"])
    }
    for row in structure["gate_node_costs"]:
        constraint[int(row["gate"])] = {
            "num_eq": int(row["extra_qubit_demand"]),
            "qubits": list(row["monitor_qubits"]),
        }
    return constraint


def partition_structure(
    structure: Mapping[str, Any], qmax: int
) -> dict[str, Any]:
    validate_structure(structure)
    if qmax not in QMAX_VALUES:
        raise SweepError(f"Qmax is outside the fixed sweep: {qmax}")
    original_qubits = int(structure["original_qubits"])
    if original_qubits > qmax:
        raise SweepError(
            f"sweep budget {qmax} is below source width {original_qubits}"
        )
    constraint = _constraint_from_structure(structure)
    batches, infeasible, metadata = partition_nodes(
        constraint,
        qmax,
        ilp_node_limit=scalability.REFINEMENT_NODE_LIMIT,
        return_metadata=True,
        refinement_time_limit_seconds=(
            scalability.MAX_REFINEMENT_TIME_LIMIT_SECONDS
        ),
        gurobi_license_retry_delays_seconds=(
            scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS
        ),
    )
    serialized_batches = [
        {
            "nodes": sorted(int(node) for node in batch["nodes"]),
            "cost": int(batch["cost"]),
        }
        for batch in batches
    ]
    infeasible_nodes = sorted(int(node) for node in infeasible)
    deployed_sequence = [
        node for batch in serialized_batches for node in batch["nodes"]
    ]
    if len(deployed_sequence) != len(set(deployed_sequence)):
        raise SweepError("partition deploys a gate node more than once")
    deployed_nodes = set(deployed_sequence)
    cost_by_gate = {
        int(row["gate"]): row for row in structure["gate_node_costs"]
    }
    if set(infeasible_nodes).intersection(deployed_nodes):
        raise SweepError("partition deploys an individually infeasible gate node")
    expected_deployed = set(cost_by_gate) - set(infeasible_nodes)
    if deployed_nodes != expected_deployed:
        raise SweepError(
            "partition does not cover every feasible gate node exactly once"
        )
    capacity = qmax - original_qubits
    for batch in serialized_batches:
        expected_cost = sum(
            int(cost_by_gate[node]["extra_qubit_demand"])
            for node in batch["nodes"]
        )
        if batch["cost"] != expected_cost or batch["cost"] > capacity:
            raise SweepError("partition batch cost violates the fixed capacity")
    candidate_gate_nodes = int(structure["candidate_gate_node_count"])
    candidate_monitor_events = int(structure["candidate_monitor_event_count"])
    deployed_monitor_events = sum(
        int(cost_by_gate[node]["monitor_event_count"])
        for node in deployed_nodes
    )
    replay_gate_executions = sum(
        int(cost_by_gate[node]["cone_gate_union_count"])
        for node in deployed_nodes
    )
    batch_costs = [batch["cost"] for batch in serialized_batches]
    original_gate_executions = (
        len(serialized_batches)
        * int(structure["original_unitary_gate_count_m"])
    )
    total_counted_operations = (
        original_gate_executions
        + replay_gate_executions
        + 2 * deployed_monitor_events
    )
    original_unitary_gate_count = int(
        structure["original_unitary_gate_count_m"]
    )
    counted_operation_multiplier = (
        total_counted_operations / original_unitary_gate_count
        if original_unitary_gate_count
        else None
    )
    result = {
        "qmax": qmax,
        "capacity_extra_qubits": qmax - original_qubits,
        "candidate_gate_node_count": candidate_gate_nodes,
        "candidate_monitor_event_count": candidate_monitor_events,
        "individually_infeasible_gate_node_count": len(infeasible_nodes),
        "individually_infeasible_gate_nodes_sha256": scalability.stable_hash(
            infeasible_nodes
        ),
        "deployed_gate_node_count": len(deployed_nodes),
        "deployed_monitor_event_count": deployed_monitor_events,
        "deployed_gate_node_coverage": (
            len(deployed_nodes) / candidate_gate_nodes
            if candidate_gate_nodes
            else 1.0
        ),
        "deployed_monitor_event_coverage": (
            deployed_monitor_events / candidate_monitor_events
            if candidate_monitor_events
            else 1.0
        ),
        "fully_deployable": len(deployed_nodes) == candidate_gate_nodes,
        "batch_count": len(serialized_batches),
        "batch_count_semantics": (
            "deterministic-ffd-followed-by-one-step-ilp-refinement;"
            "not-a-global-optimum-claim"
        ),
        "planned_peak_extra_ancillas": max(batch_costs, default=0),
        "planned_cumulative_extra_ancilla_allocations": sum(batch_costs),
        "maximum_run_qubits": (
            original_qubits + max(batch_costs)
            if batch_costs
            else None
        ),
        "planned_original_gate_executions": original_gate_executions,
        "planned_replay_gate_executions": replay_gate_executions,
        "planned_monitor_measurement_operations": deployed_monitor_events,
        "planned_monitor_reset_operations": deployed_monitor_events,
        "planned_total_counted_operations": total_counted_operations,
        "planned_counted_operation_multiplier": counted_operation_multiplier,
        "partition_batches_sha256": scalability.stable_hash(
            serialized_batches
        ),
        "partition_metadata": metadata,
    }
    for field in (
        "deployed_gate_node_coverage",
        "deployed_monitor_event_coverage",
    ):
        value = result[field]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SweepError(f"invalid sweep coverage {field}={value!r}")
    if (
        result["maximum_run_qubits"] is not None
        and result["maximum_run_qubits"] > qmax
    ):
        raise SweepError("partition exceeds Qmax")
    return result


def _rebuild_record_circuit(record: Mapping[str, Any]) -> Any:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise SweepError("record lacks its fixed attempt source")
    spec = scalability.AttemptSpec(**dict(source))
    qc = scalability._build_circuit(spec)
    circuit = record.get("circuit") or {}
    if int(qc.num_qubits) != circuit.get("original_qubits"):
        raise SweepError("rebuilt timeout circuit width drifted")
    actual_hash = scalability._circuit_canonical_sha256(qc)
    if actual_hash != circuit.get("canonical_circuit_sha256"):
        raise SweepError("rebuilt circuit checksum drifted")
    _assert_supported_static_circuit(qc)
    return qc


def _selection_nodes(record: Mapping[str, Any]) -> dict[int, list[int]]:
    selection = record.get("selection") or {}
    nodes, errors = scalability._deserialize_monitorable_nodes(
        selection.get("monitorable_nodes")
    )
    if errors:
        raise SweepError(
            "timeout selection certificate is invalid: " + "; ".join(errors)
        )
    if scalability.stable_hash(
        scalability._serialize_monitorable_nodes(nodes)
    ) != selection.get("monitorable_nodes_sha256"):
        raise SweepError("selection node checksum mismatch")
    return nodes


def evaluate_selected_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    qc = _rebuild_record_circuit(record)
    nodes = _selection_nodes(record)
    replayed_structure = build_structure_bitset(qc, nodes)
    instrumentation = record.get("instrumentation")
    if isinstance(instrumentation, Mapping):
        structure = structure_from_instrumentation(instrumentation)
        if replayed_structure != structure:
            raise SweepError(
                "forward-bitset structure replay differs from the accepted "
                f"instrumentation certificate for {record.get('attempt_id')}"
            )
        structure_origin = (
            "accepted-instrumentation-certificate-with-forward-bitset-replay"
        )
        recovered_structure = None
    else:
        if record.get("attempt_id") not in STAGE_TIMEOUT_ATTEMPT_IDS:
            raise SweepError(
                "selector-complete record lacks instrumentation outside the "
                "fixed timeout inventory"
            )
        structure = replayed_structure
        structure_origin = "exact-forward-bitset-recovery"
        recovered_structure = structure
    sweeps = [partition_structure(structure, qmax) for qmax in QMAX_VALUES]
    payload: dict[str, Any] = {
        "schema": TASK_SCHEMA,
        "source_schema": SCHEMA,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_record_sha256": record.get("record_sha256"),
        "attempt_id": record["attempt_id"],
        "attempt_index": int(record["attempt_index"]),
        "family": _family(record),
        "requested_qubits": (record.get("source") or {}).get(
            "requested_qubits"
        ),
        "actual_qubits": (record.get("circuit") or {}).get(
            "original_qubits"
        ),
        "selection_sha256": scalability.stable_hash(record.get("selection")),
        "structure_origin": structure_origin,
        "structure_sha256": structure_sha256(structure),
        "recovered_structure": recovered_structure,
        "sweeps": sweeps,
    }
    payload["task_sha256"] = scalability.stable_hash(payload)
    return payload


def verify_task(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    deep: bool,
) -> dict[str, Any]:
    stored = dict(payload)
    observed_checksum = stored.pop("task_sha256", None)
    if observed_checksum != scalability.stable_hash(stored):
        raise SweepError("task checksum is invalid")
    if payload.get("attempt_id") != record.get("attempt_id"):
        raise SweepError("task attempt identity drifted")
    if payload.get("source_record_sha256") != record.get("record_sha256"):
        raise SweepError("task source-record checksum mismatch")
    if deep:
        expected = evaluate_selected_record(record)
        if payload != expected:
            raise SweepError(
                f"deep task replay drifted for {record.get('attempt_id')}"
            )
    return {
        "attempt_id": payload["attempt_id"],
        "task_sha256": payload["task_sha256"],
        "deep": deep,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return descriptive statistics using nearest-rank quartiles."""
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "quartile_1": None,
            "median": None,
            "quartile_3": None,
            "mean": None,
            "maximum": None,
        }
    numeric = sorted(float(value) for value in values)

    def nearest_rank(probability: float) -> float:
        return numeric[math.ceil(probability * len(numeric)) - 1]

    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "quartile_1": nearest_rank(0.25),
        "median": statistics.median(numeric),
        "quartile_3": nearest_rank(0.75),
        "mean": statistics.mean(numeric),
        "maximum": max(numeric),
    }


def aggregate_tasks(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = selector_complete_extensions(records)
    by_id = {str(task.get("attempt_id")): task for task in tasks}
    expected_ids = {str(record["attempt_id"]) for record in selected}
    if set(by_id) != expected_ids or len(tasks) != len(expected_ids):
        raise SweepError("task inventory does not exactly cover the 144 cases")
    for record in selected:
        verify_task(by_id[str(record["attempt_id"])], record, deep=False)
    budget_rows = []
    for qmax in QMAX_VALUES:
        sweep_rows = []
        for task in tasks:
            matches = [
                row for row in task["sweeps"] if int(row["qmax"]) == qmax
            ]
            if len(matches) != 1:
                raise SweepError(
                    f"task {task.get('attempt_id')} lacks Qmax={qmax}"
                )
            sweep_rows.append(matches[0])
        candidate_gate_nodes = sum(
            int(row["candidate_gate_node_count"]) for row in sweep_rows
        )
        deployed_gate_nodes = sum(
            int(row["deployed_gate_node_count"]) for row in sweep_rows
        )
        candidate_monitor_events = sum(
            int(row["candidate_monitor_event_count"]) for row in sweep_rows
        )
        deployed_monitor_events = sum(
            int(row["deployed_monitor_event_count"]) for row in sweep_rows
        )
        budget_rows.append(
            {
                "qmax": qmax,
                "circuits": len(sweep_rows),
                "fully_deployable_circuits": sum(
                    bool(row["fully_deployable"]) for row in sweep_rows
                ),
                "fully_deployable_circuit_rate": (
                    sum(bool(row["fully_deployable"]) for row in sweep_rows)
                    / len(sweep_rows)
                ),
                "mean_deployed_gate_node_coverage": statistics.mean(
                    float(row["deployed_gate_node_coverage"])
                    for row in sweep_rows
                ),
                "mean_deployed_monitor_event_coverage": statistics.mean(
                    float(row["deployed_monitor_event_coverage"])
                    for row in sweep_rows
                ),
                "candidate_gate_node_count": candidate_gate_nodes,
                "deployed_gate_node_count": deployed_gate_nodes,
                "pooled_deployed_gate_node_coverage": (
                    deployed_gate_nodes / candidate_gate_nodes
                    if candidate_gate_nodes
                    else 1.0
                ),
                "candidate_monitor_event_count": candidate_monitor_events,
                "deployed_monitor_event_count": deployed_monitor_events,
                "pooled_deployed_monitor_event_coverage": (
                    deployed_monitor_events / candidate_monitor_events
                    if candidate_monitor_events
                    else 1.0
                ),
                "batch_count": _distribution(
                    [float(row["batch_count"]) for row in sweep_rows]
                ),
                "planned_peak_extra_ancillas": _distribution(
                    [
                        float(row["planned_peak_extra_ancillas"])
                        for row in sweep_rows
                    ]
                ),
                "planned_counted_operation_multiplier": _distribution(
                    [
                        float(row["planned_counted_operation_multiplier"])
                        for row in sweep_rows
                        if row["planned_counted_operation_multiplier"] is not None
                    ]
                ),
                "maximum_run_qubits": _distribution(
                    [
                        float(row["maximum_run_qubits"])
                        for row in sweep_rows
                        if row["maximum_run_qubits"] is not None
                    ]
                ),
            }
        )
    task_checksums = [
        {
            "attempt_id": task["attempt_id"],
            "task_sha256": task["task_sha256"],
        }
        for task in sorted(tasks, key=lambda item: int(item["attempt_index"]))
    ]
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "source_schema": SCHEMA,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "selector_complete_circuits": len(selected),
        "family_counts": EXPECTED_FAMILY_COUNTS,
        "recovered_instrumentation_timeout_attempts": list(
            STAGE_TIMEOUT_ATTEMPT_IDS
        ),
        "qmax_values": list(QMAX_VALUES),
        "analysis_scope": (
            "structural-instrumentation-capacity-only;"
            "instrumented-circuits-not-emitted-or-executed"
        ),
        "batch_count_semantics": (
            "deterministic-ffd-followed-by-one-step-ilp-refinement;"
            "not-a-global-optimum-claim"
        ),
        "budget_rows": budget_rows,
        "task_checksums": task_checksums,
        "task_checksums_sha256": scalability.stable_hash(task_checksums),
    }
    summary["summary_sha256"] = scalability.stable_hash(summary)
    return summary


def write_summary_csv(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "qmax",
        "circuits",
        "fully_deployable_circuits",
        "fully_deployable_circuit_rate",
        "mean_deployed_gate_node_coverage",
        "mean_deployed_monitor_event_coverage",
        "pooled_deployed_gate_node_coverage",
        "pooled_deployed_monitor_event_coverage",
        "batch_count_minimum",
        "batch_count_quartile_1",
        "batch_count_median",
        "batch_count_quartile_3",
        "batch_count_mean",
        "batch_count_maximum",
        "peak_extra_ancillas_median",
        "peak_extra_ancillas_quartile_1",
        "peak_extra_ancillas_quartile_3",
        "peak_extra_ancillas_maximum",
        "counted_operation_multiplier_median",
        "counted_operation_multiplier_quartile_1",
        "counted_operation_multiplier_quartile_3",
        "counted_operation_multiplier_maximum",
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        newline="",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["budget_rows"]:
            batch = row["batch_count"]
            peak = row["planned_peak_extra_ancillas"]
            multiplier = row["planned_counted_operation_multiplier"]
            writer.writerow(
                {
                    "qmax": row["qmax"],
                    "circuits": row["circuits"],
                    "fully_deployable_circuits": row[
                        "fully_deployable_circuits"
                    ],
                    "fully_deployable_circuit_rate": row[
                        "fully_deployable_circuit_rate"
                    ],
                    "mean_deployed_gate_node_coverage": row[
                        "mean_deployed_gate_node_coverage"
                    ],
                    "mean_deployed_monitor_event_coverage": row[
                        "mean_deployed_monitor_event_coverage"
                    ],
                    "pooled_deployed_gate_node_coverage": row[
                        "pooled_deployed_gate_node_coverage"
                    ],
                    "pooled_deployed_monitor_event_coverage": row[
                        "pooled_deployed_monitor_event_coverage"
                    ],
                    "batch_count_minimum": batch["minimum"],
                    "batch_count_quartile_1": batch["quartile_1"],
                    "batch_count_median": batch["median"],
                    "batch_count_quartile_3": batch["quartile_3"],
                    "batch_count_mean": batch["mean"],
                    "batch_count_maximum": batch["maximum"],
                    "peak_extra_ancillas_median": peak["median"],
                    "peak_extra_ancillas_quartile_1": peak["quartile_1"],
                    "peak_extra_ancillas_quartile_3": peak["quartile_3"],
                    "peak_extra_ancillas_maximum": peak["maximum"],
                    "counted_operation_multiplier_median": multiplier["median"],
                    "counted_operation_multiplier_quartile_1": multiplier[
                        "quartile_1"
                    ],
                    "counted_operation_multiplier_quartile_3": multiplier[
                        "quartile_3"
                    ],
                    "counted_operation_multiplier_maximum": multiplier[
                        "maximum"
                    ],
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def command_inventory(args: argparse.Namespace) -> int:
    records = load_source_records(args.source_artifact)
    selected = selector_complete_extensions(records)
    payload = {
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "selector_complete_circuits": len(selected),
        "family_counts": EXPECTED_FAMILY_COUNTS,
        "instrumentation_timeout_attempts": list(STAGE_TIMEOUT_ATTEMPT_IDS),
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


def command_task(args: argparse.Namespace) -> int:
    os.environ["QMON_MILP_SOLVER"] = "gurobi"
    records = load_source_records(args.source_artifact)
    selected = selector_complete_extensions(records)
    if not 0 <= args.task_index < len(selected):
        raise SweepError("task index is outside the 144-case frame")
    payload = evaluate_selected_record(selected[args.task_index])
    atomic_json(args.output, payload)
    verify_task(payload, selected[args.task_index], deep=False)
    print(
        f"wrote task={args.task_index} "
        f"attempt={payload['attempt_id']} output={args.output}"
    )
    return 0


def command_verify_task(args: argparse.Namespace) -> int:
    os.environ["QMON_MILP_SOLVER"] = "gurobi"
    records = load_source_records(args.source_artifact)
    selected = selector_complete_extensions(records)
    payload = json.loads(args.input.read_text(encoding="ascii"))
    by_id = {str(record["attempt_id"]): record for record in selected}
    attempt_id = str(payload.get("attempt_id"))
    if attempt_id not in by_id:
        raise SweepError("task is outside the 144-case frame")
    result = verify_task(payload, by_id[attempt_id], deep=args.deep)
    print(json.dumps(result, sort_keys=True))
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    records = load_source_records(args.source_artifact)
    tasks = [
        json.loads(path.read_text(encoding="ascii"))
        for path in sorted(args.inputs)
    ]
    summary = aggregate_tasks(records, tasks)
    atomic_json(args.output, summary)
    write_summary_csv(summary, args.csv_output)
    print(
        f"wrote summary={args.output} csv={args.csv_output} "
        f"sha256={summary['summary_sha256']}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--source-artifact", type=Path, required=True)
    inventory.set_defaults(handler=command_inventory)
    task = commands.add_parser("task")
    task.add_argument("--source-artifact", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--output", type=Path, required=True)
    task.set_defaults(handler=command_task)
    verify = commands.add_parser("verify-task")
    verify.add_argument("--source-artifact", type=Path, required=True)
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--deep", action="store_true")
    verify.set_defaults(handler=command_verify_task)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--source-artifact", type=Path, required=True)
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--csv-output", type=Path, required=True)
    aggregate.set_defaults(handler=command_aggregate)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError, SweepError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
