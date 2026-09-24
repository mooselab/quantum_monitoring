"""Retry only explicitly recorded resource-limited or timed-out RQ1 locations."""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import rq1_full_tightness as rq1


PLAN_SCHEMA = "qmon-rq1-retry-plan-v1"
RESULT_SCHEMA = "qmon-rq1-retry-result-v1"
RETRY_STATUSES = {"resource_limit", "timeout"}
ENTRY_FIELDS = ("attempt_id", "attempt_index", "circuit", "source_qasm_sha256")
POINT_FIELDS = ("point_ordinal", "gate", "qubit", "gate_name")


def build_plan(source, source_sha256, frame, points_per_task=8):
    """Check historical point records against current QASM before selecting them.

    A historical export may be partial. It supplies retry targets only and is
    never treated as a complete replacement for the accepted RQ1 result.
    """
    if type(points_per_task) is not int or points_per_task < 1:
        raise ValueError("points_per_task must be a positive integer")
    partial_schema = "qmon-rq1-full-tightness-partial-v1"
    if source.get("schema") not in {rq1.SCHEMA, partial_schema}:
        raise ValueError("unexpected historical RQ1 schema")
    if source["schema"] == partial_schema and source.get("complete") is not False:
        raise ValueError("partial historical export must not claim completeness")
    by_name = {entry["circuit"]: entry for entry in frame["circuits"]}
    seen_circuits, seen_points, tasks = set(), set(), []
    counts = collections.Counter()
    for circuit in source["circuit_results"]:
        if circuit["circuit"] in seen_circuits:
            raise ValueError("duplicate historical circuit")
        seen_circuits.add(circuit["circuit"])
        if not any(p["status"] in RETRY_STATUSES for p in circuit["points"]):
            continue
        rq1._check_checksum(circuit, "record_sha256", "historical circuit")
        entry = {key: circuit[key] for key in ENTRY_FIELDS}
        canonical = by_name.get(entry["circuit"])
        if canonical is None or canonical["source_qasm_sha256"] != entry["source_qasm_sha256"]:
            raise ValueError("historical circuit is absent or QASM has changed")
        path = Path(frame["qasm_dir"]) / entry["circuit"]
        if rq1.file_sha256(path) != entry["source_qasm_sha256"]:
            raise ValueError("source QASM checksum mismatch")
        qc = QuantumCircuit.from_qasm_file(str(path))
        expected = rq1.enumerate_acted_points(qc)
        if len(expected) != len(circuit["points"]):
            raise ValueError("historical point frame mismatch")
        selected = []
        for point, previous in zip(expected, circuit["points"]):
            rq1._validate_point_structure(previous, entry, point)
            if previous["point_id"] in seen_points:
                raise ValueError("duplicate historical point")
            seen_points.add(previous["point_id"])
            if previous["status"] not in RETRY_STATUSES:
                continue
            selected.append({
                **{key: previous[key] for key in POINT_FIELDS},
                "point_id": previous["point_id"],
                "previous_status": previous["status"],
                "previous_record_sha256": previous["record_sha256"],
                "previous_instrumented_qubits": previous["instrumented_qubits"],
            })
            counts[previous["status"]] += 1
        for offset in range(0, len(selected), points_per_task):
            tasks.append({
                "task_index": len(tasks), "entry": entry,
                "canonical_attempt_index": canonical["attempt_index"],
                "points": selected[offset:offset + points_per_task],
            })
    value = {
        "schema": PLAN_SCHEMA,
        "source_file_sha256": source_sha256,
        "source_export_schema": source["schema"],
        "source_export_complete": source.get("complete") is True,
        "source_export_circuits": len(source["circuit_results"]),
        "scope": "unfinished locations only; not a complete historical result",
        "corpus_sha256": frame["corpus_sha256"],
        "manifest_sha256": frame["manifest_sha256"],
        "points_per_task": points_per_task,
        "previous_status_counts": dict(sorted(counts.items())),
        "point_count": sum(counts.values()),
        "circuit_count": len({t["entry"]["circuit"] for t in tasks}),
        "tasks": tasks,
    }
    return rq1._add_checksum(value, "plan_sha256")


def validate_plan(plan, frame):
    rq1._check_checksum(plan, "plan_sha256", "retry plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("unexpected retry plan schema")
    for field in ("corpus_sha256", "manifest_sha256"):
        if plan[field] != frame[field]:
            raise ValueError(f"retry {field} mismatch")
    by_name = {entry["circuit"]: entry for entry in frame["circuits"]}
    seen, counts, circuits = set(), collections.Counter(), {}
    for index, task in enumerate(plan["tasks"]):
        if task["task_index"] != index or not task["points"]:
            raise ValueError("invalid retry task index or empty task")
        entry = task["entry"]
        current = by_name.get(entry["circuit"])
        if current is None or current["source_qasm_sha256"] != entry["source_qasm_sha256"]:
            raise ValueError("retry source mismatch")
        if task["canonical_attempt_index"] != current["attempt_index"]:
            raise ValueError("retry canonical index mismatch")
        if entry["circuit"] not in circuits:
            path = Path(frame["qasm_dir"]) / entry["circuit"]
            if rq1.file_sha256(path) != entry["source_qasm_sha256"]:
                raise ValueError("retry QASM checksum mismatch")
            circuits[entry["circuit"]] = rq1.enumerate_acted_points(
                QuantumCircuit.from_qasm_file(str(path))
            )
        for point in task["points"]:
            if point["point_id"] in seen:
                raise ValueError("duplicate retry point")
            seen.add(point["point_id"])
            if point["previous_status"] not in RETRY_STATUSES:
                raise ValueError("cannot retry a completed location")
            expected = circuits[entry["circuit"]][point["point_ordinal"]]
            if any(point[k] != expected[k] for k in POINT_FIELDS):
                raise ValueError("retry point identity mismatch")
            if point["point_id"] != rq1._point_id(entry, expected):
                raise ValueError("retry point ID mismatch")
            if not rq1._is_sha256(point["previous_record_sha256"]):
                raise ValueError("retry previous record checksum missing")
            counts[point["previous_status"]] += 1
    if dict(counts) != plan["previous_status_counts"] or sum(counts.values()) != plan["point_count"]:
        raise ValueError("retry count mismatch")
    if len(circuits) != plan["circuit_count"]:
        raise ValueError("retry circuit count mismatch")


def _worker(task, qasm_dir, maximum, connection):
    entry = task["entry"]
    try:
        path = Path(qasm_dir) / entry["circuit"]
        if rq1.file_sha256(path) != entry["source_qasm_sha256"]:
            raise ValueError("QASM changed before retry execution")
        qc = QuantumCircuit.from_qasm_file(str(path))
        error = rq1._unitary_assumption_error(qc)
        if error:
            raise ValueError(error)
        if qc.num_qubits > maximum:
            raise ValueError("original circuit exceeds retry statevector cap")
        local_values = rq1._local_quantities(qc)
        reference_circuit = qc.copy()
        reference_circuit.remove_final_measurements()
        reference = Statevector.from_instruction(reference_circuit).data
        for point in task["points"]:
            local = local_values[(point["gate"], point["qubit"])]
            monitored = rq1.force_monitor(qc, entry["circuit"], [
                point["gate"], point["gate_name"], point["qubit"],
                local["p0"], local["p1"],
            ])
            if monitored.num_qubits > maximum:
                row = rq1._terminal_point(
                    entry, point, "resource_limit",
                    category="instrumented_statevector_qubits_exceed_cap",
                    message=f"instrumented width {monitored.num_qubits}; retry cap {maximum}",
                )
                row.update(local, instrumented_qubits=monitored.num_qubits)
                rq1._add_checksum(row, "record_sha256")
            else:
                fidelity = rq1.exact_state_fidelity_to(monitored, qc.num_qubits, reference)
                row = rq1._evaluated_point(entry, point, local, monitored.num_qubits, fidelity)
            connection.send({"record": row})
        connection.send({"done": True})
    except Exception as exc:
        connection.send({"error": {"message": f"{type(exc).__name__}: {exc}"}})
    finally:
        connection.close()


def run_task(plan, frame, output, *, task_index, maximum=27, timeout=21600.0,
             point_id=None):
    validate_plan(plan, frame)
    if type(task_index) is not int or not 0 <= task_index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    if type(maximum) is not int or maximum < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid retry resource limits")
    task = dict(plan["tasks"][task_index])
    if point_id is not None:
        task["points"] = [p for p in task["points"] if p["point_id"] == point_id]
        if len(task["points"]) != 1:
            raise ValueError("requested point is not in task")
    result = {
        "schema": RESULT_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "source_file_sha256": plan["source_file_sha256"],
        "task_index": task_index, "entry": task["entry"],
        "requested_point_ids": [p["point_id"] for p in task["points"]],
        "previous_record_sha256": {p["point_id"]: p["previous_record_sha256"] for p in task["points"]},
        "maximum_statevector_qubits": maximum, "timeout_seconds": timeout,
        "fingerprints": rq1.build_fingerprint_bundle(frame, rq1.protocol_config(maximum, timeout)),
        "retry_implementation_sha256": rq1.file_sha256(__file__),
        "complete": False, "points": [],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the destination before execution; previous attempts are never reused.
    with output.open("x") as handle:
        json.dump(rq1._add_checksum(result, "result_sha256"), handle)
        handle.write("\n")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(task, frame["qasm_dir"], maximum, child))
    start = time.monotonic()
    done, fatal = False, None
    process.start()
    child.close()
    try:
        while time.monotonic() - start < timeout:
            if parent.poll(0.2):
                try:
                    message = parent.recv()
                except EOFError:
                    break
                if "record" in message:
                    row = message["record"]
                    if row["point_id"] not in result["requested_point_ids"]:
                        raise ValueError("worker returned an unrequested point")
                    result["points"].append(row)
                    rq1.atomic_write_json(output, rq1._add_checksum(result, "result_sha256"))
                    print(json.dumps({"point_id": row["point_id"], "status": row["status"]}), flush=True)
                elif "error" in message:
                    fatal = message["error"]
                    break
                elif message.get("done"):
                    done = True
                    break
            elif not process.is_alive():
                break
    finally:
        timed_out = not done and time.monotonic() - start >= timeout
        if done:
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
        process.join()
        parent.close()
    seen = {p["point_id"] for p in result["points"]}
    if len(seen) != len(result["points"]):
        raise ValueError("duplicate worker result")
    for point in task["points"]:
        if point["point_id"] not in seen:
            status, category, message = rq1._unfinished_worker_outcome(
                timed_out=timed_out, exitcode=process.exitcode,
                timeout_seconds=timeout, fatal_error=fatal,
            )
            result["points"].append(rq1._terminal_point(
                task["entry"], point, status, category=category, message=message,
            ))
    result.update(
        complete=True, elapsed_seconds=time.monotonic() - start,
        point_status_counts=dict(collections.Counter(p["status"] for p in result["points"])),
    )
    rq1.atomic_write_json(output, rq1._add_checksum(result, "result_sha256"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--source-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--points-per-task", type=int, default=8)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--point-id")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-statevector-qubits", type=int, default=27)
    run.add_argument("--timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()
    frame = rq1.resolve_circuit_corpus()
    if args.command == "prepare":
        if rq1.file_sha256(args.source) != args.source_sha256:
            parser.error("historical source file checksum mismatch")
        plan = build_plan(rq1.read_artifact(args.source), args.source_sha256,
                          frame, args.points_per_task)
        validate_plan(plan, frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(plan, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        print(json.dumps({"points": plan["point_count"], "tasks": len(plan["tasks"]),
                          "circuits": plan["circuit_count"], "plan_sha256": plan["plan_sha256"]}))
        return 0
    result = run_task(rq1.read_artifact(args.plan), frame, args.output,
                      task_index=args.task_index, maximum=args.max_statevector_qubits,
                      timeout=args.timeout_seconds, point_id=args.point_id)
    print(json.dumps({"counts": result["point_status_counts"], "elapsed_seconds": result["elapsed_seconds"]}))
    return 0 if set(result["point_status_counts"]) <= {"success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
