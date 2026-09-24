"""Full-corpus RQ1 entanglement and disturbance evaluation.

Any inconsistency stops the run with an error.  The unit of account is an
acted ``(gate, qubit)`` point.  Every point in every corpus circuit receives
exactly one result, including points that
cannot be executed under the declared statevector cap or wall-clock budget.
Successful records are tied to the exact single-monitor law
``1 - F = 3 det(rho_q)`` and are independently recomputed by validation.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    EXPECTED_CIRCUITS,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256,
    ProtocolError,
    exact_state_fidelity_to,
    file_sha256,
    force_monitor,
    resolve_circuit_corpus,
    stable_hash,
)
from fast_analysis import SKIP_OPS


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "qmon-rq1-full-tightness-v1"
LAW_TOLERANCE = 1e-12
PRIMARY_COMPARE_TOLERANCE = 5e-13
DEFAULT_MAX_STATEVECTOR_QUBITS = 20
DEFAULT_CIRCUIT_TIMEOUT_SECONDS = 21600.0
POINT_TERMINAL_STATUSES = frozenset({
    "success",
    "law_violation",
    "resource_limit",
    "timeout",
    "unsupported",
    "execution_error",
})
CIRCUIT_TERMINAL_STATUSES = frozenset({
    "success",
    "law_violation",
    "resource_limit",
    "timeout",
    "unsupported",
    "execution_error",
    "mixed_terminal",
    "parse_error",
})
SOURCE_FILES = (
    SCRIPT_DIR / "rq1_full_tightness.py",
    SCRIPT_DIR / "entanglement_disturbance.py",
    SCRIPT_DIR / "rq1_certificate_check.py",
    SCRIPT_DIR / "circuits_selection.py",
    SCRIPT_DIR / "utilities.py",
    SCRIPT_DIR / "fast_analysis.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _add_checksum(value: dict, field: str) -> dict:
    value[field] = None
    payload = dict(value)
    payload.pop(field, None)
    value[field] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return value


def _check_checksum(value: Mapping[str, Any], field: str, label: str) -> None:
    stored = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if stored != expected:
        raise ProtocolError(f"{label} checksum mismatch")


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Write sorted-key JSON through fsync and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_artifact(path: Path | str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"artifact {path} is not a JSON object")
    return value


def protocol_config(
    max_statevector_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS,
    circuit_timeout_seconds: float = DEFAULT_CIRCUIT_TIMEOUT_SECONDS,
) -> dict:
    if type(max_statevector_qubits) is not int or max_statevector_qubits < 1:
        raise ProtocolError("max_statevector_qubits must be a positive integer")
    if not math.isfinite(circuit_timeout_seconds) or circuit_timeout_seconds <= 0:
        raise ProtocolError("circuit_timeout_seconds must be finite and positive")
    return {
        "schema": SCHEMA,
        "corpus_circuits": EXPECTED_CIRCUITS,
        "point_frame": (
            "every non-SKIP_OPS instruction times every qubit acted on by that "
            "instruction, in canonical circuit/gate/qarg order"
        ),
        "sampling": "none",
        "local_quantity": "det(rho_q) by direct 2xN pure-state SVD",
        "disturbance": (
            "exact state fidelity after forced single-monitor replay; "
            "all measurement/reset branches enumerated"
        ),
        "law": "1-F = 3 det(rho_q)",
        "law_tolerance": LAW_TOLERANCE,
        "primary_compare_tolerance": PRIMARY_COMPARE_TOLERANCE,
        "max_statevector_qubits": max_statevector_qubits,
        "circuit_timeout_seconds": float(circuit_timeout_seconds),
        "timeout_scope": "one canonical circuit worker",
        "sharding": "attempt_index modulo shard_count",
        "terminal_point_statuses": sorted(POINT_TERMINAL_STATUSES),
        "resource_policy": (
            "resource-limit every affected point when original or instrumented "
            "statevector qubits exceed max_statevector_qubits"
        ),
    }


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "qiskit", "qiskit-aer", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _environment_value() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "system": platform.system(),
        "libc": list(platform.libc_ver()),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
    }


def build_fingerprint_bundle(frame: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    source_files = [
        {"path": path.name, "sha256": file_sha256(path)} for path in SOURCE_FILES
    ]
    dependencies = _dependency_versions()
    environment = _environment_value()
    config_value = copy.deepcopy(dict(config))
    return {
        "manifest": {
            "sha256": frame["manifest_sha256"],
            "path_name": Path(frame["manifest"]).name,
        },
        "corpus": {
            "sha256": frame["corpus_sha256"],
            "circuit_count": frame["circuit_count"],
            "ordered_qasm": [
                {
                    "attempt_index": entry["attempt_index"],
                    "circuit": entry["circuit"],
                    "sha256": entry["source_qasm_sha256"],
                }
                for entry in frame["circuits"]
            ],
        },
        "sources": {"sha256": stable_hash(source_files), "files": source_files},
        "dependencies": {
            "sha256": stable_hash(dependencies),
            "versions": dependencies,
        },
        "environment": {
            "sha256": stable_hash(environment),
            "value": environment,
        },
        "config": {"sha256": stable_hash(config_value), "value": config_value},
    }


def _validate_config_value(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError("artifact lacks protocol configuration")
    maximum = value.get("max_statevector_qubits")
    timeout = value.get("circuit_timeout_seconds")
    expected = protocol_config(maximum, timeout)
    if value != expected:
        raise ProtocolError("artifact protocol configuration drift")
    return expected


def enumerate_acted_points(qc: QuantumCircuit) -> list[dict[str, Any]]:
    """Enumerate the complete point frame without simulating the circuit."""

    points = []
    for gate_index, instruction in enumerate(qc.data):
        operation = instruction.operation
        if operation.name in SKIP_OPS:
            continue
        for qubit in instruction.qubits:
            qubit_index = qc.find_bit(qubit).index
            point = {
                "point_ordinal": len(points),
                "gate": int(gate_index),
                "qubit": int(qubit_index),
                "gate_name": operation.name,
            }
            points.append(point)
    identities = [(point["gate"], point["qubit"]) for point in points]
    if len(identities) != len(set(identities)):
        raise ProtocolError("circuit contains duplicate acted gate-qubit identities")
    return points


def _point_id(entry: Mapping[str, Any], point: Mapping[str, Any]) -> str:
    return (
        f"{entry['attempt_id']}:g{int(point['gate']):06d}:"
        f"q{int(point['qubit']):03d}"
    )


def _local_quantities(qc: QuantumCircuit) -> dict[tuple[int, int], dict[str, float]]:
    """Compute all local pure-state quantities in one exact prefix sweep."""

    n = qc.num_qubits
    state = Statevector.from_int(0, dims=2 ** n)
    values: dict[tuple[int, int], dict[str, float]] = {}
    for gate_index, instruction in enumerate(qc.data):
        operation = instruction.operation
        if operation.name in SKIP_OPS:
            continue
        acted = [qc.find_bit(bit).index for bit in instruction.qubits]
        state = state.evolve(operation, qargs=acted)
        amplitudes = state.data.reshape((2,) * n)
        for qubit in acted:
            axis = n - 1 - qubit
            s0 = np.take(amplitudes, 0, axis=axis).reshape(-1)
            s1 = np.take(amplitudes, 1, axis=axis).reshape(-1)
            p0 = float(np.vdot(s0, s0).real)
            p1 = float(np.vdot(s1, s1).real)
            singular_values = np.linalg.svd(
                np.stack((s0, s1), axis=0), compute_uv=False
            )
            second = float(singular_values[1] if len(singular_values) > 1 else 0.0)
            determinant = float(singular_values[0] ** 2 * second ** 2)
            if not all(math.isfinite(item) for item in (p0, p1, determinant)):
                raise ProtocolError(
                    f"non-finite local quantity at {(gate_index, qubit)}"
                )
            if determinant < -LAW_TOLERANCE or determinant > 0.25 + LAW_TOLERANCE:
                raise ProtocolError(
                    f"det(rho_q) outside physical range at {(gate_index, qubit)}"
                )
            values[(gate_index, qubit)] = {
                "p0": p0,
                "p1": p1,
                "det_rho": max(0.0, determinant),
            }
    return values


def _unitary_assumption_error(qc: QuantumCircuit) -> str | None:
    final_non_skip = max(
        (index for index, item in enumerate(qc.data)
         if item.operation.name not in SKIP_OPS),
        default=-1,
    )
    for index, instruction in enumerate(qc.data):
        operation = instruction.operation
        name = operation.name
        if getattr(operation, "condition", None) is not None:
            return f"classically conditioned operation {name} at gate {index}"
        if name == "reset":
            return f"original reset at gate {index}"
        if name == "measure" and index < final_non_skip:
            return f"nonterminal original measurement at gate {index}"
    return None


def _point_base(entry: Mapping[str, Any], point: Mapping[str, Any]) -> dict:
    return {
        "point_id": _point_id(entry, point),
        "point_ordinal": point["point_ordinal"],
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "circuit": entry["circuit"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "gate": point["gate"],
        "qubit": point["qubit"],
        "gate_name": point["gate_name"],
        "terminal": True,
        "status": None,
        "p0": None,
        "p1": None,
        "det_rho": None,
        "instrumented_qubits": None,
        "fidelity": None,
        "state_infidelity": None,
        "three_det_rho": None,
        "law_abs_error": None,
        "law_holds": None,
        "error": None,
        "record_sha256": None,
    }


def _error_payload(category: str, exc: BaseException | str) -> dict[str, str]:
    return {
        "category": category,
        "exception_type": type(exc).__name__ if isinstance(exc, BaseException) else "ProtocolOutcome",
        "message": str(exc),
    }


def _terminal_point(
    entry: Mapping[str, Any],
    point: Mapping[str, Any],
    status: str,
    *,
    category: str,
    message: str,
) -> dict:
    row = _point_base(entry, point)
    row["status"] = status
    row["error"] = _error_payload(category, message)
    return _add_checksum(row, "record_sha256")


def _evaluated_point(
    entry: Mapping[str, Any],
    point: Mapping[str, Any],
    local: Mapping[str, float],
    instrumented_qubits: int,
    fidelity: float,
) -> dict:
    row = _point_base(entry, point)
    infidelity = float(1.0 - fidelity)
    expected = float(3.0 * local["det_rho"])
    error = float(abs(infidelity - expected))
    row.update({
        "status": "success" if error <= LAW_TOLERANCE else "law_violation",
        "p0": float(local["p0"]),
        "p1": float(local["p1"]),
        "det_rho": float(local["det_rho"]),
        "instrumented_qubits": int(instrumented_qubits),
        "fidelity": float(fidelity),
        "state_infidelity": infidelity,
        "three_det_rho": expected,
        "law_abs_error": error,
        "law_holds": bool(error <= LAW_TOLERANCE),
    })
    return _add_checksum(row, "record_sha256")


def _evaluate_worker(
    entry: dict,
    qasm_path: str,
    max_statevector_qubits: int,
    connection: Any,
) -> None:
    """Stream point records so the parent can account for a killed worker."""

    try:
        qc = QuantumCircuit.from_qasm_file(qasm_path)
        points = enumerate_acted_points(qc)
        assumption_error = _unitary_assumption_error(qc)
        if assumption_error is not None:
            for point in points:
                connection.send({
                    "kind": "point",
                    "record": _terminal_point(
                        entry, point, "unsupported",
                        category="pure_unitary_assumption_failed",
                        message=assumption_error,
                    ),
                })
            connection.send({"kind": "done"})
            return
        if qc.num_qubits > max_statevector_qubits:
            message = (
                f"original circuit has {qc.num_qubits} qubits; cap is "
                f"{max_statevector_qubits}"
            )
            for point in points:
                connection.send({
                    "kind": "point",
                    "record": _terminal_point(
                        entry, point, "resource_limit",
                        category="original_statevector_qubits_exceed_cap",
                        message=message,
                    ),
                })
            connection.send({"kind": "done"})
            return

        local_values = _local_quantities(qc)
        reference_circuit = qc.copy()
        reference_circuit.remove_final_measurements()
        reference = Statevector.from_instruction(reference_circuit).data
        for point in points:
            connection.send({"kind": "phase", "point_id": _point_id(entry, point)})
            local = local_values[(point["gate"], point["qubit"])]
            try:
                monitored = force_monitor(
                    qc,
                    entry["circuit"],
                    [point["gate"], point["gate_name"], point["qubit"],
                     local["p0"], local["p1"]],
                )
                if monitored.num_qubits > max_statevector_qubits:
                    row = _terminal_point(
                        entry, point, "resource_limit",
                        category="instrumented_statevector_qubits_exceed_cap",
                        message=(
                            f"instrumented circuit has {monitored.num_qubits} qubits; "
                            f"cap is {max_statevector_qubits}"
                        ),
                    )
                    row["p0"] = local["p0"]
                    row["p1"] = local["p1"]
                    row["det_rho"] = local["det_rho"]
                    row["instrumented_qubits"] = monitored.num_qubits
                    _add_checksum(row, "record_sha256")
                else:
                    fidelity = exact_state_fidelity_to(
                        monitored, qc.num_qubits, reference
                    )
                    row = _evaluated_point(
                        entry, point, local, monitored.num_qubits, fidelity
                    )
            except BaseException as exc:
                row = _terminal_point(
                    entry, point, "execution_error",
                    category="forced_monitor_execution_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            connection.send({"kind": "point", "record": row})
        connection.send({"kind": "done"})
    except Exception as exc:
        connection.send({
            "kind": "fatal",
            "error": _error_payload("worker_failure", exc),
        })
    finally:
        connection.close()


def _circuit_status(point_records: Sequence[Mapping[str, Any]]) -> str:
    statuses = {row["status"] for row in point_records}
    if not statuses:
        return "success"
    if statuses == {"success"}:
        return "success"
    priority = (
        "timeout", "execution_error", "unsupported", "law_violation",
        "resource_limit",
    )
    for status in priority:
        if statuses == {status} or statuses <= {"success", status}:
            return status
    return "mixed_terminal"


def _build_circuit_result(
    entry: Mapping[str, Any],
    expected_points: Sequence[Mapping[str, Any]] | None,
    point_records: Sequence[Mapping[str, Any]],
    *,
    status_override: str | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict:
    ordered = sorted(point_records, key=lambda row: int(row["point_ordinal"]))
    result = {
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "circuit": entry["circuit"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "terminal": True,
        "status": status_override or _circuit_status(ordered),
        "expected_point_count": (
            None if expected_points is None else len(expected_points)
        ),
        "point_record_count": len(ordered),
        "point_status_counts": dict(sorted(collections.Counter(
            row["status"] for row in ordered
        ).items())),
        "point_frame_sha256": (
            None if expected_points is None else stable_hash([
                [point["gate"], point["qubit"], point["gate_name"]]
                for point in expected_points
            ])
        ),
        "point_records_sha256": stable_hash(ordered),
        "error": copy.deepcopy(error),
        "points": list(ordered),
        "record_sha256": None,
    }
    return _add_checksum(result, "record_sha256")


def _unfinished_worker_outcome(
    *,
    timed_out: bool,
    exitcode: int | None,
    timeout_seconds: float,
    fatal_error: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    """Classify an unfinished point without conflating crashes with OOM kills."""

    if timed_out:
        return (
            "timeout",
            "circuit_wall_clock_timeout",
            f"circuit worker exceeded {timeout_seconds} seconds",
        )
    if exitcode == -signal.SIGKILL:
        return (
            "resource_limit",
            "worker_killed_resource",
            "worker exited with SIGKILL before point result",
        )
    if exitcode == -signal.SIGSEGV:
        return (
            "execution_error",
            "worker_segmentation_fault",
            "worker exited with SIGSEGV before point result",
        )
    return (
        "execution_error",
        "worker_terminated_before_point",
        (fatal_error or {}).get("message")
        or f"worker exited with code {exitcode} before point result",
    )


def run_circuit_with_timeout(
    entry: Mapping[str, Any],
    qasm_dir: Path | str,
    *,
    max_statevector_qubits: int,
    timeout_seconds: float,
) -> dict:
    """Run one circuit and materialize a terminal row for every known point."""

    qasm_path = Path(qasm_dir) / entry["circuit"]
    try:
        if file_sha256(qasm_path) != entry["source_qasm_sha256"]:
            raise ProtocolError("source QASM checksum changed before execution")
        qc = QuantumCircuit.from_qasm_file(str(qasm_path))
        expected_points = enumerate_acted_points(qc)
    except Exception as exc:
        return _build_circuit_result(
            entry, None, [], status_override="parse_error",
            error=_error_payload("qasm_read_or_parse_failed", exc),
        )

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_evaluate_worker,
        args=(dict(entry), str(qasm_path), max_statevector_qubits, child),
    )
    records_by_id: dict[str, dict] = {}
    fatal_error = None
    timed_out = False
    started = time.monotonic()
    try:
        process.start()
    except Exception as exc:
        child.close()
        parent.close()
        rows = [
            _terminal_point(
                entry,
                point,
                "execution_error",
                category="worker_start_failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            for point in expected_points
        ]
        return _build_circuit_result(
            entry,
            expected_points,
            rows,
            error=_error_payload("worker_start_failed", exc),
        )
    child.close()
    deadline = started + timeout_seconds
    try:
        done = False
        while not done and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if parent.poll(min(0.2, remaining)):
                try:
                    message = parent.recv()
                except EOFError:
                    break
                kind = message.get("kind")
                if kind == "point":
                    row = message["record"]
                    records_by_id[row["point_id"]] = row
                elif kind == "fatal":
                    fatal_error = message.get("error")
                    done = True
                elif kind == "done":
                    done = True
            elif not process.is_alive():
                break
        if process.is_alive() and time.monotonic() >= deadline:
            timed_out = True
    finally:
        parent.close()

    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    else:
        process.join(5)

    for point in expected_points:
        identifier = _point_id(entry, point)
        if identifier in records_by_id:
            continue
        status, category, message = _unfinished_worker_outcome(
            timed_out=timed_out,
            exitcode=process.exitcode,
            timeout_seconds=timeout_seconds,
            fatal_error=fatal_error,
        )
        records_by_id[identifier] = _terminal_point(
            entry, point, status, category=category, message=message
        )
    return _build_circuit_result(
        entry,
        expected_points,
        list(records_by_id.values()),
        error=fatal_error,
    )


def shard_entries(
    frame: Mapping[str, Any], shard_index: int, shard_count: int
) -> list[dict]:
    if type(shard_count) is not int or shard_count < 1:
        raise ProtocolError("shard_count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ProtocolError("shard_index must satisfy 0 <= index < count")
    return [
        entry for entry in frame["circuits"]
        if entry["attempt_index"] % shard_count == shard_index
    ]


def _artifact(
    frame: Mapping[str, Any],
    bundle: Mapping[str, Any],
    run: Mapping[str, Any],
    circuit_results: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
) -> dict:
    ordered = sorted(circuit_results, key=lambda row: int(row["attempt_index"]))
    value = {
        "schema": SCHEMA,
        "complete": bool(complete),
        "run": copy.deepcopy(dict(run)),
        "fingerprints": copy.deepcopy(dict(bundle)),
        "circuit_result_count": len(ordered),
        "point_record_count": sum(row["point_record_count"] for row in ordered),
        "circuit_status_counts": dict(sorted(collections.Counter(
            row["status"] for row in ordered
        ).items())),
        "point_status_counts": dict(sorted(collections.Counter(
            point["status"]
            for row in ordered
            for point in row["points"]
        ).items())),
        "circuit_results_sha256": stable_hash(ordered),
        "circuit_results": list(ordered),
        "artifact_sha256": None,
    }
    return _add_checksum(value, "artifact_sha256")


def _expected_entries_for_run(frame: Mapping[str, Any], run: Mapping[str, Any]) -> list[dict]:
    kind = run.get("kind")
    if kind == "shard":
        index = run.get("shard_index")
        count = run.get("shard_count")
        expected_run = {
            "kind": "shard",
            "shard_index": index,
            "shard_count": count,
            "assignment": "attempt_index modulo shard_count",
        }
        if run != expected_run:
            raise ProtocolError("shard run metadata drift")
        return shard_entries(frame, index, count)
    if kind == "merged":
        count = run.get("shard_count")
        covered = run.get("covered_shard_indices")
        expected_run = {
            "kind": "merged",
            "shard_count": count,
            "covered_shard_indices": covered,
            "assignment": "attempt_index modulo shard_count",
        }
        if run != expected_run:
            raise ProtocolError("merged run metadata drift")
        if type(count) is not int or count < 1 or covered != list(range(count)):
            raise ProtocolError("merged artifact does not cover every declared shard")
        return list(frame["circuits"])
    raise ProtocolError(f"invalid artifact run kind: {kind!r}")


def _close(a: Any, b: float, *, label: str) -> None:
    if not isinstance(a, (int, float)) or not math.isfinite(float(a)):
        raise ProtocolError(f"non-finite {label}")
    if abs(float(a) - float(b)) > PRIMARY_COMPARE_TOLERANCE:
        raise ProtocolError(f"recomputed {label} mismatch: {a!r} != {b!r}")


def _validate_point_structure(
    row: Mapping[str, Any], entry: Mapping[str, Any], point: Mapping[str, Any]
) -> None:
    _check_checksum(row, "record_sha256", f"point {row.get('point_id')}")
    expected = {
        "point_id": _point_id(entry, point),
        "point_ordinal": point["point_ordinal"],
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "circuit": entry["circuit"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "gate": point["gate"],
        "qubit": point["qubit"],
        "gate_name": point["gate_name"],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProtocolError(
                f"point fixed-field drift for {expected['point_id']}: {field}"
            )
    if row.get("terminal") is not True:
        raise ProtocolError(f"point is not terminal: {expected['point_id']}")
    if row.get("status") not in POINT_TERMINAL_STATUSES:
        raise ProtocolError(f"invalid point status: {expected['point_id']}")
    status = row["status"]
    exact_fields = (
        "p0", "p1", "det_rho", "instrumented_qubits", "fidelity",
        "state_infidelity", "three_det_rho", "law_abs_error", "law_holds",
    )
    if status in {"timeout", "unsupported", "execution_error"}:
        if any(row.get(field) is not None for field in exact_fields):
            raise ProtocolError(
                f"non-evaluated point carries fabricated quantities: {expected['point_id']}"
            )
        if not isinstance(row.get("error"), dict):
            raise ProtocolError(f"missing point error: {expected['point_id']}")
    elif status in {"success", "law_violation"}:
        if row.get("error") is not None:
            raise ProtocolError(f"evaluated point carries an error: {expected['point_id']}")
        if any(row.get(field) is None for field in exact_fields):
            raise ProtocolError(f"evaluated point lacks quantities: {expected['point_id']}")
    elif status == "resource_limit" and not isinstance(row.get("error"), dict):
        raise ProtocolError(f"resource point lacks error: {expected['point_id']}")


def _recompute_circuit_science(
    qc: QuantumCircuit,
    entry: Mapping[str, Any],
    points: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    maximum = config["max_statevector_qubits"]
    assumption_error = _unitary_assumption_error(qc)
    local_values = None
    reference = None
    if assumption_error is None and qc.num_qubits <= maximum:
        local_values = _local_quantities(qc)
        reference_circuit = qc.copy()
        reference_circuit.remove_final_measurements()
        reference = Statevector.from_instruction(reference_circuit).data

    for point, row in zip(points, rows):
        status = row["status"]
        identity = (point["gate"], point["qubit"])
        if status == "unsupported":
            if assumption_error is None:
                raise ProtocolError(f"spurious unsupported status at {identity}")
            continue
        if status in {"timeout", "execution_error"}:
            if not isinstance(row.get("error"), dict):
                raise ProtocolError(f"missing structured terminal error at {identity}")
            continue
        if status == "resource_limit" and qc.num_qubits > maximum:
            if (row.get("error") or {}).get("category") != (
                "original_statevector_qubits_exceed_cap"
            ):
                raise ProtocolError(f"wrong original resource category at {identity}")
            continue
        if status == "resource_limit" and (row.get("error") or {}).get("category") == (
            "worker_killed_resource"
        ):
            if any(row.get(field) is not None for field in (
                "p0", "p1", "det_rho", "instrumented_qubits", "fidelity",
                "state_infidelity", "three_det_rho", "law_abs_error", "law_holds",
            )):
                raise ProtocolError(f"worker resource point carries quantities at {identity}")
            continue
        if local_values is None or reference is None:
            raise ProtocolError(f"point cannot carry computed quantities at {identity}")

        local = local_values[identity]
        monitored = force_monitor(
            qc,
            entry["circuit"],
            [point["gate"], point["gate_name"], point["qubit"],
             local["p0"], local["p1"]],
        )
        if status == "resource_limit":
            if monitored.num_qubits <= maximum:
                raise ProtocolError(f"spurious instrumented resource status at {identity}")
            if (row.get("error") or {}).get("category") != (
                "instrumented_statevector_qubits_exceed_cap"
            ):
                raise ProtocolError(f"wrong instrumented resource category at {identity}")
            _close(row.get("p0"), local["p0"], label=f"p0 at {identity}")
            _close(row.get("p1"), local["p1"], label=f"p1 at {identity}")
            _close(row.get("det_rho"), local["det_rho"], label=f"det_rho at {identity}")
            if row.get("instrumented_qubits") != monitored.num_qubits:
                raise ProtocolError(f"instrumented qubit count mismatch at {identity}")
            continue
        if monitored.num_qubits > maximum:
            raise ProtocolError(f"computed point exceeds cap without resource status at {identity}")

        fidelity = exact_state_fidelity_to(monitored, qc.num_qubits, reference)
        infidelity = 1.0 - fidelity
        expected = 3.0 * local["det_rho"]
        law_error = abs(infidelity - expected)
        recomputed_status = "success" if law_error <= LAW_TOLERANCE else "law_violation"
        if status != recomputed_status:
            raise ProtocolError(f"law status mismatch at {identity}")
        for field, value in (
            ("p0", local["p0"]),
            ("p1", local["p1"]),
            ("det_rho", local["det_rho"]),
            ("fidelity", fidelity),
            ("state_infidelity", infidelity),
            ("three_det_rho", expected),
            ("law_abs_error", law_error),
        ):
            _close(row.get(field), value, label=f"{field} at {identity}")
        if row.get("instrumented_qubits") != monitored.num_qubits:
            raise ProtocolError(f"instrumented qubit count mismatch at {identity}")
        if row.get("law_holds") is not (law_error <= LAW_TOLERANCE):
            raise ProtocolError(f"law_holds mismatch at {identity}")


def validate_artifact(
    artifact: Mapping[str, Any],
    frame: Mapping[str, Any] | None = None,
    *,
    require_complete_scope: bool = True,
    require_full_corpus: bool = False,
    recompute: bool = True,
) -> dict[str, Any]:
    """Verify an artifact and recompute point science from the fixed QASM corpus."""

    if artifact.get("schema") != SCHEMA:
        raise ProtocolError("wrong full-tightness artifact schema")
    _check_checksum(artifact, "artifact_sha256", "artifact")
    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ProtocolError("artifact lacks checksums")
    config = _validate_config_value((fingerprints.get("config") or {}).get("value"))
    if frame is None:
        frame = resolve_circuit_corpus()
    expected_bundle = build_fingerprint_bundle(frame, config)
    if fingerprints != expected_bundle:
        raise ProtocolError("artifact corpus/source/environment checksum mismatch")
    for section in ("manifest", "corpus", "sources", "dependencies", "environment", "config"):
        section_value = fingerprints.get(section) or {}
        if section == "manifest":
            digest = section_value.get("sha256")
        elif section == "corpus":
            digest = section_value.get("sha256")
        else:
            digest = section_value.get("sha256")
        if not _is_sha256(digest):
            raise ProtocolError(f"invalid {section} checksum")

    run = artifact.get("run")
    if not isinstance(run, dict):
        raise ProtocolError("artifact lacks run metadata")
    expected_entries = _expected_entries_for_run(frame, run)
    if require_full_corpus and len(expected_entries) != EXPECTED_CIRCUITS:
        raise ProtocolError("artifact is not a full 310-circuit merge")
    results = artifact.get("circuit_results")
    if not isinstance(results, list):
        raise ProtocolError("artifact lacks circuit results")
    if artifact.get("circuit_result_count") != len(results):
        raise ProtocolError("circuit_result_count mismatch")
    if artifact.get("circuit_results_sha256") != stable_hash(results):
        raise ProtocolError("circuit result checksum mismatch")
    identifiers = [row.get("attempt_id") for row in results]
    if len(identifiers) != len(set(identifiers)):
        raise ProtocolError("artifact contains duplicate circuit results")
    expected_by_id = {entry["attempt_id"]: entry for entry in expected_entries}
    extras = sorted(set(identifiers) - set(expected_by_id))
    missing = sorted(set(expected_by_id) - set(identifiers))
    if extras or (require_complete_scope and missing):
        raise ProtocolError(f"circuit frame mismatch: missing={missing}, extra={extras}")
    if artifact.get("complete") is True and missing:
        raise ProtocolError("artifact claims complete while circuits are missing")
    if require_complete_scope and artifact.get("complete") is not True:
        raise ProtocolError("artifact is an incomplete checkpoint")

    qasm_dir = Path(frame["qasm_dir"])
    total_points = 0
    circuit_counts = collections.Counter()
    point_counts = collections.Counter()
    for result in results:
        entry = expected_by_id[result["attempt_id"]]
        _check_checksum(result, "record_sha256", f"circuit {entry['circuit']}")
        for field in ("attempt_index", "attempt_id", "circuit", "source_qasm_sha256"):
            if result.get(field) != entry[field]:
                raise ProtocolError(f"circuit fixed-field drift: {field}")
        if result.get("terminal") is not True or result.get("status") not in CIRCUIT_TERMINAL_STATUSES:
            raise ProtocolError(f"invalid circuit terminal status for {entry['circuit']}")
        qasm_path = qasm_dir / entry["circuit"]
        if file_sha256(qasm_path) != entry["source_qasm_sha256"]:
            raise ProtocolError(f"QASM checksum mismatch for {entry['circuit']}")
        try:
            qc = QuantumCircuit.from_qasm_file(str(qasm_path))
            points = enumerate_acted_points(qc)
        except BaseException:
            if result.get("status") != "parse_error" or result.get("points"):
                raise ProtocolError(f"invalid parse-error accounting for {entry['circuit']}")
            if result.get("expected_point_count") is not None:
                raise ProtocolError(f"parse-error point count must be unknown for {entry['circuit']}")
            if result.get("point_record_count") != 0:
                raise ProtocolError(f"parse-error circuit contains point records for {entry['circuit']}")
            circuit_counts["parse_error"] += 1
            continue
        if result.get("status") == "parse_error":
            raise ProtocolError(f"spurious parse_error for {entry['circuit']}")
        rows = result.get("points")
        if not isinstance(rows, list):
            raise ProtocolError(f"circuit lacks point rows: {entry['circuit']}")
        if result.get("expected_point_count") != len(points) or len(rows) != len(points):
            raise ProtocolError(f"silent point omission for {entry['circuit']}")
        if result.get("point_record_count") != len(rows):
            raise ProtocolError(f"point_record_count mismatch for {entry['circuit']}")
        frame_hash = stable_hash([
            [point["gate"], point["qubit"], point["gate_name"]] for point in points
        ])
        if result.get("point_frame_sha256") != frame_hash:
            raise ProtocolError(f"point frame checksum mismatch for {entry['circuit']}")
        if result.get("point_records_sha256") != stable_hash(rows):
            raise ProtocolError(f"point records checksum mismatch for {entry['circuit']}")
        for point, row in zip(points, rows):
            _validate_point_structure(row, entry, point)
        recomputed_point_counts = dict(sorted(collections.Counter(
            row["status"] for row in rows
        ).items()))
        if result.get("point_status_counts") != recomputed_point_counts:
            raise ProtocolError(f"point status summary mismatch for {entry['circuit']}")
        if result.get("status") != _circuit_status(rows):
            raise ProtocolError(f"circuit status summary mismatch for {entry['circuit']}")
        if recompute:
            _recompute_circuit_science(qc, entry, points, rows, config)
        total_points += len(rows)
        circuit_counts[result["status"]] += 1
        point_counts.update(row["status"] for row in rows)

    if artifact.get("point_record_count") != total_points:
        raise ProtocolError("artifact point_record_count mismatch")
    if artifact.get("circuit_status_counts") != dict(sorted(circuit_counts.items())):
        raise ProtocolError("artifact circuit status summary mismatch")
    if artifact.get("point_status_counts") != dict(sorted(point_counts.items())):
        raise ProtocolError("artifact point status summary mismatch")
    return {
        "circuits": len(results),
        "points": total_points,
        "circuit_status_counts": dict(sorted(circuit_counts.items())),
        "point_status_counts": dict(sorted(point_counts.items())),
    }


def run_shard(
    frame: Mapping[str, Any],
    output: Path | str,
    *,
    shard_index: int,
    shard_count: int,
    max_statevector_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS,
    circuit_timeout_seconds: float = DEFAULT_CIRCUIT_TIMEOUT_SECONDS,
    resume: bool = False,
    runner: Callable[..., dict] = run_circuit_with_timeout,
) -> dict:
    config = protocol_config(max_statevector_qubits, circuit_timeout_seconds)
    bundle = build_fingerprint_bundle(frame, config)
    entries = shard_entries(frame, shard_index, shard_count)
    run = {
        "kind": "shard",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assignment": "attempt_index modulo shard_count",
    }
    output = Path(output)
    existing_by_id: dict[str, dict] = {}
    if output.exists():
        if not resume:
            raise ProtocolError(f"output exists: {output}; pass --resume")
        existing = read_artifact(output)
        validate_artifact(
            existing, frame, require_complete_scope=False, recompute=False
        )
        if existing.get("run") != run or existing.get("fingerprints") != bundle:
            raise ProtocolError("resume artifact run or checksum mismatch")
        existing_by_id = {
            row["attempt_id"]: row for row in existing["circuit_results"]
        }

    for position, entry in enumerate(entries, start=1):
        if entry["attempt_id"] in existing_by_id:
            continue
        print(
            f"[{position}/{len(entries)}] {entry['attempt_id']}",
            flush=True,
        )
        try:
            result = runner(
                entry,
                frame["qasm_dir"],
                max_statevector_qubits=max_statevector_qubits,
                timeout_seconds=circuit_timeout_seconds,
            )
        except Exception as exc:
            qasm_path = Path(frame["qasm_dir"]) / entry["circuit"]
            try:
                points = enumerate_acted_points(
                    QuantumCircuit.from_qasm_file(str(qasm_path))
                )
            except Exception as parse_exc:
                result = _build_circuit_result(
                    entry,
                    None,
                    [],
                    status_override="parse_error",
                    error=_error_payload("parent_runner_and_parse_failed", parse_exc),
                )
            else:
                rows = [
                    _terminal_point(
                        entry,
                        point,
                        "execution_error",
                        category="parent_runner_failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                    for point in points
                ]
                result = _build_circuit_result(
                    entry,
                    points,
                    rows,
                    error=_error_payload("parent_runner_failed", exc),
                )
        existing_by_id[entry["attempt_id"]] = result
        checkpoint = _artifact(
            frame, bundle, run, list(existing_by_id.values()), complete=False
        )
        atomic_write_json(output, checkpoint)

    final = _artifact(
        frame,
        bundle,
        run,
        [existing_by_id[entry["attempt_id"]] for entry in entries],
        complete=True,
    )
    validate_artifact(final, frame, require_complete_scope=True, recompute=False)
    atomic_write_json(output, final)
    return final


def merge_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    frame: Mapping[str, Any] | None = None,
    *,
    recompute: bool = True,
) -> dict:
    if not artifacts:
        raise ProtocolError("merge requires at least one shard artifact")
    if frame is None:
        frame = resolve_circuit_corpus()
    for artifact in artifacts:
        validate_artifact(artifact, frame, require_complete_scope=True, recompute=recompute)
        if artifact["run"].get("kind") != "shard":
            raise ProtocolError("merge accepts shard artifacts only")
    counts = {artifact["run"]["shard_count"] for artifact in artifacts}
    if len(counts) != 1:
        raise ProtocolError("merge inputs disagree on shard_count")
    shard_count = counts.pop()
    by_index = {artifact["run"]["shard_index"]: artifact for artifact in artifacts}
    if len(by_index) != len(artifacts):
        raise ProtocolError("merge inputs contain duplicate shard indices")
    if set(by_index) != set(range(shard_count)):
        raise ProtocolError("merge is missing one or more declared shards")
    first_bundle = artifacts[0]["fingerprints"]
    if any(artifact["fingerprints"] != first_bundle for artifact in artifacts[1:]):
        raise ProtocolError("merge inputs have a checksum mismatch")
    results = [
        result
        for shard_index in range(shard_count)
        for result in by_index[shard_index]["circuit_results"]
    ]
    merged = _artifact(
        frame,
        first_bundle,
        {
            "kind": "merged",
            "shard_count": shard_count,
            "covered_shard_indices": list(range(shard_count)),
            "assignment": "attempt_index modulo shard_count",
        },
        results,
        complete=True,
    )
    validate_artifact(
        merged,
        frame,
        require_complete_scope=True,
        require_full_corpus=(frame["circuit_count"] == EXPECTED_CIRCUITS),
        recompute=False,
    )
    return merged


def aggregate_artifact(
    artifact: Mapping[str, Any],
    frame: Mapping[str, Any] | None = None,
    *,
    recompute: bool = True,
    require_full_corpus: bool = True,
) -> dict:
    if frame is None:
        frame = resolve_circuit_corpus()
    validation = validate_artifact(
        artifact,
        frame,
        require_complete_scope=True,
        require_full_corpus=require_full_corpus,
        recompute=recompute,
    )
    counts = collections.Counter(validation["point_status_counts"])
    exact_evaluated = counts["success"] + counts["law_violation"]
    attempted = validation["points"]
    law_passes = counts["success"]
    return {
        "schema": "qmon-rq1-full-tightness-aggregate-v1",
        "source_artifact_sha256": artifact["artifact_sha256"],
        "corpus_sha256": artifact["fingerprints"]["corpus"]["sha256"],
        "attempted_circuit_denominator": validation["circuits"],
        "attempted_point_denominator": attempted,
        "exact_evaluated_point_denominator": exact_evaluated,
        "law_pass_count": law_passes,
        "law_violation_count": counts["law_violation"],
        "law_pass_rate_exact_evaluated": (
            None if exact_evaluated == 0 else law_passes / exact_evaluated
        ),
        "point_status_counts": validation["point_status_counts"],
        "circuit_status_counts": validation["circuit_status_counts"],
        "non_evaluated_point_count": attempted - exact_evaluated,
        "resource_limit_point_count": counts["resource_limit"],
        "timeout_point_count": counts["timeout"],
        "unsupported_point_count": counts["unsupported"],
        "execution_error_point_count": counts["execution_error"],
        "denominator_policy": (
            "attempted denominators include every enumerated point; law rate uses "
            "the separately reported exact-evaluated denominator"
        ),
    }


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)


def _load_frame(args: argparse.Namespace) -> dict:
    return resolve_circuit_corpus(args.manifest, args.qasm_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full 310-circuit RQ1 entanglement/disturbance validator"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one deterministic corpus shard")
    _add_corpus_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)
    run.add_argument(
        "--max-statevector-qubits", type=int,
        default=DEFAULT_MAX_STATEVECTOR_QUBITS,
    )
    run.add_argument(
        "--circuit-timeout-seconds", type=float,
        default=DEFAULT_CIRCUIT_TIMEOUT_SECONDS,
    )
    run.add_argument("--resume", action="store_true")

    merge = commands.add_parser("merge", help="strictly merge every shard")
    _add_corpus_arguments(merge)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.add_argument("inputs", nargs="+", type=Path)

    validate = commands.add_parser(
        "validate", help="recompute and validate an artifact from the fixed QASM corpus"
    )
    _add_corpus_arguments(validate)
    validate.add_argument("input", type=Path)
    validate.add_argument("--require-full-corpus", action="store_true")

    aggregate = commands.add_parser(
        "aggregate", help="validate and emit denominator-complete totals"
    )
    _add_corpus_arguments(aggregate)
    aggregate.add_argument("input", type=Path)
    aggregate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = _load_frame(args)
        if args.command == "run":
            artifact = run_shard(
                frame,
                args.output,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                max_statevector_qubits=args.max_statevector_qubits,
                circuit_timeout_seconds=args.circuit_timeout_seconds,
                resume=args.resume,
            )
            print(
                f"wrote {args.output}: circuits={artifact['circuit_result_count']} "
                f"points={artifact['point_record_count']} "
                f"statuses={artifact['point_status_counts']}"
            )
        elif args.command == "merge":
            if args.output.exists() and not args.overwrite:
                raise ProtocolError(
                    f"merge output exists: {args.output}; pass --overwrite"
                )
            merged = merge_artifacts(
                [read_artifact(path) for path in args.inputs], frame, recompute=True
            )
            atomic_write_json(args.output, merged)
            print(
                f"merged {len(args.inputs)} shards: circuits="
                f"{merged['circuit_result_count']} points={merged['point_record_count']}"
            )
        elif args.command == "validate":
            validation = validate_artifact(
                read_artifact(args.input),
                frame,
                require_complete_scope=True,
                require_full_corpus=args.require_full_corpus,
                recompute=True,
            )
            print(json.dumps(validation, sort_keys=True))
        elif args.command == "aggregate":
            summary = aggregate_artifact(
                read_artifact(args.input), frame, recompute=True,
                require_full_corpus=True,
            )
            if args.output is not None:
                atomic_write_json(args.output, summary)
            print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
