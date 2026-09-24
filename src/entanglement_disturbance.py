"""RQ1 tightness experiment and fixed-corpus utilities.

The experiment has two deliberately separate numerical objects:

* the selector uses the second Schmidt amplitude (cross-checked in
  ``rq1_certificate_check.py``); and
* the tightness experiment uses ``det(rho_q)`` as the physical entanglement
  score and checks the exact law ``1 - F = 3 det(rho_q)``.

This module also owns the fixed 310-circuit frame shared by RQ1 and RQ2.
The key-file hash fixes the ordered identities, while the aggregate hash of
all ``(key, source-QASM SHA-256)`` pairs fixes every source file.  A missing,
reordered, or substituted circuit is therefore a hard protocol error.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.setrecursionlimit(1_000_000)

from circuits_selection import dfs_quantum_circuit, transformation
from fast_analysis import SKIP_OPS
from utilities import generate_monitoring_circuit


SCRIPT_DIR = Path(__file__).resolve().parent
RECONSTRUCTION_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = RECONSTRUCTION_DIR.parent
DEFAULT_MANIFEST = RECONSTRUCTION_DIR / "inputs" / "certified_circuits.txt"
DEFAULT_QASM_DIR = WORKSPACE_DIR / "quantum_circuits"

EXPECTED_CIRCUITS = 310
EXPECTED_MANIFEST_SHA256 = (
    "fae150dbacc70b25efbc14c719b04e6e20d8e306d2fb703927f000b4bc6014c1"
)
EXPECTED_CORPUS_SHA256 = (
    "7347d0f3b43b2d6a28ae846d48ff5e677bf88af624261a7d505bca3eb804a5f4"
)

LOCAL_SCORE_ZERO_TOL = 1e-12
TIGHTNESS_LAW_TOL = 1e-12
TIGHTNESS_SCHEMA = "qmon-rq1-tightness-v4"
TIGHTNESS_BINS = (0.0, 0.02, 0.05, 0.10, 0.18, 0.26)
TIGHTNESS_POINTS_PER_BIN = 3
TIGHTNESS_RANK_DECIMALS = 12
SOURCE_REPLAY_FLOAT_TOL = 1e-14

# Purposefully chosen before execution: 4--6 qubits, ten benchmark families,
# and collectively broad local-entanglement support.  This is a controlled
# stress experiment, not a representative estimator of the 310-circuit pool.
TIGHTNESS_KEYS = (
    "qpeinexact_indep_qiskit_4.qasm",
    "qpeinexact_indep_qiskit_5.qasm",
    "qpeinexact_indep_qiskit_6.qasm",
    "qaoa_indep_qiskit_5.qasm",
    "qaoa_indep_qiskit_6.qasm",
    "wstate_indep_qiskit_5.qasm",
    "wstate_indep_qiskit_6.qasm",
    "su2random_indep_qiskit_4.qasm",
    "su2random_indep_qiskit_5.qasm",
    "twolocalrandom_indep_qiskit_4.qasm",
    "realamprandom_indep_qiskit_4.qasm",
    "vqe_indep_qiskit_6.qasm",
    "qftentangled_indep_qiskit_5.qasm",
    "portfoliovqe_indep_qiskit_4.qasm",
    "qnn_indep_qiskit_4.qasm",
)


class ProtocolError(RuntimeError):
    """The fixed experiment contract or a result artifact is invalid."""


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalize_manifest_key(line: str) -> str:
    key = line.strip()
    if not key:
        raise ProtocolError("blank circuit key")
    return key if key.endswith(".qasm") else key + ".qasm"


def _corpus_hash(entries: list[dict]) -> str:
    return stable_hash([
        {
            "basename": Path(entry["circuit"]).stem,
            "sha256": entry["source_qasm_sha256"],
        }
        for entry in entries
    ])


def resolve_circuit_corpus(
    manifest: Path | str = DEFAULT_MANIFEST,
    qasm_dir: Path | str = DEFAULT_QASM_DIR,
    *,
    expected_count: int = EXPECTED_CIRCUITS,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
    expected_corpus_sha256: str | None = EXPECTED_CORPUS_SHA256,
) -> dict:
    """Resolve and verify the exact circuit corpus.

    The optional expected values exist for small unit-test fixtures.  The CLI
    never exposes them and always enforces the fixed reference constants.
    """

    manifest = Path(manifest).resolve()
    qasm_dir = Path(qasm_dir).resolve()
    if not manifest.is_file():
        raise ProtocolError(f"circuit-list manifest missing: {manifest}")
    if not qasm_dir.is_dir():
        raise ProtocolError(f"QASM directory missing: {qasm_dir}")

    manifest_sha = file_sha256(manifest)
    if (expected_manifest_sha256 is not None
            and manifest_sha != expected_manifest_sha256):
        raise ProtocolError(
            "manifest checksum substitution: "
            f"{manifest_sha} != {expected_manifest_sha256}"
        )

    lines = manifest.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ProtocolError("circuit list contains a blank row")
    keys = [_normalize_manifest_key(line) for line in lines]
    if len(keys) != expected_count:
        raise ProtocolError(
            f"circuit list requires {expected_count} circuits; "
            f"found {len(keys)}"
        )
    if len(keys) != len(set(keys)):
        raise ProtocolError("circuit list contains duplicate identities")

    entries = []
    for attempt_index, key in enumerate(keys):
        source = qasm_dir / key
        if not source.is_file():
            raise ProtocolError(f"QASM missing: {source}")
        entries.append({
            "attempt_index": attempt_index,
            "attempt_id": f"canonical:{attempt_index:03d}:{key}",
            "circuit": key,
            "source_qasm_sha256": file_sha256(source),
        })

    corpus_sha = _corpus_hash(entries)
    if (expected_corpus_sha256 is not None
            and corpus_sha != expected_corpus_sha256):
        raise ProtocolError(
            "corpus checksum substitution: "
            f"{corpus_sha} != {expected_corpus_sha256}"
        )
    return {
        "manifest": str(manifest),
        "qasm_dir": str(qasm_dir),
        "circuit_count": len(entries),
        "manifest_sha256": manifest_sha,
        "corpus_sha256": corpus_sha,
        "circuits": entries,
    }


def atomic_write_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value, handle, sort_keys=True, indent=2, ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path | str, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def normalize_local_score(value, tol: float = LOCAL_SCORE_ZERO_TOL) -> float:
    """Map determinant roundoff in the numerical zero class to zero.

    This helper is retained for plots/class labels.  RQ2 does *not* use this
    thresholded value for ``E_mean`` or ``E_peak``; it uses every physical
    determinant returned by :func:`entanglement_at_all_points`.
    """

    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"non-finite local score: {score}")
    if score < -tol:
        raise ValueError(
            f"local score {score} is below the physical range by more than "
            f"the numerical tolerance {tol}"
        )
    return 0.0 if score <= tol else score


def _physical_determinant(raw_value: float) -> float:
    """Preserve positive determinants; clip only negative roundoff."""

    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite determinant: {value}")
    if value < -LOCAL_SCORE_ZERO_TOL:
        raise ValueError(f"det(rho_q) is materially negative: {value}")
    return max(0.0, value)


def entanglement_at_all_points(qc: QuantumCircuit):
    """Enumerate every ``(gate, acted-qubit)`` point exactly once.

    Returns ``[gate_index, gate_name, qubit, p0, p1, det_rho]`` records.  The
    determinant is the unthresholded physical score: tiny positive values are
    retained, while only negative floating-point roundoff is clipped to zero.
    """

    n = qc.num_qubits
    state = Statevector.from_int(0, dims=2 ** n)
    records = []
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
            overlap = np.vdot(s0, s1)
            raw_determinant = p0 * p1 - float(abs(overlap) ** 2)
            _physical_determinant(raw_determinant)
            singular_values = np.linalg.svd(
                np.stack((s0, s1), axis=0), compute_uv=False
            )
            second = float(
                singular_values[1] if len(singular_values) > 1 else 0.0
            )
            determinant = float(singular_values[0] ** 2 * second ** 2)
            records.append([
                gate_index, operation.name, qubit, p0, p1, determinant
            ])
    return records


def _exact_branches(circuit: QuantumCircuit):
    """Enumerate every branch with strictly positive computed probability."""

    circuit = circuit.copy()
    circuit.remove_final_measurements()
    n = circuit.num_qubits
    branches = [(1.0, Statevector.from_int(0, dims=2 ** n).data)]
    ignored = {"barrier", "delay", "snapshot"}

    plan = []
    segment = QuantumCircuit(n)
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ignored:
            continue
        if name in {"measure", "reset"}:
            if segment.data:
                plan.append(("segment", segment))
                segment = QuantumCircuit(n)
            plan.append((name, circuit.find_bit(instruction.qubits[0]).index))
            continue
        segment.append(
            instruction.operation,
            [circuit.find_bit(bit).index for bit in instruction.qubits],
        )
    if segment.data:
        plan.append(("segment", segment))

    for kind, payload in plan:
        if kind == "segment":
            branches = [
                (probability, Statevector(vector).evolve(payload).data)
                for probability, vector in branches
            ]
            continue

        qubit = payload
        axis = n - 1 - qubit
        reset_target = None if kind == "measure" else 0
        next_branches = []
        for branch_probability, vector in branches:
            amplitudes = vector.reshape((2,) * n)
            for outcome in (0, 1):
                slice_state = np.take(amplitudes, outcome, axis=axis)
                outcome_probability = float(
                    np.vdot(slice_state.reshape(-1), slice_state.reshape(-1)).real
                )
                if not math.isfinite(outcome_probability):
                    raise ProtocolError("non-finite measurement probability")
                if outcome_probability < 0.0:
                    raise ProtocolError("negative measurement probability")
                if outcome_probability == 0.0:
                    continue
                collapsed = np.zeros((2,) * n, dtype=complex)
                index = [slice(None)] * n
                index[axis] = outcome if reset_target is None else reset_target
                collapsed[tuple(index)] = slice_state / math.sqrt(outcome_probability)
                next_branches.append((
                    branch_probability * outcome_probability,
                    collapsed.reshape(-1),
                ))
        branches = next_branches
    return branches


def exact_state_fidelity_to(
    circuit: QuantumCircuit, n_original: int, reference_state
) -> float:
    """Exact fidelity after tracing replay ancillas, against a pure reference."""

    reference_state = np.asarray(reference_state, dtype=complex)
    fidelity = 0.0
    for probability, vector in _exact_branches(circuit):
        matrix = vector.reshape((2 ** (circuit.num_qubits - n_original),
                                 2 ** n_original))
        projected = matrix @ np.conj(reference_state)
        fidelity += probability * float(np.vdot(projected, projected).real)
    return fidelity


def force_monitor(qc: QuantumCircuit, key: str, point):
    """Insert one forced monitor using the replay implementation."""

    gate_index, gate_name, qubit, p0, p1 = point
    cone = dfs_quantum_circuit(qc, qubit, gate_index, set(), [])
    replay = transformation(qc, qubit, cone, [])
    if not replay:
        raise ProtocolError(
            f"empty replay for forced point {(gate_index, qubit)} in {key}"
        )
    operation_list = {
        key: [[gate_index, gate_name, qubit, p0, p1, replay]]
    }
    answer = {
        "picked_nodes": [gate_index],
        "picked_qubits": [qubit],
        "obj1": 1,
        "obj2": 1,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_monitoring_circuit(
            qc, key, operation_list, 10 ** 7, answer
        )


def _select_tightness_points(points):
    numerical_zero = [
        point for point in points if point[5] <= LOCAL_SCORE_ZERO_TOL
    ][:2]
    seen = {(point[0], point[2]) for point in numerical_zero}
    selected = [(point, "numerical-zero") for point in numerical_zero]

    for lower, upper in zip(TIGHTNESS_BINS[:-1], TIGHTNESS_BINS[1:]):
        candidates = [
            point for point in points
            if point[5] > LOCAL_SCORE_ZERO_TOL
            and lower <= point[5] < upper
            and (point[0], point[2]) not in seen
        ]
        candidates.sort(key=lambda point: (
            round(point[5], TIGHTNESS_RANK_DECIMALS),
            point[0],
            point[2],
        ))
        stride = max(
            1, len(candidates) // max(1, TIGHTNESS_POINTS_PER_BIN)
        )
        label = f"[{lower:.2f},{upper:.2f})"
        for point in candidates[::stride][:TIGHTNESS_POINTS_PER_BIN]:
            selected.append((point, label))
            seen.add((point[0], point[2]))

    identities = [(point[0], point[2]) for point, _ in selected]
    if len(identities) != len(set(identities)):
        raise ProtocolError("tightness selection contains duplicate points")
    return selected


def evaluate_tightness_circuit(
    key: str, qasm_path: Path | str, source_sha256: str
) -> tuple[dict, list[dict]]:
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    reference_circuit = qc.copy()
    reference_circuit.remove_final_measurements()
    reference = Statevector.from_instruction(reference_circuit).data

    all_points = entanglement_at_all_points(qc)
    selected = _select_tightness_points(all_points)
    local_rows = []
    for ordinal, (point, bucket) in enumerate(selected):
        gate_index, gate_name, qubit, p0, p1, determinant = point
        reconstructed = force_monitor(
            qc, key, [gate_index, gate_name, qubit, p0, p1]
        )
        fidelity = exact_state_fidelity_to(
            reconstructed, qc.num_qubits, reference
        )
        raw_state_infidelity = 1.0 - fidelity
        if not -TIGHTNESS_LAW_TOL <= raw_state_infidelity <= 1.0 + 1e-12:
            raise ProtocolError(
                "forced-monitor state infidelity is outside [0,1] at "
                f"{(key, gate_index, qubit)}: {raw_state_infidelity:.17g}"
            )
        state_infidelity = min(1.0, max(0.0, raw_state_infidelity))
        expected = 3.0 * determinant
        local_rows.append({
            "circuit": key,
            "source_qasm_sha256": source_sha256,
            "selection_ordinal": ordinal,
            "selection_bucket": bucket,
            "gate": int(gate_index),
            "qubit": int(qubit),
            "gate_name": gate_name,
            "det_rho": float(determinant),
            "state_infidelity": float(state_infidelity),
            "three_det_rho": float(expected),
            "law_abs_error": float(abs(state_infidelity - expected)),
            "numerically_separable": bool(
                determinant <= LOCAL_SCORE_ZERO_TOL
            ),
        })

    status = {
        "status": "complete",
        "source_qasm_sha256": source_sha256,
        "enumerated_point_count": len(all_points),
        "selected_point_count": len(selected),
        "selected_points_sha256": stable_hash([
            [row["gate"], row["qubit"], row["selection_bucket"]]
            for row in local_rows
        ]),
    }
    return status, local_rows


def _error_status(source_sha256: str, error: BaseException) -> dict:
    return {
        "status": "error",
        "source_qasm_sha256": source_sha256,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def tightness_protocol_descriptor() -> dict:
    """Return the exact controlled-tightness protocol descriptor."""

    return {
        "circuit_selection": (
            "predeclared 15-circuit, 4--6-qubit controlled stress set; "
            "not a random or representative subsample"
        ),
        "point_selection": {
            "numerical_zero_points_per_circuit": "up to 2",
            "numerical_zero_rule": f"det(rho_q) <= {LOCAL_SCORE_ZERO_TOL:g}",
            "bins": list(TIGHTNESS_BINS),
            "points_per_bin": TIGHTNESS_POINTS_PER_BIN,
            "within_bin_ranking": (
                f"det(rho_q) rounded to {TIGHTNESS_RANK_DECIMALS} decimal "
                "places, then gate index and qubit index"
            ),
            "positive_bin_rule": (
                f"det(rho_q) > {LOCAL_SCORE_ZERO_TOL:g}; numerical-zero "
                "points are excluded from every positive bin"
            ),
        },
        "disturbance_metric": "exact state infidelity 1-F",
        "law": "1-F = 3 det(rho_q)",
        "law_validation_tolerance": TIGHTNESS_LAW_TOL,
        "sampling": (
            "none; every branch with strictly positive floating-point "
            "probability is enumerated, and exactly zero-probability branches "
            "are omitted"
        ),
    }


def build_tightness_artifact(frame: dict) -> dict:
    by_key = {
        entry["circuit"]: entry for entry in frame["circuits"]
    }
    missing = sorted(set(TIGHTNESS_KEYS) - set(by_key))
    if missing:
        raise ProtocolError(
            f"tightness circuits absent from the corpus: {missing}"
        )

    records = []
    statuses = {}
    qasm_dir = Path(frame["qasm_dir"])
    for key in TIGHTNESS_KEYS:
        entry = by_key[key]
        source_sha = entry["source_qasm_sha256"]
        try:
            status, local_rows = evaluate_tightness_circuit(
                key, qasm_dir / key, source_sha
            )
        except Exception as error:
            statuses[key] = _error_status(source_sha, error)
        else:
            statuses[key] = status
            for row in local_rows:
                row["attempt_index"] = entry["attempt_index"]
                row["attempt_id"] = entry["attempt_id"]
            records.extend(local_rows)
        statuses[key]["attempt_index"] = entry["attempt_index"]
        statuses[key]["attempt_id"] = entry["attempt_id"]

    records.sort(key=lambda row: (
        TIGHTNESS_KEYS.index(row["circuit"]), row["selection_ordinal"]
    ))
    artifact = {
        "schema": TIGHTNESS_SCHEMA,
        "protocol": tightness_protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "circuit_count": frame["circuit_count"],
        },
        "circuit_status": statuses,
        "record_count": len(records),
        "records_sha256": stable_hash(records),
        "records": records,
    }
    validate_tightness_artifact(artifact, frame, require_complete=False)
    return artifact


def _finite_real(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ProtocolError(f"{label} must be a finite real number")
    return float(value)


def _require_replay_close(
    actual, expected, label: str, identity, *, tolerance=SOURCE_REPLAY_FLOAT_TOL
) -> None:
    actual = _finite_real(actual, f"{label} at {identity}")
    expected = _finite_real(expected, f"replayed {label} at {identity}")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ProtocolError(
            f"source replay mismatch for {label} at {identity}: "
            f"{actual:.17g} != {expected:.17g}"
        )


def _deep_validate_tightness_source_replay(
    frame: dict, statuses: dict, records_by_circuit: dict
) -> None:
    qasm_dir_value = frame.get("qasm_dir")
    if not isinstance(qasm_dir_value, str) or not qasm_dir_value:
        raise ProtocolError(
            "complete tightness validation requires frame qasm_dir"
        )
    qasm_dir = Path(qasm_dir_value)
    frame_entries = {
        entry["circuit"]: entry for entry in frame["circuits"]
    }

    exact_fields = (
        "attempt_index", "attempt_id", "circuit", "source_qasm_sha256",
        "selection_ordinal", "selection_bucket", "gate", "qubit",
        "gate_name", "numerically_separable",
    )
    numeric_fields = (
        "det_rho", "state_infidelity", "three_det_rho", "law_abs_error",
    )
    status_fields = (
        "status", "attempt_index", "attempt_id", "source_qasm_sha256",
        "enumerated_point_count", "selected_point_count",
        "selected_points_sha256",
    )

    for key in TIGHTNESS_KEYS:
        entry = frame_entries[key]
        qasm_path = qasm_dir / key
        if not qasm_path.is_file():
            raise ProtocolError(f"tightness source QASM missing for {key}")
        if file_sha256(qasm_path) != entry["source_qasm_sha256"]:
            raise ProtocolError(f"tightness source QASM hash drift for {key}")

        expected_status, expected_rows = evaluate_tightness_circuit(
            key, qasm_path, entry["source_qasm_sha256"]
        )
        expected_status.update({
            "attempt_index": entry["attempt_index"],
            "attempt_id": entry["attempt_id"],
        })
        for row in expected_rows:
            row.update({
                "attempt_index": entry["attempt_index"],
                "attempt_id": entry["attempt_id"],
            })

        actual_status = statuses[key]
        for field in status_fields:
            if actual_status.get(field) != expected_status.get(field):
                raise ProtocolError(
                    f"source replay mismatch for status {field} in {key}: "
                    f"{actual_status.get(field)!r} != "
                    f"{expected_status.get(field)!r}"
                )

        actual_rows = sorted(
            records_by_circuit[key], key=lambda row: row["selection_ordinal"]
        )
        if len(actual_rows) != len(expected_rows):
            raise ProtocolError(
                f"source replay selected-point count mismatch for {key}: "
                f"{len(actual_rows)} != {len(expected_rows)}"
            )
        for actual, expected in zip(actual_rows, expected_rows):
            identity = (key, expected["gate"], expected["qubit"])
            for field in exact_fields:
                if actual.get(field) != expected.get(field):
                    raise ProtocolError(
                        f"source replay mismatch for {field} at {identity}: "
                        f"{actual.get(field)!r} != {expected.get(field)!r}"
                    )
            for field in numeric_fields:
                _require_replay_close(
                    actual.get(field), expected.get(field), field, identity
                )


def validate_tightness_artifact(
    artifact: dict, frame: dict, *, require_complete: bool = True
) -> dict:
    if artifact.get("schema") != TIGHTNESS_SCHEMA:
        raise ProtocolError("wrong RQ1 tightness artifact schema")
    if artifact.get("protocol") != tightness_protocol_descriptor():
        raise ProtocolError("RQ1 tightness protocol drift")
    statuses = artifact.get("circuit_status")
    records = artifact.get("records")
    if not isinstance(statuses, dict) or not isinstance(records, list):
        raise ProtocolError("tightness artifact lacks statuses or records")
    expected_corpus = {
        "manifest_sha256": frame.get("manifest_sha256"),
        "corpus_sha256": frame.get("corpus_sha256"),
        "circuit_count": frame.get("circuit_count", len(frame["circuits"])),
    }
    if artifact.get("corpus") != expected_corpus:
        raise ProtocolError(
            f"tightness corpus provenance mismatch: "
            f"{artifact.get('corpus')} != {expected_corpus}"
        )
    if set(statuses) != set(TIGHTNESS_KEYS):
        raise ProtocolError(
            "tightness artifact must account for exactly all 15 circuits"
        )
    if artifact.get("record_count") != len(records):
        raise ProtocolError("tightness record_count does not match records")
    if artifact.get("records_sha256") != stable_hash(records):
        raise ProtocolError("tightness records checksum mismatch")

    frame_hashes = {
        entry["circuit"]: entry["source_qasm_sha256"]
        for entry in frame["circuits"]
    }
    seen = set()
    records_by_circuit = {key: [] for key in TIGHTNESS_KEYS}
    for row in records:
        key = row.get("circuit")
        if key not in records_by_circuit:
            raise ProtocolError(f"unexpected tightness circuit: {key}")
        identity = (key, row.get("gate"), row.get("qubit"))
        if identity in seen:
            raise ProtocolError(f"duplicate tightness point: {identity}")
        seen.add(identity)
        records_by_circuit[key].append(row)
        if row.get("source_qasm_sha256") != frame_hashes[key]:
            raise ProtocolError(f"source hash drift in tightness row {identity}")
        frame_entry = next(
            entry for entry in frame["circuits"] if entry["circuit"] == key
        )
        if (row.get("attempt_index") != frame_entry["attempt_index"]
                or row.get("attempt_id") != frame_entry["attempt_id"]):
            raise ProtocolError(f"attempt identity drift in tightness row {identity}")
        if (isinstance(row.get("selection_ordinal"), bool)
                or not isinstance(row.get("selection_ordinal"), int)
                or row["selection_ordinal"] < 0):
            raise ProtocolError(f"invalid selection ordinal: {identity}")
        if (isinstance(row.get("gate"), bool)
                or not isinstance(row.get("gate"), int)
                or isinstance(row.get("qubit"), bool)
                or not isinstance(row.get("qubit"), int)):
            raise ProtocolError(f"invalid tightness point identity: {identity}")
        if not isinstance(row.get("gate_name"), str):
            raise ProtocolError(f"invalid gate name: {identity}")
        if not isinstance(row.get("selection_bucket"), str):
            raise ProtocolError(f"invalid selection bucket: {identity}")

        det_rho = _finite_real(row.get("det_rho"), f"det_rho at {identity}")
        state_infidelity = _finite_real(
            row.get("state_infidelity"), f"state_infidelity at {identity}"
        )
        three_det_rho = _finite_real(
            row.get("three_det_rho"), f"three_det_rho at {identity}"
        )
        law_abs_error = _finite_real(
            row.get("law_abs_error"), f"law_abs_error at {identity}"
        )
        if not 0.0 <= det_rho <= 0.25 + 1e-12:
            raise ProtocolError(f"det(rho_q) outside physical range: {identity}")
        if not 0.0 <= state_infidelity <= 1.0 + 1e-12:
            raise ProtocolError(f"state infidelity outside range: {identity}")
        if three_det_rho < 0.0:
            raise ProtocolError(f"stored 3 det(rho_q) is negative: {identity}")
        if law_abs_error < 0.0:
            raise ProtocolError(f"stored law error is negative: {identity}")
        if abs(three_det_rho - 3.0 * det_rho) > 1e-15:
            raise ProtocolError(f"stored 3 det(rho_q) mismatch: {identity}")
        recomputed_error = abs(state_infidelity - three_det_rho)
        if not math.isclose(
            law_abs_error, recomputed_error, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ProtocolError(
                f"stored law_abs_error mismatch at {identity}: "
                f"{law_abs_error:.17g} != {recomputed_error:.17g}"
            )
        if law_abs_error > TIGHTNESS_LAW_TOL:
            raise ProtocolError(
                f"tightness law failed at {identity}: "
                f"{law_abs_error:.3e}"
            )
        expected_separable = det_rho <= LOCAL_SCORE_ZERO_TOL
        if row.get("numerically_separable") is not expected_separable:
            raise ProtocolError(
                f"numerically_separable mismatch at {identity}"
            )

    errors = []
    for key in TIGHTNESS_KEYS:
        status = statuses[key]
        frame_entry = next(
            entry for entry in frame["circuits"] if entry["circuit"] == key
        )
        if status.get("source_qasm_sha256") != frame_hashes[key]:
            raise ProtocolError(f"source hash drift in status for {key}")
        if (status.get("attempt_index") != frame_entry["attempt_index"]
                or status.get("attempt_id") != frame_entry["attempt_id"]):
            raise ProtocolError(f"attempt identity drift in status for {key}")
        if status.get("status") == "complete":
            count = status.get("selected_point_count")
            if count != len(records_by_circuit[key]):
                raise ProtocolError(
                    f"silent tightness row drop for {key}: "
                    f"expected {count}, got {len(records_by_circuit[key])}"
                )
            expected_hash = stable_hash([
                [row["gate"], row["qubit"], row["selection_bucket"]]
                for row in records_by_circuit[key]
            ])
            if status.get("selected_points_sha256") != expected_hash:
                raise ProtocolError(
                    f"selected-point checksum mismatch for {key}"
                )
        elif status.get("status") == "error":
            errors.append(key)
            if not isinstance(status.get("error"), dict):
                raise ProtocolError(f"missing structured error for {key}")
            if records_by_circuit[key]:
                raise ProtocolError(
                    f"error circuit {key} must not contribute partial rows"
                )
        else:
            raise ProtocolError(f"invalid terminal status for {key}")

    if require_complete and errors:
        raise ProtocolError(
            f"tightness artifact is not complete; errors={errors}"
        )
    if require_complete:
        _deep_validate_tightness_source_replay(
            frame, statuses, records_by_circuit
        )
    return artifact


def write_tightness_csv(path: Path | str, artifact: dict) -> None:
    rows = []
    for row in artifact["records"]:
        rows.append({
            "attempt_index": row["attempt_index"],
            "attempt_id": row["attempt_id"],
            "circuit": row["circuit"],
            "source_qasm_sha256": row["source_qasm_sha256"],
            "gate": row["gate"],
            "qubit": row["qubit"],
            "gate_name": row["gate_name"],
            "selection_bucket": row["selection_bucket"],
            "entanglement": row["det_rho"],
            "state_dist": row["state_infidelity"],
            "three_det_rho": row["three_det_rho"],
            "law_abs_error": row["law_abs_error"],
        })
    atomic_write_csv(path, list(rows[0]) if rows else [
        "attempt_index", "attempt_id", "circuit", "source_qasm_sha256",
        "gate", "qubit", "gate_name", "selection_bucket", "entanglement",
        "state_dist", "three_det_rho", "law_abs_error",
    ], rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed, exact 15-circuit RQ1 tightness stress test"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--out-json", type=Path, default=SCRIPT_DIR / "rq1_tightness.json"
    )
    parser.add_argument(
        "--out-csv", type=Path, default=SCRIPT_DIR / "ent_disturbance.csv"
    )
    args = parser.parse_args(argv)

    frame = resolve_circuit_corpus(args.manifest, args.qasm_dir)
    artifact = build_tightness_artifact(frame)
    atomic_write_json(args.out_json, artifact)
    try:
        validate_tightness_artifact(artifact, frame, require_complete=True)
    except ProtocolError as error:
        print(f"artifact rejected: {error}", file=sys.stderr)
        return 2
    write_tightness_csv(args.out_csv, artifact)
    print(
        f"wrote {args.out_json} and {args.out_csv}: "
        f"{artifact['record_count']} exact forced-monitor points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
