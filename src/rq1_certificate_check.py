"""RQ1 selector cross-check and tightness analysis.

The selector criterion and the determinant diagnostic are intentionally kept
separate:

* selection is exactly ``a2(i,q) <= 1e-12``, where ``a2`` is the second
  Schmidt amplitude used by the implemented selector; and
* ``p0*p1 - |<s0|s1>|^2`` is recomputed independently and stored as a raw,
  cancellation-prone determinant residual.  It is never treated as the
  selector threshold.

The default ``all`` command also runs the predeclared 15-circuit exact
tightness stress test from ``entanglement_disturbance.py``.  Both artifacts
stop with an error on any inconsistency and are bound to the fixed
310-circuit source corpus.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.setrecursionlimit(1_000_000)

from circuits_selection import _single_vs_rest_entangled
from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    EXPECTED_CIRCUITS as FIXED_CORPUS_CIRCUITS,
    EXPECTED_CORPUS_SHA256 as FIXED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256 as FIXED_MANIFEST_SHA256,
    ProtocolError,
    SCRIPT_DIR,
    atomic_write_json,
    build_tightness_artifact,
    file_sha256,
    resolve_circuit_corpus,
    stable_hash,
    validate_tightness_artifact,
    write_tightness_csv,
)
from fast_analysis import SKIP_OPS, analyze_per_gate_nolock


SCHEMA = "qmon-rq1-certificate-v4"
SELECTOR_A2_TOL = 1e-12
SOURCE_REPLAY_FLOAT_TOL = 2e-13
EXPECTED_ACTED_POINT_COUNT = 56_386
EXPECTED_SELECTED_POINT_COUNT = 22_332
EXPECTED_SELECTED_IDENTITY_SHA256 = (
    "0253fc962d61f4a832264d3c5dfd1d681e0f93b7c473bce161de1561c7250c11"
)
EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256 = (
    "fb14fe55d95c2df6484951457a9cd3be4526ffdd8da12722d0dab879b1183fd5"
)


def _selected_points(paths: dict) -> set[tuple[int, int]]:
    return {
        (int(gate_index), int(qubit))
        for qubit, gate_indices in paths.items()
        for gate_index in gate_indices
    }


def crosscheck_selector(qc: QuantumCircuit) -> dict:
    """Recompute every acted-on point and cross-check the implemented selector.

    The standard selector supplies the selected location set. A
    second statevector sweep re-evaluates the same Schmidt predicate and also
    computes the independent determinant residual.  Any identity mismatch is
    a hard error rather than an adjusted denominator.
    """

    selector_res, _, selector_paths = analyze_per_gate_nolock(
        qc, tol=SELECTOR_A2_TOL
    )
    selector_selected = _selected_points(selector_paths)

    n = qc.num_qubits
    state = Statevector.from_int(0, dims=2 ** n)
    recomputed_selected = set()
    all_rows = []
    acted_point_count = 0

    for gate_index, instruction in enumerate(qc.data):
        operation = instruction.operation
        if operation.name in SKIP_OPS:
            continue
        acted = [qc.find_bit(bit).index for bit in instruction.qubits]
        state = state.evolve(operation, qargs=acted)
        amplitudes = state.data.reshape((2,) * n)

        for qubit in acted:
            acted_point_count += 1
            axis = n - 1 - qubit
            s0 = np.take(amplitudes, 0, axis=axis).reshape(-1)
            s1 = np.take(amplitudes, 1, axis=axis).reshape(-1)
            entangled, selector_a2, _ = _single_vs_rest_entangled(
                state, qubit, SELECTOR_A2_TOL
            )
            singular_values = np.linalg.svd(
                np.stack((s0, s1), axis=0), compute_uv=False
            )
            independent_a2 = float(
                singular_values[1] if len(singular_values) > 1 else 0.0
            )
            independent_entangled = independent_a2 > SELECTOR_A2_TOL
            if entangled != independent_entangled:
                raise ProtocolError(
                    "selector/SVD threshold disagreement at "
                    f"{(gate_index, qubit)}: selector_a2={selector_a2:.17g}, "
                    f"independent_a2={independent_a2:.17g}"
                )
            p0 = float(np.vdot(s0, s0).real)
            p1 = float(np.vdot(s1, s1).real)
            determinant_residual = float(
                p0 * p1 - abs(np.vdot(s0, s1)) ** 2
            )
            if determinant_residual < -1e-12:
                raise ProtocolError(
                    "independent determinant is materially negative at "
                    f"{(gate_index, qubit)}: {determinant_residual:.17g}"
                )
            determinant_svd = float(
                singular_values[0] ** 2 * independent_a2 ** 2
            )
            row = {
                "gate": int(gate_index),
                "qubit": int(qubit),
                "gate_name": operation.name,
                "selector_a2": float(selector_a2),
                "independent_svd_a2": independent_a2,
                "a2_crosscheck_abs_diff": float(
                    abs(selector_a2 - independent_a2)
                ),
                "selector_passed": bool(not entangled),
                "p0": p0,
                "p1": p1,
                "determinant_residual": determinant_residual,
                "det_rho_svd": determinant_svd,
                "determinant_residual_minus_svd": float(
                    determinant_residual - determinant_svd
                ),
            }
            all_rows.append(row)
            if not entangled:
                recomputed_selected.add((gate_index, qubit))

    if selector_selected != recomputed_selected:
        missing = sorted(selector_selected - recomputed_selected)
        extra = sorted(recomputed_selected - selector_selected)
        raise ProtocolError(
            "selector cross-check mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    all_rows.sort(key=lambda row: (row["gate"], row["qubit"]))
    selected_rows = [row.copy() for row in all_rows if row["selector_passed"]]
    return {
        "acted_point_count": acted_point_count,
        "all_points": all_rows,
        "selected_points": selected_rows,
        "selector_res": selector_res,
        "selector_paths": selector_paths,
    }


def crosscheck_selected_points(qc: QuantumCircuit) -> tuple[int, list[dict]]:
    """Compatibility wrapper returning the selected subset only."""

    validation = crosscheck_selector(qc)
    return validation["acted_point_count"], validation["selected_points"]


def evaluate_circuit(entry: dict, qasm_dir: Path) -> tuple[dict, list[dict]]:
    key = entry["circuit"]
    qc = QuantumCircuit.from_qasm_file(str(qasm_dir / key))
    validation = crosscheck_selector(qc)
    acted_point_count = validation["acted_point_count"]
    local_rows = validation["selected_points"]
    for row in local_rows:
        row.update({
            "attempt_index": entry["attempt_index"],
            "attempt_id": entry["attempt_id"],
            "circuit": key,
            "source_qasm_sha256": entry["source_qasm_sha256"],
        })

    status = {
        "status": "complete",
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "qubits": qc.num_qubits,
        "acted_point_count": acted_point_count,
        "selected_point_count": len(local_rows),
        "selected_points_sha256": stable_hash([
            [row["gate"], row["qubit"]]
            for row in local_rows
        ]),
    }
    return status, local_rows


def _error_status(entry: dict, error: Exception) -> dict:
    return {
        "status": "error",
        "attempt_index": entry["attempt_index"],
        "attempt_id": entry["attempt_id"],
        "source_qasm_sha256": entry["source_qasm_sha256"],
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _certificate_summary(records: list[dict]) -> dict:
    a2_values = [row["selector_a2"] for row in records]
    independent_a2_values = [row["independent_svd_a2"] for row in records]
    a2_differences = [row["a2_crosscheck_abs_diff"] for row in records]
    residuals = [row["determinant_residual"] for row in records]
    determinant_differences = [
        row["determinant_residual_minus_svd"] for row in records
    ]
    return {
        "selected_point_count": len(records),
        "selected_identity_sha256": stable_hash([
            [row["circuit"], row["gate"], row["qubit"]]
            for row in records
        ]),
        "source_bound_selected_identity_sha256": stable_hash([
            [row["source_qasm_sha256"], row["gate"], row["qubit"]]
            for row in records
        ]),
        "max_selector_a2": max(a2_values, default=0.0),
        "max_independent_svd_a2": max(independent_a2_values, default=0.0),
        "max_a2_crosscheck_abs_diff": max(a2_differences, default=0.0),
        "determinant_residual_diagnostic": {
            "definition": "p0*p1-|<s0|s1>|^2",
            "selector_criterion": False,
            "max_raw": max(residuals, default=0.0),
            "min_raw": min(residuals, default=0.0),
            "max_absolute": max((abs(value) for value in residuals), default=0.0),
            "max_abs_difference_from_svd": max(
                (abs(value) for value in determinant_differences), default=0.0
            ),
            "count_abs_gt_1e-12": sum(
                abs(value) > 1e-12 for value in residuals
            ),
            "count_abs_gt_1e-9": sum(
                abs(value) > 1e-9 for value in residuals
            ),
        },
    }


def certificate_protocol_descriptor() -> dict:
    """Return the exact RQ1 selector protocol descriptor."""

    return {
        "selector": "analyze_per_gate_nolock",
        "selector_criterion": {
            "quantity": "second Schmidt amplitude a2(i,q)",
            "operator": "<=",
            "threshold": SELECTOR_A2_TOL,
        },
        "independent_cross_check": (
            "direct singular values of the 2 x 2^(n-1) amplitude matrix; "
            "the <=1e-12 decision and selected identity set must agree "
            "with the implemented selector; both amplitudes and their "
            "floating-point difference are retained"
        ),
        "determinant_residual": {
            "definition": "p0*p1-|<s0|s1>|^2",
            "role": (
                "independent cancellation-prone floating-point diagnostic; "
                "not the selector criterion"
            ),
        },
        "point_universe": "every gate x qubit acted on by that gate",
        "silent_skips": "forbidden",
        "publication_validation": (
            "require_complete replays every source QASM through both the "
            "implemented selector and the independent direct-SVD check, then "
            "checks every selected row, terminal status, and summary"
        ),
        "source_replay": {
            "identity_and_status_comparison": "exact",
            "numeric_absolute_tolerance": SOURCE_REPLAY_FLOAT_TOL,
            "rationale": (
                "bounds observed ARM/Accelerate versus x86/OpenBLAS roundoff; "
                "selected identities remain exact"
            ),
        },
        "expected_result": {
            "acted_point_count": EXPECTED_ACTED_POINT_COUNT,
            "selected_point_count": EXPECTED_SELECTED_POINT_COUNT,
            "selected_identity_sha256": EXPECTED_SELECTED_IDENTITY_SHA256,
            "source_bound_selected_identity_sha256": (
                EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256
            ),
        },
    }


def build_certificate_artifact(frame: dict) -> dict:
    qasm_dir = Path(frame["qasm_dir"])
    statuses = {}
    records = []
    for position, entry in enumerate(frame["circuits"], start=1):
        key = entry["circuit"]
        try:
            status, local_rows = evaluate_circuit(entry, qasm_dir)
        except Exception as error:
            statuses[key] = _error_status(entry, error)
        else:
            statuses[key] = status
            records.extend(local_rows)
        if position % 25 == 0 or position == frame["circuit_count"]:
            print(
                f"RQ1 certificate check: {position}/{frame['circuit_count']}",
                flush=True,
            )

    records.sort(key=lambda row: (
        row["attempt_index"], row["gate"], row["qubit"]
    ))
    artifact = {
        "schema": SCHEMA,
        "protocol": certificate_protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "circuit_count": frame["circuit_count"],
        },
        "circuit_status": statuses,
        "summary": _certificate_summary(records),
        "record_count": len(records),
        "records_sha256": stable_hash(records),
        "records": records,
    }
    validate_certificate_artifact(artifact, frame, require_complete=False)
    return artifact


def _finite_real(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ProtocolError(f"{label} must be a finite real number")
    return float(value)


def _require_source_close(actual, expected, field: str, identity) -> None:
    actual = _finite_real(actual, f"{field} at {identity}")
    expected = _finite_real(expected, f"replayed {field} at {identity}")
    if not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=SOURCE_REPLAY_FLOAT_TOL
    ):
        raise ProtocolError(
            f"source replay mismatch for {field} at {identity}: "
            f"{actual:.17g} != {expected:.17g}"
        )


def _deep_validate_certificate_source_replay(
    frame: dict, statuses: dict, records_by_circuit: dict
) -> None:
    qasm_dir_value = frame.get("qasm_dir")
    if not isinstance(qasm_dir_value, str) or not qasm_dir_value:
        raise ProtocolError(
            "complete certificate validation requires frame qasm_dir"
        )
    qasm_dir = Path(qasm_dir_value)
    frame_entries = {
        entry["circuit"]: entry for entry in frame["circuits"]
    }
    status_fields = (
        "status", "attempt_index", "attempt_id", "source_qasm_sha256",
        "qubits", "acted_point_count", "selected_point_count",
        "selected_points_sha256",
    )
    exact_fields = (
        "attempt_index", "attempt_id", "circuit", "source_qasm_sha256",
        "gate", "qubit", "gate_name", "selector_passed",
    )
    numeric_fields = (
        "selector_a2", "independent_svd_a2", "a2_crosscheck_abs_diff",
        "p0", "p1", "determinant_residual", "det_rho_svd",
        "determinant_residual_minus_svd",
    )

    for key, entry in frame_entries.items():
        qasm_path = qasm_dir / key
        if not qasm_path.is_file():
            raise ProtocolError(f"certificate source QASM missing for {key}")
        if file_sha256(qasm_path) != entry["source_qasm_sha256"]:
            raise ProtocolError(f"certificate source QASM hash drift for {key}")

        expected_status, expected_rows = evaluate_circuit(entry, qasm_dir)
        actual_status = statuses[key]
        for field in status_fields:
            if actual_status.get(field) != expected_status.get(field):
                raise ProtocolError(
                    f"source replay mismatch for status {field} in {key}: "
                    f"{actual_status.get(field)!r} != "
                    f"{expected_status.get(field)!r}"
                )

        actual_rows = sorted(
            records_by_circuit[key], key=lambda row: (row["gate"], row["qubit"])
        )
        expected_rows = sorted(
            expected_rows, key=lambda row: (row["gate"], row["qubit"])
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
                _require_source_close(
                    actual.get(field), expected.get(field), field, identity
                )


def validate_certificate_artifact(
    artifact: dict, frame: dict, *, require_complete: bool = True
) -> dict:
    if artifact.get("schema") != SCHEMA:
        raise ProtocolError("wrong RQ1 certificate artifact schema")
    if artifact.get("protocol") != certificate_protocol_descriptor():
        raise ProtocolError("RQ1 certificate protocol drift")
    statuses = artifact.get("circuit_status")
    records = artifact.get("records")
    if not isinstance(statuses, dict) or not isinstance(records, list):
        raise ProtocolError("certificate artifact lacks statuses or records")

    expected_corpus = {
        "manifest_sha256": frame.get("manifest_sha256"),
        "corpus_sha256": frame.get("corpus_sha256"),
        "circuit_count": frame.get("circuit_count", len(frame["circuits"])),
    }
    if artifact.get("corpus") != expected_corpus:
        raise ProtocolError(
            f"certificate corpus provenance mismatch: "
            f"{artifact.get('corpus')} != {expected_corpus}"
        )

    frame_entries = {entry["circuit"]: entry for entry in frame["circuits"]}
    if set(statuses) != set(frame_entries):
        missing = sorted(set(frame_entries) - set(statuses))
        extra = sorted(set(statuses) - set(frame_entries))
        raise ProtocolError(
            f"certificate terminal accounting mismatch: missing={missing}, "
            f"extra={extra}"
        )
    if artifact.get("record_count") != len(records):
        raise ProtocolError("certificate record_count does not match records")
    if artifact.get("records_sha256") != stable_hash(records):
        raise ProtocolError("certificate records checksum mismatch")

    records_by_circuit = {key: [] for key in frame_entries}
    identities = set()
    for row in records:
        key = row.get("circuit")
        if key not in frame_entries:
            raise ProtocolError(f"unexpected certificate circuit: {key}")
        entry = frame_entries[key]
        identity = (key, row.get("gate"), row.get("qubit"))
        if identity in identities:
            raise ProtocolError(f"duplicate selected point: {identity}")
        identities.add(identity)
        records_by_circuit[key].append(row)
        for field in ("attempt_index", "attempt_id", "source_qasm_sha256"):
            expected = entry[field]
            if row.get(field) != expected:
                raise ProtocolError(
                    f"{field} drift in selected point {identity}: "
                    f"{row.get(field)!r} != {expected!r}"
                )
        a2 = row.get("selector_a2")
        residual = row.get("determinant_residual")
        determinant_svd = row.get("det_rho_svd")
        independent_a2 = row.get("independent_svd_a2")
        a2 = _finite_real(a2, f"selector a2 at {identity}")
        if a2 < -1e-15 or a2 > SELECTOR_A2_TOL:
            raise ProtocolError(
                f"selected point violates a2 criterion at {identity}: {a2}"
            )
        independent_a2 = _finite_real(
            independent_a2, f"independent SVD a2 at {identity}"
        )
        if independent_a2 < -1e-15 or independent_a2 > SELECTOR_A2_TOL:
            raise ProtocolError(
                f"selected point violates independent SVD criterion at "
                f"{identity}: {independent_a2}"
            )
        difference = row.get("a2_crosscheck_abs_diff")
        difference = _finite_real(
            difference, f"a2 cross-check difference at {identity}"
        )
        if not math.isclose(
                    difference, abs(a2 - independent_a2),
                    rel_tol=0.0, abs_tol=1e-18,
                ):
            raise ProtocolError(f"stored a2 cross-check mismatch at {identity}")
        if row.get("selector_passed") is not True:
            raise ProtocolError(f"selected point lacks selector_passed at {identity}")
        residual = _finite_real(
            residual, f"determinant residual at {identity}"
        )
        determinant_svd = _finite_real(
            determinant_svd, f"SVD determinant at {identity}"
        )
        if not 0.0 <= determinant_svd <= 0.25 + 1e-12:
            raise ProtocolError(f"invalid SVD determinant at {identity}")
        determinant_difference = row.get("determinant_residual_minus_svd")
        determinant_difference = _finite_real(
            determinant_difference,
            f"determinant diagnostic difference at {identity}",
        )
        if not math.isclose(
                    determinant_difference, residual - determinant_svd,
                    rel_tol=0.0, abs_tol=1e-18,
                ):
            raise ProtocolError(
                f"determinant diagnostic difference mismatch at {identity}"
            )
        p0 = _finite_real(row.get("p0"), f"p0 at {identity}")
        p1 = _finite_real(row.get("p1"), f"p1 at {identity}")
        if not 0.0 <= p0 <= 1.0 + 1e-12:
            raise ProtocolError(f"p0 outside probability range at {identity}")
        if not 0.0 <= p1 <= 1.0 + 1e-12:
            raise ProtocolError(f"p1 outside probability range at {identity}")
        if not math.isclose(p0 + p1, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ProtocolError(f"p0+p1 normalization mismatch at {identity}")
        if not isinstance(row.get("gate_name"), str):
            raise ProtocolError(f"invalid gate name at {identity}")
        # Deliberately no residual-vs-selector threshold assertion here.

    errors = []
    for key, entry in frame_entries.items():
        status = statuses[key]
        for field in ("attempt_index", "attempt_id", "source_qasm_sha256"):
            if status.get(field) != entry[field]:
                raise ProtocolError(f"{field} drift in status for {key}")
        if status.get("status") == "complete":
            local_rows = records_by_circuit[key]
            if status.get("selected_point_count") != len(local_rows):
                raise ProtocolError(
                    f"silent selected-row drop for {key}: expected "
                    f"{status.get('selected_point_count')}, got {len(local_rows)}"
                )
            expected_hash = stable_hash([
                [row["gate"], row["qubit"]]
                for row in local_rows
            ])
            if status.get("selected_points_sha256") != expected_hash:
                raise ProtocolError(
                    f"selected-point checksum mismatch for {key}"
                )
            if status.get("acted_point_count", -1) < len(local_rows):
                raise ProtocolError(f"impossible acted-point count for {key}")
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

    summary = artifact.get("summary")
    if summary != _certificate_summary(records):
        raise ProtocolError("certificate summary is not derived from records")
    if require_complete and errors:
        raise ProtocolError(
            f"certificate artifact is not complete; errors={errors}"
        )
    if require_complete:
        _deep_validate_certificate_source_replay(
            frame, statuses, records_by_circuit
        )
    reference_frame = (
        frame.get("circuit_count") == FIXED_CORPUS_CIRCUITS
        and frame.get("manifest_sha256") == FIXED_MANIFEST_SHA256
        and frame.get("corpus_sha256") == FIXED_CORPUS_SHA256
    )
    if require_complete and reference_frame:
        acted_count = sum(
            status["acted_point_count"] for status in statuses.values()
        )
        expected = {
            "acted_point_count": EXPECTED_ACTED_POINT_COUNT,
            "selected_point_count": EXPECTED_SELECTED_POINT_COUNT,
            "selected_identity_sha256": EXPECTED_SELECTED_IDENTITY_SHA256,
            "source_bound_selected_identity_sha256": (
                EXPECTED_SOURCE_BOUND_SELECTED_IDENTITY_SHA256
            ),
        }
        actual = {
            "acted_point_count": acted_count,
            "selected_point_count": summary["selected_point_count"],
            "selected_identity_sha256": summary["selected_identity_sha256"],
            "source_bound_selected_identity_sha256": summary[
                "source_bound_selected_identity_sha256"
            ],
        }
        if actual != expected:
            raise ProtocolError(
                f"reference a2 selector result drift: {actual} != {expected}"
            )
    return artifact


def _run_certificate(frame: dict, output: Path) -> str | None:
    artifact = build_certificate_artifact(frame)
    atomic_write_json(output, artifact)
    try:
        validate_certificate_artifact(artifact, frame, require_complete=True)
    except ProtocolError as error:
        return str(error)
    print(
        f"wrote {output}: {artifact['record_count']} selected points over "
        f"{frame['circuit_count']} circuits"
    )
    return None


def _run_tightness(frame: dict, output: Path, csv_output: Path) -> str | None:
    artifact = build_tightness_artifact(frame)
    atomic_write_json(output, artifact)
    try:
        validate_tightness_artifact(artifact, frame, require_complete=True)
    except ProtocolError as error:
        return str(error)
    write_tightness_csv(csv_output, artifact)
    print(
        f"wrote {output} and {csv_output}: "
        f"{artifact['record_count']} exact forced-monitor points"
    )
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the final RQ1 certificate and tightness check"
    )
    parser.add_argument(
        "--component", choices=("all", "certificate", "tightness"),
        default="all",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--certificate-json", type=Path,
        default=SCRIPT_DIR / "rq1_certificate.json",
    )
    parser.add_argument(
        "--tightness-json", type=Path,
        default=SCRIPT_DIR / "rq1_tightness.json",
    )
    parser.add_argument(
        "--tightness-csv", type=Path,
        default=SCRIPT_DIR / "ent_disturbance.csv",
    )
    args = parser.parse_args(argv)

    frame = resolve_circuit_corpus(args.manifest, args.qasm_dir)
    failures = []
    if args.component in {"all", "certificate"}:
        failure = _run_certificate(frame, args.certificate_json)
        if failure:
            failures.append(f"certificate: {failure}")
    if args.component in {"all", "tightness"}:
        failure = _run_tightness(
            frame, args.tightness_json, args.tightness_csv
        )
        if failure:
            failures.append(f"tightness: {failure}")
    if failures:
        for failure in failures:
            print(f"artifact rejected: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
