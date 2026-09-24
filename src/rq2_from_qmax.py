"""Extract the RQ2 Qmax=24 results from a validated Qmax sweep.

The generated RQ2 file has the same structure as output produced directly by
``coverage_rq2``. A separate summary records the input and output checksums.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import coverage_rq2 as rq2
import qmax_sensitivity as qmax
from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    EXPECTED_CIRCUITS,
    ProtocolError,
    atomic_write_json,
    file_sha256,
    resolve_circuit_corpus,
    stable_hash,
)


SUMMARY_SCHEMA = "qmon.rq2-from-qmax.derivation-summary.v1"
PROTOCOL_PATH = SCRIPT_DIR.parent / "protocols" / "RQ2_FROM_QMAX_PROTOCOL.md"
CONVERTER_SOURCE_FILES = (Path(__file__).resolve(), PROTOCOL_PATH)

# This is the exact dictionary contract constructed by
# coverage_rq2.evaluate_circuit. An explicit allow-list rejects additions,
# removals, and accidental sensitivity-field leakage.
NATIVE_RECORD_FIELDS = (
    "attempt_index",
    "attempt_id",
    "circuit",
    "source_qasm_sha256",
    "family",
    "qmax",
    "num_qubits",
    "depth",
    "gate_count",
    "acted_point_count",
    "two_qubit_density",
    "E_mean",
    "E_peak",
    "coverage_denominators",
    "gate_node_denominator",
    "gate_qubit_event_denominator",
    "qubit_denominator",
    "normalized_depth_denominator",
    "acted_points",
    "certificate_points",
    "numerical_candidate_points",
    "prebudget_mid_points",
    "final_read_points",
    "emitted_mid_points",
    "infeasible_mid_points",
    "deployed_observable_points",
    "group_demands",
    "partition_provenance",
    "individually_feasible_gate_groups",
    "individually_infeasible_gate_groups",
    "batches",
    "infeasible_gate_groups",
    "certificate_selected_point_count",
    "certificate_selected_gate_count",
    "certificate_selected_qubit_count",
    "certificate_acted_point_coverage",
    "certificate_gate_coverage",
    "certificate_qubit_coverage",
    "certificate_depth_reach",
    "emitted_mid_point_count",
    "emitted_mid_gate_count",
    "emitted_mid_qubit_count",
    "emitted_mid_acted_point_coverage",
    "emitted_mid_gate_coverage",
    "emitted_mid_qubit_coverage",
    "emitted_mid_depth_reach",
    "final_read_point_count",
    "deployed_observable_point_count",
    "deployed_observable_gate_count",
    "deployed_observable_qubit_count",
    "deployed_observable_acted_point_coverage",
    "deployed_observable_gate_coverage",
    "deployed_observable_qubit_coverage",
    "deployed_observable_depth_reach",
    "infeasible_mid_point_count",
    "infeasible_gate_group_count",
    "batch_count",
    "source_executable_operation_count",
    "source_executable_depth",
    "selector_max_a2",
    "selector_max_independent_svd_a2",
    "selected_identity_sha256",
    "emitted_identity_sha256",
    "deployed_observable_identity_sha256",
)

SENSITIVITY_FIELDS = (
    "certificate_candidate_points",
    "certificate_candidate_groups",
    "certificate_candidate_point_count",
    "certificate_candidate_group_count",
    "mid_candidate_points",
    "mid_candidate_groups",
    "mid_candidate_point_count",
    "mid_candidate_group_count",
    "individually_feasible_mid_points",
    "individually_infeasible_mid_points",
    "individually_feasible_mid_groups",
    "individually_infeasible_mid_groups",
    "individually_feasible_mid_point_count",
    "individually_infeasible_mid_point_count",
    "individually_feasible_mid_group_count",
    "individually_infeasible_mid_group_count",
    "deployed_observable_candidate_fraction",
    "individually_feasible_mid_point_fraction",
    "individually_feasible_mid_group_fraction",
    "fully_deployable",
    "batches",
    "instrumented_batch_count",
    "source_only_final_read_run_count",
    "deployment_run_count",
    "batch_monitor_point_counts",
    "batch_monitor_group_counts",
    "peak_instrumented_width",
    "peak_instrumented_ancilla_count",
    "peak_deployment_run_width",
    "peak_deployment_run_ancilla_count",
    "cumulative_extra_ancilla_allocations",
    "planned_original_gate_executions",
    "planned_replay_gate_executions",
    "planned_monitor_measurement_operations",
    "planned_monitor_reset_operations",
    "planned_work_Bm_plus_replay",
    "planned_total_counted_operations",
    "actual_emitted_executable_operation_executions",
    "source_executable_depth",
    "peak_deployment_run_depth",
    "cumulative_deployment_run_depth",
    "peak_depth_increase",
    "peak_depth_inflation_ratio",
    "partition_provenance",
    "complete_denominators",
    "plan_semantics",
)

NATIVE_BATCH_FIELDS = (
    "batch",
    "gate_groups",
    "ancilla_cost",
    "emitted_total_qubits",
    "metadata_sha256",
    "monitor_groups",
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
    "depth_inflation_ratio",
)

QMAX_RECORD_FIELDS = frozenset({
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
if QMAX_RECORD_FIELDS != qmax.RECORD_FIELDS:
    raise RuntimeError("RQ2 converter Qmax terminal-record contract drift")


@dataclass(frozen=True)
class SourceSweep:
    """Decoded source data and its checksums."""

    path: Path
    container_schema: str
    file_sha256: str
    records_sha256: str
    run_id: str
    records: list[dict[str, Any]]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"strict JSON rejects duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> None:
    raise ProtocolError(f"strict JSON rejects non-finite number {token}")


def _strict_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ProtocolError(f"strict JSON rejects non-finite number {token}")
    return value


def _strict_json(text: str, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=_strict_float,
        )
    except ProtocolError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ProtocolError(f"{source}: invalid strict JSON: {error}") from error


def read_source_sweep(path: Path | str) -> SourceSweep:
    """Read current-schema JSON or JSONL Qmax results."""

    source = Path(path).resolve()
    if source.name == "qmax_records.pkl" or source.suffix.lower() in {
        ".pkl",
        ".pickle",
    }:
        raise ProtocolError("legacy Qmax pickle input is unconditionally rejected")
    if not source.is_file():
        raise ProtocolError(f"Qmax source artifact is not a file: {source}")
    try:
        text = source.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError(f"{source}: Qmax artifact is not strict UTF-8") from error

    suffix = source.suffix.lower()
    if suffix == ".json":
        value = _strict_json(text, str(source))
        expected_keys = {
            "schema",
            "record_schema",
            "run_id",
            "record_count",
            "records_sha256",
            "records",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ProtocolError("JSON input is not a current Qmax record-set envelope")
        if value["schema"] != qmax.RECORD_SET_SCHEMA:
            raise ProtocolError("legacy or wrong Qmax record-set schema")
        if value["record_schema"] != qmax.SCHEMA:
            raise ProtocolError("legacy or wrong Qmax record schema")
        records = value["records"]
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ProtocolError("Qmax JSON envelope lacks object records")
        if value["record_count"] != len(records):
            raise ProtocolError("Qmax JSON envelope record_count mismatch")
        if value["records_sha256"] != stable_hash(records):
            raise ProtocolError("Qmax JSON envelope records checksum mismatch")
        container_schema = qmax.RECORD_SET_SCHEMA
        envelope_run_id = value["run_id"]
    elif suffix == ".jsonl":
        if not text or not text.endswith("\n"):
            raise ProtocolError("Qmax JSONL must be nonempty and newline-terminated")
        lines = text.splitlines()
        if any(not line for line in lines):
            raise ProtocolError("Qmax JSONL contains a blank record line")
        records = []
        for line_number, line in enumerate(lines, start=1):
            value = _strict_json(line, f"{source}:{line_number}")
            if not isinstance(value, dict):
                raise ProtocolError(f"{source}:{line_number}: record is not an object")
            records.append(value)
        container_schema = f"{qmax.SCHEMA}.jsonl"
        envelope_run_id = None
    else:
        raise ProtocolError("Qmax source must be current .json or .jsonl, never pickle")

    run_ids = {
        record.get("run", {}).get("run_id")
        for record in records
        if isinstance(record.get("run"), Mapping)
    }
    if len(run_ids) != 1:
        raise ProtocolError("Qmax source records do not share one run_id")
    run_id = qmax._require_run_id(next(iter(run_ids)))
    if envelope_run_id is not None and envelope_run_id != run_id:
        raise ProtocolError("Qmax record-set envelope run_id mismatch")

    return SourceSweep(
        path=source,
        container_schema=container_schema,
        file_sha256=file_sha256(source),
        records_sha256=stable_hash(records),
        run_id=run_id,
        records=records,
    )


def _native_batch(batch: Mapping[str, Any], circuit: str) -> dict[str, Any]:
    missing = [field for field in NATIVE_BATCH_FIELDS if field not in batch]
    if missing:
        raise ProtocolError(f"Qmax batch lacks native fields for {circuit}: {missing}")
    expected_augmented = set(NATIVE_BATCH_FIELDS) | {
        "monitor_group_count",
        "monitor_point_count",
        "monitored_points",
    }
    if set(batch) != expected_augmented:
        raise ProtocolError(f"Qmax batch field contract drift for {circuit}")
    return {field: copy.deepcopy(batch[field]) for field in NATIVE_BATCH_FIELDS}


def _validate_qmax_record_envelope(record: Mapping[str, Any]) -> None:
    attempt_id = record.get("attempt_id")
    if set(record) != QMAX_RECORD_FIELDS:
        missing = sorted(QMAX_RECORD_FIELDS - set(record))
        extra = sorted(set(record) - QMAX_RECORD_FIELDS)
        raise ProtocolError(
            f"Qmax terminal-record field contract drift for {attempt_id!r}: "
            f"missing={missing}, extra={extra}"
        )
    run = record.get("run")
    if not isinstance(run, Mapping) or set(run) != {
        "shard_index", "shard_count", "run_id"
    }:
        raise ProtocolError(f"Qmax run metadata contract drift for {attempt_id!r}")
    shard_index = run["shard_index"]
    shard_count = run["shard_count"]
    if (
        type(shard_index) is not int
        or type(shard_count) is not int
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
    ):
        raise ProtocolError(f"invalid Qmax shard metadata for {attempt_id!r}")
    qmax._require_run_id(run["run_id"])
    for field in ("started_at_utc", "completed_at_utc"):
        value = record.get(field)
        if not isinstance(value, str):
            raise ProtocolError(f"missing Qmax UTC timestamp {field} for {attempt_id!r}")
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError as error:
            raise ProtocolError(
                f"invalid Qmax UTC timestamp {field} for {attempt_id!r}"
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
            raise ProtocolError(f"non-UTC Qmax timestamp {field} for {attempt_id!r}")


def _project_native_record(
    result: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = set(NATIVE_RECORD_FIELDS) | set(SENSITIVITY_FIELDS)
    if set(result) != expected_fields:
        missing = sorted(expected_fields - set(result))
        extra = sorted(set(result) - expected_fields)
        raise ProtocolError(
            f"Qmax=24 result field contract drift for {entry['circuit']}: "
            f"missing={missing}, extra={extra}"
        )

    record = {
        field: copy.deepcopy(result[field])
        for field in NATIVE_RECORD_FIELDS
        if field != "batches"
    }
    batches = result["batches"]
    if not isinstance(batches, list) or not all(
        isinstance(batch, Mapping) for batch in batches
    ):
        raise ProtocolError(f"malformed Qmax batches for {entry['circuit']}")
    record["batches"] = [
        _native_batch(batch, entry["circuit"]) for batch in batches
    ]
    for field in ("attempt_index", "attempt_id", "circuit", "source_qasm_sha256"):
        record[field] = entry[field]
    return record


def derive_native_artifact(
    records: Sequence[Mapping[str, Any]],
    frame: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    *,
    require_canonical: bool = True,
    expected_run_id: str = qmax.DEFAULT_RUN_ID,
) -> dict[str, Any]:
    """Validate the sweep and construct the standard RQ2 Qmax=24 result."""

    bundle = qmax.build_fingerprint_bundle(frame) if bundle is None else bundle
    attempts = qmax.build_attempt_grid(frame, require_canonical=require_canonical)
    if require_canonical:
        if frame.get("circuit_count") != EXPECTED_CIRCUITS:
            raise ProtocolError(f"derivation requires exactly {EXPECTED_CIRCUITS} circuits")
        if len(attempts) != qmax.EXPECTED_GRID_RECORDS:
            raise ProtocolError(
                f"derivation requires exactly {qmax.EXPECTED_GRID_RECORDS} attempts"
            )

    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Qmax terminal record is not an object")
        _validate_qmax_record_envelope(record)

    qmax.validate_record_set(
        records,
        attempts,
        frame,
        bundle,
        require_complete=True,
        require_success=True,
        expected_run_id=expected_run_id,
        deep=True,
    )
    ordered = sorted(records, key=lambda record: record["attempt_index"])
    qmax._validate_cross_budget_invariants(ordered)

    headline = [record for record in ordered if record.get("qmax") == rq2.QMAX]
    counts = Counter(record.get("circuit") for record in headline)
    expected_circuits = [entry["circuit"] for entry in frame["circuits"]]
    if len(headline) != len(expected_circuits) or set(counts) != set(expected_circuits):
        raise ProtocolError("Qmax=24 extraction does not cover the exact circuit frame")
    duplicates = sorted(circuit for circuit, count in counts.items() if count != 1)
    if duplicates:
        raise ProtocolError(f"Qmax=24 extraction is not one-per-circuit: {duplicates}")

    headline_by_circuit = {record["circuit"]: record for record in headline}
    native_records = []
    statuses: dict[str, dict[str, Any]] = {}
    qasm_dir = Path(frame["qasm_dir"])
    for entry in frame["circuits"]:
        source_record = headline_by_circuit[entry["circuit"]]
        if source_record.get("status") != "complete":
            raise ProtocolError(f"non-success Qmax=24 row for {entry['circuit']}")
        result = source_record.get("result")
        if not isinstance(result, Mapping):
            raise ProtocolError(f"missing Qmax=24 result for {entry['circuit']}")
        native = _project_native_record(result, entry)
        rq2._validate_record(native, entry, qasm_dir)
        native_records.append(native)
        statuses[entry["circuit"]] = {
            "status": "complete",
            "attempt_index": entry["attempt_index"],
            "attempt_id": entry["attempt_id"],
            "source_qasm_sha256": entry["source_qasm_sha256"],
            "record_sha256": stable_hash(native),
        }

    native_records.sort(key=lambda record: record["attempt_index"])
    artifact = {
        "schema": rq2.SCHEMA,
        "protocol": rq2.protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "circuit_count": frame["circuit_count"],
        },
        "circuit_status": statuses,
        "record_count": len(native_records),
        "records_sha256": stable_hash(native_records),
        "summary": rq2.summarize(native_records),
        "records": native_records,
    }
    return rq2.validate_artifact(artifact, frame, require_complete=True)


def _fingerprinted(value: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(value)
    return {"sha256": stable_hash(material), "value": material}


def build_derivation_summary(
    source: SourceSweep,
    ordered_records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    artifact: Mapping[str, Any],
    out_json: Path | str,
    out_csv: Path | str,
) -> dict[str, Any]:
    """Describe the validated input and generated RQ2 files."""

    missing = [str(path) for path in CONVERTER_SOURCE_FILES if not path.is_file()]
    if missing:
        raise ProtocolError(f"derivation source files missing: {missing}")
    source_files = [
        {"path": path.name, "sha256": file_sha256(path)}
        for path in CONVERTER_SOURCE_FILES
    ]
    dependencies = qmax.dependency_versions()
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "derivation": {
            "headline_qmax": rq2.QMAX,
            "run_id": source.run_id,
            "source_record_schema": qmax.SCHEMA,
            "native_artifact_schema": rq2.SCHEMA,
            "complete_grid_record_count": len(ordered_records),
            "extracted_record_count": len(artifact["records"]),
        },
        "source_sweep": {
            "basename": source.path.name,
            "run_id": source.run_id,
            "container_schema": source.container_schema,
            "file_sha256": source.file_sha256,
            "loaded_records_sha256": source.records_sha256,
            "canonical_records_sha256": stable_hash(list(ordered_records)),
            "fingerprint_bundle_sha256": stable_hash(bundle),
            "fingerprints": copy.deepcopy(dict(bundle)),
        },
        "converter": {
            "sha256": stable_hash(source_files),
            "files": source_files,
        },
        "dependencies": _fingerprinted(dependencies),
        "environment": _fingerprinted(environment),
        "outputs": {
            "native_artifact_sha256": stable_hash(artifact),
            "native_json_file_sha256": file_sha256(out_json),
            "native_csv_file_sha256": file_sha256(out_csv),
            "records_sha256": artifact["records_sha256"],
        },
    }
    summary["summary_sha256"] = stable_hash(summary)
    return summary


def _ensure_output_paths(
    paths: Sequence[Path], overwrite: bool, *, source: Path | None = None
) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ProtocolError("JSON, CSV, and summary outputs must be distinct paths")
    if source is not None and source.resolve() in set(resolved):
        raise ProtocolError("an output path must not alias the source sweep")
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ProtocolError(f"output exists; pass --overwrite: {existing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract RQ2 coverage from the Qmax=24 sensitivity rows"
    )
    parser.add_argument("source_sweep", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--run-id",
        default=qmax.DEFAULT_RUN_ID,
        help="expected run identity embedded in every Qmax source record",
    )
    parser.add_argument(
        "--out-json", type=Path, default=SCRIPT_DIR / "coverage_rq2.json"
    )
    parser.add_argument(
        "--out-csv", type=Path, default=SCRIPT_DIR / "coverage.csv"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SCRIPT_DIR / "coverage_rq2.derivation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _ensure_output_paths(
            [args.out_json, args.out_csv, args.summary],
            args.overwrite,
            source=args.source_sweep,
        )
        source = read_source_sweep(args.source_sweep)

        # This resolution is independent of every identity and hash declared by
        # the source sweep.  The bundle is rebuilt from the fixed corpus on disk.
        frame = resolve_circuit_corpus(args.manifest, args.qasm_dir)
        bundle = qmax.build_fingerprint_bundle(frame)
        run_id = qmax._require_run_id(args.run_id)
        if source.run_id != run_id:
            raise ProtocolError(
                f"Qmax source run_id mismatch: {source.run_id!r} "
                f"!= {run_id!r}"
            )
        artifact = derive_native_artifact(
            source.records,
            frame,
            bundle,
            expected_run_id=run_id,
        )

        atomic_write_json(args.out_json, artifact)
        rq2.write_csv(args.out_csv, artifact)
        ordered = sorted(source.records, key=lambda record: record["attempt_index"])
        summary = build_derivation_summary(
            source, ordered, bundle, artifact, args.out_json, args.out_csv
        )
        atomic_write_json(args.summary, summary)
        print(
            f"derived {args.out_json}, {args.out_csv}, and {args.summary}: "
            f"{artifact['record_count']} native Qmax={rq2.QMAX} records"
        )
        return 0
    except (ProtocolError, OSError, ValueError, TypeError) as error:
        print(f"RQ2-from-Qmax derivation rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
