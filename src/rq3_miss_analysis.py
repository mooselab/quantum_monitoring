"""Descriptive miss analysis for the final RQ3 v4 artifact.

This module only reads a final artifact.  It never executes an RQ3 method,
redraws a mutation, repairs an artifact, or accepts a resumable or older
result.  The current evaluator's final-artifact validator is the only entry
check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from circuit_families import canonical_family_of
import rq3_realexec_eval as rq3
from rq3_manifest import dependency_versions, stable_hash


SCHEMA = "qmon-rq3-reviewer-miss-analysis-v3"
EXPECTED_INPUT_SCHEMA = "qmon-rq3-realexec-canonical-joint-shots-v4"
SEEDS = (1, 2, 3, 4, 5)
REPORTED_METHODS = ("qmon", "statistical", "proq", "liu", "mqt")
BASELINE_METHODS = ("statistical", "proq", "liu", "mqt")
SPARSE_DENOMINATOR = 20
WILSON_Z_95 = 1.959963984540054

SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR.parent / "protocols" / "RQ3_MISS_ANALYSIS_PROTOCOL.md"

STRATIFIERS = (
    "circuit_family",
    "original_gate_arity",
    "replacement_gate_arity",
    "original_operation_class",
    "replacement_operation_class",
    "operation_replacement_class",
    "scope_class",
    "structural_miss_class",
)


class MissAnalysisError(ValueError):
    """The input or a derived denominator violates the fixed analysis
    protocol."""


def protocol_descriptor() -> dict:
    return {
        "input": {
            "schema": EXPECTED_INPUT_SCHEMA,
            "arm": "mutant",
            "validation": (
                "rq3_realexec_eval.validate_final_rq3_artifact"
            ),
            "repair_or_legacy_upgrade": False,
        },
        "denominator": (
            "intersection of records complete for QMon and the four reported "
            "baselines, restricted to canonical non-output-equivalent attempts"
        ),
        "reported_methods": list(REPORTED_METHODS),
        "seeds": list(SEEDS),
        "rate": "QMon misses / common-complete non-output-equivalent attempts",
        "seed_summary": (
            "unweighted mean and sample SD of defined per-seed rates"
        ),
        "pooled_interval": {
            "method": "Wilson score",
            "confidence": 0.95,
            "z": WILSON_Z_95,
            "role": "descriptive",
        },
        "strata": list(STRATIFIERS),
        "sparse_rule": {
            "minimum_denominator": SPARSE_DENOMINATOR,
            "required_seed_coverage": len(SEEDS),
        },
        "exploratory_inference": {
            "performed": False,
            "p_values": False,
            "bh_correction": "not-applicable-no-inferential-comparisons",
        },
        "language_policy": (
            "descriptive structural associations only; no causal claims"
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pickle_sha256(value) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _family_of(circuit: str) -> str:
    return canonical_family_of(circuit)


def _method_boolean(record: dict, method: str, field: str) -> bool:
    value = record.get("methods", {}).get(method, {}).get(field)
    if not isinstance(value, bool):
        raise MissAnalysisError(
            f"{record.get('slot_id', '<unknown>')}/{method}: {field} must be boolean"
        )
    return value


def _method_complete(record: dict, method: str) -> bool:
    result = record.get("methods", {}).get(method)
    if result is None:
        return False
    value = result.get("complete")
    if not isinstance(value, bool):
        raise MissAnalysisError(
            f"{record.get('slot_id', '<unknown>')}/{method}: "
            "complete must be boolean"
        )
    return value


def _qmon_outcome_category(detections: dict[str, bool]) -> str:
    if set(detections) != set(REPORTED_METHODS) or any(
        not isinstance(value, bool) for value in detections.values()
    ):
        raise MissAnalysisError(
            "outcome classification requires five reported boolean detections"
        )
    if detections["qmon"]:
        return "qmon-detected"
    if any(detections[method] for method in BASELINE_METHODS):
        return "qmon-miss-caught-by-baseline"
    return "qmon-miss-missed-by-all-reported-methods"


def _structural_miss_class(scope_class: str, qmon_detected: bool) -> str:
    if qmon_detected:
        return "not-a-qmon-miss"
    mapping = {
        "outside-full-scope": "no-downstream-monitorable-checkpoint",
        "deployment-gap": "monitorable-checkpoint-not-deployed-under-qmax",
        "in-scope-qmon": "deployed-checkpoint-present-no-qmon-flag",
    }
    try:
        return mapping[scope_class]
    except KeyError as error:
        raise MissAnalysisError(
            f"unexpected RQ3 scope class: {scope_class!r}"
        ) from error


def _derive_attempt_rows(artifact: dict) -> list[dict]:
    rows = []
    for record in artifact["records"]:
        if record["equivalent"]:
            continue
        circuit = record["circuit"]
        complete_methods = [
            method for method in REPORTED_METHODS
            if _method_complete(record, method)
        ]
        common_complete = len(complete_methods) == len(REPORTED_METHODS)
        if common_complete:
            qmon_detected = _method_boolean(record, "qmon", "detected")
            detections = {
                method: _method_boolean(record, method, "detected")
                for method in REPORTED_METHODS
            }
            baseline_detected = any(
                detections[method] for method in BASELINE_METHODS
            )
            outcome_category = _qmon_outcome_category(detections)
            structural_miss_class = _structural_miss_class(
                record["scope_class"], qmon_detected
            )
        else:
            qmon_detected = None
            baseline_detected = None
            outcome_category = "excluded-incomplete"
            structural_miss_class = "excluded-incomplete"

        original = record["original"]
        replacement = record["replacement"]
        rows.append({
            "slot_id": record["slot_id"],
            "circuit": circuit,
            "seed": int(record["seed"]),
            "circuit_family": _family_of(circuit),
            "original_gate_arity": int(original["num_qubits"]),
            "replacement_gate_arity": int(replacement["num_qubits"]),
            "original_operation_class": str(original["name"]),
            "replacement_operation_class": str(replacement["name"]),
            "operation_replacement_class": (
                f"{original['name']}->{replacement['name']}"
            ),
            "scope_class": record["scope_class"],
            "complete_methods": complete_methods,
            "common_complete": common_complete,
            "qmon_detected": qmon_detected,
            "qmon_miss": (
                not qmon_detected if qmon_detected is not None else None
            ),
            "baseline_detected": baseline_detected,
            "reported_outcome_category": outcome_category,
            "structural_miss_class": structural_miss_class,
        })
    return rows


def _wilson_interval(misses: int, attempts: int) -> list[float] | None:
    if attempts == 0:
        return None
    p = misses / attempts
    z2 = WILSON_Z_95 ** 2
    denominator = 1.0 + z2 / attempts
    center = (p + z2 / (2.0 * attempts)) / denominator
    radius = (
        WILSON_Z_95
        * math.sqrt(p * (1.0 - p) / attempts + z2 / (4.0 * attempts ** 2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate_summary(rows: list[dict]) -> dict:
    per_seed = []
    defined_rates = []
    represented = 0
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        misses = sum(row["qmon_miss"] for row in selected)
        rate = misses / len(selected) if selected else None
        if rate is not None:
            represented += 1
            defined_rates.append(rate)
        per_seed.append({
            "seed": seed,
            "attempts": len(selected),
            "qmon_misses": misses,
            "qmon_detections": len(selected) - misses,
            "qmon_miss_rate": rate,
        })
    attempts = len(rows)
    misses = sum(row["qmon_miss"] for row in rows)
    sparse_reasons = []
    if attempts < SPARSE_DENOMINATOR:
        sparse_reasons.append(f"denominator<{SPARSE_DENOMINATOR}")
    if represented < len(SEEDS):
        sparse_reasons.append(f"represented_seeds<{len(SEEDS)}")
    return {
        "exact_counts": {
            "attempts": attempts,
            "qmon_misses": misses,
            "qmon_detections": attempts - misses,
        },
        "per_seed": per_seed,
        "per_seed_rate_mean": (
            statistics.mean(defined_rates) if defined_rates else None
        ),
        "per_seed_rate_sample_sd": (
            statistics.stdev(defined_rates) if len(defined_rates) >= 2 else None
        ),
        "defined_seed_rate_count": len(defined_rates),
        "pooled_qmon_miss_rate": misses / attempts if attempts else None,
        "pooled_qmon_miss_rate_wilson_95": _wilson_interval(misses, attempts),
        "sparse": bool(sparse_reasons),
        "sparse_reasons": sparse_reasons,
        "interpretation": (
            "descriptive-only; sparse stratum" if sparse_reasons
            else "descriptive-only"
        ),
    }


def _stratum_sort_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summarize_rows(rows: list[dict]) -> tuple[dict, list[dict]]:
    attempted = list(rows)
    common = [row for row in attempted if row.get("common_complete") is True]
    if not common:
        raise MissAnalysisError("no common-complete non-output-equivalent attempts")
    unknown_seeds = sorted({row.get("seed") for row in attempted} - set(SEEDS))
    if unknown_seeds:
        raise MissAnalysisError(f"unexpected seed values: {unknown_seeds}")
    enriched = copy.deepcopy(common)
    strata = {}
    for field in STRATIFIERS:
        groups = defaultdict(list)
        for row in enriched:
            value = row.get(field)
            if value is None:
                value = "not-available"
            groups[_stratum_sort_key(value)].append(row)
        summaries = []
        for key in sorted(groups):
            group = groups[key]
            summaries.append({
                "value": json.loads(key),
                **_rate_summary(group),
            })
        strata[field] = summaries
    return strata, enriched


def _miss_breakdown(rows: list[dict]) -> dict[str, list[dict]]:
    dimensions = (
        "circuit_family",
        "original_operation_class",
        "original_gate_arity",
        "replacement_operation_class",
        "replacement_gate_arity",
        "operation_replacement_class",
        "scope_class",
        "structural_miss_class",
    )
    total_misses = sum(row["qmon_miss"] for row in rows)
    output: dict[str, list[dict]] = {}
    for field in dimensions:
        groups: dict[str, list[dict]] = defaultdict(list)
        values: dict[str, object] = {}
        for row in rows:
            value = row[field]
            key = _stratum_sort_key(value)
            groups[key].append(row)
            values[key] = value
        summaries = []
        for key in sorted(groups):
            summary = _rate_summary(groups[key])
            misses = summary["exact_counts"]["qmon_misses"]
            if misses == 0:
                continue
            summaries.append({
                "value": values[key],
                "share_of_all_qmon_misses": (
                    misses / total_misses if total_misses else None
                ),
                **summary,
            })
        summaries.sort(
            key=lambda row: (
                -row["exact_counts"]["qmon_misses"],
                _stratum_sort_key(row["value"]),
            )
        )
        output[field] = summaries
    return output


def _source_fingerprint() -> dict:
    files = []
    for path in (Path(__file__).resolve(), PROTOCOL_PATH):
        if not path.is_file():
            raise MissAnalysisError(f"required analysis source file is missing: {path}")
        files.append({"path": path.name, "sha256": _file_sha256(path)})
    return {"files": files, "sha256": stable_hash(files)}


def _environment_fingerprint() -> dict:
    value = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }
    return {"value": value, "sha256": stable_hash(value)}


def validate_final_artifact(artifact: dict, source: str = "RQ3 artifact") -> dict:
    if rq3.RQ3_ARTIFACT_SCHEMA != EXPECTED_INPUT_SCHEMA:
        raise MissAnalysisError(
            "current evaluator schema differs from the RQ3 v4 schema this "
            "analysis expects"
        )
    if not isinstance(artifact, dict) or artifact.get("schema") != EXPECTED_INPUT_SCHEMA:
        raise MissAnalysisError(
            f"{source}: expected final {EXPECTED_INPUT_SCHEMA}; legacy, partial, "
            "or unversioned artifacts are forbidden"
        )
    # Do not catch or translate validator errors: checksum, corpus, oracle,
    # method, and completeness failures must remain visible to the caller.
    rq3.validate_final_rq3_artifact(artifact, source)
    if artifact.get("protocol", {}).get("arm") != "mutant":
        raise MissAnalysisError(f"{source}: miss analysis requires the mutant arm")
    fingerprints = artifact.get("fingerprints")
    for field in ("sources", "qasm_corpus", "dependencies", "protocol", "circuit_budget"):
        digest = (
            fingerprints.get(field, {}).get("sha256")
            if isinstance(fingerprints, dict) else None
        )
        if not isinstance(digest, str) or len(digest) != 64:
            raise MissAnalysisError(f"{source}: malformed {field} checksum")
    return artifact


def analyze_artifact(
    artifact: dict,
    *,
    source: str = "RQ3 artifact",
    artifact_file_sha256: str | None = None,
) -> dict:
    validate_final_artifact(artifact, source)
    rows = _derive_attempt_rows(artifact)
    if {row["seed"] for row in rows} != set(SEEDS):
        raise MissAnalysisError("final artifact does not retain all five seeds")
    strata, enriched_rows = _summarize_rows(rows)

    method_complete = {
        method: sum(method in row["complete_methods"] for row in rows)
        for method in REPORTED_METHODS
    }
    equivalent_count = sum(record["equivalent"] for record in artifact["records"])
    denominator_ids = sorted(row["slot_id"] for row in enriched_rows)
    excluded_incomplete_ids = sorted(
        row["slot_id"]
        for row in rows
        if row["common_complete"] is not True
    )
    artifact_fingerprints = {
        field: artifact["fingerprints"][field]["sha256"]
        for field in ("sources", "qasm_corpus", "dependencies", "protocol", "circuit_budget")
    }
    report = {
        "schema": SCHEMA,
        "input_contract": {
            "schema": artifact["schema"],
            "arm": artifact["protocol"]["arm"],
            "validator": "rq3_realexec_eval.validate_final_rq3_artifact",
            "validation_result": "accepted-final-terminal-inventory",
        },
        "analysis_protocol": protocol_descriptor(),
        "denominators": {
            "all_canonical_records": len(artifact["records"]),
            "output_equivalent_excluded": equivalent_count,
            "attempted_non_output_equivalent": len(rows),
            "common_complete_non_output_equivalent": len(enriched_rows),
            "excluded_incomplete_non_output_equivalent": len(
                excluded_incomplete_ids
            ),
            "method_complete_non_output_equivalent": method_complete,
            "common_denominator_slot_ids_sha256": stable_hash(denominator_ids),
            "excluded_incomplete_slot_ids_sha256": stable_hash(
                excluded_incomplete_ids
            ),
            "seed_attempt_counts": {
                str(seed): sum(row["seed"] == seed for row in enriched_rows)
                for seed in SEEDS
            },
        },
        "overall": _rate_summary(enriched_rows),
        "miss_breakdown": _miss_breakdown(enriched_rows),
        "strata": strata,
        "attempts": sorted(enriched_rows, key=lambda row: row["slot_id"]),
        "exploratory_inference": {
            "performed": False,
            "comparisons": [],
            "p_values": [],
            "bh_correction": "not-applicable-no-inferential-comparisons",
        },
        "fingerprints": {
            "source": _source_fingerprint(),
            "data": {
                "artifact_file_sha256": artifact_file_sha256,
                "artifact_object_pickle_sha256": _pickle_sha256(artifact),
                "artifact_fingerprints": artifact_fingerprints,
                "canonical_manifest_sha256": artifact["protocol"][
                    "canonical_manifest_sha256"
                ],
                "analysis_rows_sha256": stable_hash(enriched_rows),
            },
            "environment": _environment_fingerprint(),
            "analysis_protocol": {
                "sha256": stable_hash(protocol_descriptor())
            },
        },
    }
    report["report_sha256"] = stable_hash(report)
    return report


def load_and_analyze(path: str | os.PathLike[str]) -> dict:
    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    artifact = pickle.loads(raw)
    return analyze_artifact(
        artifact,
        source=str(artifact_path),
        artifact_file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def write_json(path: str | os.PathLike[str], report: dict) -> None:
    payload = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict descriptive miss analysis for a final RQ3 v4 mutant artifact."
    )
    parser.add_argument("artifact", help="final validated RQ3 v4 mutant pickle")
    parser.add_argument("output", help="deterministic JSON output path")
    args = parser.parse_args(argv)
    if Path(args.artifact).resolve() == Path(args.output).resolve():
        parser.error("output must not overwrite the validated input artifact")
    report = load_and_analyze(args.artifact)
    write_json(args.output, report)
    print(
        f"wrote {args.output}: {report['overall']['exact_counts']['qmon_misses']}/"
        f"{report['overall']['exact_counts']['attempts']} QMon misses; "
        f"report_sha256={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
