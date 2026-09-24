"""Summarize MQT outcomes stored in an RQ3 mutant-result file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "qmon-rq3-mqt-diagnostics-projection-v1"
RQ3_SCHEMA = "qmon-rq3-realexec-canonical-joint-shots-v4"
CHECKSUM_FIELD = "projection_sha256"
_STATS_FIELDS = (
    "n_affected",
    "n_applicable",
    "run_fail",
    "run_unevaluable",
    "timeout",
    "error",
    "budget_exhausted",
    "applicability_budget-exhausted",
)


class MQTDiagnosticsError(ValueError):
    """MQT result rows do not have the expected diagnostic format."""


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mqt_result(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = row.get("methods", {}).get("mqt")
    if not isinstance(result, Mapping):
        raise MQTDiagnosticsError("RQ3 record lacks an MQT result")
    stats = result.get("stats")
    if not isinstance(stats, Mapping):
        raise MQTDiagnosticsError("MQT result lacks execution statistics")
    if type(result.get("complete")) is not bool:
        raise MQTDiagnosticsError("MQT completion status must be Boolean")
    reasons = stats.get("incomplete_reasons", [])
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise MQTDiagnosticsError("MQT incomplete reasons are malformed")
    if result["complete"] and reasons:
        raise MQTDiagnosticsError("complete MQT result carries incomplete reasons")
    if not result["complete"] and not reasons:
        raise MQTDiagnosticsError("incomplete MQT result lacks a reason")
    return result


def _counter(stats: Mapping[str, Any], field: str) -> int:
    value = stats.get(field, 0)
    if type(value) is not int or value < 0:
        raise MQTDiagnosticsError(f"invalid MQT counter: {field}")
    return value


def summarize(
    attempted_records: Sequence[Mapping[str, Any]],
    paired_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired flags separately from all-attempt availability."""

    attempted = list(attempted_records)
    paired = list(paired_records)
    attempted_ids = {row.get("slot_id") for row in attempted}
    paired_ids = {row.get("slot_id") for row in paired}
    if (
        None in attempted_ids
        or len(attempted_ids) != len(attempted)
        or None in paired_ids
        or len(paired_ids) != len(paired)
        or not paired_ids.issubset(attempted_ids)
    ):
        raise MQTDiagnosticsError("RQ3 slot identities are missing or inconsistent")

    attempted_results = [_mqt_result(row) for row in attempted]
    paired_results = [_mqt_result(row) for row in paired]
    paired_stats = [result["stats"] for result in paired_results]
    if any(not result["complete"] for result in paired_results):
        raise MQTDiagnosticsError("paired RQ3 row has incomplete MQT output")

    assertion_failures = sum(
        _counter(stats, "run_fail") for stats in paired_stats
    )
    unevaluable = sum(
        _counter(stats, "run_unevaluable") for stats in paired_stats
    )
    combined = assertion_failures + unevaluable
    detected = sum(result.get("detected") is True for result in paired_results)
    if combined != detected:
        raise MQTDiagnosticsError(
            "paired MQT fail/unevaluable counters disagree with detections"
        )

    seeds = sorted({row.get("seed") for row in paired})
    if any(type(seed) is not int for seed in seeds):
        raise MQTDiagnosticsError("paired RQ3 rows have invalid seed values")

    def seed_rate_summary(kind: str) -> dict[str, Any]:
        rows = []
        for seed in seeds:
            indices = [
                index for index, row in enumerate(paired)
                if row.get("seed") == seed
            ]
            denominator = len(indices)
            if kind == "state_mismatch":
                numerator = sum(
                    _counter(paired_stats[index], "run_fail")
                    for index in indices
                )
            elif kind == "unevaluable_or_refusal":
                numerator = sum(
                    _counter(paired_stats[index], "run_unevaluable")
                    for index in indices
                )
            else:
                numerator = sum(
                    paired_results[index].get("detected") is True
                    for index in indices
                )
            rows.append(
                {
                    "seed": seed,
                    "numerator": numerator,
                    "denominator": denominator,
                    "rate_percent": 100.0 * numerator / denominator,
                }
            )
        rates = [row["rate_percent"] for row in rows]
        return {
            "pooled_numerator": sum(row["numerator"] for row in rows),
            "pooled_denominator": sum(row["denominator"] for row in rows),
            "per_seed": rows,
            "unweighted_mean_percent": (
                statistics.mean(rates) if rates else None
            ),
            "sample_standard_deviation_percent": (
                statistics.stdev(rates) if len(rates) > 1
                else 0.0 if rates
                else None
            ),
        }

    incomplete_results = [
        result for result in attempted_results if not result["complete"]
    ]

    def reason_flags(result: Mapping[str, Any]) -> tuple[bool, bool, bool]:
        reasons = result["stats"].get("incomplete_reasons", [])
        timeout = any(reason.startswith("run-timeout:") for reason in reasons)
        budget = any("budget-exhausted" in reason for reason in reasons)
        error = any(reason.startswith("run-error:") for reason in reasons)
        return timeout, budget, error

    flags = [reason_flags(result) for result in incomplete_results]
    timeout_records = sum(timeout for timeout, _, _ in flags)
    budget_records = sum(budget for _, budget, _ in flags)
    error_records = sum(error for _, _, error in flags)
    overlap_records = sum(
        timeout and budget for timeout, budget, _ in flags
    )
    other_records = sum(
        not timeout and not budget and not error
        for timeout, budget, error in flags
    )
    attempted_stats = [result["stats"] for result in attempted_results]
    return {
        "paired_common_complete_non_output_equivalent_records": len(paired),
        "paired_affected_checkpoints": sum(
            _counter(stats, "n_affected") for stats in paired_stats
        ),
        "paired_applicable_checkpoints": sum(
            _counter(stats, "n_applicable") for stats in paired_stats
        ),
        "paired_assertion_failures": assertion_failures,
        "paired_unevaluable_or_refusal_outcomes": unevaluable,
        "paired_combined_fail_or_unevaluable_flags": combined,
        "per_seed_rates": {
            "state_mismatch": seed_rate_summary("state_mismatch"),
            "unevaluable_or_refusal": seed_rate_summary(
                "unevaluable_or_refusal"
            ),
            "combined_flag": seed_rate_summary("combined_flag"),
        },
        "availability": {
            "attempted_non_output_equivalent_records": len(attempted),
            "complete_records": sum(
                result["complete"] for result in attempted_results
            ),
            "terminal_incomplete_records": len(incomplete_results),
            "records_with_run_timeout": timeout_records,
            "records_with_budget_exhaustion": budget_records,
            "records_with_tool_error": error_records,
            "records_with_timeout_and_budget_exhaustion": overlap_records,
            "records_with_other_incomplete_reason": other_records,
            "run_timeout_events": sum(
                _counter(stats, "timeout") for stats in attempted_stats
            ),
            "run_error_events": sum(
                _counter(stats, "error") for stats in attempted_stats
            ),
            "run_budget_exhausted_events": sum(
                _counter(stats, "budget_exhausted")
                for stats in attempted_stats
            ),
            "applicability_budget_exhausted_events": sum(
                _counter(stats, "applicability_budget-exhausted")
                for stats in attempted_stats
            ),
        },
    }


def build_projection(
    attempted_records: Sequence[Mapping[str, Any]],
    paired_records: Sequence[Mapping[str, Any]],
    *,
    source_artifact_file_sha256: str,
) -> dict[str, Any]:
    """Create a compact checksummed projection sufficient to replay diagnostics."""

    paired_ids = {row.get("slot_id") for row in paired_records}
    projected = []
    for row in attempted_records:
        result = _mqt_result(row)
        stats = result["stats"]
        projected.append(
            {
                "slot_id": row.get("slot_id"),
                "circuit": row.get("circuit"),
                "seed": row.get("seed"),
                "paired_common_complete": row.get("slot_id") in paired_ids,
                "methods": {
                    "mqt": {
                        "complete": result["complete"],
                        "detected": result.get("detected"),
                        "stats": {
                            **{
                                field: _counter(stats, field)
                                for field in _STATS_FIELDS
                            },
                            "incomplete_reasons": copy.deepcopy(
                                stats.get("incomplete_reasons", [])
                            ),
                        },
                    }
                },
            }
        )
    paired_projected = [
        row for row in projected if row["paired_common_complete"]
    ]
    projection = {
        "schema": SCHEMA,
        "source_artifact_file_sha256": source_artifact_file_sha256,
        "record_count": len(projected),
        "records_sha256": stable_hash(projected),
        "records": projected,
        "summary": summarize(projected, paired_projected),
    }
    projection[CHECKSUM_FIELD] = stable_hash(projection)
    return projection


def validate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a projection and independently recompute its summary."""

    if value.get("schema") != SCHEMA:
        raise MQTDiagnosticsError("wrong MQT diagnostic projection schema")
    without_checksum = dict(value)
    checksum = without_checksum.pop(CHECKSUM_FIELD, None)
    if checksum != stable_hash(without_checksum):
        raise MQTDiagnosticsError("MQT diagnostic projection checksum mismatch")
    records = value.get("records")
    if (
        not isinstance(records, list)
        or value.get("record_count") != len(records)
        or value.get("records_sha256") != stable_hash(records)
    ):
        raise MQTDiagnosticsError("MQT diagnostic projection records drift")
    paired = [row for row in records if row.get("paired_common_complete") is True]
    recomputed = summarize(records, paired)
    if recomputed != value.get("summary"):
        raise MQTDiagnosticsError("MQT diagnostic projection summary drift")
    return recomputed


def projection_from_rq3_file(path: Path | str) -> dict[str, Any]:
    """Build diagnostics from the non-output-equivalent RQ3 records."""

    source = Path(path)
    payload = source.read_bytes()
    value = pickle.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != RQ3_SCHEMA:
        raise MQTDiagnosticsError("input is not an RQ3 v4 result file")
    if value.get("protocol", {}).get("arm") != "mutant":
        raise MQTDiagnosticsError("input is not the mutant arm")
    records = value.get("records")
    if not isinstance(records, list):
        raise MQTDiagnosticsError("RQ3 result file has no record list")

    attempted = [row for row in records if row.get("equivalent") is False]
    methods = tuple(value.get("protocol", {}).get("methods", ()))
    if not methods or "mqt" not in methods:
        raise MQTDiagnosticsError("RQ3 result file has no MQT method")
    paired = [
        row
        for row in attempted
        if all(
            row.get("methods", {}).get(method, {}).get("complete") is True
            for method in methods
        )
    ]
    return build_projection(
        attempted,
        paired,
        source_artifact_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize MQT outcomes in an RQ3 mutant-result file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    projection = projection_from_rq3_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(projection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(projection["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
