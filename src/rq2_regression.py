"""Circuit-level regression companion analysis for RQ2 coverage.

The command consumes a *completed and currently valid* ``coverage_rq2`` JSON
artifact.  A CSV may be supplied as a second representation; when present it
must be the exact projection written by ``coverage_rq2.write_csv``.  CSV-only
analysis is deliberately unsupported because CSV carries no protocol,
terminal-status, row-hash, or corpus-provenance checksums.

Primary models are fractional-logit quasi-maximum-likelihood regressions with
family-clustered sandwich uncertainty and deterministic family bootstrap
intervals.  Arcsine-square-root OLS with family-clustered uncertainty is a
bounded-outcome sensitivity analysis, not a replacement estimand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.stats import t as student_t

import coverage_rq2
from entanglement_disturbance import (
    DEFAULT_MANIFEST,
    DEFAULT_QASM_DIR,
    ProtocolError,
    atomic_write_json,
    file_sha256,
    resolve_circuit_corpus,
    stable_hash,
)


SCHEMA = "qmon-rq2-regression-v3"
SOURCE_SCHEMA = coverage_rq2.SCHEMA
DEFAULT_SEED = 20260714
DEFAULT_BOOTSTRAP_REPLICATES = 2000
PUBLICATION_MIN_BOOTSTRAP_REPLICATES = 1000
ARTIFACT_CHECKSUM_FIELD = "artifact_sha256"
PROTOCOL_PATH = (
    Path(__file__).resolve().parent.parent
    / "protocols"
    / "RQ2_REGRESSION_PROTOCOL.md"
)
BFGS_SUCCESS_MESSAGE = "Optimization terminated successfully."
BFGS_PRECISION_LOSS_MESSAGE = (
    "Desired error not necessarily achieved due to precision loss."
)
BFGS_SCORE_TOLERANCE = 1e-7
FINAL_SCORE_TOLERANCE = BFGS_SCORE_TOLERANCE
NEWTON_TARGET_SCORE_TOLERANCE = 1e-10
NEWTON_MAX_ITERATIONS = 25
NEWTON_MAX_BACKTRACKS = 40
PREDICTORS = ("E_mean", "E_peak", "depth", "two_qubit_density", "num_qubits")
COVERAGE_OUTCOMES = (
    "deployed_observable_gate_coverage",
    "deployed_observable_qubit_coverage",
    "deployed_observable_depth_reach",
)
OUTCOMES = (
    "deployed_observable_gate_coverage",
    "deployed_observable_depth_reach",
)
DESCRIPTIVE_ONLY_OUTCOMES = {
    "deployed_observable_qubit_coverage": (
        "reported descriptively because ceiling saturation leaves too few "
        "families with within-family variation for reliable family-aware "
        "five-predictor inference"
    ),
}
OUTCOME_SEED_OFFSETS = {
    outcome: 104729 * index for index, outcome in enumerate(COVERAGE_OUTCOMES)
}
OUTCOME_IDENTITIES = {
    "deployed_observable_gate_coverage": (
        "deployed_observable_gate_count",
        "gate_node_denominator",
    ),
    "deployed_observable_qubit_coverage": (
        "deployed_observable_qubit_count",
        "qubit_denominator",
    ),
}
SHA256_HEX_LENGTH = 64


class RegressionProtocolError(ProtocolError):
    """Raised when validation or a required statistical fit fails closed."""


def protocol_descriptor() -> dict:
    """Machine-readable specification mirrored by RQ2_REGRESSION_PROTOCOL.md."""

    return {
        "analysis_unit": "one completed circuit row",
        "source": {
            "schema": SOURCE_SCHEMA,
            "validation": (
                "coverage_rq2.validate_artifact(require_complete=True) against "
                "the fixed 310-circuit corpus"
            ),
            "csv_policy": (
                "optional exact cross-check only; CSV-only input is forbidden"
            ),
            "publication_scope": "deployed_observable coverage only",
        },
        "outcomes": {
            "validated_coverage": list(COVERAGE_OUTCOMES),
            "modeled": list(OUTCOMES),
            "descriptive_only": dict(DESCRIPTIVE_ONLY_OUTCOMES),
        },
        "predictors": list(PREDICTORS),
        "predictor_scaling": (
            "sample mean centered and sample-SD scaled once on the full data; "
            "the same scaling is retained in resamples and deletion diagnostics"
        ),
        "primary_model": {
            "name": "fractional logit QMLE",
            "mean": "E[y|x] = logistic(beta0 + x beta)",
            "criterion": "Bernoulli quasi-log-likelihood for y in [0,1]",
            "uncertainty": (
                "family-clustered sandwich covariance with finite-cluster "
                "correction and t(G-1) reference"
            ),
            "intervals": [
                "two-sided family-clustered Wald",
                "two-sided percentile nonparametric family-cluster bootstrap",
            ],
            "coefficient_scale": "log-odds change per one predictor sample SD",
            "optimizer_acceptance": (
                "finite BFGS candidate with either normal convergence or the "
                "known precision-loss status, followed for every fit by "
                "deterministic exact-Hessian Newton polishing; finite final "
                "parameters and objective, maximum absolute score at most 1e-7; "
                "the Newton polish targets 1e-10 and records floating-point "
                "stagnation within the unchanged 1e-7 acceptance gate"
            ),
        },
        "sensitivity_model": {
            "name": "arcsine-square-root transformed OLS",
            "transform": "asin(sqrt(y)); endpoints retained without continuity correction",
            "uncertainty": (
                "family-clustered sandwich covariance with finite-cluster "
                "correction and t(G-1) reference"
            ),
            "coefficient_scale": (
                "radian change in asin(sqrt(coverage)) per one predictor sample SD"
            ),
        },
        "multiplicity": {
            "family": "both modeled outcomes x 5 non-intercept primary coefficients",
            "procedure": "Benjamini-Hochberg false-discovery-rate adjustment",
            "reported": ["raw two-sided p", "BH-adjusted q"],
        },
        "multicollinearity": [
            "predictor Pearson correlation matrix",
            "variance inflation factors",
            "standardized design condition number",
        ],
        "influence": [
            "fractional-logit leverage, Pearson residual, and generalized Cook distance",
            "leave-one-family-out maximum coefficient displacement",
        ],
        "bootstrap": {
            "sampling_unit": "circuit family",
            "default_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "seed": DEFAULT_SEED,
            "minimum_success_fraction": 0.95,
        },
        "result_validation": {
            "checksum_field": ARTIFACT_CHECKSUM_FIELD,
            "validation": (
                "parse the JSON, verify the input and implementation checksums, "
                "and recompute all regression results"
            ),
        },
        "interpretation": (
            "associational only; coefficients are not causal effects"
        ),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegressionProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_strict_json(path: Path, *, label: str = "JSON artifact") -> tuple[dict, bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RegressionProtocolError(f"cannot read {label} {path}: {error}") from error
    try:
        artifact = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RegressionProtocolError(f"non-finite JSON constant: {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise RegressionProtocolError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise RegressionProtocolError(f"malformed {label}: {error}") from error
    if not isinstance(artifact, dict):
        raise RegressionProtocolError(f"{label} root must be an object")
    return artifact, payload


def _validate_json_tree(value: object, *, path: str = "artifact") -> None:
    """Reject values that strict JSON cannot faithfully represent."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegressionProtocolError(f"non-finite field at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RegressionProtocolError(f"non-string JSON key at {path}")
            _validate_json_tree(item, path=f"{path}.{key}")
        return
    raise RegressionProtocolError(
        f"non-JSON field type at {path}: {type(value).__name__}"
    )


def _add_artifact_checksum(artifact: dict) -> dict:
    """Apply the deterministic top-level content checksum in place."""

    artifact[ARTIFACT_CHECKSUM_FIELD] = None
    payload = dict(artifact)
    payload.pop(ARTIFACT_CHECKSUM_FIELD)
    _validate_json_tree(payload)
    artifact[ARTIFACT_CHECKSUM_FIELD] = stable_hash(payload)
    return artifact


def _check_artifact_checksum(artifact: dict, *, source: str) -> None:
    stored = artifact.get(ARTIFACT_CHECKSUM_FIELD)
    payload = dict(artifact)
    payload.pop(ARTIFACT_CHECKSUM_FIELD, None)
    try:
        expected = stable_hash(payload)
    except (TypeError, ValueError) as error:
        raise RegressionProtocolError(f"{source}: artifact is not strict JSON") from error
    if not _is_sha256(stored) or stored != expected:
        raise RegressionProtocolError(f"{source}: artifact checksum mismatch")


def _require_int(record: dict, field: str, *, minimum: int = 0) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RegressionProtocolError(
            f"{field} must be an integer >= {minimum} in {record.get('circuit')}"
        )
    return value


def _require_float(
    record: dict,
    field: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegressionProtocolError(
            f"{field} must be numeric in {record.get('circuit')}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise RegressionProtocolError(f"non-finite {field} in {record.get('circuit')}")
    if lower is not None and value < lower:
        raise RegressionProtocolError(f"{field} below {lower} in {record.get('circuit')}")
    if upper is not None and value > upper:
        raise RegressionProtocolError(f"{field} above {upper} in {record.get('circuit')}")
    return value


def _frame_provenance(frame: dict) -> dict:
    return {
        "manifest_sha256": frame.get("manifest_sha256"),
        "corpus_sha256": frame.get("corpus_sha256"),
        "circuit_count": frame.get("circuit_count", len(frame.get("circuits", ()))),
    }


def _validate_regression_rows(artifact: dict, frame: dict) -> None:
    """Independently check every field that enters a regression or checksum."""

    if artifact.get("schema") != SOURCE_SCHEMA or SOURCE_SCHEMA != coverage_rq2.SCHEMA:
        raise RegressionProtocolError("legacy or unknown coverage_rq2 schema")
    if artifact.get("protocol") != coverage_rq2.protocol_descriptor():
        raise RegressionProtocolError("coverage_rq2 protocol provenance drift")
    if artifact.get("corpus") != _frame_provenance(frame):
        raise RegressionProtocolError("coverage_rq2 corpus provenance drift")

    records = artifact.get("records")
    statuses = artifact.get("circuit_status")
    if not isinstance(records, list) or not isinstance(statuses, dict):
        raise RegressionProtocolError("coverage artifact lacks records or statuses")
    expected_count = frame.get("circuit_count", len(frame.get("circuits", ())))
    if artifact.get("record_count") != expected_count or len(records) != expected_count:
        raise RegressionProtocolError("coverage artifact is not complete for final reporting")
    if artifact.get("records_sha256") != stable_hash(records):
        raise RegressionProtocolError("coverage records checksum mismatch")

    entries = frame.get("circuits")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise RegressionProtocolError("resolved corpus frame is malformed")
    entry_by_circuit = {entry.get("circuit"): entry for entry in entries}
    if len(entry_by_circuit) != expected_count or set(statuses) != set(entry_by_circuit):
        raise RegressionProtocolError("coverage terminal accounting drift")

    observed_circuits = set()
    observed_attempts = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise RegressionProtocolError(f"coverage record {position} is not an object")
        circuit = record.get("circuit")
        attempt_index = _require_int(record, "attempt_index")
        if circuit not in entry_by_circuit or circuit in observed_circuits:
            raise RegressionProtocolError(f"duplicate or unexpected circuit row: {circuit}")
        if attempt_index in observed_attempts or attempt_index != position:
            raise RegressionProtocolError("coverage records are not in attempt order")
        observed_circuits.add(circuit)
        observed_attempts.add(attempt_index)

        entry = entry_by_circuit[circuit]
        for field in ("attempt_index", "attempt_id", "circuit", "source_qasm_sha256"):
            if record.get(field) != entry.get(field):
                raise RegressionProtocolError(f"{field} provenance drift in {circuit}")
        if not _is_sha256(record.get("source_qasm_sha256")):
            raise RegressionProtocolError(f"invalid source hash in {circuit}")
        family = record.get("family")
        if not isinstance(family, str) or not family or family != coverage_rq2.family_of(circuit):
            raise RegressionProtocolError(f"family identity drift in {circuit}")
        if record.get("qmax") != coverage_rq2.QMAX:
            raise RegressionProtocolError(f"Qmax drift in {circuit}")

        e_mean = _require_float(record, "E_mean", lower=0.0, upper=0.25 + 1e-12)
        e_peak = _require_float(record, "E_peak", lower=0.0, upper=0.25 + 1e-12)
        if e_mean > e_peak + 1e-15:
            raise RegressionProtocolError(f"E_mean exceeds E_peak in {circuit}")
        _require_int(record, "depth", minimum=1)
        _require_int(record, "num_qubits", minimum=1)
        _require_int(record, "gate_count", minimum=1)
        _require_float(record, "two_qubit_density", lower=0.0, upper=1.0)

        for outcome in COVERAGE_OUTCOMES:
            _require_float(record, outcome, lower=0.0, upper=1.0)
        for outcome, (numerator_field, denominator_field) in OUTCOME_IDENTITIES.items():
            numerator = _require_int(record, numerator_field)
            denominator = _require_int(record, denominator_field, minimum=1)
            if numerator > denominator:
                raise RegressionProtocolError(f"{outcome} numerator exceeds denominator")
            expected = numerator / denominator
            if not math.isclose(float(record[outcome]), expected, rel_tol=0.0, abs_tol=1e-15):
                raise RegressionProtocolError(f"derived outcome mismatch after recomputation: {outcome}")
        if record.get("gate_node_denominator") != record.get("gate_count"):
            raise RegressionProtocolError(f"gate denominator drift in {circuit}")
        if record.get("qubit_denominator") != record.get("num_qubits"):
            raise RegressionProtocolError(f"qubit denominator drift in {circuit}")
        depth_denominator = _require_int(record, "normalized_depth_denominator", minimum=1)
        if depth_denominator != record["depth"]:
            raise RegressionProtocolError(f"depth denominator drift in {circuit}")
        depth_reach = float(record["deployed_observable_depth_reach"])
        scaled_depth = depth_reach * depth_denominator
        if not math.isclose(scaled_depth, round(scaled_depth), rel_tol=0.0, abs_tol=1e-12):
            raise RegressionProtocolError(f"depth reach is not a source-layer ratio in {circuit}")

        status = statuses[circuit]
        if not isinstance(status, dict) or status.get("status") != "complete":
            raise RegressionProtocolError(f"non-complete terminal status for {circuit}")
        for field in ("attempt_index", "attempt_id", "source_qasm_sha256"):
            if status.get(field) != entry.get(field):
                raise RegressionProtocolError(f"status {field} provenance drift in {circuit}")
        if status.get("record_sha256") != stable_hash(record):
            raise RegressionProtocolError(f"row checksum mismatch for {circuit}")

    if observed_circuits != set(entry_by_circuit):
        raise RegressionProtocolError("silent circuit-row drop")


def _csv_projection(record: dict) -> dict[str, object]:
    projected = {field: record.get(field) for field in coverage_rq2.CSV_FIELDS}
    projected.update({
        "mean_ent": record["E_mean"],
        "peak_ent": record["E_peak"],
        "node_cov": record["deployed_observable_gate_coverage"],
        "emitted_node_cov": record["emitted_mid_gate_coverage"],
        "certificate_node_cov": record["certificate_gate_coverage"],
        "deployed_node_cov": record["deployed_observable_gate_coverage"],
    })
    return projected


def _csv_text(value: object) -> str:
    return "" if value is None else str(value)


def _validate_csv_projection(path: Path, records: list[dict]) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RegressionProtocolError(f"cannot read coverage CSV {path}: {error}") from error
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RegressionProtocolError("coverage CSV is not UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames != coverage_rq2.CSV_FIELDS:
        raise RegressionProtocolError("coverage CSV header/schema drift")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RegressionProtocolError("coverage CSV contains duplicate columns")
    rows = list(reader)
    if len(rows) != len(records):
        raise RegressionProtocolError("coverage JSON/CSV row-count mismatch")
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        expected = _csv_projection(record)
        for field in coverage_rq2.CSV_FIELDS:
            if row.get(field) != _csv_text(expected[field]):
                raise RegressionProtocolError(
                    f"coverage JSON/CSV mismatch at row {index}, field {field}"
                )
    return payload


def load_validated_coverage(
    json_path: Path | str,
    *,
    csv_path: Path | str | None = None,
    manifest: Path | str = DEFAULT_MANIFEST,
    qasm_dir: Path | str = DEFAULT_QASM_DIR,
) -> tuple[dict, dict]:
    """Load and verify source coverage, returning artifact and checksums."""

    json_path = Path(json_path).resolve()
    csv_path = Path(csv_path).resolve() if csv_path is not None else None
    artifact, json_payload = _read_strict_json(json_path, label="coverage JSON")

    # Resolve before and after producer validation to detect source provenance
    # changes during validation.  The corpus resolver also verifies all
    # QASM bytes against the fixed corpus constants.
    frame_before = resolve_circuit_corpus(manifest, qasm_dir)
    _validate_regression_rows(artifact, frame_before)
    coverage_rq2.validate_artifact(artifact, frame_before, require_complete=True)
    frame_after = resolve_circuit_corpus(manifest, qasm_dir)
    if _frame_provenance(frame_after) != _frame_provenance(frame_before):
        raise RegressionProtocolError("corpus provenance changed during validation")
    if frame_after.get("circuits") != frame_before.get("circuits"):
        raise RegressionProtocolError("corpus circuit identities changed during validation")
    if json_path.read_bytes() != json_payload:
        raise RegressionProtocolError("coverage JSON changed during validation")

    csv_payload = None
    if csv_path is not None:
        csv_payload = _validate_csv_projection(csv_path, artifact["records"])
        if csv_path.read_bytes() != csv_payload:
            raise RegressionProtocolError("coverage CSV changed during validation")

    fingerprints = {
        "coverage_json_sha256": _sha256_bytes(json_payload),
        "coverage_csv_sha256": (
            _sha256_bytes(csv_payload) if csv_payload is not None else None
        ),
        "coverage_schema": artifact["schema"],
        "native_coverage_artifact_sha256": stable_hash(artifact),
        "coverage_records_sha256": artifact["records_sha256"],
        "coverage_protocol_sha256": stable_hash(artifact["protocol"]),
        "manifest_sha256": artifact["corpus"]["manifest_sha256"],
        "corpus_sha256": artifact["corpus"]["corpus_sha256"],
    }
    return artifact, fingerprints


def _standardized_design(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [[float(record[predictor]) for predictor in PREDICTORS] for record in records],
        dtype=float,
    )
    if not np.all(np.isfinite(raw)):
        raise RegressionProtocolError("non-finite predictor matrix")
    means = np.mean(raw, axis=0)
    scales = np.std(raw, axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        bad = [PREDICTORS[index] for index in np.flatnonzero(scales <= 0.0)]
        raise RegressionProtocolError(f"zero-variance predictors: {bad}")
    standardized = (raw - means) / scales
    design = np.column_stack([np.ones(len(records)), standardized])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RegressionProtocolError("predictor design is rank deficient")
    return design, means, scales


def _validated_bfgs_candidate(
    result: object,
    objective,
    gradient,
    *,
    parameter_count: int,
) -> tuple[np.ndarray, float, float, int, dict]:
    """Validate a BFGS candidate before deterministic Newton polishing."""

    failures = []
    success = getattr(result, "success", None)
    status = getattr(result, "status", None)
    message = getattr(result, "message", None)
    success_is_bool = isinstance(success, (bool, np.bool_))
    status_is_int = (
        not isinstance(status, (bool, np.bool_))
        and isinstance(status, (int, np.integer))
    )
    if not isinstance(message, str) or not message.strip():
        failures.append("message is missing")
        message_value = None
    else:
        message_value = message.strip()
    normal_convergence = (
        success_is_bool
        and bool(success)
        and status_is_int
        and int(status) == 0
        and message_value == BFGS_SUCCESS_MESSAGE
    )
    precision_loss_candidate = (
        success_is_bool
        and not bool(success)
        and status_is_int
        and int(status) == 2
        and message_value == BFGS_PRECISION_LOSS_MESSAGE
    )
    if not normal_convergence and not precision_loss_candidate:
        failures.append("BFGS termination state is not an allowed candidate")

    parameters = None
    try:
        candidate = np.asarray(getattr(result, "x", None), dtype=float)
    except (TypeError, ValueError):
        failures.append("parameters are not numeric")
    else:
        if candidate.shape != (parameter_count,):
            failures.append("parameter vector has the wrong shape")
        elif not np.all(np.isfinite(candidate)):
            failures.append("parameters are non-finite")
        else:
            parameters = candidate
            if float(np.max(np.abs(parameters))) > 30.0:
                failures.append("absolute coefficient exceeds separation guard")

    reported_objective = getattr(result, "fun", None)
    if (
        isinstance(reported_objective, (bool, np.bool_))
        or not isinstance(reported_objective, (int, float, np.integer, np.floating))
        or not math.isfinite(float(reported_objective))
    ):
        failures.append("reported objective is missing or non-finite")
        reported_objective_value = None
    else:
        reported_objective_value = float(reported_objective)

    objective_value = None
    gradient_norm = None
    if parameters is not None:
        objective_value = float(objective(parameters))
        score = np.asarray(gradient(parameters), dtype=float)
        if not math.isfinite(objective_value):
            failures.append("recomputed objective is non-finite")
        if score.shape != (parameter_count,) or not np.all(np.isfinite(score)):
            failures.append("recomputed score is malformed or non-finite")
        else:
            gradient_norm = float(np.max(np.abs(score)))
        if (
            reported_objective_value is not None
            and math.isfinite(objective_value)
            and not math.isclose(
                reported_objective_value,
                objective_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            failures.append("reported and recomputed objectives disagree")

    iterations = getattr(result, "nit", None)
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or int(iterations) < 0
    ):
        failures.append("iteration count is missing or invalid")

    if failures:
        raise RegressionProtocolError(
            "fractional-logit BFGS candidate rejected: "
            + "; ".join(failures)
            + f"; success={success!r}, status={status!r}, message={message!r}"
        )
    return (
        parameters,
        objective_value,
        gradient_norm,
        int(iterations),
        {
            "success": bool(success),
            "status": int(status),
            "message": message_value,
            "termination": (
                "normal_convergence"
                if normal_convergence
                else "precision_loss_candidate"
            ),
            "gradient_inf_norm": gradient_norm,
        },
    )


def _newton_polish(
    design: np.ndarray,
    outcome: np.ndarray,
    initial: np.ndarray,
    objective,
    gradient,
) -> tuple[np.ndarray, float, float, dict]:
    """Polish the same convex objective with its exact Hessian."""

    beta = np.asarray(initial, dtype=float).copy()
    initial_objective = float(objective(beta))
    if not math.isfinite(initial_objective):
        raise RegressionProtocolError("Newton polishing received non-finite objective")
    backtracking_steps = 0
    roundoff_stationary_steps = 0
    accepted_steps = 0
    best_beta = beta.copy()
    best_objective = initial_objective
    best_gradient_norm = math.inf
    termination = "iteration_limit"
    for iteration in range(NEWTON_MAX_ITERATIONS + 1):
        score = np.asarray(gradient(beta), dtype=float)
        if score.shape != (design.shape[1],) or not np.all(np.isfinite(score)):
            raise RegressionProtocolError(
                "Newton polishing produced malformed or non-finite score"
            )
        gradient_norm = float(np.max(np.abs(score)))
        criterion = float(objective(beta))
        if not math.isfinite(criterion):
            raise RegressionProtocolError(
                "Newton polishing produced non-finite objective"
            )
        if gradient_norm < best_gradient_norm:
            best_beta = beta.copy()
            best_objective = criterion
            best_gradient_norm = gradient_norm
        if gradient_norm <= NEWTON_TARGET_SCORE_TOLERANCE:
            objective_roundoff = max(
                1e-12,
                64.0 * np.finfo(float).eps * max(1.0, abs(initial_objective)),
            )
            if criterion > initial_objective + objective_roundoff:
                raise RegressionProtocolError(
                    "Newton polishing increased the objective"
                )
            return (
                beta,
                criterion,
                gradient_norm,
                {
                    "success": True,
                    "iterations": accepted_steps,
                    "backtracking_steps": backtracking_steps,
                    "roundoff_stationary_steps": roundoff_stationary_steps,
                    "target_score_tolerance": NEWTON_TARGET_SCORE_TOLERANCE,
                    "acceptance_score_tolerance": FINAL_SCORE_TOLERANCE,
                    "objective_roundoff_tolerance": objective_roundoff,
                    "gradient_inf_norm": gradient_norm,
                    "target_reached": True,
                    "termination": "target_reached",
                },
            )
        if iteration == NEWTON_MAX_ITERATIONS:
            break

        fitted = expit(design @ beta)
        weights = fitted * (1.0 - fitted)
        information = design.T @ (weights[:, None] * design)
        if not np.all(np.isfinite(information)):
            raise RegressionProtocolError(
                "Newton polishing produced non-finite information matrix"
            )
        condition_number = float(np.linalg.cond(information))
        if not math.isfinite(condition_number) or condition_number > 1e14:
            raise RegressionProtocolError(
                "Newton polishing information matrix is ill-conditioned"
            )
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as error:
            raise RegressionProtocolError(
                "Newton polishing information solve failed"
            ) from error
        directional_decrease = float(score @ step)
        if not math.isfinite(directional_decrease) or directional_decrease <= 0.0:
            raise RegressionProtocolError(
                "Newton polishing did not produce a descent direction"
            )

        step_scale = 1.0
        accepted = False
        for _ in range(NEWTON_MAX_BACKTRACKS + 1):
            candidate = beta - step_scale * step
            candidate_objective = float(objective(candidate))
            candidate_score = np.asarray(gradient(candidate), dtype=float)
            candidate_gradient_norm = (
                float(np.max(np.abs(candidate_score)))
                if candidate_score.shape == (design.shape[1],)
                and np.all(np.isfinite(candidate_score))
                else math.inf
            )
            objective_roundoff = max(
                1e-12,
                64.0 * np.finfo(float).eps * max(1.0, abs(criterion)),
            )
            armijo_acceptance = (
                math.isfinite(candidate_objective)
                and candidate_objective
                <= criterion - 1e-4 * step_scale * directional_decrease
            )
            roundoff_stationary_acceptance = (
                math.isfinite(candidate_objective)
                and candidate_objective <= criterion + objective_roundoff
                and candidate_gradient_norm <= NEWTON_TARGET_SCORE_TOLERANCE
                and candidate_gradient_norm < gradient_norm
            )
            if armijo_acceptance or roundoff_stationary_acceptance:
                beta = candidate
                accepted = True
                accepted_steps += 1
                if roundoff_stationary_acceptance and not armijo_acceptance:
                    roundoff_stationary_steps += 1
                break
            step_scale *= 0.5
            backtracking_steps += 1
        if not accepted:
            termination = "line_search_stagnation"
            break

    objective_roundoff = max(
        1e-12,
        64.0 * np.finfo(float).eps * max(1.0, abs(initial_objective)),
    )
    if (
        best_gradient_norm <= FINAL_SCORE_TOLERANCE
        and best_objective <= initial_objective + objective_roundoff
    ):
        return (
            best_beta,
            best_objective,
            best_gradient_norm,
            {
                "success": True,
                "iterations": accepted_steps,
                "backtracking_steps": backtracking_steps,
                "roundoff_stationary_steps": roundoff_stationary_steps,
                "target_score_tolerance": NEWTON_TARGET_SCORE_TOLERANCE,
                "acceptance_score_tolerance": FINAL_SCORE_TOLERANCE,
                "objective_roundoff_tolerance": objective_roundoff,
                "gradient_inf_norm": best_gradient_norm,
                "target_reached": False,
                "termination": (
                    f"{termination}_within_acceptance_gate"
                ),
            },
        )
    raise RegressionProtocolError(
        "Newton polishing did not reach the unchanged score tolerance "
        f"{FINAL_SCORE_TOLERANCE:.1e}; best score "
        f"{best_gradient_norm:.17g}"
    )


def _fractional_logit_fit(design: np.ndarray, outcome: np.ndarray) -> dict:
    if design.ndim != 2 or outcome.ndim != 1 or len(outcome) != len(design):
        raise RegressionProtocolError("malformed fractional-logit design or outcome")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(outcome)):
        raise RegressionProtocolError("non-finite fractional-logit design or outcome")
    if len(outcome) <= design.shape[1]:
        raise RegressionProtocolError("insufficient rows for fractional logit")
    if np.any((outcome < 0.0) | (outcome > 1.0)) or np.ptp(outcome) <= 0.0:
        raise RegressionProtocolError("fractional-logit outcome must vary within [0,1]")

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - outcome * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - outcome)

    initial = np.zeros(design.shape[1], dtype=float)
    clipped_mean = float(np.clip(np.mean(outcome), 1e-6, 1.0 - 1e-6))
    initial[0] = math.log(clipped_mean / (1.0 - clipped_mean))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": BFGS_SCORE_TOLERANCE, "maxiter": 4000},
    )
    (
        bfgs_beta,
        _bfgs_criterion,
        _bfgs_gradient_norm,
        bfgs_iterations,
        bfgs_optimizer,
    ) = (
        _validated_bfgs_candidate(
            result,
            objective,
            gradient,
            parameter_count=design.shape[1],
        )
    )
    beta, criterion, gradient_norm, newton_optimizer = _newton_polish(
        design,
        outcome,
        bfgs_beta,
        objective,
        gradient,
    )
    if float(np.max(np.abs(beta))) > 30.0:
        raise RegressionProtocolError(
            "absolute coefficient exceeds separation guard after Newton polishing"
        )
    fitted = expit(design @ beta)
    weights = np.clip(fitted * (1.0 - fitted), 1e-12, None)
    information = design.T @ (weights[:, None] * design)
    condition_number = float(np.linalg.cond(information))
    if not math.isfinite(condition_number) or condition_number > 1e14:
        raise RegressionProtocolError("fractional-logit information matrix is ill-conditioned")
    bread = np.linalg.inv(information)
    if not all(
        np.all(np.isfinite(value)) for value in (fitted, weights, information, bread)
    ):
        raise RegressionProtocolError("fractional-logit fit produced non-finite matrices")
    return {
        "beta": beta,
        "fitted": fitted,
        "weights": weights,
        "bread": bread,
        "criterion": criterion,
        "gradient_inf_norm": gradient_norm,
        "iterations": bfgs_iterations + newton_optimizer["iterations"],
        "optimizer": {
            "method": "BFGS with deterministic exact-Hessian Newton polishing",
            "bfgs": bfgs_optimizer,
            "newton_polish": newton_optimizer,
        },
    }


def _cluster_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    bread: np.ndarray,
    families: np.ndarray,
) -> tuple[np.ndarray, int]:
    unique_families = np.unique(families)
    n_rows, n_parameters = design.shape
    n_clusters = len(unique_families)
    if n_clusters <= n_parameters:
        raise RegressionProtocolError(
            f"family-cluster covariance needs more than {n_parameters} families"
        )
    meat = np.zeros((n_parameters, n_parameters), dtype=float)
    for family in unique_families:
        selected = families == family
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (n_clusters / (n_clusters - 1.0)) * (
        (n_rows - 1.0) / (n_rows - n_parameters)
    )
    covariance = correction * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise RegressionProtocolError("invalid family-cluster covariance")
    return covariance, n_clusters


def _family_bootstrap(
    design: np.ndarray,
    outcome: np.ndarray,
    families: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    unique_families = np.unique(families)
    family_rows = [np.flatnonzero(families == family) for family in unique_families]
    rng = np.random.default_rng(seed)
    estimates = []
    failures = 0
    for _ in range(replicates):
        sampled = rng.integers(0, len(unique_families), size=len(unique_families))
        indices = np.concatenate([family_rows[index] for index in sampled])
        try:
            estimates.append(_fractional_logit_fit(design[indices], outcome[indices])["beta"])
        except RegressionProtocolError:
            failures += 1
    required = math.ceil(0.95 * replicates)
    if len(estimates) < required:
        raise RegressionProtocolError(
            f"family bootstrap recovered {len(estimates)}/{replicates} fits; "
            f"requires {required}"
        )
    return np.asarray(estimates), failures


def _coefficient_rows(
    beta: np.ndarray,
    covariance: np.ndarray,
    n_clusters: int,
    alpha: float,
    bootstrap: np.ndarray | None = None,
) -> list[dict]:
    standard_errors = np.sqrt(np.diag(covariance))
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=n_clusters - 1))
    names = ("intercept",) + PREDICTORS
    rows = []
    for index, name in enumerate(names):
        statistic = float(beta[index] / standard_errors[index])
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n_clusters - 1))
        row = {
            "term": name,
            "estimate": float(beta[index]),
            "cluster_standard_error": float(standard_errors[index]),
            "cluster_t": statistic,
            "cluster_df": n_clusters - 1,
            "p_raw": p_value,
            "wald_ci": [
                float(beta[index] - critical * standard_errors[index]),
                float(beta[index] + critical * standard_errors[index]),
            ],
        }
        if bootstrap is not None:
            row["family_bootstrap_percentile_ci"] = [
                float(np.quantile(bootstrap[:, index], alpha / 2.0)),
                float(np.quantile(bootstrap[:, index], 1.0 - alpha / 2.0)),
            ]
        rows.append(row)
    return rows


def _bh_adjust(values: Iterable[float]) -> list[float]:
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise RegressionProtocolError("invalid p-values for BH adjustment")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return [float(value) for value in adjusted]


def _multicollinearity(design: np.ndarray) -> dict:
    standardized = design[:, 1:]
    correlation = np.corrcoef(standardized, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)
    if np.min(eigenvalues) <= 1e-12:
        raise RegressionProtocolError("predictor correlation matrix is singular")
    inverse = np.linalg.inv(correlation)
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "predictor_order": list(PREDICTORS),
        "correlation_matrix": [[float(value) for value in row] for row in correlation],
        "variance_inflation_factors": {
            predictor: float(inverse[index, index])
            for index, predictor in enumerate(PREDICTORS)
        },
        "standardized_design_condition_number": condition_number,
        "minimum_correlation_eigenvalue": float(np.min(eigenvalues)),
        "flags": {
            "any_vif_above_5": bool(np.any(np.diag(inverse) > 5.0)),
            "any_vif_above_10": bool(np.any(np.diag(inverse) > 10.0)),
            "condition_number_above_30": condition_number > 30.0,
        },
    }


def _influence_diagnostics(
    fit: dict,
    design: np.ndarray,
    outcome: np.ndarray,
    families: np.ndarray,
    records: list[dict],
) -> dict:
    weighted_design = design * np.sqrt(fit["weights"])[:, None]
    leverage = np.einsum(
        "ij,jk,ik->i", weighted_design, fit["bread"], weighted_design
    )
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
    pearson = (outcome - fit["fitted"]) / np.sqrt(fit["weights"])
    cook = (pearson ** 2 * leverage) / (
        design.shape[1] * np.maximum((1.0 - leverage) ** 2, 1e-24)
    )
    deviance = np.sign(outcome - fit["fitted"]) * np.sqrt(np.maximum(
        2.0 * (
            xlogy(outcome, outcome / fit["fitted"])
            + xlogy(1.0 - outcome, (1.0 - outcome) / (1.0 - fit["fitted"]))
        ),
        0.0,
    ))
    top_indices = np.argsort(cook, kind="stable")[::-1][: min(10, len(records))]
    row_threshold = 4.0 / len(records)
    leverage_threshold = 2.0 * design.shape[1] / len(records)

    deletion_rows = []
    failures = []
    for family in np.unique(families):
        retained = families != family
        try:
            deleted_fit = _fractional_logit_fit(design[retained], outcome[retained])
        except RegressionProtocolError as error:
            failures.append({"family": str(family), "error": str(error)})
            continue
        delta = deleted_fit["beta"] - fit["beta"]
        deletion_rows.append({
            "family": str(family),
            "circuit_count": int(np.sum(~retained)),
            "max_abs_coefficient_change": float(np.max(np.abs(delta[1:]))),
            "coefficient_changes": {
                predictor: float(delta[index + 1])
                for index, predictor in enumerate(PREDICTORS)
            },
        })
    deletion_rows.sort(
        key=lambda row: (-row["max_abs_coefficient_change"], row["family"])
    )
    return {
        "thresholds": {
            "cook_4_over_n": row_threshold,
            "leverage_2p_over_n": leverage_threshold,
        },
        "counts_above_threshold": {
            "cook": int(np.sum(cook > row_threshold)),
            "leverage": int(np.sum(leverage > leverage_threshold)),
            "absolute_pearson_residual_above_2": int(np.sum(np.abs(pearson) > 2.0)),
        },
        "maxima": {
            "leverage": float(np.max(leverage)),
            "absolute_pearson_residual": float(np.max(np.abs(pearson))),
            "absolute_deviance_residual": float(np.max(np.abs(deviance))),
            "cook_distance": float(np.max(cook)),
        },
        "top_circuits_by_cook": [
            {
                "circuit": records[index]["circuit"],
                "family": records[index]["family"],
                "leverage": float(leverage[index]),
                "pearson_residual": float(pearson[index]),
                "deviance_residual": float(deviance[index]),
                "cook_distance": float(cook[index]),
            }
            for index in top_indices
        ],
        "leave_one_family_out": {
            "ranked": deletion_rows,
            "failed": failures,
        },
    }


def _sensitivity_fit(
    design: np.ndarray,
    outcome: np.ndarray,
    families: np.ndarray,
    alpha: float,
) -> dict:
    transformed = np.arcsin(np.sqrt(outcome))
    cross_product = design.T @ design
    if np.linalg.cond(cross_product) > 1e14:
        raise RegressionProtocolError("sensitivity OLS design is ill-conditioned")
    bread = np.linalg.inv(cross_product)
    beta = bread @ design.T @ transformed
    residual = transformed - design @ beta
    covariance, n_clusters = _cluster_covariance(design, residual, bread, families)
    return {
        "model": "arcsine-square-root OLS",
        "coefficient_scale": (
            "radian change in asin(sqrt(coverage)) per predictor sample SD"
        ),
        "coefficients": _coefficient_rows(beta, covariance, n_clusters, alpha),
        "fit": {
            "r_squared": float(
                1.0 - np.sum(residual ** 2)
                / np.sum((transformed - np.mean(transformed)) ** 2)
            ),
            "family_clusters": n_clusters,
        },
    }


def _validate_analysis_configuration(
    bootstrap_replicates: object,
    seed: object,
    alpha: object,
    *,
    minimum_replicates: int = 20,
) -> None:
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < minimum_replicates
    ):
        raise RegressionProtocolError(
            f"bootstrap_replicates must be at least {minimum_replicates}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RegressionProtocolError("seed must be a nonnegative integer")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 0.2
    ):
        raise RegressionProtocolError("alpha must lie strictly between 0 and 0.2")


def _outcome_support(
    records: list[dict],
    families: np.ndarray,
) -> dict:
    support = {}
    unique_families = np.unique(families)
    for outcome_name in COVERAGE_OUTCOMES:
        outcome = np.asarray(
            [float(record[outcome_name]) for record in records],
            dtype=float,
        )
        within_family_variation = 0
        families_with_any_nonunit_value = 0
        for family in unique_families:
            selected = outcome[families == family]
            within_family_variation += int(np.ptp(selected) > 0.0)
            families_with_any_nonunit_value += int(np.any(selected < 1.0))
        support[outcome_name] = {
            "role": (
                "modeled"
                if outcome_name in OUTCOMES
                else "descriptive_only"
            ),
            "circuit_count": len(outcome),
            "unique_value_count": int(len(np.unique(outcome))),
            "exact_zero_count": int(np.sum(outcome == 0.0)),
            "exact_one_count": int(np.sum(outcome == 1.0)),
            "exact_one_fraction": float(np.mean(outcome == 1.0)),
            "family_count": len(unique_families),
            "families_with_within_family_variation": within_family_variation,
            "families_with_any_nonunit_value": families_with_any_nonunit_value,
        }
        if outcome_name in DESCRIPTIVE_ONLY_OUTCOMES:
            support[outcome_name]["reason"] = DESCRIPTIVE_ONLY_OUTCOMES[
                outcome_name
            ]
    return support


def analyze_records(
    records: list[dict],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """Fit all prespecified models to already verified circuit rows."""

    _validate_analysis_configuration(bootstrap_replicates, seed, alpha)
    if len(records) < 2:
        raise RegressionProtocolError("at least two circuit rows are required")

    design, means, scales = _standardized_design(records)
    families = np.asarray([record["family"] for record in records], dtype=object)
    if len(np.unique(families)) <= design.shape[1]:
        raise RegressionProtocolError("too few circuit families for clustered inference")

    primary = {}
    sensitivity = {}
    p_locations = []
    p_values = []
    for outcome_name in OUTCOMES:
        outcome = np.asarray([float(record[outcome_name]) for record in records])
        fit = _fractional_logit_fit(design, outcome)
        covariance, n_clusters = _cluster_covariance(
            design, outcome - fit["fitted"], fit["bread"], families
        )
        bootstrap, failures = _family_bootstrap(
            design,
            outcome,
            families,
            replicates=bootstrap_replicates,
            seed=seed + OUTCOME_SEED_OFFSETS[outcome_name],
        )
        coefficient_rows = _coefficient_rows(
            fit["beta"], covariance, n_clusters, alpha, bootstrap
        )
        primary[outcome_name] = {
            "model": "fractional logit QMLE",
            "coefficient_scale": "log-odds per predictor sample SD",
            "coefficients": coefficient_rows,
            "fit": {
                "bernoulli_quasi_negative_log_likelihood": fit["criterion"],
                "gradient_inf_norm": fit["gradient_inf_norm"],
                "iterations": fit["iterations"],
                "optimizer": fit["optimizer"],
                "family_clusters": n_clusters,
                "bootstrap_requested": bootstrap_replicates,
                "bootstrap_successful": int(len(bootstrap)),
                "bootstrap_failed": failures,
            },
            "influence": _influence_diagnostics(
                fit, design, outcome, families, records
            ),
        }
        sensitivity[outcome_name] = _sensitivity_fit(
            design, outcome, families, alpha
        )
        for coefficient_index, coefficient in enumerate(coefficient_rows[1:], start=1):
            p_locations.append((outcome_name, coefficient_index))
            p_values.append(coefficient["p_raw"])

    adjusted = _bh_adjust(p_values)
    for (outcome_name, coefficient_index), q_value in zip(
        p_locations, adjusted, strict=True
    ):
        primary[outcome_name]["coefficients"][coefficient_index]["p_bh"] = q_value

    analysis_rows = [
        {
            "attempt_index": record["attempt_index"],
            "attempt_id": record["attempt_id"],
            "circuit": record["circuit"],
            "source_qasm_sha256": record["source_qasm_sha256"],
            "family": record["family"],
            **{
                field: record[field]
                for field in PREDICTORS + COVERAGE_OUTCOMES
            },
        }
        for record in records
    ]
    return {
        "sample": {
            "circuit_count": len(records),
            "family_count": len(np.unique(families)),
            "family_counts": dict(sorted(Counter(families).items())),
            "analysis_rows_sha256": stable_hash(analysis_rows),
        },
        "predictor_standardization": {
            predictor: {"mean": float(means[index]), "sample_sd": float(scales[index])}
            for index, predictor in enumerate(PREDICTORS)
        },
        "multicollinearity": _multicollinearity(design),
        "outcome_support": _outcome_support(records, families),
        "descriptive_only_outcomes": dict(DESCRIPTIVE_ONLY_OUTCOMES),
        "primary_models": primary,
        "sensitivity_models": sensitivity,
        "multiple_testing": {
            "procedure": "Benjamini-Hochberg",
            "family_size": len(p_values),
            "included_terms": list(PREDICTORS),
            "included_outcomes": list(OUTCOMES),
        },
    }


def _validate_source_fingerprints(artifact: dict, fingerprints: dict) -> None:
    required = {
        "coverage_json_sha256",
        "coverage_csv_sha256",
        "coverage_schema",
        "native_coverage_artifact_sha256",
        "coverage_records_sha256",
        "coverage_protocol_sha256",
        "manifest_sha256",
        "corpus_sha256",
    }
    if not isinstance(fingerprints, dict) or set(fingerprints) != required:
        raise RegressionProtocolError("incomplete native coverage checksum bundle")
    for field in required - {"coverage_csv_sha256", "coverage_schema"}:
        if not _is_sha256(fingerprints[field]):
            raise RegressionProtocolError(f"malformed source checksum: {field}")
    csv_digest = fingerprints["coverage_csv_sha256"]
    if csv_digest is not None and not _is_sha256(csv_digest):
        raise RegressionProtocolError("malformed source checksum: coverage_csv_sha256")
    corpus = artifact.get("corpus")
    if not isinstance(corpus, dict):
        raise RegressionProtocolError("native coverage artifact lacks corpus provenance")
    expected = {
        "coverage_schema": artifact.get("schema"),
        "native_coverage_artifact_sha256": stable_hash(artifact),
        "coverage_records_sha256": artifact.get("records_sha256"),
        "coverage_protocol_sha256": stable_hash(artifact.get("protocol")),
        "manifest_sha256": corpus.get("manifest_sha256"),
        "corpus_sha256": corpus.get("corpus_sha256"),
    }
    for field, value in expected.items():
        if fingerprints.get(field) != value:
            raise RegressionProtocolError(
                f"native coverage content/provenance checksum mismatch: {field}"
            )


def _runtime_fingerprints() -> dict:
    source_paths = (
        ("regression_implementation", Path(__file__).resolve()),
        ("regression_protocol", PROTOCOL_PATH.resolve()),
        ("coverage_validator", Path(coverage_rq2.__file__).resolve()),
    )
    source_files = []
    for role, path in source_paths:
        if not path.is_file():
            raise RegressionProtocolError(f"required source file is missing: {path}")
        source_files.append(
            {"role": role, "name": path.name, "sha256": file_sha256(path)}
        )
    dependencies = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    # Bind numerical semantics, not the host kernel build string.  The latter
    # changes across otherwise equivalent machines and prevents independent
    # revalidation without protecting any scientific computation.
    environment = {
        "float64_epsilon": float(np.finfo(np.float64).eps),
    }
    return {
        "protocol": {
            "schema": SCHEMA,
            "sha256": stable_hash(protocol_descriptor()),
        },
        "sources": {
            "files": source_files,
            "sha256": stable_hash(source_files),
        },
        "dependencies": {
            "versions": dependencies,
            "sha256": stable_hash(dependencies),
        },
        "environment": environment,
        "environment_sha256": stable_hash(environment),
    }


# Runtime checksums that define the experiment. A difference here means the
# artifact was produced under a different protocol or a different floating
# point environment, so validation rejects it.
STRICT_RUNTIME_FINGERPRINT_FIELDS = (
    "protocol",
    "environment",
    "environment_sha256",
)
# Runtime checksums that only record the machine of origin. Editing a comment
# in a source file or upgrading a package moves them without touching a single
# reported number, so validation reports a difference and continues.
ENVIRONMENT_RUNTIME_FINGERPRINT_FIELDS = ("sources", "dependencies")


def _with_recorded_environment(expected: dict, recorded: dict) -> dict:
    """Copy the recorded source and dependency stamps into a recomputation.

    Only the two stamps that record the machine of origin are taken from the
    result being checked, and only when the result actually carries them.
    Everything else, including the analysis and every content checksum, still
    has to equal what this machine recomputed.
    """

    fingerprints = dict(expected["fingerprints"])
    for field in ENVIRONMENT_RUNTIME_FINGERPRINT_FIELDS:
        if field in recorded:
            fingerprints[field] = recorded[field]
    adjusted = dict(expected)
    adjusted["fingerprints"] = fingerprints
    return _add_artifact_checksum(adjusted)


def build_output(
    artifact: dict,
    source_fingerprints: dict,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict:
    """Build the deterministic, machine-readable regression artifact."""

    _validate_source_fingerprints(artifact, source_fingerprints)
    config = {
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "alpha": alpha,
    }
    analysis = analyze_records(
        artifact["records"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        alpha=alpha,
    )
    fingerprints = {
        "source": dict(source_fingerprints),
        "data": {
            "analysis_rows_sha256": analysis["sample"]["analysis_rows_sha256"],
            "family_counts_sha256": stable_hash(analysis["sample"]["family_counts"]),
        },
        "configuration": {
            **config,
            "sha256": stable_hash(config),
        },
        **_runtime_fingerprints(),
    }
    output = {
        "schema": SCHEMA,
        "protocol": protocol_descriptor(),
        "interpretation": (
            "All estimates describe conditional associations in this circuit corpus; "
            "they are not causal effects."
        ),
        "fingerprints": fingerprints,
        "validation": {
            "source_schema_current": True,
            "producer_validation_complete": True,
            "all_terminal_statuses_complete": True,
            "csv_cross_checked": source_fingerprints["coverage_csv_sha256"] is not None,
        },
        "analysis": analysis,
        ARTIFACT_CHECKSUM_FIELD: None,
    }
    return _add_artifact_checksum(output)


def _configuration_from_artifact(
    artifact: dict,
    *,
    source: str,
    require_publication: bool,
) -> dict:
    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise RegressionProtocolError(f"{source}: missing checksum bundle")
    bundle = fingerprints.get("configuration")
    required = {"bootstrap_replicates", "seed", "alpha", "sha256"}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise RegressionProtocolError(f"{source}: malformed configuration checksum")
    config = {field: bundle[field] for field in ("bootstrap_replicates", "seed", "alpha")}
    minimum = PUBLICATION_MIN_BOOTSTRAP_REPLICATES if require_publication else 20
    _validate_analysis_configuration(
        config["bootstrap_replicates"],
        config["seed"],
        config["alpha"],
        minimum_replicates=minimum,
    )
    if bundle["sha256"] != stable_hash(config):
        raise RegressionProtocolError(f"{source}: configuration checksum mismatch")
    return config


def validate_regression_artifact(
    artifact: dict,
    coverage_artifact: dict,
    source_fingerprints: dict,
    *,
    source: str = "regression artifact",
    require_publication: bool = True,
) -> dict:
    """Independently verify and recompute a final regression artifact."""

    if not isinstance(artifact, dict):
        raise RegressionProtocolError(f"{source}: root must be an object")
    _validate_json_tree(artifact, path=source)
    if artifact.get("schema") != SCHEMA:
        raise RegressionProtocolError(f"{source}: legacy or unknown regression schema")
    if artifact.get("protocol") != protocol_descriptor():
        raise RegressionProtocolError(f"{source}: regression protocol drift")
    _check_artifact_checksum(artifact, source=source)
    _validate_source_fingerprints(coverage_artifact, source_fingerprints)

    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise RegressionProtocolError(f"{source}: missing checksum bundle")
    if fingerprints.get("source") != source_fingerprints:
        raise RegressionProtocolError(
            f"{source}: bound native coverage input checksum mismatch"
        )
    runtime = _runtime_fingerprints()
    # The protocol descriptor and the numerical environment define the
    # experiment and must match exactly. The source-file and dependency
    # checksums record the machine that produced the artifact; they move
    # after a documentation edit, a comment change, or a package upgrade
    # without changing any number, so a difference there is reported as a
    # warning and the recomputation below adopts the recorded stamps.
    for field in STRICT_RUNTIME_FINGERPRINT_FIELDS:
        if fingerprints.get(field) != runtime[field]:
            raise RegressionProtocolError(
                f"{source}: current {field} checksum mismatch"
            )
    for field in ENVIRONMENT_RUNTIME_FINGERPRINT_FIELDS:
        if fingerprints.get(field) != runtime[field]:
            warnings.warn(
                f"{source}: {field} checksum differs from the recorded value "
                "(documentation or environment change); results are "
                "unaffected",
                RuntimeWarning,
                stacklevel=2,
            )
    config = _configuration_from_artifact(
        artifact,
        source=source,
        require_publication=require_publication,
    )

    expected = _with_recorded_environment(
        build_output(coverage_artifact, source_fingerprints, **config),
        fingerprints,
    )
    if artifact.get("analysis") != expected["analysis"]:
        raise RegressionProtocolError(
            f"{source}: recomputed analysis mismatch; coefficient, predictor, "
            "or model field is missing or tampered"
        )
    if artifact.get("fingerprints") != expected["fingerprints"]:
        raise RegressionProtocolError(
            f"{source}: recomputed data/configuration checksum mismatch"
        )
    if artifact != expected:
        raise RegressionProtocolError(
            f"{source}: recomputed regression artifact mismatch"
        )
    return {
        "schema": SCHEMA,
        "validation": "recomputed-and-valid",
        "artifact_sha256": artifact[ARTIFACT_CHECKSUM_FIELD],
        "native_coverage_artifact_sha256": source_fingerprints[
            "native_coverage_artifact_sha256"
        ],
        "analysis_rows_sha256": artifact["analysis"]["sample"][
            "analysis_rows_sha256"
        ],
        "publication_bootstrap_minimum_met": (
            config["bootstrap_replicates"] >= PUBLICATION_MIN_BOOTSTRAP_REPLICATES
        ),
    }


def validate_regression_output(
    regression_json: Path | str,
    coverage_json: Path | str,
    *,
    coverage_csv: Path | str | None = None,
    manifest: Path | str = DEFAULT_MANIFEST,
    qasm_dir: Path | str = DEFAULT_QASM_DIR,
    require_publication: bool = True,
) -> dict:
    """Strictly load and independently validate a regression artifact file."""

    regression_path = Path(regression_json).resolve()
    candidate, payload = _read_strict_json(
        regression_path, label="regression JSON"
    )
    coverage_artifact, source_fingerprints = load_validated_coverage(
        coverage_json,
        csv_path=coverage_csv,
        manifest=manifest,
        qasm_dir=qasm_dir,
    )
    validation = validate_regression_artifact(
        candidate,
        coverage_artifact,
        source_fingerprints,
        source=str(regression_path),
        require_publication=require_publication,
    )
    if regression_path.read_bytes() != payload:
        raise RegressionProtocolError("regression JSON changed during validation")
    return validation


def _add_coverage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        help="optional exact coverage_rq2 CSV cross-check; CSV-only input is invalid",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit or independently validate verified RQ2 regressions"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="fit the regression models")
    _add_coverage_arguments(run)
    run.add_argument("--out-json", type=Path, required=True)
    run.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--alpha", type=float, default=0.05)

    validate = commands.add_parser(
        "validate", help="verify inputs and recompute the regression results"
    )
    _add_coverage_arguments(validate)
    validate.add_argument("--regression-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command == "run"
        and args.bootstrap_replicates < PUBLICATION_MIN_BOOTSTRAP_REPLICATES
    ):
        parser.error(
            "the final-run CLI requires at least 999 family bootstrap replicates"
        )

    try:
        if args.command == "run":
            artifact, source_fingerprints = load_validated_coverage(
                args.coverage_json,
                csv_path=args.coverage_csv,
                manifest=args.manifest,
                qasm_dir=args.qasm_dir,
            )
            output = build_output(
                artifact,
                source_fingerprints,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
                alpha=args.alpha,
            )
            atomic_write_json(args.out_json, output)
            print(
                f"wrote {args.out_json}: "
                f"{output['analysis']['sample']['circuit_count']} circuits, "
                f"{output['analysis']['sample']['family_count']} families, "
                f"checksum={output[ARTIFACT_CHECKSUM_FIELD]}"
            )
        else:
            validation = validate_regression_output(
                args.regression_json,
                args.coverage_json,
                coverage_csv=args.coverage_csv,
                manifest=args.manifest,
                qasm_dir=args.qasm_dir,
                require_publication=True,
            )
            print(json.dumps(validation, sort_keys=True))
    except (
        OSError,
        ValueError,
        np.linalg.LinAlgError,
        RegressionProtocolError,
        ProtocolError,
    ) as error:
        print(f"RQ2 regression rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
