"""Synthetic and robustness tests for the RQ2 regression analysis.

Run from ``reconstruction/`` with::

    python -m unittest -v test_rq2_regression.py

No coverage experiment or QASM execution is performed.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import coverage_rq2
import rq2_regression as regression
from entanglement_disturbance import stable_hash


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def synthetic_fixture(family_count: int = 12, rows_per_family: int = 4):
    """Return a full-rank, bounded synthetic artifact and matching frame."""

    records = []
    circuits = []
    statuses = {}
    for family_index in range(family_count):
        family = f"family{family_index:02d}"
        family_shift = ((family_index % 5) - 2) * 0.08
        for member in range(rows_per_family):
            index = len(records)
            circuit = f"{family}_indep_qiskit_{member + 2}.qasm"
            source_hash = f"{index + 1:064x}"
            num_qubits = 2 + member + family_index % 3
            depth = 12 + 3 * family_index + 2 * member + (family_index * member) % 5
            gate_count = 83 + 2 * family_index + 7 * member
            density = ((2 * family_index + 3 * member) % 11) / 10.0
            e_mean = 0.008 + 0.0035 * ((family_index + 2 * member) % 13)
            e_mean += 0.0002 * family_index
            e_peak = e_mean + 0.025 + 0.0015 * ((3 * family_index + member) % 7)

            gate_probability = _logistic(
                -0.8 + 5.0 * e_mean - 2.0 * e_peak + 0.012 * depth
                + 0.35 * density - 0.035 * num_qubits + family_shift
            )
            qubit_probability = _logistic(
                0.1 - 3.0 * e_mean + 2.0 * e_peak - 0.004 * depth
                + 0.2 * density + 0.025 * num_qubits - family_shift
            )
            depth_probability = _logistic(
                -0.2 + 2.5 * e_mean + e_peak + 0.006 * depth
                - 0.25 * density + 0.02 * num_qubits + family_shift
            )
            gate_count_observed = round(gate_count * gate_probability)
            qubit_count_observed = round(num_qubits * qubit_probability)
            depth_layer_observed = round(depth * depth_probability)
            if index == 0:
                gate_count_observed = 0
            if index == family_count * rows_per_family - 1:
                gate_count_observed = gate_count

            entry = {
                "attempt_index": index,
                "attempt_id": f"canonical:{index:03d}:{circuit}",
                "circuit": circuit,
                "source_qasm_sha256": source_hash,
            }
            record = {
                **entry,
                "family": family,
                "qmax": coverage_rq2.QMAX,
                "num_qubits": num_qubits,
                "depth": depth,
                "gate_count": gate_count,
                "gate_node_denominator": gate_count,
                "qubit_denominator": num_qubits,
                "normalized_depth_denominator": depth,
                "two_qubit_density": density,
                "E_mean": e_mean,
                "E_peak": e_peak,
                "deployed_observable_gate_count": gate_count_observed,
                "deployed_observable_qubit_count": qubit_count_observed,
                "deployed_observable_gate_coverage": (
                    gate_count_observed / gate_count
                ),
                "deployed_observable_qubit_coverage": (
                    qubit_count_observed / num_qubits
                ),
                "deployed_observable_depth_reach": depth_layer_observed / depth,
                "emitted_mid_gate_coverage": gate_count_observed / gate_count,
                "certificate_gate_coverage": min(
                    1.0, gate_count_observed / gate_count + 0.05
                ),
            }
            records.append(record)
            circuits.append(entry)
            statuses[circuit] = {
                "status": "complete",
                "attempt_index": index,
                "attempt_id": entry["attempt_id"],
                "source_qasm_sha256": source_hash,
                "record_sha256": stable_hash(record),
            }

    corpus = {
        "manifest_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "circuit_count": len(records),
    }
    artifact = {
        "schema": coverage_rq2.SCHEMA,
        "protocol": coverage_rq2.protocol_descriptor(),
        "corpus": corpus,
        "circuit_status": statuses,
        "record_count": len(records),
        "records_sha256": stable_hash(records),
        "summary": {"synthetic": True},
        "records": records,
    }
    frame = {
        **corpus,
        "circuits": circuits,
        "qasm_dir": "/synthetic/not-read",
    }
    return artifact, frame


def update_checksum(artifact: dict) -> None:
    artifact["record_count"] = len(artifact["records"])
    artifact["records_sha256"] = stable_hash(artifact["records"])
    for record in artifact["records"]:
        artifact["circuit_status"][record["circuit"]]["record_sha256"] = stable_hash(record)


def source_fingerprints(artifact: dict) -> dict:
    return {
        "coverage_json_sha256": "c" * 64,
        "coverage_csv_sha256": None,
        "coverage_schema": artifact["schema"],
        "native_coverage_artifact_sha256": stable_hash(artifact),
        "coverage_records_sha256": artifact["records_sha256"],
        "coverage_protocol_sha256": stable_hash(artifact["protocol"]),
        "manifest_sha256": artifact["corpus"]["manifest_sha256"],
        "corpus_sha256": artifact["corpus"]["corpus_sha256"],
    }


def update_regression_checksum(artifact: dict) -> None:
    payload = dict(artifact)
    payload.pop(regression.ARTIFACT_CHECKSUM_FIELD, None)
    artifact[regression.ARTIFACT_CHECKSUM_FIELD] = stable_hash(payload)


class StatisticalProtocolTests(unittest.TestCase):
    def test_bv_replacement_is_same_algorithm_family(self):
        self.assertEqual(
            coverage_rq2.family_of("bv_alt_indep_qiskit_12.qasm"),
            "bv",
        )
        self.assertEqual(
            coverage_rq2.family_of("bv_indep_qiskit_14.qasm"),
            "bv",
        )

    def test_runtime_fingerprint_does_not_bind_host_kernel_string(self):
        with mock.patch.object(
            regression.platform,
            "platform",
            side_effect=AssertionError("host kernel must not enter the checksum"),
        ):
            fingerprints = regression._runtime_fingerprints()
        self.assertEqual(
            fingerprints["environment"]["float64_epsilon"],
            float(np.finfo(np.float64).eps),
        )
        self.assertNotIn("platform", fingerprints["environment"])

    def test_every_runtime_fingerprint_field_is_classified(self):
        strict = set(regression.STRICT_RUNTIME_FINGERPRINT_FIELDS)
        reported = set(regression.ENVIRONMENT_RUNTIME_FINGERPRINT_FIELDS)
        self.assertEqual(strict & reported, set())
        self.assertEqual(
            strict | reported, set(regression._runtime_fingerprints())
        )

    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.frame = synthetic_fixture()

    def test_fractional_logit_handles_zero_and_one_and_is_deterministic(self):
        first = regression.analyze_records(
            copy.deepcopy(self.artifact["records"]),
            bootstrap_replicates=39,
            seed=917,
        )
        second = regression.analyze_records(
            copy.deepcopy(self.artifact["records"]),
            bootstrap_replicates=39,
            seed=917,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["multiple_testing"]["family_size"], 10)
        self.assertEqual(first["sample"]["family_count"], 12)
        self.assertEqual(
            first["multiple_testing"]["included_outcomes"],
            list(regression.OUTCOMES),
        )
        self.assertEqual(
            set(first["outcome_support"]),
            set(regression.COVERAGE_OUTCOMES),
        )
        self.assertEqual(
            first["outcome_support"][
                "deployed_observable_qubit_coverage"
            ]["role"],
            "descriptive_only",
        )
        for outcome in regression.OUTCOMES:
            model = first["primary_models"][outcome]
            self.assertEqual(model["model"], "fractional logit QMLE")
            optimizer = model["fit"]["optimizer"]
            self.assertEqual(
                optimizer["method"],
                "BFGS with deterministic exact-Hessian Newton polishing",
            )
            self.assertEqual(
                optimizer["bfgs"]["termination"], "normal_convergence"
            )
            self.assertTrue(optimizer["newton_polish"]["success"])
            self.assertLessEqual(
                optimizer["newton_polish"]["gradient_inf_norm"],
                regression.FINAL_SCORE_TOLERANCE,
            )
            self.assertGreaterEqual(model["fit"]["bootstrap_successful"], 38)
            self.assertEqual(len(model["coefficients"]), 6)
            for coefficient in model["coefficients"][1:]:
                self.assertIn("p_bh", coefficient)
                self.assertGreaterEqual(coefficient["p_bh"], coefficient["p_raw"])
                self.assertEqual(len(coefficient["family_bootstrap_percentile_ci"]), 2)
            self.assertIn("leave_one_family_out", model["influence"])
            self.assertEqual(
                first["sensitivity_models"][outcome]["model"],
                "arcsine-square-root OLS",
            )

    def test_standardized_coefficients_have_declared_scaling(self):
        result = regression.analyze_records(
            copy.deepcopy(self.artifact["records"]),
            bootstrap_replicates=29,
            seed=11,
        )
        self.assertEqual(set(result["predictor_standardization"]), set(regression.PREDICTORS))
        self.assertTrue(
            all(
                values["sample_sd"] > 0
                for values in result["predictor_standardization"].values()
            )
        )
        diagnostics = result["multicollinearity"]
        self.assertEqual(diagnostics["predictor_order"], list(regression.PREDICTORS))
        self.assertEqual(len(diagnostics["variance_inflation_factors"]), 5)

    def test_machine_artifact_is_strict_json_and_fingerprinted(self):
        fingerprints = source_fingerprints(self.artifact)
        output = regression.build_output(
            copy.deepcopy(self.artifact),
            fingerprints,
            bootstrap_replicates=39,
            seed=917,
        )
        encoded = json.dumps(output, sort_keys=True, allow_nan=False)
        self.assertGreater(len(encoded), 1000)
        self.assertEqual(output["schema"], regression.SCHEMA)
        self.assertEqual(
            output["fingerprints"]["data"]["analysis_rows_sha256"],
            output["analysis"]["sample"]["analysis_rows_sha256"],
        )
        self.assertRegex(
            output["fingerprints"]["environment_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(output["artifact_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            output["fingerprints"]["source"][
                "native_coverage_artifact_sha256"
            ],
            stable_hash(self.artifact),
        )
        for section in ("protocol", "sources", "dependencies"):
            self.assertRegex(
                output["fingerprints"][section]["sha256"], r"^[0-9a-f]{64}$"
            )

    def test_every_optimizer_result_field_is_checked_fail_closed(self):
        design, _, _ = regression._standardized_design(self.artifact["records"])
        outcome = np.asarray(
            [
                record["deployed_observable_gate_coverage"]
                for record in self.artifact["records"]
            ],
            dtype=float,
        )
        original_minimize = regression.minimize
        mutations = {
            "failed_success": lambda result: setattr(result, "success", False),
            "failed_status": lambda result: setattr(result, "status", 2),
            "hostile_message": lambda result: setattr(
                result, "message", "precision loss"
            ),
            "nonfinite_parameter": lambda result: setattr(
                result, "x", np.full_like(result.x, np.nan)
            ),
            "nonfinite_objective": lambda result: setattr(result, "fun", np.nan),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                def poisoned_minimize(*args, **kwargs):
                    result = original_minimize(*args, **kwargs)
                    mutate(result)
                    return result

                with mock.patch.object(
                    regression, "minimize", side_effect=poisoned_minimize
                ):
                    with self.assertRaisesRegex(
                        regression.RegressionProtocolError,
                        "BFGS candidate rejected",
                    ):
                        regression._fractional_logit_fit(design, outcome)

    def test_bfgs_precision_loss_is_polished_without_relaxing_score_gate(self):
        design, _, _ = regression._standardized_design(self.artifact["records"])
        outcome = np.asarray(
            [
                record["deployed_observable_gate_coverage"]
                for record in self.artifact["records"]
            ],
            dtype=float,
        )
        original_minimize = regression.minimize

        def precision_loss_minimize(*args, **kwargs):
            result = original_minimize(*args, **kwargs)
            result.success = False
            result.status = 2
            result.message = regression.BFGS_PRECISION_LOSS_MESSAGE
            return result

        with mock.patch.object(
            regression, "minimize", side_effect=precision_loss_minimize
        ):
            fit = regression._fractional_logit_fit(design, outcome)
        self.assertEqual(
            fit["optimizer"]["bfgs"]["termination"],
            "precision_loss_candidate",
        )
        self.assertTrue(fit["optimizer"]["newton_polish"]["success"])
        self.assertLessEqual(
            fit["gradient_inf_norm"], regression.FINAL_SCORE_TOLERANCE
        )

    def test_failed_required_optimizer_prevents_artifact_build(self):
        fingerprints = source_fingerprints(self.artifact)
        original_minimize = regression.minimize

        def failed_minimize(*args, **kwargs):
            result = original_minimize(*args, **kwargs)
            result.success = False
            result.status = 2
            result.message = "precision loss"
            return result

        with mock.patch.object(
            regression, "minimize", side_effect=failed_minimize
        ):
            with self.assertRaisesRegex(
                regression.RegressionProtocolError, "BFGS candidate rejected"
            ):
                regression.build_output(
                    copy.deepcopy(self.artifact),
                    fingerprints,
                    bootstrap_replicates=20,
                )

    def test_rank_deficiency_fails_closed(self):
        records = copy.deepcopy(self.artifact["records"])
        for record in records:
            record["E_peak"] = record["E_mean"]
        with self.assertRaisesRegex(regression.RegressionProtocolError, "rank deficient"):
            regression.analyze_records(records, bootstrap_replicates=20)

    def test_too_few_family_clusters_fails_closed(self):
        artifact, _ = synthetic_fixture(family_count=6, rows_per_family=5)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "too few"):
            regression.analyze_records(artifact["records"], bootstrap_replicates=20)

    def test_bh_adjustment_matches_known_example(self):
        adjusted = regression._bh_adjust([0.01, 0.04, 0.03, 0.002])
        expected = [0.02, 0.04, 0.04, 0.008]
        for actual, target in zip(adjusted, expected, strict=True):
            self.assertAlmostEqual(actual, target)


class RegressionArtifactValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coverage_artifact, _ = synthetic_fixture()
        cls.source = source_fingerprints(cls.coverage_artifact)
        cls.output = regression.build_output(
            copy.deepcopy(cls.coverage_artifact),
            copy.deepcopy(cls.source),
            bootstrap_replicates=23,
            seed=481,
        )

    def validate(self, candidate):
        return regression.validate_regression_artifact(
            candidate,
            copy.deepcopy(self.coverage_artifact),
            copy.deepcopy(self.source),
            source="modified test input",
            require_publication=False,
        )

    def test_checksummed_artifact_passes_independent_recomputation(self):
        validation = self.validate(copy.deepcopy(self.output))
        self.assertEqual(
            validation["validation"], "recomputed-and-valid"
        )
        self.assertFalse(validation["publication_bootstrap_minimum_met"])

    def test_coefficient_tamper_is_rejected_by_checksum(self):
        tampered = copy.deepcopy(self.output)
        tampered["analysis"]["primary_models"][regression.OUTCOMES[0]][
            "coefficients"
        ][1]["estimate"] += 0.5
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "artifact checksum mismatch"
        ):
            self.validate(tampered)

    def test_checksummed_coefficient_tamper_is_rejected_by_recomputation(self):
        tampered = copy.deepcopy(self.output)
        tampered["analysis"]["primary_models"][regression.OUTCOMES[0]][
            "coefficients"
        ][1]["estimate"] += 0.5
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "recomputed analysis mismatch"
        ):
            self.validate(tampered)

    def test_checksummed_predictor_tamper_is_rejected_by_recomputation(self):
        tampered = copy.deepcopy(self.output)
        tampered["analysis"]["predictor_standardization"]["E_mean"]["mean"] += 0.01
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "recomputed analysis mismatch"
        ):
            self.validate(tampered)

    def test_checksummed_input_hash_tamper_is_rejected(self):
        tampered = copy.deepcopy(self.output)
        tampered["fingerprints"]["source"][
            "native_coverage_artifact_sha256"
        ] = "f" * 64
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "coverage input checksum mismatch"
        ):
            self.validate(tampered)

    def test_dependency_checksum_change_is_reported_not_rejected(
        self,
    ):
        drifted = copy.deepcopy(self.output)
        drifted["fingerprints"]["dependencies"]["sha256"] = "e" * 64
        update_regression_checksum(drifted)
        with self.assertWarnsRegex(
            RuntimeWarning, "dependencies checksum differs"
        ):
            validation = self.validate(drifted)
        self.assertEqual(
            validation["validation"], "recomputed-and-valid"
        )

    def test_source_checksum_change_is_reported_not_rejected(self):
        drifted = copy.deepcopy(self.output)
        drifted["fingerprints"]["sources"]["files"][0]["sha256"] = "d" * 64
        update_regression_checksum(drifted)
        with self.assertWarnsRegex(RuntimeWarning, "sources checksum differs"):
            validation = self.validate(drifted)
        self.assertEqual(
            validation["validation"], "recomputed-and-valid"
        )

    def test_protocol_checksum_tamper_is_rejected(self):
        tampered = copy.deepcopy(self.output)
        tampered["fingerprints"]["protocol"]["sha256"] = "c" * 64
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError,
            "protocol checksum mismatch",
        ):
            self.validate(tampered)

    def test_environment_checksum_tamper_is_rejected(self):
        tampered = copy.deepcopy(self.output)
        tampered["fingerprints"]["environment_sha256"] = "b" * 64
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError,
            "environment_sha256 checksum mismatch",
        ):
            self.validate(tampered)

    def test_checksummed_missing_coefficient_field_is_rejected(self):
        tampered = copy.deepcopy(self.output)
        del tampered["analysis"]["primary_models"][regression.OUTCOMES[0]][
            "coefficients"
        ][1]["estimate"]
        update_regression_checksum(tampered)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "missing or tampered"
        ):
            self.validate(tampered)

    def test_nonfinite_output_field_is_rejected_before_checksum_check(self):
        tampered = copy.deepcopy(self.output)
        tampered["analysis"]["primary_models"][regression.OUTCOMES[0]][
            "coefficients"
        ][1]["estimate"] = math.nan
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "non-finite field"
        ):
            self.validate(tampered)

    def test_cli_exposes_independent_validate_subcommand(self):
        args = regression.build_parser().parse_args(
            [
                "validate",
                "--coverage-json",
                "coverage.json",
                "--regression-json",
                "regression.json",
            ]
        )
        self.assertEqual(args.command, "validate")

    def test_file_validator_strictly_loads_and_recomputes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "regression.json"
            output_path.write_text(
                json.dumps(self.output, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with mock.patch.object(
                regression,
                "load_validated_coverage",
                return_value=(
                    copy.deepcopy(self.coverage_artifact),
                    copy.deepcopy(self.source),
                ),
            ):
                validation = regression.validate_regression_output(
                    output_path,
                    Path(directory) / "coverage.json",
                    require_publication=False,
                )
        self.assertEqual(
            validation["artifact_sha256"], self.output["artifact_sha256"]
        )


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.artifact, self.frame = synthetic_fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.json_path = self.root / "coverage.json"
        self.csv_path = self.root / "coverage.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, artifact=None):
        artifact = self.artifact if artifact is None else artifact
        self.json_path.write_text(
            json.dumps(artifact, sort_keys=True, allow_nan=False), encoding="utf-8"
        )

    def write_csv(self, artifact=None):
        artifact = self.artifact if artifact is None else artifact
        coverage_rq2.write_csv(self.csv_path, artifact)

    def load(self, *, csv_cross_check=False):
        with (
            mock.patch.object(
                regression,
                "resolve_circuit_corpus",
                side_effect=[copy.deepcopy(self.frame), copy.deepcopy(self.frame)],
            ),
            mock.patch.object(
                coverage_rq2, "validate_artifact", return_value=self.artifact
            ) as validator,
        ):
            result = regression.load_validated_coverage(
                self.json_path,
                csv_path=self.csv_path if csv_cross_check else None,
            )
        validator.assert_called_once()
        self.assertTrue(validator.call_args.kwargs["require_complete"])
        return result

    def test_current_schema_complete_artifact_and_csv_are_accepted(self):
        self.write_json()
        self.write_csv()
        artifact, fingerprints = self.load(csv_cross_check=True)
        self.assertEqual(artifact["record_count"], 48)
        self.assertRegex(fingerprints["coverage_json_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(fingerprints["coverage_csv_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            fingerprints["native_coverage_artifact_sha256"], stable_hash(artifact)
        )

    def test_legacy_schema_is_rejected_before_modeling(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["schema"] = "qmon-rq2-coverage-v1"
        self.write_json(artifact)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "legacy"):
            self.load()

    def test_protocol_drift_is_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["protocol"]["qmax"] = 23
        self.write_json(artifact)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "protocol"):
            self.load()

    def test_corpus_provenance_drift_is_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["corpus"]["corpus_sha256"] = "0" * 64
        self.write_json(artifact)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "corpus provenance"):
            self.load()

    def test_checksummed_outcome_tamper_is_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["records"][3]["deployed_observable_gate_coverage"] = 0.123
        update_checksum(artifact)
        self.write_json(artifact)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "derived outcome mismatch"):
            self.load()

    def test_checksummed_completed_row_drop_is_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        dropped = artifact["records"].pop()
        artifact["circuit_status"].pop(dropped["circuit"])
        update_checksum(artifact)
        self.write_json(artifact)
        with self.assertRaisesRegex(
            regression.RegressionProtocolError, "not complete for final reporting"
        ):
            self.load()

    def test_csv_tamper_is_rejected(self):
        self.write_json()
        self.write_csv()
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        column = rows[0].index("E_mean")
        rows[2][column] = "0.249"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        with self.assertRaisesRegex(regression.RegressionProtocolError, "JSON/CSV mismatch"):
            self.load(csv_cross_check=True)

    def test_duplicate_json_key_is_rejected(self):
        self.json_path.write_text(
            f'{{"schema":"{coverage_rq2.SCHEMA}","schema":"duplicate"}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(regression.RegressionProtocolError, "duplicate JSON key"):
            regression.load_validated_coverage(self.json_path)

    def test_nonfinite_json_constant_is_rejected(self):
        self.json_path.write_text('{"value":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(regression.RegressionProtocolError, "non-finite"):
            regression.load_validated_coverage(self.json_path)


if __name__ == "__main__":
    unittest.main()
