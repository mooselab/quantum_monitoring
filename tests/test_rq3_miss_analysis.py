import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rq3_miss_analysis as subject


def _artifact():
    digest = "a" * 64
    return {
        "schema": subject.EXPECTED_INPUT_SCHEMA,
        "protocol": {
            "arm": "mutant",
            "canonical_manifest_sha256": "b" * 64,
        },
        "fingerprints": {
            field: {"sha256": digest}
            for field in (
                "sources",
                "qasm_corpus",
                "dependencies",
                "protocol",
                "circuit_budget",
            )
        },
        "records": [
            {"slot_id": f"circuit.qasm|s{seed}|m0", "equivalent": False}
            for seed in subject.SEEDS
        ],
    }


def _row(seed, *, miss=False, common=True, suffix="0"):
    complete = list(subject.REPORTED_METHODS) if common else ["qmon"]
    return {
        "slot_id": f"family_indep_qiskit_3.qasm|s{seed}|m{suffix}",
        "circuit": "family_indep_qiskit_3.qasm",
        "seed": seed,
        "circuit_family": "family",
        "original_gate_arity": 1,
        "replacement_gate_arity": 1,
        "original_operation_class": "x",
        "replacement_operation_class": "h",
        "operation_replacement_class": "x->h",
        "scope_class": "in-scope-qmon",
        "complete_methods": complete,
        "common_complete": common,
        "qmon_detected": (not miss) if common else None,
        "qmon_miss": miss if common else None,
        "baseline_detected": miss if common else None,
        "reported_outcome_category": (
            "excluded-incomplete"
            if not common
            else "qmon-miss-caught-by-baseline"
            if miss
            else "qmon-detected"
        ),
        "structural_miss_class": (
            "excluded-incomplete"
            if not common
            else "deployed-checkpoint-present-no-qmon-flag"
            if miss else "not-a-qmon-miss"
        ),
    }


class StrictInputTests(unittest.TestCase):
    def test_bv_replacement_uses_existing_bv_family(self):
        self.assertEqual(
            subject._family_of("bv_alt_indep_qiskit_12.qasm"),
            "bv",
        )

    def test_legacy_artifact_is_rejected_before_current_validator(self):
        with mock.patch.object(
            subject.rq3, "validate_final_rq3_artifact"
        ) as validator:
            with self.assertRaisesRegex(subject.MissAnalysisError, "legacy"):
                subject.validate_final_artifact([], "legacy")
        validator.assert_not_called()

    def test_partial_artifact_validator_failure_is_not_swallowed(self):
        artifact = _artifact()
        with mock.patch.object(
            subject.rq3,
            "validate_final_rq3_artifact",
            side_effect=ValueError("exact complete 238-circuit reference corpus"),
        ):
            with self.assertRaisesRegex(ValueError, "238-circuit"):
                subject.validate_final_artifact(artifact, "partial")

    def test_mixed_fingerprint_validator_failure_is_not_swallowed(self):
        artifact = _artifact()
        with mock.patch.object(
            subject.rq3,
            "validate_final_rq3_artifact",
            side_effect=ValueError("sources checksum mismatch"),
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                subject.validate_final_artifact(artifact, "mixed")

    def test_malformed_fingerprint_fails_after_validator(self):
        artifact = _artifact()
        artifact["fingerprints"]["dependencies"]["sha256"] = "tampered"
        with mock.patch.object(
            subject.rq3, "validate_final_rq3_artifact", return_value=artifact
        ):
            with self.assertRaisesRegex(subject.MissAnalysisError, "dependencies"):
                subject.validate_final_artifact(artifact, "tampered")

    def test_wrong_arm_fails_closed(self):
        artifact = _artifact()
        artifact["protocol"]["arm"] = "false-alarm"
        with mock.patch.object(
            subject.rq3, "validate_final_rq3_artifact", return_value=artifact
        ):
            with self.assertRaisesRegex(subject.MissAnalysisError, "mutant arm"):
                subject.validate_final_artifact(artifact, "false-alarm")


class DenominatorAndRateTests(unittest.TestCase):
    def test_common_complete_denominator_excludes_incomplete_attempt(self):
        rows = [
            _row(seed, miss=seed in (1, 3)) for seed in subject.SEEDS
        ]
        rows.append(_row(1, miss=True, common=False, suffix="1"))
        strata, enriched = subject._summarize_rows(rows)

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(enriched), 5)
        overall = subject._rate_summary(enriched)
        self.assertEqual(
            overall["exact_counts"],
            {"attempts": 5, "qmon_misses": 2, "qmon_detections": 3},
        )
        self.assertAlmostEqual(overall["per_seed_rate_mean"], 0.4)
        self.assertAlmostEqual(
            overall["per_seed_rate_sample_sd"],
            subject.statistics.stdev([1.0, 0.0, 1.0, 0.0, 0.0]),
        )
        self.assertEqual(len(strata["circuit_family"][0]["per_seed"]), 5)

    def test_sparse_stratum_and_missing_seed_are_explicit(self):
        summary = subject._rate_summary([_row(1, miss=True)])
        self.assertTrue(summary["sparse"])
        self.assertIn("denominator<20", summary["sparse_reasons"])
        self.assertIn("represented_seeds<5", summary["sparse_reasons"])
        self.assertIsNone(summary["per_seed"][1]["qmon_miss_rate"])
        lower, upper = summary["pooled_qmon_miss_rate_wilson_95"]
        self.assertLess(lower, upper)
        self.assertEqual(upper, 1.0)

    def test_only_reviewer_requested_strata_are_emitted(self):
        rows = [_row(seed) for seed in subject.SEEDS]
        strata, _ = subject._summarize_rows(rows)
        self.assertEqual(set(strata), set(subject.STRATIFIERS))
        self.assertNotIn("mutation_position_quantile", strata)
        self.assertNotIn("candidate_checkpoint_count", strata)


class CategoryAndOutputTests(unittest.TestCase):
    def _detections(self, **updates):
        values = {method: False for method in subject.REPORTED_METHODS}
        values.update(updates)
        return values

    def test_outcome_categories_are_exhaustive(self):
        self.assertEqual(
            subject._qmon_outcome_category(self._detections(qmon=True)),
            "qmon-detected",
        )
        self.assertEqual(
            subject._qmon_outcome_category(self._detections(proq=True)),
            "qmon-miss-caught-by-baseline",
        )
        self.assertEqual(
            subject._qmon_outcome_category(self._detections()),
            "qmon-miss-missed-by-all-reported-methods",
        )

    def test_structural_miss_classes_are_scope_derived(self):
        self.assertEqual(
            subject._structural_miss_class("outside-full-scope", False),
            "no-downstream-monitorable-checkpoint",
        )
        self.assertEqual(
            subject._structural_miss_class("deployment-gap", False),
            "monitorable-checkpoint-not-deployed-under-qmax",
        )
        self.assertEqual(
            subject._structural_miss_class("in-scope-qmon", False),
            "deployed-checkpoint-present-no-qmon-flag",
        )
        self.assertEqual(
            subject._structural_miss_class("in-scope-qmon", True),
            "not-a-qmon-miss",
        )

    def test_report_and_json_are_deterministic_and_noninferential(self):
        artifact = _artifact()
        rows = [_row(seed, miss=seed in (2, 5)) for seed in subject.SEEDS]
        with mock.patch.object(
            subject.rq3, "validate_final_rq3_artifact", return_value=artifact
        ), mock.patch.object(subject, "_derive_attempt_rows", return_value=rows):
            first = subject.analyze_artifact(
                copy.deepcopy(artifact), artifact_file_sha256="c" * 64
            )
            second = subject.analyze_artifact(
                copy.deepcopy(artifact), artifact_file_sha256="c" * 64
            )
        self.assertEqual(first, second)
        self.assertFalse(first["exploratory_inference"]["performed"])
        self.assertEqual(first["exploratory_inference"]["p_values"], [])
        self.assertEqual(len(first["report_sha256"]), 64)
        unhashed = dict(first)
        report_hash = unhashed.pop("report_sha256")
        self.assertEqual(report_hash, subject.stable_hash(unhashed))
        self.assertEqual(
            first["denominators"]["common_complete_non_output_equivalent"], 5
        )
        self.assertEqual(
            first["denominators"][
                "excluded_incomplete_non_output_equivalent"
            ],
            0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            one = Path(temporary) / "one.json"
            two = Path(temporary) / "two.json"
            subject.write_json(one, first)
            subject.write_json(two, second)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            decoded = json.loads(one.read_text(encoding="ascii"))
        self.assertEqual(decoded["report_sha256"], first["report_sha256"])

    def test_terminal_incomplete_attempt_is_excluded_not_counted_as_miss(self):
        artifact = _artifact()
        rows = [_row(seed, miss=seed == 2) for seed in subject.SEEDS]
        rows.append(_row(1, miss=True, common=False, suffix="1"))
        with mock.patch.object(
            subject.rq3, "validate_final_rq3_artifact", return_value=artifact
        ), mock.patch.object(subject, "_derive_attempt_rows", return_value=rows):
            report = subject.analyze_artifact(
                artifact, artifact_file_sha256="d" * 64
            )
        self.assertEqual(
            report["overall"]["exact_counts"],
            {"attempts": 5, "qmon_misses": 1, "qmon_detections": 4},
        )
        self.assertEqual(
            report["denominators"][
                "excluded_incomplete_non_output_equivalent"
            ],
            1,
        )

    def test_cli_refuses_to_overwrite_input_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact.pkl"
            artifact.write_bytes(b"not read because path collision is checked first")
            with self.assertRaises(SystemExit):
                subject.main([str(artifact), str(artifact)])


if __name__ == "__main__":
    unittest.main()
