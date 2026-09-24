from __future__ import annotations

import copy
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

import mqt_diagnostics as subject


def row(slot, *, complete, detected, stats, paired=False, seed=1):
    return {
        "slot_id": slot,
        "circuit": "c.qasm",
        "seed": seed,
        "paired_common_complete": paired,
        "methods": {
            "mqt": {
                "complete": complete,
                "detected": detected,
                "stats": {
                    "n_affected": 0,
                    "n_applicable": 0,
                    "run_fail": 0,
                    "run_unevaluable": 0,
                    "timeout": 0,
                    "error": 0,
                    "budget_exhausted": 0,
                    "applicability_budget-exhausted": 0,
                    "incomplete_reasons": [],
                    **stats,
                },
            }
        },
    }


class MQTDiagnosticsTests(unittest.TestCase):
    def test_paired_and_availability_denominators_are_separate(self):
        paired = row(
            "paired",
            complete=True,
            detected=True,
            paired=True,
            stats={"run_fail": 1},
        )
        timeout = row(
            "timeout",
            complete=False,
            detected=False,
            stats={
                "timeout": 2,
                "incomplete_reasons": ["run-timeout:(1, 0)"],
            },
        )
        budget = row(
            "budget",
            complete=False,
            detected=False,
            stats={
                "budget_exhausted": 3,
                "applicability_budget-exhausted": 4,
                "incomplete_reasons": [
                    "run-budget-exhausted:(2, 0)",
                    "applicability-budget-exhausted:(3, 0)",
                ],
            },
        )
        summary = subject.summarize(
            [paired, timeout, budget], [paired]
        )
        self.assertEqual(
            summary["paired_combined_fail_or_unevaluable_flags"], 1
        )
        self.assertEqual(
            summary["availability"]["terminal_incomplete_records"], 2
        )
        self.assertEqual(
            summary["availability"]["run_timeout_events"], 2
        )
        self.assertEqual(
            summary["availability"]["run_budget_exhausted_events"], 3
        )
        self.assertEqual(
            summary["availability"][
                "applicability_budget_exhausted_events"
            ],
            4,
        )

    def test_projection_rejects_tampering(self):
        paired = row(
            "paired",
            complete=True,
            detected=True,
            stats={"run_unevaluable": 1},
        )
        projection = subject.build_projection(
            [paired],
            [paired],
            source_artifact_file_sha256="a" * 64,
        )
        self.assertEqual(
            subject.validate_projection(projection)[
                "paired_unevaluable_or_refusal_outcomes"
            ],
            1,
        )
        tampered = copy.deepcopy(projection)
        tampered["summary"]["paired_assertion_failures"] = 9
        with self.assertRaisesRegex(
            subject.MQTDiagnosticsError, "checksum mismatch"
        ):
            subject.validate_projection(tampered)

    def test_seed_rates_use_unweighted_mean_and_sample_sd(self):
        rows = [
            row(
                "s1-mismatch", complete=True, detected=True, paired=True,
                seed=1, stats={"run_fail": 1},
            ),
            row(
                "s1-refusal", complete=True, detected=True, paired=True,
                seed=1, stats={"run_unevaluable": 1},
            ),
            row(
                "s2-pass", complete=True, detected=False, paired=True,
                seed=2, stats={},
            ),
            row(
                "s2-refusal", complete=True, detected=True, paired=True,
                seed=2, stats={"run_unevaluable": 1},
            ),
        ]
        rates = subject.summarize(rows, rows)["per_seed_rates"]
        self.assertEqual(
            rates["state_mismatch"]["unweighted_mean_percent"], 25.0
        )
        self.assertAlmostEqual(
            rates["state_mismatch"][
                "sample_standard_deviation_percent"
            ],
            35.35533905932738,
        )
        self.assertEqual(
            rates["unevaluable_or_refusal"]["unweighted_mean_percent"],
            50.0,
        )
        self.assertEqual(
            rates["combined_flag"]["unweighted_mean_percent"], 75.0
        )

    def test_incomplete_record_requires_reason(self):
        invalid = row("invalid", complete=False, detected=False, stats={})
        with self.assertRaisesRegex(
            subject.MQTDiagnosticsError, "lacks a reason"
        ):
            subject.summarize([invalid], [])

    def test_applicability_timeout_is_not_counted_as_run_timeout(self):
        applicability_timeout = row(
            "applicability-timeout",
            complete=False,
            detected=False,
            stats={
                "incomplete_reasons": ["applicability-timeout:(1, 0)"],
            },
        )
        summary = subject.summarize([applicability_timeout], [])
        self.assertEqual(
            summary["availability"]["records_with_run_timeout"], 0
        )
        self.assertEqual(
            summary["availability"]["records_with_other_incomplete_reason"],
            1,
        )

    def test_projection_from_rq3_file_uses_non_equivalent_complete_records(self):
        included = row(
            "included",
            complete=True,
            detected=True,
            stats={"run_fail": 1},
        )
        included["equivalent"] = False
        included["methods"]["qmon"] = {"complete": True}
        excluded = copy.deepcopy(included)
        excluded["slot_id"] = "equivalent"
        excluded["equivalent"] = True
        value = {
            "schema": subject.RQ3_SCHEMA,
            "protocol": {"arm": "mutant", "methods": ["qmon", "mqt"]},
            "records": [included, excluded],
        }
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rq3.pkl"
            source.write_bytes(payload)
            projection = subject.projection_from_rq3_file(source)
        self.assertEqual(projection["record_count"], 1)
        self.assertEqual(
            projection["source_artifact_file_sha256"],
            hashlib.sha256(payload).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
