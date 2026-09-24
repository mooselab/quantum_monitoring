"""Retry selection must preserve successful locations and original evidence."""

import copy
import tempfile
import unittest
from pathlib import Path

import rq1_full_tightness as rq1
import rq1_retry as retry
from test_rq1_full_tightness import _bell, _direct_result, _make_frame


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = _make_frame(self.root, {"bell.qasm": _bell()})
        original = _direct_result(self.frame["circuits"][0], self.frame["qasm_dir"], maximum=2)
        self.source = {"schema": rq1.SCHEMA, "circuit_results": [original], "complete": False}
        self.plan = retry.build_plan(self.source, "a" * 64, self.frame, points_per_task=1)

    def test_selects_only_unfinished_and_preserves_original(self):
        original = copy.deepcopy(self.source)
        retry.validate_plan(self.plan, self.frame)
        wanted = {p["point_id"] for p in self.source["circuit_results"][0]["points"]
                  if p["status"] in retry.RETRY_STATUSES}
        actual = {p["point_id"] for t in self.plan["tasks"] for p in t["points"]}
        self.assertEqual(actual, wanted)
        self.assertTrue(actual)
        self.assertEqual(self.source, original)
        self.assertFalse(self.plan["source_export_complete"])

    def test_rejects_changed_historical_record(self):
        self.source["circuit_results"][0]["points"][0]["gate"] += 1
        with self.assertRaisesRegex(rq1.ProtocolError, "checksum"):
            retry.build_plan(self.source, "a" * 64, self.frame)

    def test_partial_export_supplies_targets_but_not_complete_baseline(self):
        self.source["schema"] = "qmon-rq1-full-tightness-partial-v1"
        plan = retry.build_plan(self.source, "a" * 64, self.frame)
        self.assertFalse(plan["source_export_complete"])
        self.assertEqual(plan["point_count"], self.plan["point_count"])
        self.source["complete"] = True
        with self.assertRaisesRegex(ValueError, "completeness"):
            retry.build_plan(self.source, "a" * 64, self.frame)

    def test_rejects_changed_plan(self):
        self.plan["tasks"][0]["points"][0]["gate"] += 1
        with self.assertRaisesRegex(rq1.ProtocolError, "checksum"):
            retry.validate_plan(self.plan, self.frame)

    def test_rejects_duplicate_even_with_recomputed_plan_checksum(self):
        self.plan["tasks"][1]["points"] = copy.deepcopy(self.plan["tasks"][0]["points"])
        rq1._add_checksum(self.plan, "plan_sha256")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            retry.validate_plan(self.plan, self.frame)

    def test_rejects_completed_location(self):
        self.plan["tasks"][0]["points"][0]["previous_status"] = "success"
        rq1._add_checksum(self.plan, "plan_sha256")
        with self.assertRaisesRegex(ValueError, "completed"):
            retry.validate_plan(self.plan, self.frame)

    def test_rejects_changed_qasm(self):
        path = Path(self.frame["qasm_dir"]) / "bell.qasm"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "QASM"):
            retry.validate_plan(self.plan, self.frame)

    def test_real_retry_completes_only_one_requested_point(self):
        output = self.root / "retry.json"
        result = retry.run_task(self.plan, self.frame, output,
                                task_index=0, maximum=4, timeout=30)
        self.assertEqual(result["point_status_counts"], {"success": 1})
        self.assertEqual(result["requested_point_ids"], [result["points"][0]["point_id"]])
        self.assertLessEqual(result["points"][0]["law_abs_error"], 1e-12)
        rq1._check_checksum(rq1.read_artifact(output), "result_sha256", "retry output")
        with self.assertRaises(FileExistsError):
            retry.run_task(self.plan, self.frame, output, task_index=0)

    def test_timeout_preserves_requested_point_identity(self):
        result = retry.run_task(self.plan, self.frame, self.root / "timeout.json",
                                task_index=0, maximum=4, timeout=0.001)
        self.assertEqual(result["point_status_counts"], {"timeout": 1})
        self.assertEqual(result["points"][0]["point_id"], result["requested_point_ids"][0])


if __name__ == "__main__":
    unittest.main()
