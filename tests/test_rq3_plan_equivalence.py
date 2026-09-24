"""Stored RQ3 plans may be alternative optima of the batch-partition MILP."""

import unittest

import rq3_realexec_eval


def _plan(batches):
    return {
        "qmax": 24,
        "original_qubits": 8,
        "batches": batches,
        "plan_sha256": rq3_realexec_eval.stable_hash({"b": batches}),
        "deployed_checkpoint_count": 6,
    }


class PlanEquivalenceTests(unittest.TestCase):

    def test_regrouped_plan_warns_but_validates(self):
        stored = _plan([{"batch": 0, "nodes": [0, 1, 2], "cost": 16},
                        {"batch": 1, "nodes": [3, 4, 5], "cost": 16}])
        recomputed = _plan([{"batch": 0, "nodes": [0, 1, 5], "cost": 15},
                            {"batch": 1, "nodes": [2, 3, 4], "cost": 16}])
        with self.assertWarnsRegex(RuntimeWarning, "alternative MILP optimum"):
            rq3_realexec_eval._check_plan_equivalent(stored, recomputed, "loc")

    def test_different_checkpoint_set_is_rejected(self):
        stored = _plan([{"batch": 0, "nodes": [0, 1, 2], "cost": 16},
                        {"batch": 1, "nodes": [3, 4, 5], "cost": 16}])
        recomputed = _plan([{"batch": 0, "nodes": [0, 1, 2], "cost": 16},
                            {"batch": 1, "nodes": [3, 4, 6], "cost": 16}])
        with self.assertRaisesRegex(ValueError, "different checkpoint set"):
            rq3_realexec_eval._check_plan_equivalent(stored, recomputed, "loc")

    def test_different_batch_count_is_rejected(self):
        stored = _plan([{"batch": 0, "nodes": [0, 1, 2, 3, 4, 5], "cost": 16}])
        recomputed = _plan([{"batch": 0, "nodes": [0, 1, 2], "cost": 16},
                            {"batch": 1, "nodes": [3, 4, 5], "cost": 16}])
        with self.assertRaisesRegex(ValueError, "uses 1 batches"):
            rq3_realexec_eval._check_plan_equivalent(stored, recomputed, "loc")

    def test_other_field_difference_is_rejected(self):
        stored = _plan([{"batch": 0, "nodes": [0, 1], "cost": 16}])
        recomputed = _plan([{"batch": 0, "nodes": [0, 1], "cost": 16}])
        recomputed["deployed_checkpoint_count"] = 7
        with self.assertRaisesRegex(ValueError, "deployed_checkpoint_count"):
            rq3_realexec_eval._check_plan_equivalent(stored, recomputed, "loc")


if __name__ == "__main__":
    unittest.main()
