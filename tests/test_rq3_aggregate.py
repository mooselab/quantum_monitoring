import contextlib
import copy
import io
import pathlib
import tempfile
import unittest
from unittest import mock

import rq3_realexec_eval as rq3
from rq3_realexec_eval import METHODS, aggregate, _common_complete_records


def _method_result(*, detected, complete=True, false_alarm=False):
    return {
        "complete": complete,
        "detected": detected,
        "loc_precision": 1.0,
        "false_alarm": false_alarm,
        "stats": {"n_exec": 1},
    }


def _mutant_record(
        *, detected, incomplete_method=None, circuit="a.qasm", seed=1,
        k=2, k_qmon=2, batches=1, in_scope_full=True,
        in_scope_qmon=True):
    methods = {
        method: _method_result(
            detected=detected if method == "qmon" else False,
            complete=method != incomplete_method,
        )
        for method in METHODS
    }
    return {
        "equivalent": False,
        "circuit": circuit,
        "seed": seed,
        "K": k,
        "K_qmon": k_qmon,
        "B": batches,
        "qmon_plan": {"plan_sha256": f"plan-{circuit}"},
        "in_scope_full": in_scope_full,
        "in_scope_qmon": in_scope_qmon,
        "deployment_gap": in_scope_full and not in_scope_qmon,
        "scope_class": (
            "in-scope-qmon" if in_scope_qmon
            else "deployment-gap" if in_scope_full
            else "outside-full-scope"
        ),
        "methods": methods,
    }


def _false_alarm_record(*, qmon_alarm, incomplete_method=None):
    return {
        "methods": {
            method: _method_result(
                detected=False,
                complete=method != incomplete_method,
                false_alarm=(
                    None if method == "mqt"
                    else qmon_alarm if method == "qmon" else False
                ),
            )
            for method in METHODS
        }
    }


class Rq3AggregationTests(unittest.TestCase):
    def _valid_plan(self):
        value = {
            "qmax": rq3.CANONICAL_QMAX,
            "original_qubits": 2,
            "batches": [{"batch": 0, "nodes": [0], "cost": 1}],
            "infeasible_nodes": [],
            "individually_infeasible_gate_nodes": [],
            "individually_infeasible_checkpoint_nodes": [],
            "all_monitorable_checkpoint_count": 2,
            "prebudget_instrumentable_checkpoint_count": 1,
            "deployed_checkpoint_count": 2,
            "deployed_mid_checkpoint_count": 1,
            "deployed_final_read_count": 1,
            "coverage_scope": "test fixture",
            "monitor_groups": [{"gate_index": 0, "qubits": [0]}],
            "grouped_monitor_channel": dict(rq3.GROUPED_MONITOR_CHANNEL),
            "final_read_policy": dict(rq3.FINAL_READ_POLICY),
        }
        plan = dict(value)
        plan["plan_sha256"] = rq3.stable_hash(value)
        return plan

    def test_qmax_plan_requires_fixed_final_read_policy(self):
        plan = self._valid_plan()
        rq3._validate_qmax_plan(plan, "fixture")
        plan["final_read_policy"]["assignment"] = "every batch"
        plan_value = dict(plan)
        plan_value.pop("plan_sha256")
        plan["plan_sha256"] = rq3.stable_hash(plan_value)
        with self.assertRaisesRegex(ValueError, "final-read policy mismatch"):
            rq3._validate_qmax_plan(plan, "fixture")

    def test_uses_one_common_method_complete_denominator(self):
        common = _mutant_record(detected=True)
        qmon_only = _mutant_record(
            detected=False, incomplete_method="proq"
        )
        fa_common = _false_alarm_record(qmon_alarm=False)
        fa_qmon_only = _false_alarm_record(
            qmon_alarm=True, incomplete_method="proq"
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            summary = aggregate(
                [common, qmon_only], [fa_common, fa_qmon_only]
            )

        self.assertEqual(summary["attempted_non_output_equivalent"], 2)
        self.assertEqual(
            summary["common_complete_non_output_equivalent"], 1
        )
        self.assertEqual(
            summary["method_complete_non_output_equivalent"]["qmon"], 2
        )
        self.assertEqual(
            summary["method_complete_non_output_equivalent"]["proq"], 1
        )
        self.assertEqual(summary["false_alarm_attempted"], 2)
        self.assertEqual(summary["false_alarm_common_complete"], 1)
        self.assertEqual(summary["false_alarm_method_complete"]["qmon"], 2)
        self.assertEqual(summary["false_alarm_method_complete"]["proq"], 1)
        self.assertEqual(summary["false_alarm_method_empirical"]["mqt"], 0)
        output = stdout.getvalue()
        self.assertIn("1/2 non-output-equivalent", output)
        self.assertIn("paired=1/2, complete=2/2", output)
        self.assertIn("FALSE ALARMS: 2 attempted circuits, 1", output)

    def test_missing_completion_flag_fails_closed(self):
        record = _mutant_record(detected=True)
        del record["methods"]["mqt"]["complete"]
        self.assertEqual(_common_complete_records([record]), [])
        with contextlib.redirect_stdout(io.StringIO()):
            summary = aggregate([record])
        completion = summary["completion"]["mutant_non_output_equivalent"]
        self.assertEqual(completion["common_method_complete"], 0)
        self.assertEqual(completion["excluded_from_paired_statistics"], 1)

    def test_scope_denominator_is_full_not_deployed(self):
        gap = _mutant_record(
            detected=False, in_scope_full=True, in_scope_qmon=False
        )
        with contextlib.redirect_stdout(io.StringIO()):
            summary = aggregate([gap])
        self.assertEqual(summary["common_complete_in_scope_full"], 1)
        self.assertEqual(summary["common_complete_in_scope_qmon"], 0)
        self.assertEqual(summary["common_complete_deployment_gap"], 1)

    def test_plan_costs_are_unique_circuit_weighted(self):
        records = [
            _mutant_record(
                detected=False, circuit="small.qasm", k=4,
                k_qmon=2, batches=1,
            ),
            _mutant_record(
                detected=False, circuit="large.qasm", seed=1, k=20,
                k_qmon=10, batches=9,
            ),
            _mutant_record(
                detected=False, circuit="large.qasm", seed=2, k=20,
                k_qmon=10, batches=9,
            ),
            _mutant_record(
                detected=False, circuit="large.qasm", seed=3, k=20,
                k_qmon=10, batches=9,
            ),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            summary = aggregate(records)
        costs = summary["unique_circuit_plan_costs"]
        self.assertEqual(costs["unique_circuits"], 2)
        self.assertEqual(costs["B"], {
            "mean": 5.0,
            "median": 5.0,
            "percentile_75_nearest_rank": 9,
            "maximum": 9,
        })
        self.assertEqual(costs["K_qmon"], {
            "mean": 6.0,
            "median": 6.0,
            "percentile_75_nearest_rank": 10,
            "maximum": 10,
        })
        self.assertEqual(costs["K"], {
            "mean": 12.0,
            "median": 12.0,
            "percentile_75_nearest_rank": 20,
            "maximum": 20,
        })


class Rq3MethodValidatorTests(unittest.TestCase):
    def setUp(self):
        self.record = {"K": 2, "K_qmon": 1, "B": 1, "mut_gate": 0}
        self.signature = {
            "nodes": ((0, 0), (2, 0)),
            "qmon_nodes": ((2, 0),),
            "cone": {
                (0, 0): frozenset({0}),
                (2, 0): frozenset({0, 1, 2}),
            },
        }
        method = "statistical"
        self.result = {
            "complete": True,
            "asserted_subsystem": rq3.METHOD_ASSERTED_SUBSYSTEM[method],
            "pruning_scope": rq3.PRUNING_SCOPE_RULE,
            "bonferroni_family_rule": rq3.BONFERRONI_FAMILY_RULE[method],
            "execution_shortcut_policy": rq3.EXECUTION_SHORTCUT_POLICY[method],
            "execution_mode": rq3.MUTANT_EXECUTION_MODE[method],
            "detected": True,
            "first_node": (2, 0),
            "loc_hit": True,
            "loc_precision": 1.0 / 3.0,
            "executed_program_count": 1,
            "counterfactual_marginal_count": 0,
            "stats": {
                "n_exec": 1,
                "n_null_shortcuts": 0,
                "n_analytical_marginals": 0,
                "complete": True,
                "incomplete_reasons": [],
                "family_size": 2,
                "alpha_node": rq3.ALPHA / 2,
                "bonferroni_family_rule": rq3.BONFERRONI_FAMILY_RULE[method],
                "execution_shortcut_policy": rq3.EXECUTION_SHORTCUT_POLICY[
                    method
                ],
                "execution_mode": rq3.MUTANT_EXECUTION_MODE[method],
            },
        }

    def validate(self, result):
        return rq3._validate_mutant_method_result(
            "statistical", result, self.record, self.signature, "fixture"
        )

    def test_current_valid_method_record_remains_accepted(self):
        self.assertIsNone(self.validate(copy.deepcopy(self.result)))

    def test_detected_none_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["detected"] = None
        with self.assertRaisesRegex(ValueError, "detected must be a bool"):
            self.validate(hostile)

    def test_negative_localization_precision_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["loc_precision"] = -0.01
        with self.assertRaisesRegex(ValueError, "localization precision"):
            self.validate(hostile)

    def test_method_and_stats_completion_disagreement_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["complete"] = False
        with self.assertRaisesRegex(ValueError, "disagrees with stats.complete"):
            self.validate(hostile)

    def test_truthy_non_boolean_stats_completion_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["stats"]["complete"] = 1
        with self.assertRaisesRegex(ValueError, "stats.complete must be a bool"):
            self.validate(hostile)

    def test_forged_localization_cone_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["loc_hit"] = False
        hostile["loc_precision"] = 0.0
        with self.assertRaisesRegex(ValueError, "recomputed cone"):
            self.validate(hostile)

    def test_first_node_outside_method_scan_set_is_rejected(self):
        hostile = copy.deepcopy(self.result)
        hostile["first_node"] = (99, 0)
        with self.assertRaisesRegex(ValueError, "outside its scan set"):
            self.validate(hostile)

    def test_completion_predicate_rejects_truthy_non_boolean(self):
        record = {"methods": {method: {"complete": True}
                              for method in METHODS}}
        record["methods"]["qmon"]["complete"] = 1
        self.assertEqual(rq3._common_complete_records([record]), [])


class Rq3TerminalOutcomeTests(unittest.TestCase):
    KEY = "graphstate_indep_qiskit_14.qasm"

    def _mqt_timeout_records(self):
        return [
            {
                "circuit": self.KEY,
                "slot_id": f"{self.KEY}:slot-{index}",
                "equivalent": False,
                "methods": {
                    "mqt": {
                        "complete": False,
                        "stats": {
                            "incomplete_reasons": [
                                "run-timeout:(5, 0)"
                            ]
                        },
                    }
                },
            }
            for index in range(15)
        ]

    def _slots(self):
        return [
            {"slot_id": f"{self.KEY}:slot-{index}"}
            for index in range(15)
        ]

    def test_mqt_single_call_timeouts_are_fixed_at_120_seconds(self):
        with mock.patch.object(
            rq3, "load_manifest", return_value={"manifest_sha256": "1" * 64}
        ), mock.patch.object(
            rq3, "runtime_fingerprints", return_value={}
        ):
            protocol, _ = rq3._rq3_artifact_protocol(
                ("mqt",), "mutant", rq3.MUTANT_CIRCUIT_BUDGET_SECONDS
            )
        caps = protocol["thresholds_and_caps"]
        self.assertEqual(rq3.MQT_APPLICABLE_TIMEOUT, 120)
        self.assertEqual(rq3.MQT_RUN_TIMEOUT, 120)
        self.assertEqual(caps["mqt_applicability_timeout_seconds"], 120)
        self.assertEqual(caps["mqt_run_timeout_seconds"], 120)

    def test_mqt_timeout_is_terminal_when_all_slots_are_recorded(self):
        records = self._mqt_timeout_records()
        with mock.patch.object(
            rq3, "canonical_mutation_slots", return_value=self._slots()
        ):
            status = rq3._terminal_circuit_status(
                records, ("mqt",), arm="mutant"
            )
        self.assertEqual(status["status"], "terminal-incomplete")
        self.assertEqual(status["incomplete_methods"], ["mqt"])
        self.assertTrue(status["terminal"])

    def test_resume_does_not_repeat_terminal_mqt_timeout(self):
        records = self._mqt_timeout_records()
        with mock.patch.object(
            rq3, "canonical_mutation_slots", return_value=self._slots()
        ):
            status = rq3._terminal_circuit_status(
                records, ("mqt",), arm="mutant"
            )
        artifact = {
            "protocol": {"test": "same"},
            "records": records,
            "circuit_status": {self.KEY: status},
        }
        expected = {
            "protocol": {"test": "same"},
            "records": [],
            "circuit_status": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "mqt.pkl"
            output.touch()
            with mock.patch.object(
                rq3, "load_rq3_artifact", return_value=artifact
            ), mock.patch.object(
                rq3, "new_rq3_artifact", return_value=expected
            ), mock.patch.object(
                rq3, "canonical_mutation_slots", return_value=self._slots()
            ), mock.patch.object(
                rq3, "evaluate_circuit"
            ) as evaluate, mock.patch.object(
                rq3, "save_rq3_artifact"
            ) as save:
                resumed = rq3.run_rq3_keys(
                    [self.KEY], output, methods=("mqt",), arm="mutant"
                )
        self.assertIs(resumed, artifact)
        evaluate.assert_not_called()
        save.assert_not_called()

    def test_strict_merge_accepts_terminal_mqt_timeout_inventory(self):
        slots = self._slots()

        def records(method, complete):
            return [
                {
                    "circuit": self.KEY,
                    "slot_id": slot["slot_id"],
                    "seed": index // 3 + 1,
                    "slot_ordinal": index % 3,
                    "equivalent": False,
                    "methods": {method: {"complete": complete}},
                }
                for index, slot in enumerate(slots)
            ]

        fingerprint = {
            field: {"sha256": "1" * 64}
            for field in (
                "sources", "qasm_corpus", "dependencies", "protocol",
                "circuit_budget",
            )
        }

        def artifact(method, complete):
            return {
                "protocol": {
                    "arm": "mutant",
                    "methods": [method],
                    "qmax": rq3.CANONICAL_QMAX,
                    "shots": rq3.SHOTS,
                    "alpha_familywise": rq3.ALPHA,
                    "branch_cap": rq3.BRANCH_CAP,
                    "circuit_budget_seconds": (
                        rq3.MUTANT_CIRCUIT_BUDGET_SECONDS
                    ),
                    "canonical_manifest_sha256": "2" * 64,
                },
                "fingerprints": copy.deepcopy(fingerprint),
                "records": records(method, complete),
                "circuit_status": {},
            }

        merged_base = {
            "protocol": {"arm": "mutant"},
            "fingerprints": copy.deepcopy(fingerprint),
            "records": [],
            "circuit_status": {},
        }
        with mock.patch.object(
            rq3, "load_rq3_artifact",
            side_effect=[artifact("qmon", True), artifact("mqt", False)],
        ), mock.patch.object(
            rq3, "new_rq3_artifact", return_value=merged_base
        ), mock.patch.object(
            rq3, "canonical_mutation_slots", return_value=slots
        ), mock.patch.object(
            rq3, "validate_rq3_artifact"
        ), mock.patch.object(
            rq3, "validate_final_rq3_artifact"
        ) as validate_final:
            merged = rq3.merge_rq3_artifacts(
                ["fast.pkl", "mqt.pkl"], strict=True
            )

        status = merged["circuit_status"][self.KEY]
        self.assertEqual(status["status"], "terminal-incomplete")
        self.assertEqual(status["incomplete_methods"], ["mqt"])
        self.assertEqual(len(merged["records"]), 15)
        validate_final.assert_called_once_with(merged, "merged artifact")

    def test_false_alarm_incomplete_record_is_also_terminal(self):
        record = {"methods": {"qmon": {"complete": False}}}
        status = rq3._terminal_circuit_status(
            [record], ("qmon",), arm="false-alarm"
        )
        self.assertEqual(status["status"], "terminal-incomplete")
        self.assertEqual(status["records"], 1)

    def test_false_alarm_resume_preserves_terminal_incomplete_record(self):
        record = {
            "circuit": self.KEY,
            "methods": {"qmon": {"complete": False}},
        }
        status = rq3._terminal_circuit_status(
            [record], ("qmon",), arm="false-alarm"
        )
        artifact = {
            "protocol": {"test": "same"},
            "records": [record],
            "circuit_status": {self.KEY: status},
        }
        expected = {
            "protocol": {"test": "same"},
            "records": [],
            "circuit_status": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "false-alarm.pkl"
            output.touch()
            with mock.patch.object(
                rq3, "load_rq3_artifact", return_value=artifact
            ), mock.patch.object(
                rq3, "new_rq3_artifact", return_value=expected
            ), mock.patch.object(
                rq3, "false_alarm_circuit"
            ) as evaluate, mock.patch.object(
                rq3, "save_rq3_artifact"
            ) as save:
                resumed = rq3.run_rq3_keys(
                    [self.KEY], output, methods=("qmon",),
                    arm="false-alarm"
                )
        self.assertIs(resumed, artifact)
        evaluate.assert_not_called()
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
