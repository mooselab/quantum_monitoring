"""Focused invariants for multi_mutation_realexec_eval.

Run from the workspace root with:
    PYTHONPATH=reconstruction/src python -m unittest \
        reconstruction/tests/test_multi_mutation_realexec.py

Or from reconstruction/ with:
    PYTHONPATH=src python -m unittest tests/test_multi_mutation_realexec.py
"""

import sys
import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from qiskit import QuantumCircuit

import multi_mutation_realexec_eval as subject
import rq3_manifest
import rq3_realexec_eval as rq3


def _toy_qmon_plan(gate=4):
    plan_value = {
        "circuit": "unit.qasm",
        "source_qasm_sha256": "unit-source",
        "qmax": 24,
        "original_qubits": 2,
        "batches": [{"batch": 0, "nodes": [gate], "cost": 1}],
        "infeasible_nodes": [],
        "individually_infeasible_gate_nodes": [],
        "individually_infeasible_checkpoint_nodes": [],
        "all_monitorable_checkpoint_count": 1,
        "prebudget_instrumentable_checkpoint_count": 1,
        "deployed_checkpoint_count": 1,
        "deployed_mid_checkpoint_count": 1,
        "deployed_final_read_count": 0,
        "coverage_scope": "unit-test",
        "monitor_groups": [{"gate_index": gate, "qubits": [0]}],
        "grouped_monitor_channel": dict(rq3.GROUPED_MONITOR_CHANNEL),
        "final_read_policy": dict(rq3.FINAL_READ_POLICY),
    }
    plan = copy.deepcopy(plan_value)
    plan["plan_sha256"] = rq3_manifest.stable_hash(plan_value)
    return plan


def _toy_method_result(detected=False, complete=False):
    flagged = [(4, 0)] if detected else []
    return {
        "applicable": True,
        "evaluated": True,
        "complete": complete,
        "terminal": True,
        "detected": detected,
        "first_node": flagged[0] if flagged else None,
        "flagged_nodes": flagged,
        "n_flagged": len(flagged),
        "loc_any": detected,
        "loc_all": detected,
        "any_fault_in_scope": True,
        "all_faults_in_scope": True,
        "in_scope_faults": [4],
        "faults_in_first_cone": [4] if detected else [],
        "faults_in_flagged_union": [4] if detected else [],
        "first_cone_size": 1 if detected else 0,
        "flagged_union_cone_size": 1 if detected else 0,
        "executions": 1,
        "stats": {
            "n_exec": 1,
            "n_checked": 1,
            "n_decisions": 1,
            "n_affected": 1,
            "n_null_shortcuts": 0,
            "n_errors": 0,
            "n_flagged": len(flagged),
            "shots": subject.SHOTS,
            "alpha_node": subject.ALPHA,
            "complete": complete,
            "incomplete_reasons": [],
        },
    }


def _toy_record(methods=None, *, complete=False):
    if methods is None:
        methods = {"qmon": _toy_method_result(complete=complete)}
    return {
        "circuit": "unit.qasm",
        "seed": 1,
        "mode": "m1",
        "status": "ok",
        "attempted": True,
        "applicable": True,
        "executed": True,
        "terminal": True,
        "complete": complete,
        "fault_count": 1,
        "gates": [4],
        "fault_gates": [4],
        "application_order": [4],
        "candidate_count": 1,
        "K": 1,
        "K_qmon": 1,
        "B": 1,
        "shots": subject.SHOTS,
        "alpha_familywise": subject.ALPHA,
        "output_equivalent": False,
        "output_l1": 0.5,
        "masked": False,
        "masking_checked": False,
        "in_scope": True,
        "any_fault_in_scope": True,
        "all_faults_in_scope": True,
        "in_scope_faults": [4],
        "qmon_plan": _toy_qmon_plan(),
        "methods": methods,
    }


def _toy_protocol_artifact(record, methods=("qmon",)):
    protocol = {
        "methods": list(methods),
        "modes": [record["mode"]],
        "seeds": [record["seed"]],
        "qmax": 24,
        "circuit_budget_seconds": None,
    }
    fingerprints = {
        field: {"sha256": f"unit-{field}"}
        for field in (
            "sources", "qasm_corpus", "dependencies", "protocol",
            "circuit_budget",
        )
    }
    return {
        "schema": subject.SCHEMA,
        "created_at": "unit",
        "updated_at": "unit",
        "protocol": protocol,
        "fingerprints": fingerprints,
        "records": [record],
        "circuit_status": {},
    }, protocol, fingerprints


def _toy_inapplicable_record():
    reason = "unit-test protocol inapplicability"
    method = {
        "applicable": False,
        "evaluated": False,
        "complete": True,
        "terminal": True,
        "reason": reason,
        "detected": None,
        "first_node": None,
        "flagged_nodes": [],
        "n_flagged": 0,
        "loc_any": None,
        "loc_all": None,
        "any_fault_in_scope": False,
        "all_faults_in_scope": False,
        "in_scope_faults": [],
        "executions": 1,
        "stats": {},
    }
    record = _toy_record(methods={"qmon": method}, complete=True)
    record.update({
        "mode": "clus",
        "status": "inapplicable",
        "applicable": False,
        "executed": False,
        "fault_count": 0,
        "gates": [],
        "fault_gates": [],
        "application_order": [],
        "output_equivalent": None,
        "output_l1": None,
        "masked": None,
        "in_scope": False,
        "any_fault_in_scope": False,
        "all_faults_in_scope": False,
        "in_scope_faults": [],
        "inapplicable_reason": reason,
    })
    return record


def _toy_error_record():
    method = {
        "applicable": True,
        "evaluated": False,
        "complete": False,
        "terminal": True,
        "reason": "error",
        "detected": None,
        "first_node": None,
        "flagged_nodes": [],
        "n_flagged": 0,
        "loc_any": None,
        "loc_all": None,
        "any_fault_in_scope": True,
        "all_faults_in_scope": True,
        "in_scope_faults": [4],
        "executions": 1,
        "stats": {},
    }
    record = _toy_record(methods={"qmon": method})
    record.update({
        "status": "error",
        "executed": False,
        "output_equivalent": None,
        "output_l1": None,
        "masked": None,
        "error": {"type": "RuntimeError", "message": "unit failure"},
    })
    return record


def _validate_synthetic_record(record, *, strict=False):
    artifact, protocol, fingerprints = _toy_protocol_artifact(record)
    with mock.patch.object(
        subject, "_artifact_protocol",
        return_value=(protocol, fingerprints),
    ):
        if strict:
            return subject._strict_validate_merged(artifact)
        return subject.validate_artifact(artifact, "synthetic modified fixture")


def _toy_qmon_diag(gate=4):
    return {
        "complete": True,
        "batches": [{
            "batch": 0,
            "nodes": (gate,),
            "cost": 1,
            "emitted_qubits": 3,
            "monitor_groups": ({
                "gate_index": gate,
                "qubits": (0,),
                "replay_source_gate_indices": (),
                "ancilla_slice": (0,),
                "ancilla_qubits": (2,),
            },),
            "replay_fail_closed": True,
            "complete": True,
            "shot_table_sha256": "unit-shot-table",
            "incomplete_reasons": (),
        }],
    }


class MultiMutationRealExecTests(unittest.TestCase):
    def test_affected_when_any_fault_is_in_original_cone(self):
        self.assertTrue(subject.any_fault_in_cone({4, 19}, {1, 4, 8}))
        self.assertTrue(subject.any_fault_in_cone({4, 19}, {19}))
        self.assertFalse(subject.any_fault_in_cone({4, 19}, {1, 8}))

    def test_selection_and_replacements_are_deterministic(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.25, 1)
        candidates = [0, 1, 2]

        first_variants = subject.select_variants(circuit, candidates, 2, 17)
        second_variants = subject.select_variants(circuit, candidates, 2, 17)
        self.assertEqual(first_variants, second_variants)

        first, manifest = subject.mutate_many_with_manifest(
            circuit, [0, 1], key="unit.qasm", seed=17
        )
        second, repeated_manifest = subject.mutate_many_with_manifest(
            circuit, [0, 1], key="unit.qasm", seed=17
        )
        self.assertEqual(manifest, repeated_manifest)
        self.assertEqual(
            [subject.operation_signature(first, index) for index in range(3)],
            [subject.operation_signature(second, index) for index in range(3)],
        )

        for replacement in manifest:
            _, single_manifest = subject.mutate_many_with_manifest(
                circuit,
                [replacement["gate"]],
                key="unit.qasm",
                seed=17,
            )
            self.assertEqual(
                replacement["replacement"],
                single_manifest[0]["replacement"],
            )

    def test_singleton_qmon_context_delegates_to_main_rq3(self):
        sentinel = object()
        mutant, oracle = object(), object()
        with mock.patch.object(
            subject.rq3, "make_qmon_ctx", return_value=sentinel
        ) as delegated:
            actual = subject.make_qmon_ctx_many(
                mutant, "unit.qasm", oracle, {23}
            )
        self.assertIs(actual, sentinel)
        delegated.assert_called_once_with(mutant, "unit.qasm", oracle, 23)

    def test_singleton_qmon_context_forwards_unique_mutation_identity(self):
        sentinel = object()
        mutant, oracle = object(), object()
        with mock.patch.object(
            subject.rq3, "make_qmon_ctx", return_value=sentinel
        ) as delegated:
            actual = subject.make_qmon_ctx_many(
                mutant,
                "unit.qasm",
                oracle,
                {23},
                sample_tag="canonical:007:unit-slot",
            )
        self.assertIs(actual, sentinel)
        delegated.assert_called_once_with(
            mutant, "unit.qasm", oracle, "canonical:007:unit-slot"
        )

    def test_full_scan_preserves_first_flag_and_unions_later_cones(self):
        nodes = [(1, 0), (2, 0), (3, 0)]

        class Oracle:
            key = "unit"
            qmon_nodes = nodes
            K_qmon = 3
            K = 3
            p1e = {node: 0.0 for node in nodes}
            cone = {
                nodes[0]: {10},
                nodes[1]: {20},
                nodes[2]: {10, 20},
            }

        reads = {nodes[0]: 1.0, nodes[1]: 1.0, nodes[2]: 0.0}
        context = {
            "reads": reads.__getitem__,
            "raw_p1": {node: 0.0 for node in nodes},
        }

        def flag_on_one(node, p1_read, oracle, **kwargs):
            del node, oracle, kwargs
            return p1_read > 0.5

        with mock.patch.object(
            subject.rq3, "qmon_node_flag", side_effect=flag_on_one
        ) as decision:
            detected, first, flagged, stats = subject.scan_method_full(
                "qmon", None, Oracle(), "tag", {10, 20}, context
            )

        self.assertTrue(detected)
        self.assertEqual(first, nodes[0])
        self.assertEqual(flagged, nodes[:2])
        self.assertEqual(stats["n_checked"], 3)
        self.assertEqual(decision.call_count, 3)

        localization = subject.localization_metrics(
            flagged, Oracle.cone, {10, 20}
        )
        self.assertTrue(localization["loc_any"])
        self.assertTrue(localization["loc_all"])
        self.assertEqual(
            localization["faults_in_flagged_union"], [10, 20]
        )

    def test_method_chunks_merge_on_the_same_deterministic_record(self):
        record = _toy_record(methods={})
        qmon = subject.new_artifact(
            methods=("qmon",), modes=("m1",), seeds=(1,), qmax=24
        )
        qmon_record = copy.deepcopy(record)
        qmon_record["methods"] = {
            "qmon": _toy_method_result(detected=True)
        }
        qmon["records"] = [qmon_record]

        statistical = subject.new_artifact(
            methods=("statistical",), modes=("m1",), seeds=(1,), qmax=24
        )
        statistical_record = copy.deepcopy(record)
        statistical_record["methods"] = {
            "statistical": _toy_method_result(detected=False)
        }
        statistical["records"] = [statistical_record]

        with tempfile.TemporaryDirectory() as directory:
            qmon_path = Path(directory) / "qmon.pkl"
            statistical_path = Path(directory) / "statistical.pkl"
            subject.atomic_pickle_dump(qmon, qmon_path)
            subject.atomic_pickle_dump(statistical, statistical_path)
            merged = subject.merge_artifacts(
                [qmon_path, statistical_path], strict=False
            )

        self.assertEqual(len(merged["records"]), 1)
        self.assertEqual(
            set(merged["records"][0]["methods"]),
            {"qmon", "statistical"},
        )
        self.assertEqual(
            merged["protocol"]["methods"], ["qmon", "statistical"]
        )

    def test_canonical_manifest_counts_and_three_arm_identity(self):
        manifest = rq3_manifest.load_manifest(verify_runtime=False)
        self.assertEqual(
            manifest["counts"],
            {
                "circuits": 238,
                "slots": 3570,
                "output_equivalent": 710,
                "non_output_equivalent": 2860,
            },
        )
        rq3_manifest.assert_manifest_replay(manifest)
        key = "dj_indep_qiskit_3.qasm"
        expected = [
            slot["slot_id"]
            for slot in rq3_manifest.slots_for_circuit(manifest, key)
        ]
        canonical = rq3_manifest.slots_for_circuit(manifest, key)
        slots = rq3.canonical_mutation_slots(key, manifest)
        self.assertEqual([slot["slot_id"] for slot in slots], expected)
        self.assertEqual(slots, canonical)

    def test_main_rq3_binds_manifest_and_rejects_partial_final_artifact(self):
        manifest = rq3_manifest.load_manifest(verify_runtime=False)
        slot = manifest["slots"][0]
        plan_value = {
            "circuit": slot["circuit"],
            "source_qasm_sha256": slot["source_qasm_sha256"],
            "qmax": 24,
            "original_qubits": 1,
            "batches": [],
            "infeasible_nodes": [],
            "individually_infeasible_gate_nodes": [],
            "individually_infeasible_checkpoint_nodes": [],
            "all_monitorable_checkpoint_count": 0,
            "prebudget_instrumentable_checkpoint_count": 0,
            "deployed_checkpoint_count": 0,
            "deployed_mid_checkpoint_count": 0,
            "deployed_final_read_count": 0,
            "coverage_scope": "unit-test",
            "monitor_groups": [],
            "grouped_monitor_channel": dict(rq3.GROUPED_MONITOR_CHANNEL),
            "final_read_policy": dict(rq3.FINAL_READ_POLICY),
        }
        plan = dict(plan_value)
        plan["plan_sha256"] = rq3_manifest.stable_hash(plan_value)
        record = {
            "slot_id": slot["slot_id"],
            "circuit": slot["circuit"],
            "seed": int(slot["seed"]),
            "slot_ordinal": int(slot["ordinal"]),
            "mut_gate": int(slot["gate"]),
            "original": slot["original"],
            "replacement": slot["replacement"],
            "mutant_qasm_sha256": slot["mutant_qasm_sha256"],
            "equivalent": bool(slot["output_equivalent"]),
            "output_l1": slot["output_l1"],
            "manifest_sha256": manifest["manifest_sha256"],
            "K": 0,
            "K_qmon": 0,
            "B": 0,
            "in_scope_full": False,
            "in_scope_qmon": False,
            "deployment_gap": False,
            "scope_class": "outside-full-scope",
            "qmon_plan": plan,
        }
        expected_plan_value = copy.deepcopy(plan_value)
        expected_plan_value.update({
            "original_qubits": 1,
            "batches": [{"batch": 0, "nodes": [int(slot["gate"])],
                         "cost": 1}],
            "all_monitorable_checkpoint_count": 1,
            "prebudget_instrumentable_checkpoint_count": 1,
            "deployed_checkpoint_count": 1,
            "deployed_mid_checkpoint_count": 1,
            "monitor_groups": [{"gate_index": int(slot["gate"]),
                                "qubits": [0]}],
        })
        expected_plan = copy.deepcopy(expected_plan_value)
        expected_plan["plan_sha256"] = rq3_manifest.stable_hash(
            expected_plan_value
        )
        node = (int(slot["gate"]), 0)
        signature = {
            "K": 1,
            "K_qmon": 1,
            "B": 1,
            "qmon_plan": expected_plan,
            "nodes": (node,),
            "qmon_nodes": (node,),
            "cone": {node: frozenset({int(slot["gate"])})},
        }
        with mock.patch.object(
            rq3, "_canonical_oracle_signature", return_value=signature
        ):
            with self.assertRaisesRegex(ValueError, "stored K"):
                rq3._validate_canonical_mutant_record(record, "unit-test")

            valid = copy.deepcopy(record)
            valid.update({
                "K": 1,
                "K_qmon": 1,
                "B": 1,
                "in_scope_full": True,
                "in_scope_qmon": True,
                "deployment_gap": False,
                "scope_class": "in-scope-qmon",
                "qmon_plan": expected_plan,
            })
            rq3._validate_canonical_mutant_record(valid, "unit-test")
            scope_drift = copy.deepcopy(valid)
            scope_drift["in_scope_full"] = False
            with self.assertRaisesRegex(ValueError, "in_scope_full"):
                rq3._validate_canonical_mutant_record(
                    scope_drift, "unit-test"
                )

        tampered = copy.deepcopy(record)
        tampered["mutant_qasm_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "mutant_qasm_sha256"):
            rq3._validate_canonical_mutant_record(tampered, "unit-test")

        partial = rq3.new_rq3_artifact()
        with self.assertRaisesRegex(ValueError, "238-circuit"):
            rq3.validate_complete_rq3_artifact(partial, "unit-test")

    def test_manifest_tamper_fails_after_self_hash_is_recomputed(self):
        manifest = copy.deepcopy(rq3_manifest.load_manifest(
            verify_runtime=False))
        replacement = manifest["slots"][0]["replacement"]
        replacement["name"] = (
            "y" if replacement["name"] != "y" else "x")
        unhashed = copy.deepcopy(manifest)
        unhashed.pop("manifest_sha256")
        manifest["manifest_sha256"] = rq3_manifest.stable_hash(unhashed)
        with self.assertRaisesRegex(AssertionError, "replacement drift"):
            rq3_manifest.validate_manifest(
                manifest, verify_runtime=False, verify_replay=True)

    def test_main_rq3_oracle_signature_is_recomputed_from_qasm(self):
        manifest = rq3_manifest.load_manifest(verify_runtime=False)
        entry = manifest["circuits"][0]
        key = entry["circuit"]
        node = (3, 0)

        class FakeOracle:
            K = 1
            K_qmon = 1
            B = 1
            nodes = [node]
            qmon_nodes = [node]
            cone = {node: {0, 3}}
            qmon_plan = {
                "source_qasm_sha256": entry["source_qasm_sha256"],
                "plan_sha256": "recomputed-plan",
            }

            @staticmethod
            def eligible():
                return True

        with mock.patch.object(rq3, "Oracle", return_value=FakeOracle()) as made:
            signature = rq3._canonical_oracle_signature(key, cache={})
        made.assert_called_once_with(key, 24)
        self.assertEqual(signature["K"], 1)
        self.assertEqual(signature["qmon_plan"]["plan_sha256"],
                         "recomputed-plan")

    def test_qmon_pernode_schema_forbids_claimed_circuit_executions(self):
        result = {
            "execution_mode": rq3.MUTANT_EXECUTION_MODE["qmon_pernode"],
            "executed_program_count": 0,
            "counterfactual_marginal_count": 2,
            "stats": {"n_exec": 0, "n_analytical_marginals": 2},
        }
        rq3._validate_mutant_method_result(
            "qmon_pernode", result, "unit-test"
        )
        legacy = copy.deepcopy(result)
        legacy["executions"] = 2
        with self.assertRaisesRegex(ValueError, "legacy executions"):
            rq3._validate_mutant_method_result(
                "qmon_pernode", legacy, "unit-test"
            )
        false_claim = copy.deepcopy(result)
        false_claim["executed_program_count"] = 2
        with self.assertRaisesRegex(ValueError, "executed-program"):
            rq3._validate_mutant_method_result(
                "qmon_pernode", false_claim, "unit-test"
            )

    def test_mqt_false_alarm_is_conditional_definition_not_empirical(self):
        node = (3, 0)

        class FakeOracle:
            qc = object()
            K = 1
            K_qmon = 1
            B = 1
            nodes = [node]
            qmon_nodes = [node]
            qmon_plan = {"plan_sha256": "unit-plan"}

            @staticmethod
            def eligible():
                return True

        with mock.patch.object(rq3, "Oracle", return_value=FakeOracle()):
            record = rq3.false_alarm_circuit(
                "unit.qasm", methods=("mqt",)
            )
        result = record["methods"]["mqt"]
        self.assertEqual(result["execution_class"], "definition-shortcut")
        self.assertEqual(result["reference_applicability_status"],
                         "not-evaluated")
        self.assertEqual(result["status"],
                         "definition-shortcut-not-empirical")
        self.assertIsNone(result["false_alarm"])
        self.assertFalse(result["conditional_false_alarm"])
        self.assertEqual(result["n_executed_programs"], 0)
        rq3._validate_false_alarm_method_result(
            "mqt", result, record, "unit-test"
        )
        masquerading = copy.deepcopy(result)
        masquerading["false_alarm"] = False
        with self.assertRaisesRegex(ValueError, "masquerades as empirical"):
            rq3._validate_false_alarm_method_result(
                "mqt", masquerading, record, "unit-test"
            )

    def test_rq3_protocol_fingerprints_all_caps_and_thresholds(self):
        artifact = rq3.new_rq3_artifact(methods=("qmon",))
        parameters = artifact["protocol"]["thresholds_and_caps"]
        self.assertEqual(parameters["qmax"], 24)
        self.assertEqual(parameters["mqt_assert_eq_similarity"], 0.99)
        self.assertEqual(parameters["branch_cap"], 64)
        self.assertEqual(parameters["instrumentation_internal_qubit_limit"],
                         10 ** 7)
        self.assertEqual(parameters["false_alarm_real_run_k_cap"], 200)
        self.assertEqual(parameters["false_alarm_shot_seed"], 1)
        self.assertEqual(
            artifact["fingerprints"]["circuit_budget"]["value"], parameters
        )
        with self.assertRaisesRegex(ValueError, "fixed protocol parameters"):
            rq3.false_alarm_circuit(
                "unit.qasm", methods=("mqt",), real_run_k_cap=201
            )

    def test_replay_emission_error_marks_qmon_batch_incomplete(self):
        node = (0, 0)

        class Oracle:
            key = "unit.qasm"
            qmax = 24
            nodes = [node]
            qmon_nodes = [node]
            emitted = [node]
            lastgate = []
            batches = [{"nodes": [0], "cost": 0}]
            batch_of = {0: 0}
            op_list = {"unit.qasm": []}
            qmon_plan = {"plan_sha256": "plan"}
            K = 1
            K_qmon = 1
            B = 1
            cone = {node: {0}}
            p1e = {node: 0.0}

        with mock.patch.object(
                rq3, "generate_monitoring_circuit",
                side_effect=RuntimeError("unsupported replay gate")):
            context = rq3.make_qmon_joint_ctx(
                QuantumCircuit(1), "unit.qasm", Oracle(), "tamper",
                include_raw=False)
            self.assertFalse(context["materialize"]())
        self.assertFalse(context["diag"]["complete"])
        self.assertIn("unsupported replay gate",
                      " ".join(context["diag"]["incomplete_reasons"]))

    def test_canonical_multi_modes_are_nested_prefixes(self):
        manifest = rq3_manifest.load_manifest(verify_runtime=False)
        key = "dj_indep_qiskit_3.qasm"
        variants = subject.canonical_nested_variants(manifest, key, 1)
        slots = rq3_manifest.slots_for_circuit(manifest, key)[:3]
        self.assertEqual([mode for mode, _ in variants], ["m1", "m2", "m3"])
        self.assertEqual(variants[0][1], slots[:1])
        self.assertEqual(variants[1][1], slots[:2])
        self.assertEqual(variants[2][1], slots[:3])

    def test_cluster_none_emits_explicit_inapplicable_terminal_record(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.x(0)
        circuit.z(0)
        selected = subject.select_variants(circuit, [0, 1, 2], 2, 1)
        self.assertEqual(selected[-1], ("clus", []))

        node = (2, 0)

        class FakeOracle:
            qc = circuit
            nodes = [node]
            cone = {node: {0, 1, 2}}
            K = 1
            K_qmon = 1
            B = 1
            qmon_plan = {}

            @staticmethod
            def eligible():
                return True

        with mock.patch.object(
                subject, "build_oracle", return_value=FakeOracle()), \
             mock.patch.object(
                 subject, "canonical_nested_variants", return_value=[]):
            records = subject.evaluate_circuit(
                "unit.qasm",
                seeds=(1,),
                modes=("clus",),
                methods=("qmon", "mqt"),
                manifest={"manifest_sha256": "unit-manifest"},
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "inapplicable")
        self.assertTrue(record["attempted"])
        self.assertFalse(record["applicable"])
        self.assertFalse(record["executed"])
        self.assertTrue(record["terminal"])
        self.assertTrue(record["complete"])
        self.assertIsNone(record["output_equivalent"])
        self.assertEqual(record["gates"], [])
        self.assertEqual(set(record["methods"]), {"qmon", "mqt"})
        subject._validate_record_semantics(
            record, {"qmon", "mqt"}, "produced cluster inapplicable"
        )
        for result in record["methods"].values():
            self.assertFalse(result["applicable"])
            self.assertFalse(result["evaluated"])
            self.assertTrue(result["complete"])
            self.assertTrue(result["terminal"])

    def test_circuit_inapplicability_emits_every_declared_mode(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        class FakeOracle:
            qc = circuit
            nodes = []
            cone = {}
            K = 0
            K_qmon = 0
            B = None
            qmon_plan = {}

            @staticmethod
            def eligible():
                return False

        slots = []
        for ordinal in range(3):
            slots.append({
                "gate": ordinal,
                "slot_id": f"unit-s1-m{ordinal}",
                "original": {"name": "x", "params": [], "qubits": [0]},
                "replacement": {"name": "z", "params": [], "qubits": [0]},
            })
        nested = [
            ("m1", slots[:1]), ("m2", slots[:2]), ("m3", slots[:3])
        ]
        with mock.patch.object(
                subject, "build_oracle", return_value=FakeOracle()), \
             mock.patch.object(
                 subject, "canonical_nested_variants", return_value=nested):
            records = subject.evaluate_circuit(
                "unit.qasm",
                seeds=(1,),
                modes=subject.MODES,
                methods=("qmon",),
                manifest={"manifest_sha256": "unit-manifest"},
            )

        self.assertEqual(
            {(record["seed"], record["mode"]) for record in records},
            {(1, mode) for mode in subject.MODES},
        )
        self.assertTrue(all(
            record["status"] == "inapplicable"
            and record["terminal"]
            and record["complete"]
            for record in records
        ))
        for record in records:
            subject._validate_record_semantics(
                record, {"qmon"}, "produced circuit inapplicable"
            )

    def test_identity_coverage_rejects_silently_omitted_cluster_slot(self):
        records = [{"circuit": "unit.qasm", "seed": 1, "mode": "m1"}]
        with self.assertRaisesRegex(ValueError, "coverage failure"):
            subject._validate_identity_coverage(
                records,
                keys=("unit.qasm",),
                seeds=(1,),
                modes=("m1", "clus"),
                location="unit-test",
            )

    def test_artifact_rejects_malformed_inapplicable_cluster_terminal(self):
        record = _toy_inapplicable_record()
        record["executed"] = True
        artifact, protocol, fingerprints = _toy_protocol_artifact(record)
        with mock.patch.object(
            subject, "_artifact_protocol",
            return_value=(protocol, fingerprints),
        ):
            with self.assertRaisesRegex(ValueError, "malformed inapplicable"):
                subject.validate_artifact(artifact, "unit-test")

    def test_deep_validator_accepts_a_consistent_synthetic_record(self):
        completed = _toy_record(complete=True)
        completed["methods"]["qmon"]["diag"] = _toy_qmon_diag()
        for expected, record in (
            ("ok", _toy_record()),
            ("ok", completed),
            ("error", _toy_error_record()),
            ("inapplicable", _toy_inapplicable_record()),
        ):
            with self.subTest(status=expected):
                validated = _validate_synthetic_record(record)
                self.assertEqual(
                    validated["records"][0]["status"], expected
                )

    def test_deep_validator_rejects_hostile_method_state_and_accounting(self):
        corruptions = {
            "not-evaluated-ok-method": lambda method: method.update(
                evaluated=False
            ),
            "nullable-detection": lambda method: method.update(detected=None),
            "localization-without-detection": lambda method: method.update(
                loc_any=True
            ),
            "negative-flag-count": lambda method: method.update(n_flagged=-1),
            "negative-executions": lambda method: method.update(executions=-1),
            "negative-added-cost": lambda method: method.update(
                execution_cost=-0.25
            ),
            "negative-added-precision": lambda method: method.update(
                loc_precision=-0.01
            ),
            "nonfinite-added-cost": lambda method: method.update(
                execution_cost=float("nan")
            ),
            "null-added-precision": lambda method: method.update(
                loc_precision=None
            ),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                record = _toy_record()
                corrupt(record["methods"]["qmon"])
                with self.assertRaises(ValueError):
                    _validate_synthetic_record(record)

    def test_deep_validator_rejects_nonboolean_completeness(self):
        record = _toy_record()
        record["complete"] = 0
        with self.assertRaisesRegex(ValueError, "complete must be a strict bool"):
            _validate_synthetic_record(record)

    def test_deep_validator_rejects_error_and_inapplicable_branch_leaks(self):
        hostile_error = _toy_error_record()
        hostile_error["output_equivalent"] = False
        hostile_error["output_l1"] = 0.5
        with self.assertRaisesRegex(ValueError, "malformed error status"):
            _validate_synthetic_record(hostile_error)

        missing_error = _toy_error_record()
        del missing_error["error"]
        with self.assertRaisesRegex(ValueError, "malformed error payload"):
            _validate_synthetic_record(missing_error)

        hostile_inapplicable = _toy_inapplicable_record()
        hostile_inapplicable["methods"]["qmon"]["detected"] = False
        with self.assertRaisesRegex(ValueError, "inapplicable method branch"):
            _validate_synthetic_record(hostile_inapplicable)

    def test_strict_merge_enters_the_same_deep_validator_first(self):
        record = _toy_record()
        record["methods"]["qmon"]["detected"] = None
        with self.assertRaisesRegex(ValueError, "evaluated method branch"):
            _validate_synthetic_record(record, strict=True)

    def test_checkpoint_completion_rejects_truthy_nonboolean_method_state(self):
        record = _toy_record(complete=True)
        record["methods"]["qmon"]["complete"] = "yes"
        self.assertFalse(subject._circuit_records_complete(
            [record], methods=("qmon",), modes=("m1",), seeds=(1,)
        ))

    def test_resume_keeps_terminal_method_incomplete_circuit(self):
        methods, modes, seeds = ("qmon",), ("m1",), (1,)
        artifact = subject.new_artifact(
            methods=methods, modes=modes, seeds=seeds, qmax=24
        )
        artifact["records"] = [_toy_record()]
        artifact["circuit_status"]["unit.qasm"] = {
            "status": "complete",
            "all_methods_complete": False,
        }
        completed = copy.deepcopy(artifact["records"][0])
        completed["methods"]["qmon"]["complete"] = True
        completed["methods"]["qmon"]["stats"]["complete"] = True
        completed["methods"]["qmon"]["diag"] = _toy_qmon_diag()
        completed["complete"] = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pkl"
            subject.atomic_pickle_dump(artifact, path)
            with mock.patch.object(
                subject, "evaluate_circuit", return_value=[completed]
            ) as evaluate:
                resumed = subject.run_keys(
                    ["unit.qasm"],
                    out=path,
                    methods=methods,
                    modes=modes,
                    seeds=seeds,
                    qmax=24,
                    resume=True,
                )
        evaluate.assert_not_called()
        self.assertFalse(resumed["records"][0]["methods"]["qmon"]["complete"])
        self.assertEqual(
            resumed["circuit_status"]["unit.qasm"]["status"], "complete"
        )

    def test_resume_reexecutes_missing_identity_frame(self):
        methods, modes, seeds = ("qmon",), ("m1", "m2"), (1,)
        artifact = subject.new_artifact(
            methods=methods, modes=modes, seeds=seeds, qmax=24
        )
        artifact["records"] = [_toy_record()]
        artifact["circuit_status"]["unit.qasm"] = {"status": "incomplete"}
        completed = copy.deepcopy(artifact["records"][0])
        completed["methods"]["qmon"]["complete"] = True
        completed["methods"]["qmon"]["stats"]["complete"] = True
        completed["methods"]["qmon"]["diag"] = _toy_qmon_diag()
        completed["complete"] = True
        second = copy.deepcopy(completed)
        second["mode"] = "m2"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pkl"
            subject.atomic_pickle_dump(artifact, path)
            with mock.patch.object(
                subject, "evaluate_circuit", return_value=[completed, second]
            ) as evaluate:
                resumed = subject.run_keys(
                    ["unit.qasm"], out=path, methods=methods,
                    modes=modes, seeds=seeds, qmax=24, resume=True,
                )
        evaluate.assert_called_once()
        self.assertEqual(len(resumed["records"]), 2)
        self.assertTrue(
            resumed["circuit_status"]["unit.qasm"]["all_methods_complete"]
        )

    def test_merge_rejects_conflicting_slot_metadata(self):
        first = subject.new_artifact(
            methods=("qmon",), modes=("m1",), seeds=(1,), qmax=24
        )
        second = subject.new_artifact(
            methods=("statistical",), modes=("m1",), seeds=(1,), qmax=24
        )
        base = _toy_record(methods={})
        first_record = copy.deepcopy(base)
        first_record["methods"] = {"qmon": _toy_method_result()}
        second_record = copy.deepcopy(base)
        second_record["gates"] = [5]
        second_record["fault_gates"] = [5]
        second_record["application_order"] = [5]
        second_record["in_scope_faults"] = [5]
        second_record["methods"] = {
            "statistical": _toy_method_result()
        }
        second_record["methods"]["statistical"]["in_scope_faults"] = [5]
        first["records"] = [first_record]
        second["records"] = [second_record]
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.pkl"
            two = Path(directory) / "two.pkl"
            subject.atomic_pickle_dump(first, one)
            subject.atomic_pickle_dump(second, two)
            with self.assertRaises(ValueError):
                subject.merge_artifacts([one, two], strict=False)

    def test_mqt_timeout_is_not_cached_and_budget_is_fresh(self):
        node = (2, 0)

        class FakeOracle:
            _mqt_semantic_applicability = {}
            qc = object()
            amps = {node: object()}

        oracle = FakeOracle()
        with mock.patch.object(rq3, "_mqt_qasm", return_value="qasm"), \
             mock.patch.object(rq3, "_mqt_call", side_effect=["timeout", "pass"]):
            first = rq3.MQTMutantBudget(timeout_limit=1, wall_limit=10)
            second = rq3.MQTMutantBudget(timeout_limit=1, wall_limit=10)
            self.assertEqual(
                rq3.Oracle.mqt_applicability_status(oracle, node, first),
                "timeout",
            )
            self.assertNotIn(node, oracle._mqt_semantic_applicability)
            self.assertEqual(
                rq3.Oracle.mqt_applicability_status(oracle, node, second),
                "pass",
            )
            self.assertEqual(second.timeouts, 0)

    def test_qmon_incomplete_context_is_not_a_miss(self):
        node = (1, 0)

        class Oracle:
            key = "unit"
            qmon_nodes = [node]
            K_qmon = 1
            K = 1
            cone = {node: {0}}
            p1e = {node: 0.0}

        context = {
            "materialize": lambda: False,
            "diag": {"complete": False,
                     "incomplete_reasons": ["branch-cap"]},
        }
        detected, first, flagged, stats = subject.scan_method_full(
            "qmon", None, Oracle(), "tag", {0}, context
        )
        self.assertFalse(detected)
        self.assertIsNone(first)
        self.assertEqual(flagged, [])
        self.assertFalse(stats["complete"])

    def test_noncanonical_qmax_is_rejected(self):
        with self.assertRaises(ValueError):
            subject.new_artifact(qmax=20)
        with self.assertRaises(ValueError):
            subject.evaluate_circuit(
                "dj_indep_qiskit_3.qasm",
                seeds=(1,), modes=("m1",), methods=("qmon",), qmax=20,
            )
        artifact = rq3.new_rq3_artifact(methods=("qmon",))
        rq3.validate_rq3_artifact(artifact)
        malformed = copy.deepcopy(artifact)
        malformed["protocol"]["qmax"] = 20
        with self.assertRaises(ValueError):
            rq3.validate_rq3_artifact(malformed)
        with self.assertRaises(ValueError):
            rq3.validate_rq3_artifact([])

    def test_independent_qubit_fault_is_outside_all_baseline_scopes(self):
        node = (0, 0)

        class Oracle:
            key = "independent-two-qubit"
            nodes = [node]
            qmon_nodes = [node]
            K = 1
            K_qmon = 1
            p1e = {node: 0.0}
            cone = {node: {0}}

            @staticmethod
            def deterministic(_node):
                return True

        circuit = QuantumCircuit(2)
        circuit.id(0)  # checkpoint cone contains gate 0 only
        circuit.x(1)   # fault gate 1 acts on the independent qubit

        with mock.patch.object(rq3, "run_statistical") as statistical, \
             mock.patch.object(rq3, "run_proq") as proq, \
             mock.patch.object(rq3, "run_liu") as liu, \
             mock.patch.object(rq3, "run_mqt_status") as mqt:
            for method in ("statistical", "proq", "liu", "mqt"):
                detected, first, stats = rq3.scan_method(
                    method, circuit, Oracle(), "independent", 1
                )
                self.assertFalse(detected, method)
                self.assertIsNone(first, method)
                self.assertEqual(stats["n_exec"], 0, method)
                self.assertEqual(stats["n_null_shortcuts"], 1, method)
                self.assertTrue(stats["complete"], method)
        statistical.assert_not_called()
        proq.assert_not_called()
        liu.assert_not_called()
        mqt.assert_not_called()
        for method in ("statistical", "proq", "liu", "mqt"):
            self.assertIn("one-qubit", rq3.METHOD_ASSERTED_SUBSYSTEM[method])
            self.assertNotIn("global", rq3.METHOD_ASSERTED_SUBSYSTEM[method])
        self.assertEqual(
            rq3.EXACT_BINOMIAL_CONVENTION["equality"],
            "pvalue == alpha_node does not flag",
        )

    def test_exact_binomial_alpha_equality_does_not_flag(self):
        node = (0, 0)

        class Oracle:
            p1e = {node: 0.5}

            @staticmethod
            def deterministic(_node):
                return False

        result = mock.Mock(pvalue=rq3.ALPHA)
        with mock.patch.object(rq3, "binomtest", return_value=result) as tested:
            self.assertFalse(
                rq3.qmon_count_flag(
                    node, 4, Oracle(), shots=8, alpha_node=rq3.ALPHA
                )
            )
        tested.assert_called_once_with(4, 8, 0.5, alternative="two-sided")


if __name__ == "__main__":
    unittest.main()
