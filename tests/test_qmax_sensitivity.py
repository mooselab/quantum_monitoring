"""Focused tests for the Qmax threshold sensitivity protocol."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from qiskit import QuantumCircuit, qasm2


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import coverage_rq2 as rq2
import batch_partition
import qmax_sensitivity as qmax
import rq1_certificate_check as rq1
from entanglement_disturbance import (
    ProtocolError,
    file_sha256,
    resolve_circuit_corpus,
)
from utilities import filter_operation_list


class QmaxProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = resolve_circuit_corpus()
        cls.grid = qmax.build_attempt_grid(cls.frame)
        cls.bundle = qmax.build_fingerprint_bundle(cls.frame)

    def test_budget_grid_is_exact_and_headline_remains_24(self):
        self.assertEqual(qmax.QMAX_VALUES, (16, 20, 24, 28, 32, 36, 40))
        self.assertEqual(qmax.HEADLINE_QMAX, 24)
        self.assertEqual(len(self.grid), 310 * 7)
        self.assertEqual(
            Counter(attempt.qmax for attempt in self.grid),
            Counter({budget: 310 for budget in qmax.QMAX_VALUES}),
        )
        self.assertEqual(
            [attempt.attempt_index for attempt in self.grid],
            list(range(310 * 7)),
        )
        protocol = qmax.protocol_descriptor()
        self.assertEqual(protocol["qmax_values"], list(qmax.QMAX_VALUES))
        self.assertEqual(protocol["headline_rq1_rq3_qmax"], 24)
        self.assertEqual(protocol["record_schema"], qmax.SCHEMA)

    def test_same_two_qubit_gate_uses_one_union_group_and_real_emitter(self):
        entry = next(
            entry
            for entry in self.frame["circuits"]
            if entry["circuit"] == "qft_indep_qiskit_3.qasm"
        )
        key = entry["circuit"]
        qc = QuantumCircuit.from_qasm_file(
            str(Path(self.frame["qasm_dir"]) / key)
        )
        validation = rq1.crosscheck_selector(qc)
        operation_list, prebudget, _ = rq2._build_operation_list(
            qc, key, validation["selected_points"]
        )
        with qmax._rq2_qmax(16):
            groups, batches, infeasible, provenance = rq2._group_demands(
                qc, key, operation_list, prebudget
            )
            emitted, emitted_batches = rq2._emit_batches(
                qc, key, operation_list, groups, batches
            )
        self.assertFalse(infeasible)
        self.assertTrue(emitted_batches)
        self.assertEqual(
            provenance["solver"]["backend"], batch_partition.requested_backend()
        )
        first_batch = emitted_batches[0]
        self.assertGreater(first_batch["replay_gate_count"], 0)
        self.assertEqual(
            first_batch["monitor_measurement_count"],
            first_batch["monitor_point_count"],
        )
        self.assertEqual(
            first_batch["monitor_reset_count"],
            first_batch["monitor_point_count"],
        )
        self.assertGreaterEqual(first_batch["depth_inflation_ratio"], 1.0)

        rows = operation_list[key]
        witness = None
        for group in groups:
            source_instruction = qc.data[group["gate"]]
            if (
                source_instruction.operation.num_qubits != 2
                or len(group["members"]) != 2
            ):
                continue
            pointwise_costs = []
            gate_rows = [row for row in rows if row[0] == group["gate"]]
            for row in gate_rows:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, extra = filter_operation_list(key, {key: [row]}, 10 ** 7)
                pointwise_costs.append(len(extra[group["gate"]]))
            if (
                len(pointwise_costs) == 2
                and pointwise_costs[0] == pointwise_costs[1]
                and group["ancilla_demand"] < sum(pointwise_costs)
            ):
                witness = (group, pointwise_costs)
                break

        self.assertIsNotNone(witness)
        group, pointwise_costs = witness
        self.assertEqual(len(pointwise_costs), 2)
        self.assertLess(group["ancilla_demand"], sum(pointwise_costs))
        self.assertTrue(
            {(group["gate"], qubit) for qubit in group["members"]}.issubset(
                emitted
            )
        )
        self.assertEqual(
            sum(candidate["gate"] == group["gate"] for candidate in groups), 1
        )

    def test_deep_group_validation_rejects_repartitioned_stored_plan(self):
        entry = next(
            entry
            for entry in self.frame["circuits"]
            if entry["circuit"] == "qft_indep_qiskit_3.qasm"
        )
        key = entry["circuit"]
        qc = QuantumCircuit.from_qasm_file(
            str(Path(self.frame["qasm_dir"]) / key)
        )
        validation = rq1.crosscheck_selector(qc)
        operation_list, prebudget, _ = rq2._build_operation_list(
            qc, key, validation["selected_points"]
        )
        with qmax._rq2_qmax(16):
            groups, _batches, infeasible, provenance = rq2._group_demands(
                qc, key, operation_list, prebudget
            )
            split_batches = [
                {
                    "gate_groups": [group["gate"]],
                    "ancilla_cost": group["ancilla_demand"],
                }
                for group in groups
                if group["gate"] not in infeasible
            ]
            forged_provenance = copy.deepcopy(provenance)
            forged_provenance["final_batch_count"] = len(split_batches)
            override = {
                "batches": split_batches,
                "infeasible_gate_groups": infeasible,
                "partition_provenance": forged_provenance,
            }
            with self.assertRaisesRegex(ProtocolError, "partition certificate"):
                rq2._group_demands(
                    qc,
                    key,
                    operation_list,
                    prebudget,
                    partition_override=override,
                )

    def test_group_alone_infeasible_cannot_be_rescued(self):
        plan = qmax.partition_group_demands(
            15,
            [{"gate": 7, "members": [0, 1], "ancilla_demand": 2}],
            16,
        )
        self.assertEqual(plan["batches"], [])
        self.assertEqual(plan["feasible_groups"], [])
        self.assertEqual(plan["infeasible_gates"], [7])
        self.assertEqual(
            plan["infeasible_groups"],
            [{"gate": 7, "members": [0, 1], "ancilla_demand": 2}],
        )

    def test_distinct_feasible_groups_cross_batches(self):
        groups = [
            {"gate": 2, "members": [0], "ancilla_demand": 2},
            {"gate": 5, "members": [1], "ancilla_demand": 2},
        ]
        plan = qmax.partition_group_demands(14, groups, 16)
        self.assertEqual(plan["feasible_groups"], groups)
        self.assertEqual(plan["infeasible_groups"], [])
        self.assertEqual(len(plan["batches"]), 2)
        self.assertEqual(
            {tuple(batch["nodes"]) for batch in plan["batches"]},
            {(2,), (5,)},
        )

    def _final_read_fixture(self, directory: Path):
        key = "final-read-only.qasm"
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.x(1)
        qc.measure([0, 1], [0, 1])
        source = directory / key
        source.write_text(qasm2.dumps(qc), encoding="utf-8")
        entry = {
            "attempt_index": 0,
            "attempt_id": f"canonical:000:{key}",
            "circuit": key,
            "source_qasm_sha256": file_sha256(source),
        }
        frame = {
            "manifest": str(directory / "fixture-manifest.txt"),
            "manifest_sha256": "1" * 64,
            "corpus_sha256": "2" * 64,
            "circuit_count": 1,
            "circuits": [entry],
            "qasm_dir": str(directory),
        }
        attempt = qmax.build_attempt_grid(
            frame, require_canonical=False
        )[0]
        bundle = qmax.build_fingerprint_bundle(frame)
        return frame, attempt, bundle

    def test_final_read_convention_and_source_only_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, attempt, _ = self._final_read_fixture(Path(temporary))
            result = qmax.evaluate_budget(attempt, frame["qasm_dir"])
        self.assertEqual(result["final_read_points"], [[0, 0], [1, 1]])
        self.assertEqual(result["mid_candidate_points"], [])
        self.assertEqual(result["batches"], [])
        self.assertEqual(result["deployed_observable_points"], [[0, 0], [1, 1]])
        self.assertEqual(result["instrumented_batch_count"], 0)
        self.assertEqual(result["source_only_final_read_run_count"], 1)
        self.assertEqual(result["deployment_run_count"], 1)
        self.assertEqual(result["peak_deployment_run_width"], 2)
        self.assertEqual(result["peak_deployment_run_ancilla_count"], 0)
        self.assertEqual(result["planned_replay_gate_executions"], 0)
        self.assertEqual(
            result["planned_work_Bm_plus_replay"], result["gate_count"]
        )
        self.assertEqual(result["peak_depth_inflation_ratio"], 1.0)

    def test_record_tamper_is_caught_by_checksum_and_deep_derivation(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, attempt, bundle = self._final_read_fixture(Path(temporary))
            record = qmax.evaluate_attempt(
                attempt, frame["qasm_dir"], bundle
            )
            self.assertEqual(record["status"], "complete")
            qmax.validate_record(record, attempt, frame, bundle, deep=True)

            tampered = copy.deepcopy(record)
            tampered["result"]["deployment_run_count"] = 2
            with self.assertRaisesRegex(ProtocolError, "checksum mismatch"):
                qmax.validate_record(
                    tampered, attempt, frame, bundle, deep=True
                )
            qmax._add_record_checksum(tampered)
            with self.assertRaisesRegex(ProtocolError, "derived Qmax"):
                qmax.validate_record(
                    tampered, attempt, frame, bundle, deep=True
                )

    def test_record_from_another_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, attempt, bundle = self._final_read_fixture(Path(temporary))
            run_id = "a" * 64
            record = qmax.evaluate_attempt(
                attempt,
                frame["qasm_dir"],
                bundle,
                run_id=run_id,
            )
            qmax.validate_record(
                record,
                attempt,
                frame,
                bundle,
                expected_run_id=run_id,
                deep=True,
            )
            foreign = copy.deepcopy(record)
            foreign["run"]["run_id"] = "b" * 64
            qmax._add_record_checksum(foreign)
            with self.assertRaisesRegex(ProtocolError, "run_id mismatch"):
                qmax.validate_record(
                    foreign,
                    attempt,
                    frame,
                    bundle,
                    expected_run_id=run_id,
                    deep=True,
                )

    def test_deep_validation_does_not_resolve_optional_partition_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, attempt, bundle = self._final_read_fixture(Path(temporary))
            record = qmax.evaluate_attempt(
                attempt, frame["qasm_dir"], bundle
            )
            self.assertEqual(record["status"], "complete")
            with patch.object(
                rq2,
                "partition_nodes",
                side_effect=AssertionError("validator reran timed refinement"),
            ):
                qmax.validate_record(
                    record, attempt, frame, bundle, deep=True
                )

    def test_publication_attempt_timeout_covers_one_solve_and_validation(self):
        self.assertEqual(qmax.ATTEMPT_TIMEOUT_SECONDS, 7200)
        protocol = qmax.protocol_descriptor()
        self.assertEqual(protocol["per_attempt_timeout_seconds"], 7200)
        self.assertEqual(
            protocol["transient_license_retry_delays_seconds"],
            list(batch_partition.GUROBI_LICENSE_RETRY_DELAYS_SECONDS),
        )
        self.assertIn("without rerunning", protocol["partition_validation"])
        parser = qmax.build_parser()
        args = parser.parse_args(["run", "--output", "unused.jsonl"])
        self.assertEqual(args.attempt_timeout_seconds, 7200)
        args.attempt_timeout_seconds = 7199
        with self.assertRaisesRegex(ProtocolError, "requires.*7200"):
            qmax.run_protocol(args)
        if hasattr(qmax.signal, "SIGALRM"):
            previous_handler = qmax.signal.getsignal(qmax.signal.SIGALRM)
            previous_timer = qmax.signal.getitimer(qmax.signal.ITIMER_REAL)
            with self.assertRaisesRegex(TimeoutError, "exceeded"):
                with qmax._attempt_timeout(0.01):
                    qmax.time.sleep(0.05)
            self.assertEqual(
                qmax.signal.getsignal(qmax.signal.SIGALRM), previous_handler
            )
            restored_timer = qmax.signal.getitimer(qmax.signal.ITIMER_REAL)
            self.assertAlmostEqual(restored_timer[0], previous_timer[0], places=2)
            self.assertAlmostEqual(restored_timer[1], previous_timer[1], places=2)

    def test_checksummed_self_consistent_empty_selector_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, attempt, bundle = self._final_read_fixture(Path(temporary))
            record = qmax.evaluate_attempt(
                attempt, frame["qasm_dir"], bundle
            )
            self.assertEqual(record["status"], "complete")
            tampered = copy.deepcopy(record)
            result = tampered["result"]
            for field in (
                "certificate_points",
                "numerical_candidate_points",
                "final_read_points",
                "deployed_observable_points",
            ):
                result[field] = []
            result.update({
                "certificate_selected_point_count": 0,
                "certificate_selected_gate_count": 0,
                "certificate_selected_qubit_count": 0,
                "certificate_acted_point_coverage": 0.0,
                "certificate_gate_coverage": 0.0,
                "certificate_qubit_coverage": 0.0,
                "certificate_depth_reach": 0.0,
                "final_read_point_count": 0,
                "deployed_observable_point_count": 0,
                "deployed_observable_gate_count": 0,
                "deployed_observable_qubit_count": 0,
                "deployed_observable_acted_point_coverage": 0.0,
                "deployed_observable_gate_coverage": 0.0,
                "deployed_observable_qubit_coverage": 0.0,
                "deployed_observable_depth_reach": 0.0,
                "selected_identity_sha256": qmax.stable_hash([]),
                "deployed_observable_identity_sha256": qmax.stable_hash([]),
            })
            result.update(qmax._derive_sensitivity_fields(result))
            qmax._add_record_checksum(tampered)
            with self.assertRaisesRegex(
                ProtocolError, "source-recomputed scientific field mismatch"
            ):
                qmax.validate_record(
                    tampered, attempt, frame, bundle, deep=True
                )

    def _error_record_fixture(self):
        attempts = [
            qmax.BudgetAttempt(
                attempt_index=index,
                attempt_id=f"qmax:16:test:{index}",
                circuit_index=index,
                budget_index=0,
                circuit=f"fixture-{index}.qasm",
                qmax=16,
                source_qasm_sha256=f"{index + 3:x}" * 64,
                source_attempt_id=f"test:{index}",
            )
            for index in range(2)
        ]
        frame = {
            "manifest": "fixture-manifest.txt",
            "manifest_sha256": "1" * 64,
            "corpus_sha256": "2" * 64,
            "circuit_count": 2,
            "circuits": [],
            "qasm_dir": tempfile.gettempdir(),
        }
        bundle = qmax.build_fingerprint_bundle(frame)
        records = [
            qmax.terminal_error_record(
                attempt, bundle, RuntimeError(f"fixture {attempt.attempt_index}")
            )
            for attempt in attempts
        ]
        return frame, attempts, bundle, records

    def test_missing_duplicate_and_mixed_source_records_are_rejected(self):
        frame, attempts, bundle, records = self._error_record_fixture()
        with self.assertRaisesRegex(ProtocolError, "partial Qmax grid"):
            qmax.validate_record_set(
                records[:1],
                attempts,
                frame,
                bundle,
                require_complete=True,
            )
        with self.assertRaisesRegex(ProtocolError, "duplicate Qmax"):
            qmax.validate_record_set(
                [records[0], records[0]],
                [attempts[0]],
                frame,
                bundle,
                require_complete=True,
            )

        mixed = copy.deepcopy(records)
        mixed[1]["fingerprints"]["sources_sha256"] = "f" * 64
        qmax._add_record_checksum(mixed[1])
        with self.assertRaisesRegex(ProtocolError, "mixed sources_sha256"):
            qmax.validate_record_set(
                mixed,
                attempts,
                frame,
                bundle,
                require_complete=True,
            )

    def test_missing_budget_and_old_schema_are_rejected(self):
        frame, attempts, bundle, records = self._error_record_fixture()
        missing_budget = copy.deepcopy(records[0])
        missing_budget.pop("qmax")
        qmax._add_record_checksum(missing_budget)
        with self.assertRaisesRegex(
            ProtocolError, "field contract drift.*missing=.*qmax"
        ):
            qmax.validate_record(
                missing_budget, attempts[0], frame, bundle, deep=True
            )

        old_schema = copy.deepcopy(records[0])
        old_schema["schema"] = "qmon.qmax-sensitivity.v1"
        qmax._add_record_checksum(old_schema)
        with self.assertRaisesRegex(ProtocolError, "legacy or wrong"):
            qmax.validate_record(
                old_schema, attempts[0], frame, bundle, deep=True
            )

        extra = copy.deepcopy(records[0])
        extra["undeclared"] = True
        qmax._add_record_checksum(extra)
        with self.assertRaisesRegex(ProtocolError, "field contract drift"):
            qmax.validate_record(extra, attempts[0], frame, bundle, deep=True)

        boolean_elapsed = copy.deepcopy(records[0])
        boolean_elapsed["elapsed_seconds"] = False
        qmax._add_record_checksum(boolean_elapsed)
        with self.assertRaisesRegex(ProtocolError, "invalid elapsed"):
            qmax.validate_record(
                boolean_elapsed, attempts[0], frame, bundle, deep=True
            )

    def test_legacy_pickle_rejected_and_json_formats_are_atomic(self):
        _, _, _, records = self._error_record_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            legacy = directory / "qmax_records.pkl"
            legacy.write_bytes(b"legacy-list-pickle")
            with self.assertRaisesRegex(ProtocolError, "legacy qmax_records"):
                qmax.read_records(legacy)

            for suffix in (".jsonl", ".json"):
                output = directory / f"records{suffix}"
                qmax.write_records(records, output)
                self.assertEqual(qmax.read_records(output), records)
                self.assertFalse(list(directory.glob(f".{output.name}.*.tmp")))

    def test_qmax_reader_rejects_duplicate_keys_and_nonfinite_constants(self):
        _, _, _, records = self._error_record_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            duplicate = directory / "duplicate.jsonl"
            encoded = json.dumps(records[0], separators=(",", ":"))
            duplicate.write_text(
                '{"qmax":999,' + encoded[1:] + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ProtocolError, "duplicate JSON key"):
                qmax.read_records(duplicate)

            nonfinite = directory / "nonfinite.jsonl"
            nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "non-finite JSON"):
                qmax.read_records(nonfinite)

            overflow = directory / "overflow.jsonl"
            overflow.write_text('{"value":1e999}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "non-finite JSON"):
                qmax.read_records(overflow)

    def test_complete_310_by_7_identity_gate(self):
        identities = [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
            }
            for attempt in self.grid
        ]
        qmax._validate_identity_coverage(
            identities, self.grid, require_complete=True
        )
        with self.assertRaisesRegex(ProtocolError, "partial Qmax grid"):
            qmax._validate_identity_coverage(
                identities[:-1], self.grid, require_complete=True
            )
        with self.assertRaisesRegex(ProtocolError, "partial Qmax grid"):
            qmax.aggregate_records(
                [],
                self.frame,
                self.bundle,
                attempts=self.grid,
                deep=False,
            )


if __name__ == "__main__":
    unittest.main()
