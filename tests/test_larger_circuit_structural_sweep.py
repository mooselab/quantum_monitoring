"""Focused tests for the larger-circuit structural Qmax sweep."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import unittest

from qiskit import QuantumCircuit

import larger_circuit_structural_sweep as sweep
import scalability_v2 as scalability


class ForwardBitsetConeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_solver = os.environ.get("QMON_MILP_SOLVER")
        os.environ["QMON_MILP_SOLVER"] = "cbc"

    def tearDown(self) -> None:
        if self.previous_solver is None:
            os.environ.pop("QMON_MILP_SOLVER", None)
        else:
            os.environ["QMON_MILP_SOLVER"] = self.previous_solver

    def assert_matches_canonical(
        self, circuit: QuantumCircuit, nodes: dict[int, list[int]]
    ) -> None:
        expected = scalability.build_instrumentation_metrics(
            circuit,
            nodes,
            refinement_node_limit=0,
        )
        observed = sweep.build_structure_bitset(circuit, nodes)
        self.assertEqual(
            observed, sweep.structure_from_instrumentation(expected)
        )

    def test_single_and_two_qubit_cones_match_backward_reachability(self) -> None:
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)
        circuit.cx(1, 2)
        circuit.z(3)
        circuit.cx(2, 3)
        nodes = {
            gate: [
                circuit.find_bit(qubit).index
                for qubit in instruction.qubits
            ]
            for gate, instruction in enumerate(circuit.data)
        }
        self.assert_matches_canonical(circuit, nodes)

    def test_terminal_measurement_removes_only_free_final_reads(self) -> None:
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)
        circuit.barrier()
        circuit.measure(range(3), range(3))
        nodes = {0: [0], 1: [0, 1], 2: [2]}
        self.assert_matches_canonical(circuit, nodes)
        structure = sweep.build_structure_bitset(circuit, nodes)
        self.assertEqual(structure["free_final_read_event_count"], 3)
        self.assertEqual(structure["candidate_monitor_event_count"], 1)

    def test_same_gate_group_uses_union_cone_once(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        nodes = {2: [0, 1], 3: [1, 2]}
        self.assert_matches_canonical(circuit, nodes)
        structure = sweep.build_structure_bitset(circuit, nodes)
        self.assertEqual(
            [row["gate"] for row in structure["gate_node_costs"]],
            [2, 3],
        )
        self.assertEqual(
            structure["gate_node_costs"][0]["monitor_event_count"], 2
        )

    def test_entangle_then_uncompute_cones_remain_structural(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(1, 2)
        nodes = {3: [0], 4: [1, 2]}
        self.assert_matches_canonical(circuit, nodes)

    def test_mid_circuit_measurement_is_rejected(self) -> None:
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        with self.assertRaisesRegex(
            sweep.SweepError, "mid-circuit measurement"
        ):
            sweep._assert_supported_static_circuit(circuit)

    def test_reset_is_rejected(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.reset(0)
        with self.assertRaisesRegex(sweep.SweepError, "reset"):
            sweep._assert_supported_static_circuit(circuit)


class RetainedSourceResultTests(unittest.TestCase):
    """The sweep reads the retained 144-circuit result file."""

    SUBSET_PATH = (
        pathlib.Path(__file__).resolve().parents[1]
        / "results"
        / "larger_circuits"
        / "larger_circuits_144.jsonl"
    )
    SUMMARY_PATH = SUBSET_PATH.with_name("structural_summary.json")

    def test_only_the_144_record_input_is_accepted(self) -> None:
        self.assertEqual(
            sweep.SOURCE_RECORD_COUNT, sweep.SELECTOR_COMPLETE_EXTENSION_COUNT
        )

    def test_retained_source_loads_and_covers_the_144_cases(
        self,
    ) -> None:
        if not self.SUBSET_PATH.exists():
            self.skipTest("the shipped selector-complete subset is absent")
        self.assertEqual(
            sweep.file_sha256(self.SUBSET_PATH), sweep.SOURCE_ARTIFACT_SHA256
        )
        records = sweep.load_source_records(self.SUBSET_PATH)
        self.assertEqual(len(records), sweep.SOURCE_RECORD_COUNT)
        selected = sweep.selector_complete_extensions(records)
        self.assertEqual(
            len(selected), sweep.SELECTOR_COMPLETE_EXTENSION_COUNT
        )

    def test_other_source_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "unpinned.jsonl"
            path.write_text('{"attempt_id": "extension:ae:q15"}\n')
            with self.assertRaisesRegex(
                sweep.SweepError, "checksum mismatch"
            ):
                sweep.load_source_records(path)

    def test_result_field_update_preserves_values_and_integrity(self) -> None:
        manifest_path = self.SUBSET_PATH.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        seal = manifest.pop("manifest_sha256")
        self.assertEqual(seal, scalability.stable_hash(manifest))
        self.assertEqual(manifest["records_jsonl"]["size"], self.SUBSET_PATH.stat().st_size)
        self.assertEqual(
            manifest["records_jsonl"]["sha256"], sweep.SOURCE_ARTIFACT_SHA256
        )
        changed_format = manifest["format_change"]
        self.assertEqual(changed_format["record_schema_after"], scalability.SCHEMA_VERSION)
        excluded = {"schema_version", "provenance", "fingerprints", "record_sha256"}
        values_digest = hashlib.sha256()
        with self.SUBSET_PATH.open("rb") as source:
            for line, identity in zip(source, manifest["records"], strict=True):
                record = json.loads(line)
                self.assertEqual(record["attempt_id"], identity["attempt_id"])
                self.assertEqual(record["schema_version"], scalability.SCHEMA_VERSION)
                self.assertEqual(record["record_sha256"], identity["record_sha256"])
                self.assertEqual(record["record_sha256"], scalability._record_sha256(record))
                self.assertEqual(
                    hashlib.sha256(line.rstrip(b"\n")).hexdigest(),
                    identity["record_line_sha256"],
                )
                config = record["provenance"]["protocol_config"]
                self.assertEqual(config["schema_version"], scalability.SCHEMA_VERSION)
                self.assertEqual(
                    record["fingerprints"]["config_sha256"],
                    scalability.stable_hash(config),
                )
                self.assertEqual(
                    config["base_criterion_reference"], scalability.BASE_CRITERION_REFERENCE
                )
                self.assertEqual(
                    config["base_mps_crosscheck_reference"],
                    scalability.BASE_MPS_CROSSCHECK_REFERENCE,
                )
                values = {key: value for key, value in record.items() if key not in excluded}
                values_digest.update(
                    json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
                    + b"\n"
                )
        self.assertEqual(
            values_digest.hexdigest(), changed_format["unchanged_result_values_sha256"]
        )

    def test_reported_summary_matches_the_retained_source(self) -> None:
        summary = json.loads(self.SUMMARY_PATH.read_text(encoding="ascii"))
        source = summary["source"]
        self.assertEqual(
            source["records_sha256"], sweep.SOURCE_ARTIFACT_SHA256
        )
        plot_path = self.SUBSET_PATH.with_name(source["plot_data"])
        self.assertEqual(
            source["plot_data_sha256"],
            hashlib.sha256(plot_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(summary["population"]["circuits"], 144)
        rows = summary["structural_planning"]["reported_rounded_values"]
        self.assertEqual([row["qmax"] for row in rows], list(sweep.QMAX_VALUES))
        self.assertEqual(rows[-1]["fully_deployable"], 144)
        self.assertEqual(rows[-1]["selected_group_coverage_percent"], 100.0)


class SweepPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_solver = os.environ.get("QMON_MILP_SOLVER")
        os.environ["QMON_MILP_SOLVER"] = "cbc"

    def tearDown(self) -> None:
        if self.previous_solver is None:
            os.environ.pop("QMON_MILP_SOLVER", None)
        else:
            os.environ["QMON_MILP_SOLVER"] = self.previous_solver

    def test_sweep_excludes_qmax_24(self) -> None:
        self.assertEqual(sweep.QMAX_VALUES, (60, 70, 80, 90, 100))
        self.assertNotIn(24, sweep.QMAX_VALUES)

    def test_family_inventory_sums_to_144(self) -> None:
        self.assertEqual(sum(sweep.EXPECTED_FAMILY_COUNTS.values()), 144)
        self.assertEqual(len(sweep.EXPECTED_FAMILY_COUNTS), 27)

    def test_coverage_is_monotone_with_capacity(self) -> None:
        structure = {
            "original_qubits": 50,
            "original_unitary_gate_count_m": 20,
            "monitorable_node_count": 4,
            "monitorable_gate_node_count": 4,
            "candidate_monitor_event_count": 4,
            "candidate_gate_node_count": 4,
            "terminally_measured_qubits": [],
            "free_final_read_event_count": 0,
            "diagnostic_selected_per_event_cone_gate_occurrences": 20,
            "diagnostic_selected_per_event_cone_qubit_occurrences": 20,
            "diagnostic_candidate_per_event_cone_gate_occurrences": 20,
            "diagnostic_candidate_per_event_cone_qubit_occurrences": 20,
            "diagnostic_selected_per_event_extra_cone_qubit_occurrences": 16,
            "gate_node_costs": [
                {
                    "gate": gate,
                    "monitor_qubits": [gate % 2],
                    "monitor_event_count": 1,
                    "extra_qubit_demand": demand,
                    "cone_qubit_union_count": demand + 1,
                    "cone_gate_union_count": gate + 1,
                }
                for gate, demand in enumerate((5, 15, 25, 45))
            ],
        }
        rows = [
            sweep.partition_structure(structure, qmax)
            for qmax in sweep.QMAX_VALUES
        ]
        coverages = [row["deployed_gate_node_coverage"] for row in rows]
        self.assertEqual(coverages, sorted(coverages))
        self.assertEqual(rows[0]["deployed_gate_node_count"], 1)
        self.assertEqual(rows[-1]["deployed_gate_node_count"], 4)
        self.assertFalse(rows[0]["fully_deployable"])
        self.assertTrue(rows[-1]["fully_deployable"])

    def test_structure_tamper_is_rejected(self) -> None:
        structure = {
            "original_qubits": 2,
            "original_unitary_gate_count_m": 1,
            "monitorable_node_count": 1,
            "monitorable_gate_node_count": 1,
            "candidate_monitor_event_count": 2,
            "candidate_gate_node_count": 1,
            "terminally_measured_qubits": [],
            "free_final_read_event_count": 0,
            "diagnostic_selected_per_event_cone_gate_occurrences": 1,
            "diagnostic_selected_per_event_cone_qubit_occurrences": 1,
            "diagnostic_candidate_per_event_cone_gate_occurrences": 1,
            "diagnostic_candidate_per_event_cone_qubit_occurrences": 1,
            "diagnostic_selected_per_event_extra_cone_qubit_occurrences": 0,
            "gate_node_costs": [
                {
                    "gate": 0,
                    "monitor_qubits": [0],
                    "monitor_event_count": 1,
                    "extra_qubit_demand": 0,
                    "cone_qubit_union_count": 1,
                    "cone_gate_union_count": 1,
                }
            ],
        }
        with self.assertRaisesRegex(
            sweep.SweepError, "candidate monitor-event count drifted"
        ):
            sweep.validate_structure(structure)

    def test_distribution_reports_nearest_rank_quartiles(self) -> None:
        observed = sweep._distribution([1, 2, 3, 4])
        self.assertEqual(observed["quartile_1"], 1)
        self.assertEqual(observed["median"], 2.5)
        self.assertEqual(observed["quartile_3"], 3)


if __name__ == "__main__":
    unittest.main()
