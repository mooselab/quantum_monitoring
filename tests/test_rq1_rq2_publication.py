"""Focused tests for the final RQ1/RQ2 rerun path.

Run from ``reconstruction/`` with::

    python -m unittest -v test_rq1_rq2_publication.py
"""

from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qiskit import QuantumCircuit, qasm2

import coverage_rq2
import rq1_certificate_check as rq1
from entanglement_disturbance import (
    EXPECTED_CIRCUITS,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256,
    ProtocolError,
    TIGHTNESS_KEYS,
    TIGHTNESS_SCHEMA,
    _exact_branches,
    _select_tightness_points,
    atomic_write_json,
    build_tightness_artifact,
    file_sha256,
    resolve_circuit_corpus,
    stable_hash,
    tightness_protocol_descriptor,
    validate_tightness_artifact,
)
from fast_analysis import analyze_per_gate_nolock


def fixture_corpus_hash(keys, qasm_dir):
    return stable_hash([
        {
            "basename": Path(key).stem,
            "sha256": file_sha256(qasm_dir / key),
        }
        for key in keys
    ])


def certificate_artifact(frame, status, records):
    return {
        "schema": rq1.SCHEMA,
        "protocol": rq1.certificate_protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame.get("manifest_sha256"),
            "corpus_sha256": frame.get("corpus_sha256"),
            "circuit_count": frame.get("circuit_count", len(frame["circuits"])),
        },
        "circuit_status": {
            frame["circuits"][0]["circuit"]: copy.deepcopy(status)
        },
        "summary": rq1._certificate_summary(records),
        "record_count": len(records),
        "records_sha256": stable_hash(records),
        "records": records,
    }


def coverage_artifact(frame, status, record):
    records = [record]
    return {
        "schema": coverage_rq2.SCHEMA,
        "protocol": coverage_rq2.protocol_descriptor(),
        "corpus": {
            "manifest_sha256": frame.get("manifest_sha256"),
            "corpus_sha256": frame.get("corpus_sha256"),
            "circuit_count": frame.get("circuit_count", len(frame["circuits"])),
        },
        "circuit_status": {record["circuit"]: copy.deepcopy(status)},
        "record_count": 1,
        "records_sha256": stable_hash(records),
        "summary": coverage_rq2.summarize(records),
        "records": records,
    }


def update_coverage_checksums(artifact):
    artifact["records_sha256"] = stable_hash(artifact["records"])
    artifact["summary"] = coverage_rq2.summarize(artifact["records"])
    for record in artifact["records"]:
        artifact["circuit_status"][record["circuit"]]["record_sha256"] = (
            stable_hash(record)
        )


class CircuitCorpusTests(unittest.TestCase):
    def test_repository_frame_contains_exactly_310_circuits(self):
        frame = resolve_circuit_corpus()
        self.assertEqual(frame["circuit_count"], EXPECTED_CIRCUITS)
        self.assertEqual(frame["manifest_sha256"], EXPECTED_MANIFEST_SHA256)
        self.assertEqual(frame["corpus_sha256"], EXPECTED_CORPUS_SHA256)
        self.assertEqual(
            len({entry["attempt_id"] for entry in frame["circuits"]}),
            EXPECTED_CIRCUITS,
        )

    def test_missing_and_substituted_qasm_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qasm_dir = root / "qasm"
            qasm_dir.mkdir()
            manifest = root / "manifest.txt"
            manifest.write_text("alpha\nbeta\n", encoding="utf-8")
            (qasm_dir / "alpha.qasm").write_text("alpha", encoding="utf-8")
            (qasm_dir / "beta.qasm").write_text("beta", encoding="utf-8")
            manifest_hash = file_sha256(manifest)
            corpus_hash = fixture_corpus_hash(
                ["alpha.qasm", "beta.qasm"], qasm_dir
            )

            frame = resolve_circuit_corpus(
                manifest,
                qasm_dir,
                expected_count=2,
                expected_manifest_sha256=manifest_hash,
                expected_corpus_sha256=corpus_hash,
            )
            self.assertEqual(frame["circuit_count"], 2)

            (qasm_dir / "beta.qasm").unlink()
            with self.assertRaisesRegex(ProtocolError, "QASM missing"):
                resolve_circuit_corpus(
                    manifest,
                    qasm_dir,
                    expected_count=2,
                    expected_manifest_sha256=manifest_hash,
                    expected_corpus_sha256=corpus_hash,
                )

            (qasm_dir / "beta.qasm").write_text(
                "substituted beta", encoding="utf-8"
            )
            with self.assertRaisesRegex(ProtocolError, "corpus checksum"):
                resolve_circuit_corpus(
                    manifest,
                    qasm_dir,
                    expected_count=2,
                    expected_manifest_sha256=manifest_hash,
                    expected_corpus_sha256=corpus_hash,
                )

    def test_manifest_row_drop_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qasm_dir = root / "qasm"
            qasm_dir.mkdir()
            manifest = root / "manifest.txt"
            manifest.write_text("alpha\n", encoding="utf-8")
            (qasm_dir / "alpha.qasm").write_text("alpha", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "requires 2 circuits"):
                resolve_circuit_corpus(
                    manifest,
                    qasm_dir,
                    expected_count=2,
                    expected_manifest_sha256=None,
                    expected_corpus_sha256=None,
                )

    def test_missing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ProtocolError, "manifest missing"):
                resolve_circuit_corpus(
                    root / "absent.txt",
                    root,
                    expected_count=1,
                    expected_manifest_sha256=None,
                    expected_corpus_sha256=None,
                )

    def test_json_writer_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            value = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
            atomic_write_json(path, value)
            first = path.read_bytes()
            atomic_write_json(path, value)
            self.assertEqual(first, path.read_bytes())


class SelectorConsistencyTests(unittest.TestCase):
    def test_source_replay_tolerance_covers_backend_roundoff_only(self):
        rq1._require_source_close(
            2.1027550526066466e-14,
            1.1027518695667238e-13,
            "selector_a2",
            ("fixture.qasm", 0, 0),
        )
        with self.assertRaisesRegex(ProtocolError, "source replay mismatch"):
            rq1._require_source_close(
                0.0,
                3.0e-13,
                "selector_a2",
                ("fixture.qasm", 0, 0),
            )

    def test_implemented_selector_matches_independent_direct_svd(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        validation = rq1.crosscheck_selector(qc)

        self.assertEqual(validation["acted_point_count"], 5)
        self.assertEqual(
            {(row["gate"], row["qubit"])
             for row in validation["selected_points"]},
            {(0, 0), (2, 0), (2, 1)},
        )
        for row in validation["all_points"]:
            self.assertAlmostEqual(
                row["selector_a2"], row["independent_svd_a2"], places=13
            )

    def test_selected_point_wrapper_and_coverage_use_same_crosscheck(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        expected = rq1.crosscheck_selector(qc)
        self.assertEqual(
            rq1.crosscheck_selected_points(qc),
            (expected["acted_point_count"], expected["selected_points"]),
        )
        self.assertIs(coverage_rq2.crosscheck_selector, rq1.crosscheck_selector)

    def test_selector_path_drop_is_a_hard_error(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        result, events, paths = analyze_per_gate_nolock(qc)
        self.assertEqual(paths[0], [0])
        with mock.patch.object(
            rq1, "analyze_per_gate_nolock",
            return_value=(result, events, {0: [], 1: []}),
        ):
            with self.assertRaisesRegex(ProtocolError, "cross-check mismatch"):
                rq1.crosscheck_selector(qc)

    def test_determinant_residual_is_not_a_selector_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            qasm_dir = Path(temporary)
            key = "product.qasm"
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)
            (qasm_dir / key).write_text(qasm2.dumps(qc), encoding="utf-8")
            entry = {
                "attempt_index": 0,
                "attempt_id": "canonical:000:product.qasm",
                "circuit": key,
                "source_qasm_sha256": file_sha256(qasm_dir / key),
            }
            frame = {"circuits": [entry], "qasm_dir": str(qasm_dir)}
            status, records = rq1.evaluate_circuit(entry, qasm_dir)
            records[0]["determinant_residual"] = 5.163e-15
            records[0]["determinant_residual_minus_svd"] = (
                records[0]["determinant_residual"] - records[0]["det_rho_svd"]
            )
            artifact = certificate_artifact(frame, status, records)
            rq1.validate_certificate_artifact(
                artifact, frame, require_complete=True
            )
            self.assertEqual(records[0]["selector_a2"], 0.0)

            drifted = copy.deepcopy(artifact)
            drifted["protocol"]["selector_criterion"]["threshold"] = 1e-9
            with self.assertRaisesRegex(ProtocolError, "protocol drift"):
                rq1.validate_certificate_artifact(
                    drifted, frame, require_complete=True
                )

            wrong_corpus = copy.deepcopy(artifact)
            wrong_corpus["corpus"]["corpus_sha256"] = "0" * 64
            with self.assertRaisesRegex(ProtocolError, "corpus provenance"):
                rq1.validate_certificate_artifact(
                    wrong_corpus, frame, require_complete=True
                )

    def test_a2_borderline_above_threshold_is_consistently_rejected(self):
        qc = QuantumCircuit(2)
        qc.ry(2.2e-12, 0)
        qc.cx(0, 1)
        direct_a2 = 1.1e-12
        determinant = direct_a2 ** 2
        self.assertGreater(direct_a2, rq1.SELECTOR_A2_TOL)
        self.assertLess(determinant, rq1.SELECTOR_A2_TOL)
        validation = rq1.crosscheck_selector(qc)
        rows = [
            row for row in validation["all_points"]
            if row["gate"] == 1 and row["qubit"] in (0, 1)
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertGreater(row["selector_a2"], rq1.SELECTOR_A2_TOL)
            self.assertGreater(
                row["independent_svd_a2"], rq1.SELECTOR_A2_TOL
            )
            self.assertFalse(row["selector_passed"])
            self.assertLess(row["det_rho_svd"], rq1.SELECTOR_A2_TOL)
        self.assertFalse(
            any(row["gate"] == 1 for row in validation["selected_points"])
        )

    def test_duplicate_selected_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            qasm_dir = Path(temporary)
            key = "duplicate.qasm"
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)
            (qasm_dir / key).write_text(qasm2.dumps(qc), encoding="utf-8")
            entry = {
                "attempt_index": 0,
                "attempt_id": "canonical:000:duplicate.qasm",
                "circuit": key,
                "source_qasm_sha256": file_sha256(qasm_dir / key),
            }
            frame = {"circuits": [entry], "qasm_dir": str(qasm_dir)}
            status, records = rq1.evaluate_circuit(entry, qasm_dir)
            records.append(copy.deepcopy(records[0]))
            artifact = certificate_artifact(frame, status, records)
            with self.assertRaisesRegex(ProtocolError, "duplicate selected point"):
                rq1.validate_certificate_artifact(
                    artifact, frame, require_complete=True
                )

    def test_selected_point_row_drop_is_rejected_after_checksum_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            qasm_dir = Path(temporary)
            key = "two-points.qasm"
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.z(0)
            qc.x(1)
            (qasm_dir / key).write_text(qasm2.dumps(qc), encoding="utf-8")
            entry = {
                "attempt_index": 0,
                "attempt_id": "canonical:000:two-points.qasm",
                "circuit": key,
                "source_qasm_sha256": file_sha256(qasm_dir / key),
            }
            frame = {"circuits": [entry], "qasm_dir": str(qasm_dir)}
            status, records = rq1.evaluate_circuit(entry, qasm_dir)
            artifact = certificate_artifact(frame, status, records)
            rq1.validate_certificate_artifact(
                artifact, frame, require_complete=True
            )

            artifact["records"].pop()
            artifact["record_count"] = len(artifact["records"])
            artifact["records_sha256"] = stable_hash(artifact["records"])
            artifact["summary"] = rq1._certificate_summary(artifact["records"])
            local_rows = artifact["records"]
            artifact["circuit_status"][key]["selected_point_count"] = len(
                local_rows
            )
            artifact["circuit_status"][key]["selected_points_sha256"] = (
                stable_hash([[row["gate"], row["qubit"]] for row in local_rows])
            )
            with self.assertRaisesRegex(ProtocolError, "source replay"):
                rq1.validate_certificate_artifact(
                    artifact, frame, require_complete=True
                )

    def test_selected_scientific_fields_are_bound_to_source_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            qasm_dir = Path(temporary)
            key = "source-bound.qasm"
            qc = QuantumCircuit(2)
            qc.x(0)
            qc.h(1)
            (qasm_dir / key).write_text(qasm2.dumps(qc), encoding="utf-8")
            entry = {
                "attempt_index": 0,
                "attempt_id": "canonical:000:source-bound.qasm",
                "circuit": key,
                "source_qasm_sha256": file_sha256(qasm_dir / key),
            }
            frame = {"circuits": [entry], "qasm_dir": str(qasm_dir)}
            status, records = rq1.evaluate_circuit(entry, qasm_dir)
            baseline = certificate_artifact(frame, status, records)
            rq1.validate_certificate_artifact(
                baseline, frame, require_complete=True
            )

            mutations = {}
            gate_mutation = copy.deepcopy(baseline)
            gate_mutation["records"][0]["gate_name"] = "forged_gate"
            mutations["gate_name"] = gate_mutation

            probability_mutation = copy.deepcopy(baseline)
            probability_mutation["records"][0]["p0"] = 0.25
            probability_mutation["records"][0]["p1"] = 0.75
            mutations["p0_p1"] = probability_mutation

            schmidt_mutation = copy.deepcopy(baseline)
            row = schmidt_mutation["records"][0]
            row["independent_svd_a2"] = 5e-13
            row["a2_crosscheck_abs_diff"] = abs(
                row["selector_a2"] - row["independent_svd_a2"]
            )
            row["det_rho_svd"] = 2.5e-25
            row["determinant_residual_minus_svd"] = (
                row["determinant_residual"] - row["det_rho_svd"]
            )
            mutations["schmidt"] = schmidt_mutation

            for label, artifact in mutations.items():
                with self.subTest(label=label):
                    artifact["record_count"] = len(artifact["records"])
                    artifact["records_sha256"] = stable_hash(
                        artifact["records"]
                    )
                    artifact["summary"] = rq1._certificate_summary(
                        artifact["records"]
                    )
                    with self.assertRaisesRegex(ProtocolError, "source replay"):
                        rq1.validate_certificate_artifact(
                            artifact, frame, require_complete=True
                        )


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full_frame = resolve_circuit_corpus()
        cls.entry = next(
            entry for entry in full_frame["circuits"]
            if entry["circuit"] == "qft_indep_qiskit_3.qasm"
        )
        cls.qasm_dir = Path(full_frame["qasm_dir"])
        cls.frame = {
            "circuits": [cls.entry],
            "circuit_count": 1,
            "qasm_dir": str(cls.qasm_dir),
        }
        cls.status, cls.record = coverage_rq2.evaluate_circuit(
            cls.entry, cls.qasm_dir
        )

    def test_real_emitter_preserves_same_gate_groups(self):
        grouped = [
            group for group in self.record["group_demands"]
            if len(group["members"]) > 1
        ]
        self.assertTrue(grouped)
        self.assertGreater(self.record["emitted_mid_point_count"], 0)
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        coverage_rq2.validate_artifact(
            artifact, self.frame, require_complete=True
        )
        self.assertEqual(
            self.record["numerical_candidate_points"],
            self.record["certificate_points"],
        )
        self.assertEqual(
            self.record["coverage_denominators"]["gate_node_count"],
            self.record["gate_count"],
        )
        self.assertEqual(
            self.record["gate_qubit_event_denominator"],
            self.record["acted_point_count"],
        )
        feasible = self.record["individually_feasible_gate_groups"]
        infeasible = self.record["individually_infeasible_gate_groups"]
        self.assertEqual(
            [group["gate"] for group in feasible + infeasible],
            [group["gate"] for group in self.record["group_demands"]],
        )

        drifted = copy.deepcopy(artifact)
        drifted["protocol"]["qmax"] = 20
        with self.assertRaisesRegex(ProtocolError, "protocol drift"):
            coverage_rq2.validate_artifact(
                drifted, self.frame, require_complete=True
            )

        wrong_corpus = copy.deepcopy(artifact)
        wrong_corpus["corpus"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "corpus provenance"):
            coverage_rq2.validate_artifact(
                wrong_corpus, self.frame, require_complete=True
            )

    def test_duplicate_replay_union_is_rejected(self):
        source = Path(coverage_rq2.__file__).read_text(encoding="utf-8")
        guard = "if replay_indices != tuple(sorted(set(replay_indices))):"
        self.assertEqual(source.count(guard), 1)

        qc = QuantumCircuit.from_qasm_file(
            str(self.qasm_dir / self.entry["circuit"])
        )
        validation = rq1.crosscheck_selector(qc)
        operation_list, prebudget, _ = coverage_rq2._build_operation_list(
            qc, self.entry["circuit"], validation["selected_points"]
        )
        groups, batches, _, _ = coverage_rq2._group_demands(
            qc, self.entry["circuit"], operation_list, prebudget
        )
        original_emitter = coverage_rq2.generate_monitoring_circuit

        def duplicate_replay_index(*args, **kwargs):
            emitted = original_emitter(*args, **kwargs)
            metadata_groups = (emitted.metadata or {}).get(
                "qmon_monitor_groups", ()
            )
            for group in metadata_groups:
                indices = list(group["replay_source_gate_indices"])
                if indices:
                    group["replay_source_gate_indices"] = indices + [indices[0]]
                    return emitted
            self.fail("fixture generated no replay source indices")

        with mock.patch.object(
            coverage_rq2,
            "generate_monitoring_circuit",
            side_effect=duplicate_replay_index,
        ):
            with self.assertRaisesRegex(ProtocolError, "non-deduplicated"):
                coverage_rq2._emit_batches(
                    qc,
                    self.entry["circuit"],
                    operation_list,
                    groups,
                    batches,
                )

    def test_zero_emitted_coverage_row_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            qasm_dir = Path(temporary)
            key = "final-read-only.qasm"
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.x(1)
            qc.measure([0, 1], [0, 1])
            (qasm_dir / key).write_text(qasm2.dumps(qc), encoding="utf-8")
            entry = {
                "attempt_index": 0,
                "attempt_id": "canonical:000:final-read-only.qasm",
                "circuit": key,
                "source_qasm_sha256": file_sha256(qasm_dir / key),
            }
            status, record = coverage_rq2.evaluate_circuit(entry, qasm_dir)
            self.assertEqual(status["status"], "complete")
            self.assertEqual(record["emitted_mid_point_count"], 0)
            self.assertEqual(record["final_read_point_count"], 2)
            self.assertEqual(record["deployed_observable_point_count"], 2)
            self.assertEqual(record["deployed_observable_qubit_count"], 2)
            self.assertEqual(record["deployed_observable_qubit_coverage"], 1.0)
            self.assertEqual(record["deployed_observable_depth_reach"], 1.0)

            frame = {
                "circuits": [entry],
                "circuit_count": 1,
                "qasm_dir": str(qasm_dir),
            }
            artifact = coverage_artifact(frame, status, record)
            coverage_rq2.validate_artifact(
                artifact, frame, require_complete=True
            )

    def test_deployed_qubit_and_depth_metrics_are_recomputed(self):
        for field, replacement, expected_message in (
            ("deployed_observable_qubit_count", -1, "count mismatch"),
            ("deployed_observable_qubit_coverage", 0.123, "coverage mismatch"),
            ("deployed_observable_depth_reach", 0.123, "depth reach mismatch"),
        ):
            with self.subTest(field=field):
                artifact = coverage_artifact(
                    self.frame, self.status, copy.deepcopy(self.record)
                )
                artifact["records"][0][field] = replacement
                artifact["records_sha256"] = stable_hash(artifact["records"])
                artifact["summary"] = coverage_rq2.summarize(
                    artifact["records"]
                )
                artifact["circuit_status"][self.entry["circuit"]][
                    "record_sha256"
                ] = stable_hash(artifact["records"][0])
                with self.assertRaisesRegex(ProtocolError, expected_message):
                    coverage_rq2.validate_artifact(
                        artifact, self.frame, require_complete=True
                    )

    def test_boolean_gate_qubit_and_count_substitutes_are_rejected(self):
        cases = (
            ("point gate", ("acted_points", 0, 0), False, "malformed acted_points"),
            ("point qubit", ("acted_points", 1, 1), True, "malformed acted_points"),
            ("group gate", ("group_demands", 0, "gate"), False,
             "malformed group demand"),
            ("group qubit", ("group_demands", 1, "members", 0), True,
             "malformed group demand"),
            ("group demand", ("group_demands", 0, "ancilla_demand"), False,
             "malformed group demand"),
            ("batch gate", ("batches", 0, "gate_groups", 0), False,
             "malformed generated batch"),
            ("batch index", ("batches", 0, "batch"), False,
             "malformed generated batch"),
            ("nested monitor gate", ("batches", 0, "monitor_groups", 0, "gate"),
             False, "malformed generated monitor group"),
            ("batch count", ("batch_count",), True, "batch_count mismatch"),
        )
        for label, path, replacement, message in cases:
            with self.subTest(label=label):
                artifact = coverage_artifact(
                    self.frame, self.status, copy.deepcopy(self.record)
                )
                target = artifact["records"][0]
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                update_coverage_checksums(artifact)
                with self.assertRaisesRegex(ProtocolError, message):
                    coverage_rq2.validate_artifact(
                        artifact, self.frame, require_complete=True
                    )

    def test_boolean_artifact_counts_and_attempt_identity_are_rejected(self):
        cases = (
            ("record count", ("record_count",), True, "record_count mismatch"),
            ("corpus count", ("corpus", "circuit_count"), True,
             "corpus provenance"),
            ("summary count", ("summary", "circuit_count"), True,
             "summary is not derived"),
            (
                "protocol solver thread count",
                ("protocol", "partition_solver", "threads"),
                True,
                "protocol drift",
            ),
            (
                "status attempt index",
                ("circuit_status", self.entry["circuit"], "attempt_index"),
                False,
                "attempt_index drift",
            ),
        )
        for label, path, replacement, message in cases:
            with self.subTest(label=label):
                artifact = coverage_artifact(
                    self.frame, self.status, copy.deepcopy(self.record)
                )
                target = artifact
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                with self.assertRaisesRegex(ProtocolError, message):
                    coverage_rq2.validate_artifact(
                        artifact, self.frame, require_complete=True
                    )

    def test_denominator_conflation_is_rejected(self):
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        artifact["records"][0]["coverage_denominators"][
            "gate_qubit_event_count"
        ] = artifact["records"][0]["gate_count"]
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "coverage denominator"):
            coverage_rq2.validate_artifact(
                artifact, self.frame, require_complete=True
            )

    def test_csv_node_alias_uses_deployed_publication_scope(self):
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coverage.csv"
            coverage_rq2.write_csv(path, artifact)
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(
            float(row["node_cov"]),
            self.record["deployed_observable_gate_coverage"],
        )
        self.assertEqual(
            float(row["certificate_node_cov"]),
            self.record["certificate_gate_coverage"],
        )
        self.assertEqual(
            int(row["gate_qubit_event_denominator"]),
            self.record["acted_point_count"],
        )

    def test_silent_circuit_row_drop_is_rejected(self):
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        coverage_rq2.validate_artifact(
            artifact, self.frame, require_complete=True
        )
        artifact["records"] = []
        artifact["record_count"] = 0
        artifact["records_sha256"] = stable_hash([])
        artifact["summary"] = coverage_rq2.summarize([])
        with self.assertRaisesRegex(ProtocolError, "silent RQ2 circuit-row drop"):
            coverage_rq2.validate_artifact(
                artifact, self.frame, require_complete=True
            )

    def test_silent_emitted_point_drop_is_rejected(self):
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        artifact["records"][0]["emitted_mid_points"].pop()
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "Qmax scope accounting"):
            coverage_rq2.validate_artifact(
                artifact, self.frame, require_complete=True
            )

    def test_incomplete_feasible_batch_is_rejected(self):
        artifact = coverage_artifact(
            self.frame, self.status, copy.deepcopy(self.record)
        )
        self.assertTrue(artifact["records"][0]["batches"])
        artifact["records"][0]["batches"].pop()
        artifact["records"][0]["batch_count"] -= 1
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "feasible gate groups"):
            coverage_rq2.validate_artifact(
                artifact, self.frame, require_complete=True
            )


class TightnessArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        qasm_dir = Path(cls.temporary.name)
        circuits = []
        qc = QuantumCircuit(2)
        qc.ry(0.7, 0)
        qc.cx(0, 1)
        source = qasm2.dumps(qc)
        for index, key in enumerate(TIGHTNESS_KEYS):
            path = qasm_dir / key
            path.write_text(source, encoding="utf-8")
            circuits.append({
                "attempt_index": index,
                "attempt_id": f"canonical:{index:03d}:{key}",
                "circuit": key,
                "source_qasm_sha256": file_sha256(path),
            })
        cls.frame = {
            "circuits": circuits,
            "qasm_dir": str(qasm_dir),
            "circuit_count": len(circuits),
            "manifest_sha256": "fixture-manifest",
            "corpus_sha256": "fixture-corpus",
        }
        cls.artifact = build_tightness_artifact(cls.frame)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _artifact(self):
        return self.frame, copy.deepcopy(self.artifact)

    def test_tightness_row_drop_is_rejected_after_status_checksum_update(self):
        frame, artifact = self._artifact()
        validate_tightness_artifact(artifact, frame, require_complete=True)
        removed = artifact["records"].pop()
        key = removed["circuit"]
        local_rows = [
            row for row in artifact["records"] if row["circuit"] == key
        ]
        status = artifact["circuit_status"][key]
        status["selected_point_count"] = len(local_rows)
        status["selected_points_sha256"] = stable_hash([
            [row["gate"], row["qubit"], row["selection_bucket"]]
            for row in local_rows
        ])
        artifact["record_count"] = len(artifact["records"])
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "source replay"):
            validate_tightness_artifact(
                artifact, frame, require_complete=True
            )

    def test_tightness_formula_fields_must_be_mutually_consistent(self):
        frame, artifact = self._artifact()
        row = artifact["records"][0]
        row["state_infidelity"] = 0.9
        row["law_abs_error"] = 0.0
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "law_abs_error mismatch"):
            validate_tightness_artifact(
                artifact, frame, require_complete=True
            )

    def test_tightness_negative_law_error_is_rejected(self):
        frame, artifact = self._artifact()
        artifact["records"][0]["law_abs_error"] = -1e-16
        artifact["records_sha256"] = stable_hash(artifact["records"])
        with self.assertRaisesRegex(ProtocolError, "law error is negative"):
            validate_tightness_artifact(
                artifact, frame, require_complete=True
            )

    def test_numerical_zero_points_never_reenter_positive_bins(self):
        points = [
            [index, "h", 0, 1.0, 0.0, determinant]
            for index, determinant in enumerate(
                [0.0, 0.0, 5e-13, 9e-13, 1e-6, 0.01, 0.03]
            )
        ]
        selected = _select_tightness_points(points)
        zero_rows = [
            (point, bucket) for point, bucket in selected
            if point[5] <= 1e-12
        ]
        self.assertLessEqual(len(zero_rows), 2)
        self.assertTrue(
            all(bucket == "numerical-zero" for _, bucket in zero_rows)
        )
        self.assertFalse(any(
            bucket != "numerical-zero" and point[5] <= 1e-12
            for point, bucket in selected
        ))

    def test_tightness_ranking_is_stable_under_sub_quantum_roundoff(self):
        common = [
            [0, "h", 0, 0.5, 0.5, 0.12500000000000003],
            [1, "h", 0, 0.5, 0.5, 0.12499999999999997],
            [2, "h", 0, 0.5, 0.5, 0.12500000000000001],
            [3, "h", 0, 0.5, 0.5, 0.12499999999999999],
            [4, "h", 0, 0.5, 0.5, 0.12500000000000000],
            [5, "h", 0, 0.5, 0.5, 0.12500000000000002],
        ]
        perturbed = copy.deepcopy(common)
        perturbed[0][5], perturbed[1][5] = perturbed[1][5], perturbed[0][5]
        first = [
            (point[0], point[2], bucket)
            for point, bucket in _select_tightness_points(common)
        ]
        second = [
            (point[0], point[2], bucket)
            for point, bucket in _select_tightness_points(perturbed)
        ]
        self.assertEqual(first, second)

    def test_strictly_positive_tiny_measurement_branch_is_not_pruned(self):
        qc = QuantumCircuit(1, 1)
        qc.ry(2e-8, 0)
        qc.measure(0, 0)
        qc.x(0)
        branches = _exact_branches(qc)
        probabilities = [probability for probability, _ in branches]
        self.assertEqual(len(probabilities), 2)
        self.assertGreater(min(probabilities), 0.0)
        self.assertLess(min(probabilities), 1e-14)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=14)

    def test_tightness_protocol_drift_is_rejected(self):
        frame, artifact = self._artifact()
        validate_tightness_artifact(artifact, frame, require_complete=True)
        artifact["protocol"]["law"] = "1-F = det(rho_q)"
        with self.assertRaisesRegex(ProtocolError, "protocol drift"):
            validate_tightness_artifact(
                artifact, frame, require_complete=True
            )

    def test_tightness_corpus_provenance_drift_is_rejected(self):
        frame, artifact = self._artifact()
        artifact["corpus"]["circuit_count"] -= 1
        with self.assertRaisesRegex(ProtocolError, "corpus provenance"):
            validate_tightness_artifact(
                artifact, frame, require_complete=True
            )


if __name__ == "__main__":
    unittest.main()
