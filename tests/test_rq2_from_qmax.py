"""Tests for extracting the RQ2 Qmax=24 data from a Qmax sweep."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from qiskit import QuantumCircuit, qasm2


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import coverage_rq2 as rq2
import qmax_sensitivity as qmax
import rq2_from_qmax as derive
from entanglement_disturbance import (
    ProtocolError,
    atomic_write_json,
    resolve_circuit_corpus,
    stable_hash,
)


class RQ2FromQmaxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.qasm_dir = self.root / "qasm"
        self.qasm_dir.mkdir()
        self.key = "emitting_fixture.qasm"

        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(0)
        circuit.measure([0, 1], [0, 1])
        (self.qasm_dir / self.key).write_text(
            qasm2.dumps(circuit), encoding="utf-8"
        )
        self.manifest = self.root / "manifest.txt"
        self.manifest.write_text(self.key + "\n", encoding="utf-8")
        self.frame = resolve_circuit_corpus(
            self.manifest,
            self.qasm_dir,
            expected_count=1,
            expected_manifest_sha256=None,
            expected_corpus_sha256=None,
        )
        self.grid = qmax.build_attempt_grid(
            self.frame, require_canonical=False
        )
        self.bundle = qmax.build_fingerprint_bundle(self.frame)
        self.records = [
            qmax.evaluate_attempt(attempt, self.qasm_dir, self.bundle)
            for attempt in self.grid
        ]
        self.assertTrue(all(record["status"] == "complete" for record in self.records))

    def tearDown(self):
        self.temporary.cleanup()

    def _derive(self, records=None):
        return derive.derive_native_artifact(
            self.records if records is None else records,
            self.frame,
            self.bundle,
            require_canonical=False,
        )

    def _direct_artifact(self):
        status, record = rq2.evaluate_circuit(
            self.frame["circuits"][0], self.qasm_dir
        )
        artifact = {
            "schema": rq2.SCHEMA,
            "protocol": rq2.protocol_descriptor(),
            "corpus": {
                "manifest_sha256": self.frame["manifest_sha256"],
                "corpus_sha256": self.frame["corpus_sha256"],
                "circuit_count": 1,
            },
            "circuit_status": {self.key: status},
            "record_count": 1,
            "records_sha256": stable_hash([record]),
            "summary": rq2.summarize([record]),
            "records": [record],
        }
        return rq2.validate_artifact(
            artifact, self.frame, require_complete=True
        )

    def test_direct_and_derived_native_outputs_are_identical_with_real_batch(self):
        headline = next(record for record in self.records if record["qmax"] == 24)
        self.assertTrue(headline["result"]["batches"])
        self.assertIn("monitored_points", headline["result"]["batches"][0])

        direct = self._direct_artifact()
        derived = self._derive()
        self.assertEqual(derived, direct)
        self.assertEqual(
            set(derived["records"][0]), set(derive.NATIVE_RECORD_FIELDS)
        )
        self.assertNotIn("monitored_points", derived["records"][0]["batches"][0])
        self.assertEqual(self._derive(list(reversed(self.records))), derived)

        direct_csv = self.root / "direct.csv"
        derived_csv = self.root / "derived.csv"
        rq2.write_csv(direct_csv, direct)
        rq2.write_csv(derived_csv, derived)
        self.assertEqual(derived_csv.read_bytes(), direct_csv.read_bytes())

    def test_missing_duplicate_error_and_mixed_source_are_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "partial Qmax grid"):
            self._derive(self.records[:-1])

        duplicate = self.records[:-1] + [self.records[0]]
        with self.assertRaisesRegex(ProtocolError, "duplicate Qmax"):
            self._derive(duplicate)

        failed = copy.deepcopy(self.records)
        failed[0] = qmax.terminal_error_record(
            self.grid[0], self.bundle, RuntimeError("synthetic failure")
        )
        with self.assertRaisesRegex(ProtocolError, "all-success"):
            self._derive(failed)

        mixed = copy.deepcopy(self.records)
        mixed[-1]["fingerprints"]["sources_sha256"] = "f" * 64
        qmax._add_record_checksum(mixed[-1])
        with self.assertRaisesRegex(ProtocolError, "mixed sources_sha256"):
            self._derive(mixed)

        foreign = copy.deepcopy(self.records)
        foreign[-1]["run"]["run_id"] = "b" * 64
        qmax._add_record_checksum(foreign[-1])
        with self.assertRaisesRegex(ProtocolError, "run_id mismatch"):
            self._derive(foreign)

    def test_tamper_and_result_contract_drift_are_rejected(self):
        tampered = copy.deepcopy(self.records)
        row = next(record for record in tampered if record["qmax"] == 24)
        row["result"]["deployed_observable_gate_count"] += 1
        qmax._add_record_checksum(row)
        with self.assertRaisesRegex(ProtocolError, "mismatch"):
            self._derive(tampered)

        extra = copy.deepcopy(self.records)
        row = next(record for record in extra if record["qmax"] == 24)
        row["result"]["unrecognized_scientific_field"] = 1
        qmax._add_record_checksum(row)
        with self.assertRaisesRegex(ProtocolError, "field contract drift"):
            self._derive(extra)

        envelope_extra = copy.deepcopy(self.records)
        envelope_extra[0]["unrecognized_terminal_field"] = 1
        qmax._add_record_checksum(envelope_extra[0])
        with self.assertRaisesRegex(ProtocolError, "terminal-record field contract"):
            self._derive(envelope_extra)

    def test_reader_rejects_legacy_duplicate_keys_nonfinite_and_bad_checksum(self):
        legacy = self.root / "qmax_records.pkl"
        legacy.write_bytes(b"pickle")
        with self.assertRaisesRegex(ProtocolError, "legacy"):
            derive.read_source_sweep(legacy)

        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "duplicate object key"):
            derive.read_source_sweep(duplicate)

        nonfinite = self.root / "nonfinite.jsonl"
        nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "non-finite"):
            derive.read_source_sweep(nonfinite)

        envelope = self.root / "records.json"
        qmax.write_records(self.records, envelope)
        value = json.loads(envelope.read_text(encoding="utf-8"))
        value["records_sha256"] = "0" * 64
        atomic_write_json(envelope, value)
        with self.assertRaisesRegex(ProtocolError, "records checksum mismatch"):
            derive.read_source_sweep(envelope)

        old = self.root / "old.json"
        value["records_sha256"] = stable_hash(value["records"])
        value["record_schema"] = "qmon.qmax-sensitivity.v1"
        atomic_write_json(old, value)
        with self.assertRaisesRegex(ProtocolError, "legacy or wrong"):
            derive.read_source_sweep(old)

    def test_atomic_outputs_and_derivation_summary(self):
        source_path = self.root / "records.json"
        qmax.write_records(self.records, source_path)
        source = derive.read_source_sweep(source_path)
        artifact = self._derive(source.records)
        out_json = self.root / "native.json"
        out_csv = self.root / "native.csv"
        summary_path = self.root / "summary.json"

        atomic_write_json(out_json, artifact)
        rq2.write_csv(out_csv, artifact)
        summary = derive.build_derivation_summary(
            source,
            sorted(source.records, key=lambda record: record["attempt_index"]),
            self.bundle,
            artifact,
            out_json,
            out_csv,
        )
        atomic_write_json(summary_path, summary)

        self.assertEqual(summary["schema"], derive.SUMMARY_SCHEMA)
        checksummed = dict(summary)
        checksum = checksummed.pop("summary_sha256")
        self.assertEqual(checksum, stable_hash(checksummed))
        self.assertEqual(
            summary["source_sweep"]["file_sha256"], source.file_sha256
        )
        self.assertEqual(
            summary["outputs"]["records_sha256"], artifact["records_sha256"]
        )
        self.assertFalse(list(self.root.glob(".*.tmp")))
        with self.assertRaisesRegex(ProtocolError, "must not alias"):
            derive._ensure_output_paths(
                [source_path, out_csv, summary_path],
                overwrite=True,
                source=source_path,
            )


if __name__ == "__main__":
    unittest.main()
