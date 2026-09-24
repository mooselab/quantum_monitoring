"""Robustness tests for the full-corpus RQ1 tightness protocol.

These tests use tiny QASM fixtures for execution.  They verify the real
310-circuit frame but never launch the full experiment.
"""

from __future__ import annotations

import copy
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qiskit import QuantumCircuit, qasm2

import rq1_full_tightness as rq1


class EnvironmentFingerprintTests(unittest.TestCase):
    def test_environment_fingerprint_excludes_kernel_patch_release(self):
        with mock.patch.object(
            rq1.platform,
            "release",
            side_effect=AssertionError("kernel release must not enter the checksum"),
        ):
            environment = rq1._environment_value()
        self.assertNotIn("release", environment)
        self.assertEqual(environment["libc"], list(rq1.platform.libc_ver()))


class _CaptureConnection:
    def __init__(self):
        self.messages = []

    def send(self, value):
        self.messages.append(value)

    def close(self):
        pass


def _fixture_corpus_hash(keys, qasm_dir):
    return rq1.stable_hash([
        {"basename": Path(key).stem, "sha256": rq1.file_sha256(qasm_dir / key)}
        for key in keys
    ])


def _make_frame(root: Path, circuits: dict[str, QuantumCircuit]):
    qasm_dir = root / "qasm"
    qasm_dir.mkdir()
    manifest = root / "manifest.txt"
    keys = list(circuits)
    manifest.write_text("\n".join(keys) + "\n", encoding="utf-8")
    for key, circuit in circuits.items():
        (qasm_dir / key).write_text(qasm2.dumps(circuit), encoding="utf-8")
    return rq1.resolve_circuit_corpus(
        manifest,
        qasm_dir,
        expected_count=len(keys),
        expected_manifest_sha256=rq1.file_sha256(manifest),
        expected_corpus_sha256=_fixture_corpus_hash(keys, qasm_dir),
    )


def _direct_result(entry, qasm_dir, maximum=10):
    path = Path(qasm_dir) / entry["circuit"]
    qc = QuantumCircuit.from_qasm_file(str(path))
    points = rq1.enumerate_acted_points(qc)
    connection = _CaptureConnection()
    rq1._evaluate_worker(dict(entry), str(path), maximum, connection)
    fatal = [message for message in connection.messages if message["kind"] == "fatal"]
    if fatal:
        raise AssertionError(fatal)
    rows = [
        message["record"]
        for message in connection.messages
        if message["kind"] == "point"
    ]
    return rq1._build_circuit_result(entry, points, rows)


def _artifact_for(frame, maximum=10, results=None, *, shard_index=0, shard_count=1):
    config = rq1.protocol_config(maximum, 30.0)
    bundle = rq1.build_fingerprint_bundle(frame, config)
    entries = rq1.shard_entries(frame, shard_index, shard_count)
    if results is None:
        results = [_direct_result(entry, frame["qasm_dir"], maximum) for entry in entries]
    return rq1._artifact(
        frame,
        bundle,
        {
            "kind": "shard",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "assignment": "attempt_index modulo shard_count",
        },
        results,
        complete=True,
    )


def _bell():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def _product():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.z(0)
    return circuit


class FixedFrameTests(unittest.TestCase):
    def test_repository_frame_is_exactly_the_fixed_310(self):
        frame = rq1.resolve_circuit_corpus()
        self.assertEqual(frame["circuit_count"], rq1.EXPECTED_CIRCUITS)
        self.assertEqual(frame["manifest_sha256"], rq1.EXPECTED_MANIFEST_SHA256)
        self.assertEqual(frame["corpus_sha256"], rq1.EXPECTED_CORPUS_SHA256)
        self.assertEqual(len(frame["circuits"]), 310)

    def test_worker_signals_do_not_conflate_crash_with_resource_limit(self):
        killed = rq1._unfinished_worker_outcome(
            timed_out=False,
            exitcode=-signal.SIGKILL,
            timeout_seconds=30.0,
            fatal_error=None,
        )
        crashed = rq1._unfinished_worker_outcome(
            timed_out=False,
            exitcode=-signal.SIGSEGV,
            timeout_seconds=30.0,
            fatal_error=None,
        )
        self.assertEqual(killed[:2], ("resource_limit", "worker_killed_resource"))
        self.assertEqual(
            crashed[:2], ("execution_error", "worker_segmentation_fault")
        )

    def test_sharding_is_disjoint_and_exhaustive(self):
        frame = rq1.resolve_circuit_corpus()
        shards = [rq1.shard_entries(frame, index, 17) for index in range(17)]
        flattened = [entry for shard in shards for entry in shard]
        self.assertEqual(len(flattened), 310)
        self.assertEqual(
            {entry["attempt_id"] for entry in flattened},
            {entry["attempt_id"] for entry in frame["circuits"]},
        )
        for index, shard in enumerate(shards):
            self.assertTrue(all(entry["attempt_index"] % 17 == index for entry in shard))


class EnumerationAndLawTests(unittest.TestCase):
    def test_enumerates_every_acted_qarg_and_skips_measurements(self):
        circuit = _bell()
        points = rq1.enumerate_acted_points(circuit)
        self.assertEqual(
            [(row["gate"], row["qubit"], row["gate_name"]) for row in points],
            [(0, 0, "h"), (1, 0, "cx"), (1, 1, "cx")],
        )
        self.assertEqual([row["point_ordinal"] for row in points], [0, 1, 2])

    def test_exact_forced_monitor_obeys_law_at_all_bell_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            result = _direct_result(frame["circuits"][0], frame["qasm_dir"])
            self.assertEqual(result["point_record_count"], 3)
            self.assertEqual(result["point_status_counts"], {"success": 3})
            entangled = result["points"][1:]
            for row in entangled:
                self.assertAlmostEqual(row["det_rho"], 0.25, places=13)
                self.assertAlmostEqual(row["state_infidelity"], 0.75, places=13)
                self.assertLessEqual(row["law_abs_error"], rq1.LAW_TOLERANCE)

    def test_original_resource_cap_accounts_for_every_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            result = _direct_result(
                frame["circuits"][0], frame["qasm_dir"], maximum=1
            )
            self.assertEqual(result["point_record_count"], 3)
            self.assertEqual(result["point_status_counts"], {"resource_limit": 3})
            self.assertTrue(all(
                row["error"]["category"]
                == "original_statevector_qubits_exceed_cap"
                for row in result["points"]
            ))

    def test_parent_timeout_materializes_all_unfinished_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            result = rq1.run_circuit_with_timeout(
                frame["circuits"][0],
                frame["qasm_dir"],
                max_statevector_qubits=10,
                timeout_seconds=1e-9,
            )
            self.assertEqual(result["point_record_count"], 3)
            self.assertEqual(result["point_status_counts"], {"timeout": 3})


class IndependentValidationTests(unittest.TestCase):
    def test_valid_artifact_recomputes_from_qasm(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame)
            validation = rq1.validate_artifact(artifact, frame, recompute=True)
            self.assertEqual(validation["circuits"], 1)
            self.assertEqual(validation["points"], 3)

    def test_checksummed_primary_quantity_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame)
            result = copy.deepcopy(artifact["circuit_results"][0])
            row = result["points"][1]
            row["det_rho"] = 0.20
            row["three_det_rho"] = 0.60
            row["state_infidelity"] = 0.60
            row["fidelity"] = 0.40
            row["law_abs_error"] = 0.0
            rq1._add_checksum(row, "record_sha256")
            result["point_records_sha256"] = rq1.stable_hash(result["points"])
            rq1._add_checksum(result, "record_sha256")
            forged = _artifact_for(frame, results=[result])
            with self.assertRaisesRegex(rq1.ProtocolError, "recomputed det_rho.*mismatch"):
                rq1.validate_artifact(forged, frame, recompute=True)

    def test_checksummed_row_drop_is_detected_from_qasm_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            entry = frame["circuits"][0]
            original = _direct_result(entry, frame["qasm_dir"])
            forged_result = rq1._build_circuit_result(
                entry,
                rq1.enumerate_acted_points(_bell())[:-1],
                original["points"][:-1],
            )
            forged = _artifact_for(frame, results=[forged_result])
            with self.assertRaisesRegex(rq1.ProtocolError, "silent point omission"):
                rq1.validate_artifact(forged, frame, recompute=True)

    def test_fake_resource_limit_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame)
            result = copy.deepcopy(artifact["circuit_results"][0])
            row = result["points"][0]
            row.update({
                "status": "resource_limit",
                "p0": None,
                "p1": None,
                "det_rho": None,
                "instrumented_qubits": None,
                "fidelity": None,
                "state_infidelity": None,
                "three_det_rho": None,
                "law_abs_error": None,
                "law_holds": None,
                "error": {
                    "category": "original_statevector_qubits_exceed_cap",
                    "exception_type": "ProtocolOutcome",
                    "message": "forged",
                },
            })
            rq1._add_checksum(row, "record_sha256")
            forged_result = rq1._build_circuit_result(
                frame["circuits"][0], rq1.enumerate_acted_points(_bell()),
                result["points"],
            )
            forged = _artifact_for(frame, results=[forged_result])
            with self.assertRaisesRegex(rq1.ProtocolError, "spurious instrumented resource"):
                rq1.validate_artifact(forged, frame, recompute=True)

    def test_fingerprint_tamper_fails_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame)
            artifact["fingerprints"]["environment"]["value"]["machine"] = "forged"
            artifact["fingerprints"]["environment"]["sha256"] = rq1.stable_hash(
                artifact["fingerprints"]["environment"]["value"]
            )
            rq1._add_checksum(artifact, "artifact_sha256")
            with self.assertRaisesRegex(rq1.ProtocolError, "checksum mismatch"):
                rq1.validate_artifact(artifact, frame, recompute=False)

    def test_qasm_substitution_after_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame)
            path = Path(frame["qasm_dir"]) / "bell.qasm"
            path.write_text(qasm2.dumps(_product()), encoding="utf-8")
            with self.assertRaisesRegex(rq1.ProtocolError, "QASM checksum mismatch"):
                rq1.validate_artifact(artifact, frame, recompute=False)


class ShardResumeAndAggregateTests(unittest.TestCase):
    def test_merge_requires_each_shard_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {
                "bell.qasm": _bell(),
                "product.qasm": _product(),
            })
            shard0 = _artifact_for(frame, shard_index=0, shard_count=2)
            shard1 = _artifact_for(frame, shard_index=1, shard_count=2)
            merged = rq1.merge_artifacts([shard1, shard0], frame, recompute=True)
            validation = rq1.validate_artifact(merged, frame, recompute=True)
            self.assertEqual(validation["circuits"], 2)
            with self.assertRaisesRegex(rq1.ProtocolError, "missing"):
                rq1.merge_artifacts([shard0], frame, recompute=False)
            with self.assertRaisesRegex(rq1.ProtocolError, "duplicate shard"):
                rq1.merge_artifacts([shard0, shard0], frame, recompute=False)

    def test_resume_retains_terminal_circuits_and_runs_only_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _make_frame(root, {
                "bell.qasm": _bell(),
                "product.qasm": _product(),
            })
            config = rq1.protocol_config(10, 30.0)
            bundle = rq1.build_fingerprint_bundle(frame, config)
            run = {
                "kind": "shard",
                "shard_index": 0,
                "shard_count": 1,
                "assignment": "attempt_index modulo shard_count",
            }
            first = _direct_result(frame["circuits"][0], frame["qasm_dir"])
            checkpoint = rq1._artifact(
                frame, bundle, run, [first], complete=False
            )
            output = root / "checkpoint.json"
            rq1.atomic_write_json(output, checkpoint)
            calls = []

            def runner(entry, qasm_dir, **kwargs):
                calls.append(entry["attempt_id"])
                return _direct_result(entry, qasm_dir, maximum=kwargs["max_statevector_qubits"])

            final = rq1.run_shard(
                frame,
                output,
                shard_index=0,
                shard_count=1,
                max_statevector_qubits=10,
                circuit_timeout_seconds=30.0,
                resume=True,
                runner=runner,
            )
            self.assertEqual(calls, [frame["circuits"][1]["attempt_id"]])
            self.assertTrue(final["complete"])
            self.assertEqual(final["circuit_result_count"], 2)
            before = output.read_bytes()
            calls.clear()
            rq1.run_shard(
                frame,
                output,
                shard_index=0,
                shard_count=1,
                max_statevector_qubits=10,
                circuit_timeout_seconds=30.0,
                resume=True,
                runner=runner,
            )
            self.assertEqual(calls, [])
            self.assertEqual(before, output.read_bytes())

    def test_aggregate_keeps_resource_points_in_attempted_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = _make_frame(Path(temporary), {"bell.qasm": _bell()})
            artifact = _artifact_for(frame, maximum=1)
            summary = rq1.aggregate_artifact(
                artifact,
                frame,
                recompute=True,
                require_full_corpus=False,
            )
            self.assertEqual(summary["attempted_point_denominator"], 3)
            self.assertEqual(summary["exact_evaluated_point_denominator"], 0)
            self.assertEqual(summary["resource_limit_point_count"], 3)
            self.assertEqual(summary["non_evaluated_point_count"], 3)
            self.assertIsNone(summary["law_pass_rate_exact_evaluated"])

    def test_parent_runner_exception_still_accounts_for_every_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = _make_frame(root, {"bell.qasm": _bell()})

            def broken_runner(*args, **kwargs):
                raise RuntimeError("injected parent failure")

            artifact = rq1.run_shard(
                frame,
                root / "failed.json",
                shard_index=0,
                shard_count=1,
                max_statevector_qubits=10,
                circuit_timeout_seconds=30.0,
                runner=broken_runner,
            )
            self.assertEqual(artifact["point_record_count"], 3)
            self.assertEqual(
                artifact["point_status_counts"], {"execution_error": 3}
            )

    def test_atomic_writer_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            value = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
            rq1.atomic_write_json(path, value)
            first = path.read_bytes()
            rq1.atomic_write_json(path, value)
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
