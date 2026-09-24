"""Focused tests for the Appendix C scalability-v2 protocol.

Run in the requested environment:

    conda run -n qmon2 python -m unittest -v test_scalability_v2.py

The unit-test process explicitly uses CBC so structural tests never consume a
Gurobi license or wait through its retry schedule.  A dedicated validation
test still verifies that full final artifacts require Gurobi.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
import signal
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import Initialize

# Keep the test suite deterministic and independent of Gurobi availability.
# This is set before importing scalability_v2 because that module records
# a checksum of the selected partition backend.
os.environ["QMON_MILP_SOLVER"] = "cbc"

import scalability_v2 as scalability
import batch_partition as batching


class RuntimeFingerprintTests(unittest.TestCase):
    def test_comparison_commands_select_matching_handlers(self):
        parser = scalability.build_parser()
        for command, handler in (
            (
                "compare-base-criteria",
                scalability.compare_base_criteria_protocol,
            ),
            (
                "compare-base-mps",
                scalability.compare_base_mps_protocol,
            ),
        ):
            with self.subTest(command=command):
                arguments = parser.parse_args([
                    command, "--output", "comparison.jsonl"
                ])
                self.assertIs(arguments.handler, handler)
                self.assertEqual(arguments.output, Path("comparison.jsonl"))

    def test_runtime_fingerprint_excludes_kernel_patch_release(self):
        with patch.object(
            scalability.platform,
            "release",
            side_effect=AssertionError("kernel release must not enter the checksum"),
        ):
            runtime = scalability._runtime_value()
        self.assertNotIn("release", runtime)
        self.assertEqual(runtime["libc"], list(scalability.platform.libc_ver()))

    def test_import_does_not_mutate_shared_partitioner_retry_policy(self):
        self.assertEqual(
            batching.GUROBI_LICENSE_RETRY_DELAYS_SECONDS,
            (15, 30, 60, 120, 240, 480),
        )
        self.assertEqual(
            scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS,
            (5, 10, 20, 40),
        )

    def test_protocol_records_its_bounded_gurobi_policy_per_call(self):
        with patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ):
            protocol = scalability._protocol_config()
            shared_default = batching.solver_provenance()
        self.assertEqual(
            protocol["partition_solver"][
                "transient_license_retry_delays_seconds"
            ],
            [5, 10, 20, 40],
        )
        self.assertEqual(
            shared_default["transient_license_retry_delays_seconds"],
            [15, 30, 60, 120, 240, 480],
        )


FINGERPRINT_FIELDS = (
    "corpus_sha256",
    "manifest_sha256",
    "dependencies_sha256",
    "runtime_sha256",
    "sources_sha256",
    "config_sha256",
)


def fingerprints(character: str = "a") -> dict[str, str]:
    return {field: character * 64 for field in FINGERPRINT_FIELDS}


def passing_crosscheck(point_count: int = 1) -> dict:
    return {
        "required": True,
        "performed": True,
        "passed": True,
        "point_count_mps": point_count,
        "point_count_statevector": point_count,
        "mps_selected_event_count": 0,
        "statevector_selected_event_count": 0,
        "mps_selected_events": [],
        "statevector_selected_events": [],
        "mps_selected_events_sha256": scalability.stable_hash([]),
        "statevector_selected_events_sha256": scalability.stable_hash([]),
        "selection_mismatch_count": 0,
        "missing_mps_point_count": 0,
        "missing_statevector_point_count": 0,
        "maximum_absolute_a2_difference": 0.0,
        "maximum_absolute_determinant_difference": 0.0,
    }


def minimal_mps_diagnostics(point_count: int) -> dict:
    return {
        "point_count": point_count,
        "configured_max_bond_dimension": None,
        "configured_max_bond_dimension_mode": "unbounded",
        "configured_truncation_threshold": scalability.MPS_TRUNCATION_THRESHOLD,
        "configured_mps_lapack": scalability.MPS_LAPACK,
        "observed_max_chi": 1,
        "observed_max_chi_available": True,
        "observed_max_chi_method": "save_matrix_product_state test evidence",
        "observed_max_chi_unavailable_reason": None,
    }


def terminal_record(
    status: str,
    *,
    attempt_id: str = "base:synthetic",
    instrumentation: dict | None = None,
    original_qubits: int = 3,
    original_m_operations: int | None = None,
) -> dict:
    error = None
    circuit = None
    selection = None
    if status == "success":
        if original_m_operations is None:
            original_m_operations = (
                instrumentation.get("original_unitary_gate_count_m", 1)
                if instrumentation is not None
                else 1
            )
        circuit = {
            "evidence_stage": "eligibility_fingerprint_complete",
            "original_qubits": original_qubits,
            "original_m_operations": original_m_operations,
            "original_depth": 1,
            "total_instructions": 1,
            "canonical_circuit_sha256": "c" * 64,
            "fingerprint_method": (
                "canonical instruction stream plus per-unitary matrix SHA-256"
            ),
            "requested_qubits": None,
            "eligibility": {"eligible": True},
        }
        selection = {
            "mps_runtime_seconds": 0.01,
            "mps_peak_rss_bytes": 1024,
            "mps_rss_measurement": {
                "available": True,
                "method": "synthetic high-water RSS",
            },
            "pre_selection_high_water_rss_bytes": 512,
            "validation_runtime_seconds": 0.0,
            "total_selection_certificate_runtime_seconds": 0.01,
            "total_selection_certificate_peak_rss_bytes": 1024,
            "total_rss_measurement": {
                "available": True,
                "method": "synthetic high-water RSS",
            },
            "monitorable_node_count": 0,
            "monitorable_gate_node_count": 0,
            "monitorable_nodes": [],
            "monitorable_nodes_sha256": scalability.stable_hash([]),
            "point_scores": [],
            "point_scores_sha256": scalability.stable_hash([]),
            "point_score_definition": {
                "selection": "second_schmidt_amplitude_a2",
                "selection_tolerance": (
                    scalability.SCHMIDT_AMPLITUDE_TOLERANCE
                ),
                "a2_extraction": "aer_mps_local_tensor_svd",
                "determinant_role": "physical_diagnostic_only",
            },
            "mps": {
                "configured_max_bond_dimension": None,
                "configured_max_bond_dimension_mode": "unbounded",
                "configured_truncation_threshold": (
                    scalability.MPS_TRUNCATION_THRESHOLD
                ),
                "configured_mps_lapack": True,
                "observed_max_chi": 1,
                "observed_max_chi_available": True,
                "observed_max_chi_method": "save_matrix_product_state test evidence",
                "observed_max_chi_unavailable_reason": None,
            },
            "statevector_crosscheck": {
                "required": False,
                "performed": False,
                "passed": None,
            },
        }
        if instrumentation is None:
            partition_solver = scalability.solver_provenance()
            instrumentation = {
                "qmax": 24,
                "partitioner": "batch_partition.partition_nodes",
                "partition_solver": partition_solver,
                "partition_solver_sha256": scalability.stable_hash(
                    partition_solver
                ),
                "partition_metadata": {
                    "solver": partition_solver,
                    "capacity": 24 - original_qubits,
                    "item_count": 0,
                    "feasible_item_count": 0,
                    "infeasible_item_count": 0,
                    "ffd_batch_count": 0,
                    "final_batch_count": 0,
                    "refinement": {
                        "attempted": False,
                        "node_limit": 120,
                        "target_batch_count": None,
                        "time_limit_seconds": (
                            scalability.MAX_REFINEMENT_TIME_LIMIT_SECONDS
                        ),
                        "solver_status": "not-run",
                        "resolved": None,
                        "fallback_used": False,
                        "improved": False,
                    },
                },
                "partition_budget": {
                    "refinement_node_limit": scalability.REFINEMENT_NODE_LIMIT,
                    "refinement_time_limit_seconds": (
                        scalability.MAX_REFINEMENT_TIME_LIMIT_SECONDS
                    ),
                    "license_retry_backoff_budget_seconds": float(
                        sum(scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
                    ),
                    "completion_reserve_seconds": (
                        scalability.INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
                    ),
                },
                "partition_certificate_kind": "structural_ancilla_width",
                "emitted_circuits_executed_or_width_checked": False,
                "operation_accounting": dict(
                    scalability.INSTRUMENTATION_OPERATION_ACCOUNTING
                ),
                "original_qubits": original_qubits,
                "original_unitary_gate_count_m": original_m_operations,
                "monitorable_node_count": 0,
                "monitorable_gate_node_count": 0,
                "candidate_monitor_event_count": 0,
                "candidate_gate_node_count": 0,
                "individually_feasible_monitor_event_count": 0,
                "individually_feasible_gate_node_count": 0,
                "deployed_monitor_event_count": 0,
                "deployed_gate_node_count": 0,
                "terminally_measured_qubits": [],
                "free_final_read_event_count": 0,
                "planned_peak_extra_ancillas": 0,
                "planned_cumulative_extra_ancilla_allocations": 0,
                "planned_original_gate_executions": 0,
                "planned_replay_gate_executions": 0,
                "planned_monitor_measurement_operations": 0,
                "planned_monitor_reset_operations": 0,
                "planned_total_counted_operations": 0,
                "diagnostic_selected_per_event_cone_gate_occurrences": 0,
                "diagnostic_selected_per_event_cone_qubit_occurrences": 0,
                "diagnostic_candidate_per_event_cone_gate_occurrences": 0,
                "diagnostic_candidate_per_event_cone_qubit_occurrences": 0,
                "diagnostic_selected_per_event_extra_cone_qubit_occurrences": 0,
                "individually_infeasible_gate_node_count": 0,
                "infeasible_gate_nodes": [],
                "gate_node_costs": [],
                "batch_count": 0,
                "maximum_run_qubits": None,
                "batches": [],
                "runtime_seconds": 0.0,
            }
        else:
            instrumentation = copy.deepcopy(instrumentation)
            instrumentation.setdefault("runtime_seconds", 0.0)
    else:
        outcome = {
            "timeout": (
                "selection",
                "wall_clock_timeout",
                "ParentProcessOutcome",
            ),
            "build_error": (
                "build",
                "circuit_build_failed",
                "SyntheticBuildError",
            ),
            "analysis_error": (
                "selection",
                "mps_analysis_failed",
                "SyntheticAnalysisError",
            ),
            "resource_limit": ("selection", "memory_error", "MemoryError"),
            "unsupported": (
                "build",
                "generator_rejected",
                "UnsupportedAttempt",
            ),
        }[status]
        phase, category, exception_type = outcome
        trace = "synthetic traceback"
        error = {
            "phase": phase,
            "category": category,
            "exception_type": exception_type,
            "message": status,
            "exception_evidence_sha256": (
                scalability._exception_evidence_sha256(
                    exception_type, status
                )
            ),
            "traceback": trace,
            "traceback_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        }
    record = {
        "schema_version": scalability.SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_index": 0,
        "source": {"source_kind": "test", "attempt_id": attempt_id},
        "qmax": 24,
        "status": status,
        "status_semantics": "analysis_terminal_status",
        "deployment_status": (
            scalability._deployment_status(circuit, instrumentation)
            if status == "success"
            else "not_evaluated"
        ),
        "record_complete": True,
        "record_sha256": None,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "completed_at_utc": "2026-01-01T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "phase_elapsed_seconds": (
            {"selection": 0.01, "instrumentation": 0.0}
            if status == "success"
            else {}
        ),
        "last_phase": "complete" if status == "success" else error["phase"],
        "error": error,
        "error_category": None if error is None else error["category"],
        "circuit": circuit,
        "selection": selection,
        "instrumentation": instrumentation,
        "fingerprints": fingerprints(),
        "provenance": {
            "protocol_config": {
                "qmax": 24,
                "partitioner_qmax": 24,
                "attempt_timeout_default_seconds": (
                    scalability.DEFAULT_TIMEOUT_SECONDS
                ),
                "refinement_node_limit": scalability.REFINEMENT_NODE_LIMIT,
                "refinement_max_time_limit_seconds": (
                    scalability.MAX_REFINEMENT_TIME_LIMIT_SECONDS
                ),
                "instrumentation_completion_reserve_seconds": (
                    scalability.INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
                ),
                "gurobi_license_retry_delays_seconds": list(
                    scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS
                ),
                "infrastructure_outcome_policy": (
                    "checkpoint as incomplete, retry once immediately, and exclude "
                    "from terminal publication artifacts"
                ),
                "partition_solver": scalability.solver_provenance(),
                "partition_solver_sha256": scalability.stable_hash(
                    scalability.solver_provenance()
                ),
                "publication_partition_solver_backend": "gurobi",
                "status_semantics": (
                    "status is the analysis terminal status; deployment_status "
                    "is the separate operational Qmax outcome"
                ),
                "instrumentation_operation_accounting": dict(
                    scalability.INSTRUMENTATION_OPERATION_ACCOUNTING
                ),
                "mps_max_bond_dimension": None,
                "mps_lapack": True,
                "schmidt_amplitude_tolerance": (
                    scalability.SCHMIDT_AMPLITUDE_TOLERANCE
                ),
                "density_matrix_validation_tolerance": (
                    scalability.DENSITY_MATRIX_TOLERANCE
                ),
            }
        },
        "run": {
            "mode": "test",
            "shard_index": 0,
            "shard_count": 1,
            "attempt_timeout_seconds": scalability.DEFAULT_TIMEOUT_SECONDS,
            "infrastructure_retries_before_outcome": 0,
        },
    }
    return scalability._add_record_checksum(record)


def bind_record(record: dict, attempt: scalability.AttemptSpec) -> dict:
    record["attempt_id"] = attempt.attempt_id
    record["attempt_index"] = attempt.attempt_index
    record["source"] = asdict(attempt)
    if record.get("status") == "success" and attempt.source_kind == "canonical_qasm":
        record["selection"]["statevector_crosscheck"] = passing_crosscheck()
    return scalability._add_record_checksum(record)


class CanonicalFrameTests(unittest.TestCase):
    def test_resolves_manifest_exactly_and_reports_directory_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qasm = root / "qasm"
            qasm.mkdir()
            manifest = root / "certified.txt"
            manifest.write_text("alpha\nbeta\n", encoding="utf-8")
            (qasm / "alpha.qasm").write_text("alpha", encoding="utf-8")
            (qasm / "beta.qasm").write_text("beta", encoding="utf-8")
            (qasm / "not_certified.qasm").write_text("extra", encoding="utf-8")

            frame = scalability.resolve_canonical_frame(
                manifest, qasm, expected_count=2
            )

            self.assertEqual(frame["basenames"], ["alpha", "beta"])
            self.assertEqual(frame["circuit_count"], 2)
            self.assertEqual(
                frame["excluded_extra_qasm_basenames"], ["not_certified"]
            )
            self.assertEqual(len(frame["corpus_sha256"]), 64)

    def test_rejects_duplicate_missing_and_requested_extra_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qasm = root / "qasm"
            qasm.mkdir()
            manifest = root / "certified.txt"
            (qasm / "alpha.qasm").write_text("alpha", encoding="utf-8")

            manifest.write_text("alpha\nalpha\n", encoding="utf-8")
            with self.assertRaisesRegex(scalability.ProtocolError, "duplicate"):
                scalability.resolve_canonical_frame(manifest, qasm, expected_count=2)

            manifest.write_text("alpha\nbeta\n", encoding="utf-8")
            with self.assertRaisesRegex(scalability.ProtocolError, "missing"):
                scalability.resolve_canonical_frame(manifest, qasm, expected_count=2)

            (qasm / "beta.qasm").write_text("beta", encoding="utf-8")
            with self.assertRaisesRegex(scalability.ProtocolError, "differs"):
                scalability.resolve_canonical_frame(
                    manifest,
                    qasm,
                    expected_count=2,
                    requested_basenames=["alpha", "outside"],
                )

    def test_repository_frame_is_exactly_310_and_grid_reaches_50(self):
        frame = scalability.resolve_canonical_frame()
        grid = scalability.build_attempt_grid(frame)

        self.assertEqual(frame["circuit_count"], 310)
        self.assertEqual(
            frame["manifest_sha256"], scalability.EXPECTED_MANIFEST_SHA256
        )
        self.assertEqual(frame["corpus_sha256"], scalability.EXPECTED_CORPUS_SHA256)
        self.assertEqual(
            len(grid),
            310
            + len(scalability.EXTENSION_FAMILIES)
            * len(scalability.EXTENSION_SIZES),
        )
        extension = [attempt for attempt in grid if attempt.family is not None]
        self.assertEqual(max(attempt.requested_qubits for attempt in extension), 50)
        self.assertEqual(
            {attempt.family for attempt in extension},
            set(scalability.EXTENSION_FAMILIES),
        )
        smoke = scalability.select_attempts(
            grid, smoke=True, shard_index=0, shard_count=1
        )
        self.assertEqual(
            [attempt.attempt_id for attempt in smoke],
            list(scalability.SMOKE_ATTEMPT_IDS),
        )

    def test_rejects_310_entry_manifest_and_corpus_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            substituted_manifest = root / "certified_circuits.txt"
            substituted_manifest.write_text(
                scalability.DEFAULT_MANIFEST.read_text(encoding="utf-8")
                + "\n# byte-level substitution\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                scalability.ProtocolError, "manifest checksum substitution"
            ):
                scalability.resolve_canonical_frame(
                    substituted_manifest, scalability.DEFAULT_QASM_DIR
                )

            substituted_qasm = root / "qasm"
            substituted_qasm.mkdir()
            basenames = [
                line.strip()
                for line in scalability.DEFAULT_MANIFEST.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            for basename in basenames:
                source = scalability.DEFAULT_QASM_DIR / f"{basename}.qasm"
                (substituted_qasm / source.name).symlink_to(source)
            first = substituted_qasm / f"{basenames[0]}.qasm"
            original = first.resolve().read_text(encoding="utf-8")
            first.unlink()
            first.write_text(original + "\n// corpus substitution\n", encoding="utf-8")
            with self.assertRaisesRegex(
                scalability.ProtocolError, "corpus checksum substitution"
            ):
                scalability.resolve_canonical_frame(
                    scalability.DEFAULT_MANIFEST, substituted_qasm
                )


class GeneratorReproducibilityTests(unittest.TestCase):
    def test_all_34_q15_generator_hashes_and_statuses_match_two_rebuilds(self):
        self.assertEqual(
            set(scalability.Q15_GENERATOR_REFERENCE),
            set(scalability.EXTENSION_FAMILIES),
        )
        passes = [
            {
                family: scalability.check_q15_generator(family)
                for family in scalability.EXTENSION_FAMILIES
            }
            for _ in range(2)
        ]
        self.assertEqual(passes[0], scalability.Q15_GENERATOR_REFERENCE)
        self.assertEqual(passes[1], scalability.Q15_GENERATOR_REFERENCE)
        self.assertEqual(passes[0], passes[1])

    def test_known_cross_family_duplicates_match_at_all_supported_sizes(self):
        for size, expected_hash in scalability.KNOWN_DUPLICATE_EXTENSION_HASHES.items():
            hashes = []
            for family in ("cdkm_ripple_carry_adder", "full_adder"):
                spec = scalability.AttemptSpec(
                    f"extension:{family}:q{size}",
                    0,
                    "mqt_bench_extension",
                    f"{family}_q{size}",
                    family=family,
                    requested_qubits=size,
                )
                circuit = scalability._build_circuit(spec)
                scalability.validate_circuit_eligibility(circuit)
                hashes.append(scalability._circuit_canonical_sha256(circuit))
            self.assertEqual(hashes, [expected_hash, expected_hash])


class RecordValidationTests(unittest.TestCase):
    def test_all_declared_terminal_statuses_have_complete_shapes(self):
        for status in sorted(scalability.STATUSES):
            with self.subTest(status=status):
                record = terminal_record(status)
                self.assertEqual(scalability.validate_record(record), [])

        incomplete = terminal_record("success")
        incomplete["record_complete"] = False
        self.assertIn(
            "record_complete is not true",
            scalability.validate_record(incomplete),
        )

        malformed = terminal_record("analysis_error")
        malformed["error"] = None
        self.assertTrue(
            any(
                "structured error" in error
                for error in scalability.validate_record(malformed)
            )
        )

    def test_rejects_status_phase_category_cross_products(self):
        bad_category = terminal_record("timeout")
        bad_category["error"]["category"] = "mps_analysis_failed"
        bad_category["error_category"] = "mps_analysis_failed"
        scalability._add_record_checksum(bad_category)
        self.assertTrue(
            any(
                "phase/category contract" in error
                for error in scalability.validate_record(bad_category)
            )
        )

        bad_phase = terminal_record("build_error")
        bad_phase["error"]["phase"] = "selection"
        bad_phase["last_phase"] = "selection"
        scalability._add_record_checksum(bad_phase)
        self.assertTrue(
            any(
                "status x phase x category" in error
                for error in scalability.validate_record(bad_phase)
            )
        )

        fake_oom = terminal_record("resource_limit")
        fake_oom["error"]["category"] = "mps_backend_out_of_memory"
        fake_oom["error_category"] = "mps_backend_out_of_memory"
        fake_oom["error"]["exception_type"] = "SyntheticAnalysisError"
        fake_oom["error"]["exception_evidence_sha256"] = (
            scalability._exception_evidence_sha256(
                "SyntheticAnalysisError", fake_oom["error"]["message"]
            )
        )
        scalability._add_record_checksum(fake_oom)
        self.assertIn(
            "MPS out-of-memory category lacks backend memory evidence",
            scalability.validate_record(fake_oom),
        )

    def test_runtime_and_rss_are_range_checked_but_not_replay_certificates(self):
        negative_runtime = terminal_record("success")
        negative_runtime["selection"]["mps_runtime_seconds"] = -1.0
        scalability._add_record_checksum(negative_runtime)
        self.assertIn(
            "selection mps_runtime_seconds is missing or invalid",
            scalability.validate_record(negative_runtime),
        )

        nonmonotone_rss = terminal_record("success")
        nonmonotone_rss["selection"]["mps_peak_rss_bytes"] = 2048
        nonmonotone_rss["selection"][
            "total_selection_certificate_peak_rss_bytes"
        ] = 1024
        scalability._add_record_checksum(nonmonotone_rss)
        self.assertIn(
            "selection total peak RSS is below the MPS peak RSS",
            scalability.validate_record(nonmonotone_rss),
        )

    def test_rejects_missing_old_and_mixed_qmax(self):
        valid = terminal_record("timeout", attempt_id="base:a")
        old = terminal_record("timeout", attempt_id="base:b")
        old["attempt_index"] = 1
        old["qmax"] = 20
        with self.assertRaisesRegex(scalability.ProtocolError, "qmax=20"):
            scalability.validate_record_set([valid, old])

        missing = terminal_record("timeout")
        del missing["qmax"]
        self.assertTrue(
            any("expected exactly 24" in error for error in scalability.validate_record(missing))
        )

        mixed_metadata = terminal_record("timeout")
        mixed_metadata["provenance"]["protocol_config"]["qmax"] = 20
        self.assertTrue(
            any(
                "protocol_config qmax" in error
                for error in scalability.validate_record(mixed_metadata)
            )
        )

        malformed_width = terminal_record("success")
        malformed_width["circuit"]["original_qubits"] = "24"
        scalability._add_record_checksum(malformed_width)
        errors = scalability.validate_record(malformed_width)
        self.assertTrue(
            any("original_qubits is missing or invalid" in error for error in errors)
        )
        self.assertTrue(any("deployment_status differs" in error for error in errors))

    def test_requires_fingerprints_and_detects_drift(self):
        record = terminal_record("timeout")
        del record["fingerprints"]["config_sha256"]
        self.assertTrue(
            any(
                "checksum config_sha256" in error
                for error in scalability.validate_record(record)
            )
        )

        record = terminal_record("timeout")
        expected = fingerprints("b")
        errors = scalability.validate_record(
            record, expected_fingerprints=expected
        )
        self.assertTrue(any("checksum mismatch" in error for error in errors))

        non_hex = terminal_record("timeout")
        non_hex["fingerprints"]["runtime_sha256"] = "z" * 64
        self.assertTrue(
            any(
                "invalid checksum runtime_sha256" in error
                for error in scalability.validate_record(non_hex)
            )
        )

    def test_validates_record_checksum_traceback_hash_and_shard_metadata(self):
        unchecked_tamper = terminal_record("timeout")
        unchecked_tamper["error"]["message"] = "changed after completion"
        self.assertIn(
            "record_sha256 integrity checksum mismatch",
            scalability.validate_record(unchecked_tamper),
        )

        bad_traceback = terminal_record("timeout")
        bad_traceback["error"]["traceback"] = "fabricated trace"
        scalability._add_record_checksum(bad_traceback)
        self.assertIn(
            "error traceback_sha256 does not match traceback",
            scalability.validate_record(bad_traceback),
        )

        bad_shard = terminal_record("timeout")
        bad_shard["attempt_index"] = 2
        bad_shard["run"] = {"mode": "full", "shard_index": 1, "shard_count": 2}
        scalability._add_record_checksum(bad_shard)
        self.assertIn(
            "record attempt is assigned to the wrong full-run shard",
            scalability.validate_record(bad_shard),
        )

        impossible_shard = terminal_record("timeout")
        impossible_shard["run"]["shard_index"] = 1
        impossible_shard["run"]["shard_count"] = 1
        scalability._add_record_checksum(impossible_shard)
        self.assertIn(
            "record shard_index is outside the declared shard_count",
            scalability.validate_record(impossible_shard),
        )

        smoke_record = terminal_record("timeout")
        smoke_record["run"]["mode"] = "smoke"
        scalability._add_record_checksum(smoke_record)
        with self.assertRaisesRegex(scalability.ProtocolError, "outside run mode"):
            scalability.validate_record_set(
                [smoke_record], expected_run_mode="full"
            )

        shard_zero = terminal_record("timeout", attempt_id="base:zero")
        shard_zero["run"] = {"mode": "full", "shard_index": 0, "shard_count": 2}
        scalability._add_record_checksum(shard_zero)
        shard_one = terminal_record("timeout", attempt_id="base:one")
        shard_one["attempt_index"] = 1
        shard_one["run"] = {"mode": "full", "shard_index": 1, "shard_count": 3}
        scalability._add_record_checksum(shard_one)
        with self.assertRaisesRegex(scalability.ProtocolError, "mixes shard_count"):
            scalability.validate_record_set(
                [shard_zero, shard_one], expected_run_mode="full"
            )

    def test_deep_validation_rejects_checksummed_fabricated_scientific_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.qasm"
            circuit = QuantumCircuit(1)
            circuit.h(0)
            qasm2.dump(circuit, path)
            attempt = scalability.AttemptSpec(
                "base:synthetic",
                0,
                "canonical_qasm",
                "synthetic",
                basename="synthetic",
                qasm_path=str(path),
                qasm_sha256=scalability.file_sha256(path),
            )
            frame = scalability.resolve_canonical_frame()
            bundle = scalability.build_fingerprint_bundle(frame)
            record = scalability.evaluate_attempt(
                attempt,
                bundle,
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )
            self.assertEqual(record["status"], "success")
            self.assertEqual(
                scalability.validate_record(
                    record, expected_attempt=attempt, recompute_science=True
                ),
                [],
            )

            fabricated_chi = copy.deepcopy(record)
            fabricated_chi["selection"]["mps"]["observed_max_chi"] += 1
            scalability._add_record_checksum(fabricated_chi)
            self.assertIn(
                "stored observed maximum chi differs from MPS recomputation",
                scalability.validate_record(
                    fabricated_chi,
                    expected_attempt=attempt,
                    recompute_science=True,
                ),
            )

            non_replayable_measurements = copy.deepcopy(record)
            non_replayable_measurements["selection"]["mps_runtime_seconds"] += 0.001
            non_replayable_measurements["selection"][
                "total_selection_certificate_runtime_seconds"
            ] += 0.001
            non_replayable_measurements["phase_elapsed_seconds"]["selection"] = (
                non_replayable_measurements["selection"][
                    "total_selection_certificate_runtime_seconds"
                ]
            )
            for field in (
                "pre_selection_high_water_rss_bytes",
                "mps_peak_rss_bytes",
                "total_selection_certificate_peak_rss_bytes",
            ):
                non_replayable_measurements["selection"][field] += 4096
            scalability._add_record_checksum(non_replayable_measurements)
            self.assertEqual(
                scalability.validate_record(
                    non_replayable_measurements,
                    expected_attempt=attempt,
                    recompute_science=True,
                ),
                [],
            )

            fabricated_hash = copy.deepcopy(record)
            fabricated_hash["circuit"]["canonical_circuit_sha256"] = "d" * 64
            scalability._add_record_checksum(fabricated_hash)
            self.assertTrue(
                any(
                    "recomputed circuit canonical_circuit_sha256" in error
                    for error in scalability.validate_record(
                        fabricated_hash,
                        expected_attempt=attempt,
                        recompute_science=True,
                    )
                )
            )

            fabricated_zero = copy.deepcopy(record)
            fabricated_zero["selection"]["monitorable_nodes"] = []
            fabricated_zero["selection"]["monitorable_nodes_sha256"] = (
                scalability.stable_hash([])
            )
            fabricated_zero["selection"]["monitorable_node_count"] = 0
            fabricated_zero["selection"]["monitorable_gate_node_count"] = 0
            fabricated_zero["instrumentation"] = (
                scalability.build_instrumentation_metrics(circuit, {})
            )
            fabricated_zero["instrumentation"]["runtime_seconds"] = 0.0
            scalability._add_record_checksum(fabricated_zero)
            errors = scalability.validate_record(
                fabricated_zero,
                expected_attempt=attempt,
                recompute_science=True,
            )
            self.assertTrue(
                any(
                    "stored monitorable nodes differ" in error
                    for error in errors
                )
            )

            path.write_text(
                path.read_text(encoding="utf-8") + "\n// source tamper\n",
                encoding="utf-8",
            )
            errors = scalability.validate_record(
                record, expected_attempt=attempt, recompute_science=True
            )
            self.assertTrue(
                any("QASM checksum mismatch" in error for error in errors)
            )

    def test_deep_validation_replays_stored_partition_without_resolving(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stored-plan.qasm"
            circuit = QuantumCircuit(1)
            circuit.h(0)
            qasm2.dump(circuit, path)
            attempt = scalability.AttemptSpec(
                "base:stored-plan",
                0,
                "canonical_qasm",
                "stored-plan",
                basename="stored-plan",
                qasm_path=str(path),
                qasm_sha256=scalability.file_sha256(path),
            )
            bundle = scalability.build_fingerprint_bundle(
                scalability.resolve_canonical_frame()
            )
            record = scalability.evaluate_attempt(
                attempt,
                bundle,
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )
            self.assertEqual(record["status"], "success")
            with patch.object(
                scalability,
                "partition_nodes",
                side_effect=AssertionError("validator reran timed refinement"),
            ):
                self.assertEqual(
                    scalability.validate_record(
                        record,
                        expected_attempt=attempt,
                        recompute_science=True,
                    ),
                    [],
                )

    def test_deep_validation_accepts_only_reproducible_build_error_evidence(self):
        attempt = scalability.AttemptSpec(
            "extension:synthetic:q15",
            310,
            "mqt_bench_extension",
            "synthetic_q15",
            family="synthetic",
            requested_qubits=15,
        )
        bundle = scalability.build_fingerprint_bundle(
            scalability.resolve_canonical_frame()
        )
        with patch.object(
            scalability,
            "_build_circuit",
            side_effect=RuntimeError("deterministic generator failure"),
        ):
            record = scalability.evaluate_attempt(
                attempt,
                bundle,
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )
            errors = scalability.validate_record(
                record, expected_attempt=attempt, recompute_science=True
            )

        self.assertEqual(record["status"], "build_error")
        self.assertEqual(record["error_category"], "circuit_build_failed")
        self.assertEqual(errors, [])

        with patch.object(
            scalability,
            "_build_circuit",
            side_effect=RuntimeError("different generator failure"),
        ):
            errors = scalability.validate_record(
                record, expected_attempt=attempt, recompute_science=True
            )
        self.assertTrue(
            any("exception evidence differs" in error for error in errors)
        )

    def test_sigsegv_is_analysis_error_not_resource_limit(self):
        self.assertEqual(
            scalability._worker_exit_outcome(-signal.SIGSEGV),
            ("analysis_error", "worker_segmentation_fault"),
        )
        self.assertEqual(
            scalability._worker_exit_outcome(-signal.SIGKILL),
            ("resource_limit", "worker_killed_resource_limit"),
        )

    def test_full_publication_artifact_rejects_local_cbc_backend(self):
        record = terminal_record("timeout")
        record["run"]["mode"] = "full"
        scalability._add_record_checksum(record)
        with self.assertRaisesRegex(
            scalability.ProtocolError, "QMON_MILP_SOLVER=gurobi"
        ):
            scalability.validate_record_set(
                [record], expected_run_mode="full"
            )

    def test_canonical_success_requires_unbounded_mps_and_crosscheck(self):
        attempt = scalability.AttemptSpec(
            "base:alpha",
            0,
            "canonical_qasm",
            "alpha",
            basename="alpha",
            qasm_path="/canonical/alpha.qasm",
            qasm_sha256="1" * 64,
        )
        record = bind_record(terminal_record("success"), attempt)
        self.assertEqual(
            scalability.validate_record(record, expected_attempt=attempt), []
        )

        capped = copy.deepcopy(record)
        capped["selection"]["mps"]["configured_max_bond_dimension"] = 256
        self.assertTrue(
            any(
                "finite MPS maximum" in error
                for error in scalability.validate_record(
                    capped, expected_attempt=attempt
                )
            )
        )

        mismatch = copy.deepcopy(record)
        mismatch["selection"]["statevector_crosscheck"]["passed"] = False
        mismatch["selection"]["statevector_crosscheck"][
            "selection_mismatch_count"
        ] = 1
        self.assertTrue(
            any(
                "statevector cross-check passed is not true" in error
                for error in scalability.validate_record(
                    mismatch, expected_attempt=attempt
                )
            )
        )

        tampered_events = copy.deepcopy(record)
        tampered_events["selection"]["statevector_crosscheck"][
            "statevector_selected_events"
        ] = [[0, 0]]
        tampered_events["selection"]["statevector_crosscheck"][
            "statevector_selected_event_count"
        ] = 1
        tampered_events["selection"]["statevector_crosscheck"][
            "statevector_selected_events_sha256"
        ] = scalability.stable_hash([(0, 0)])
        scalability._add_record_checksum(tampered_events)
        self.assertTrue(
            any(
                "statevector and MPS selected events differ" in error
                for error in scalability.validate_record(
                    tampered_events, expected_attempt=attempt
                )
            )
        )

    def test_record_set_is_bound_to_exact_attempt_specs(self):
        base = scalability.AttemptSpec(
            "base:alpha",
            0,
            "canonical_qasm",
            "alpha",
            basename="alpha",
            qasm_path="/canonical/alpha.qasm",
            qasm_sha256="1" * 64,
        )
        extension = scalability.AttemptSpec(
            "extension:ghz:q15",
            310,
            "mqt_bench_extension",
            "ghz_q15",
            family="ghz",
            requested_qubits=15,
        )
        expected = {
            base.attempt_id: base,
            extension.attempt_id: extension,
        }
        base_record = bind_record(terminal_record("timeout"), base)
        extension_record = bind_record(terminal_record("unsupported"), extension)
        self.assertEqual(
            scalability.validate_record_set(
                [base_record, extension_record], expected_attempts=expected
            ),
            {"timeout": 1, "unsupported": 1},
        )

        tampered = copy.deepcopy(base_record)
        tampered["attempt_index"] = 999
        with self.assertRaisesRegex(scalability.ProtocolError, "attempt_index"):
            scalability.validate_record_set(
                [tampered, extension_record], expected_attempts=expected
            )

        tampered = copy.deepcopy(base_record)
        tampered["source"]["qasm_sha256"] = "2" * 64
        with self.assertRaisesRegex(scalability.ProtocolError, "fixed source"):
            scalability.validate_record_set(
                [tampered, extension_record], expected_attempts=expected
            )

        tampered = copy.deepcopy(extension_record)
        tampered["source"]["family"] = "qft"
        tampered["source"]["requested_qubits"] = 18
        with self.assertRaisesRegex(scalability.ProtocolError, "fixed source"):
            scalability.validate_record_set(
                [base_record, tampered], expected_attempts=expected
            )

        swapped_base = copy.deepcopy(base_record)
        swapped_extension = copy.deepcopy(extension_record)
        swapped_base["source"], swapped_extension["source"] = (
            swapped_extension["source"],
            swapped_base["source"],
        )
        with self.assertRaisesRegex(scalability.ProtocolError, "fixed source"):
            scalability.validate_record_set(
                [swapped_base, swapped_extension], expected_attempts=expected
            )

    def test_resume_retains_every_complete_terminal_status(self):
        attempts = [
            scalability.AttemptSpec("base:a", 0, "canonical_qasm", "a"),
            scalability.AttemptSpec("base:b", 1, "canonical_qasm", "b"),
            scalability.AttemptSpec("base:c", 2, "canonical_qasm", "c"),
            scalability.AttemptSpec("base:d", 3, "canonical_qasm", "d"),
            scalability.AttemptSpec("base:e", 4, "canonical_qasm", "e"),
        ]
        success = bind_record(terminal_record("success"), attempts[0])
        timeout = bind_record(terminal_record("timeout"), attempts[1])
        resource_limit = bind_record(
            terminal_record("resource_limit"), attempts[2]
        )
        unsupported = bind_record(terminal_record("unsupported"), attempts[3])
        incomplete = bind_record(terminal_record("analysis_error"), attempts[4])
        incomplete["record_complete"] = False
        pending = scalability.pending_attempts(
            attempts,
            {
                "base:a": success,
                "base:b": timeout,
                "base:c": resource_limit,
                "base:d": unsupported,
                "base:e": incomplete,
            },
            expected_fingerprints=fingerprints(),
        )
        self.assertEqual([attempt.attempt_id for attempt in pending], ["base:e"])

        incomplete["qmax"] = 20
        with self.assertRaisesRegex(scalability.ProtocolError, "expected exactly 24"):
            scalability.pending_attempts(
                attempts,
                {
                    "base:a": success,
                    "base:b": timeout,
                    "base:c": resource_limit,
                    "base:d": unsupported,
                    "base:e": incomplete,
                },
                expected_fingerprints=fingerprints(),
            )


class WorkerAndDeploymentTests(unittest.TestCase):
    @staticmethod
    def bundle() -> dict:
        return scalability.build_fingerprint_bundle(
            scalability.resolve_canonical_frame()
        )

    @staticmethod
    def extension_attempt() -> scalability.AttemptSpec:
        return scalability.AttemptSpec(
            "extension:synthetic:q25",
            310,
            "mqt_bench_extension",
            "synthetic_q25",
            family="synthetic",
            requested_qubits=25,
        )

    def evaluate_with_selection(
        self,
        circuit: QuantumCircuit,
        nodes: dict[int, list[int]],
        scores: dict[tuple[int, int], dict[str, float]],
    ) -> dict:
        with patch.object(scalability, "_build_circuit", return_value=circuit), patch.object(
            scalability,
            "_select_monitorable_nodes_mps_details",
            return_value=(
                nodes,
                minimal_mps_diagnostics(len(scores)),
                scores,
            ),
        ):
            return scalability.evaluate_attempt(
                self.extension_attempt(),
                self.bundle(),
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )

    def test_worker_streams_actual_circuit_evidence_and_parent_failure_keeps_it(self):
        class RecordingConnection:
            def __init__(self):
                self.messages = []
                self.closed = False

            def send(self, value):
                self.messages.append(value)

            def close(self):
                self.closed = True

        circuit = QuantumCircuit(7)
        circuit.h(0)
        connection = RecordingConnection()
        attempt = self.extension_attempt()
        bundle = self.bundle()
        with patch.object(scalability, "_build_circuit", return_value=circuit), patch.object(
            scalability,
            "_select_monitorable_nodes_mps_details",
            return_value=({}, minimal_mps_diagnostics(0), {}),
        ):
            scalability._worker_entry(
                asdict(attempt), bundle, "test", 0, 1, connection
            )

        evidence_messages = [
            message
            for message in connection.messages
            if message["kind"] == "circuit_evidence"
        ]
        self.assertEqual(
            [message["circuit"]["evidence_stage"] for message in evidence_messages],
            [
                "build_complete",
                "eligibility_complete",
                "eligibility_fingerprint_complete",
            ],
        )
        self.assertTrue(
            all(message["circuit"]["original_qubits"] == 7 for message in evidence_messages)
        )
        self.assertTrue(connection.closed)

        selection_messages = [
            message
            for message in connection.messages
            if message["kind"] == "selection_evidence"
        ]
        self.assertEqual(len(selection_messages), 1)
        selection_message = selection_messages[0]
        self.assertEqual(selection_message["selection"]["monitorable_nodes"], [])
        self.assertEqual(
            selection_message["selection"]["mps"]["observed_max_chi"], 1
        )
        self.assertIn("mps_runtime_seconds", selection_message["selection"])
        self.assertIn("mps_peak_rss_bytes", selection_message["selection"])

        timeout = scalability._parent_failure_record(
            attempt,
            bundle,
            run_mode="test",
            shard_index=0,
            shard_count=1,
            status="timeout",
            phase="selection",
            category="wall_clock_timeout",
            message="timed out",
            elapsed=1.0,
            circuit_evidence=evidence_messages[0]["circuit"],
        )
        self.assertEqual(timeout["circuit"]["original_qubits"], 7)
        self.assertEqual(timeout["circuit"]["evidence_stage"], "build_complete")
        self.assertEqual(scalability.validate_record(timeout), [])

        instrumentation_timeout = scalability._parent_failure_record(
            attempt,
            bundle,
            run_mode="test",
            shard_index=0,
            shard_count=1,
            status="timeout",
            phase="instrumentation",
            category="wall_clock_timeout",
            message="timed out during instrumentation",
            elapsed=2.0,
            circuit_evidence=evidence_messages[-1]["circuit"],
            selection_evidence=selection_message["selection"],
            phase_elapsed_seconds=selection_message["phase_elapsed_seconds"],
        )
        self.assertIsNone(instrumentation_timeout["instrumentation"])
        self.assertEqual(
            instrumentation_timeout["selection"], selection_message["selection"]
        )
        self.assertEqual(
            instrumentation_timeout["selection"]["mps"]["observed_max_chi"], 1
        )
        self.assertEqual(scalability.validate_record(instrumentation_timeout), [])
        self.assertEqual(
            scalability.selection_outcome(instrumentation_timeout),
            "selector_complete",
        )
        self.assertEqual(
            scalability.figure_data_rows([instrumentation_timeout])[0][
                "selection_outcome"
            ],
            "selector_complete",
        )

        instrumentation_resource_limit = scalability._parent_failure_record(
            attempt,
            bundle,
            run_mode="test",
            shard_index=0,
            shard_count=1,
            status="resource_limit",
            phase="instrumentation",
            category="worker_killed_resource_limit",
            message="worker was killed during instrumentation",
            elapsed=2.0,
            circuit_evidence=evidence_messages[-1]["circuit"],
            selection_evidence=selection_message["selection"],
            phase_elapsed_seconds=selection_message["phase_elapsed_seconds"],
        )
        self.assertEqual(
            scalability.validate_record(instrumentation_resource_limit), []
        )
        self.assertEqual(
            scalability.selection_outcome(instrumentation_resource_limit),
            "selector_complete",
        )

    def test_mps_backend_failure_taxonomy_distinguishes_numerical_and_oom(self):
        numerical = SimpleNamespace(
            success=False,
            status="ERROR",
            results=[SimpleNamespace(status="Wrong SVD calculations: A != USV*")],
        )
        with self.assertRaises(scalability.MPSBackendNumericalFailure):
            scalability._raise_mps_backend_failure(numerical)

        oom = SimpleNamespace(
            success=False,
            status="ERROR: Insufficient memory for MPS state",
            results=[],
        )
        with self.assertRaises(scalability.MPSBackendResourceLimit):
            scalability._raise_mps_backend_failure(oom)

        other = SimpleNamespace(
            success=False,
            status="ERROR: unsupported backend state",
            results=[],
        )
        with self.assertRaises(scalability.MPSBackendFailure):
            scalability._raise_mps_backend_failure(other)

        circuit = QuantumCircuit(2)
        circuit.h(0)
        for exception, status, category in (
            (
                scalability.MPSBackendNumericalFailure("Wrong SVD calculations"),
                "analysis_error",
                "mps_backend_numerical_failure",
            ),
            (
                scalability.MPSBackendResourceLimit("Insufficient memory"),
                "resource_limit",
                "mps_backend_out_of_memory",
            ),
        ):
            with self.subTest(category=category), patch.object(
                scalability, "_build_circuit", return_value=circuit
            ), patch.object(
                scalability,
                "_select_monitorable_nodes_mps_details",
                side_effect=exception,
            ):
                record = scalability.evaluate_attempt(
                    self.extension_attempt(),
                    self.bundle(),
                    run_mode="test",
                    shard_index=0,
                    shard_count=1,
                )
            self.assertEqual(record["status"], status)
            self.assertEqual(record["error_category"], category)
            self.assertEqual(scalability.validate_record(record), [])

    def test_gurobi_license_failure_is_incomplete_retryable_infrastructure(self):
        fake_gurobi_error = type(
            "GurobiError", (RuntimeError,), {"__module__": "gurobipy"}
        )
        circuit = QuantumCircuit(2)
        circuit.h(0)
        attempt = self.extension_attempt()
        bundle = self.bundle()
        with patch.object(
            scalability, "_build_circuit", return_value=circuit
        ), patch.object(
            scalability,
            "_select_monitorable_nodes_mps_details",
            return_value=({}, minimal_mps_diagnostics(0), {}),
        ), patch.object(
            scalability,
            "build_instrumentation_metrics",
            side_effect=fake_gurobi_error(
                "Too many sessions, 5 active sessions for WLS"
            ),
        ):
            record = scalability.evaluate_attempt(
                attempt,
                bundle,
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )

        self.assertEqual(record["status"], "infrastructure_error")
        self.assertEqual(
            record["error_category"], "gurobi_license_infrastructure_failure"
        )
        self.assertFalse(record["record_complete"])
        self.assertIsNotNone(record["selection"])
        self.assertTrue(
            any(
                "cannot be a final terminal" in error
                for error in scalability.validate_record(record)
            )
        )
        pending = scalability.pending_attempts(
            [attempt],
            {attempt.attempt_id: record},
            expected_fingerprints=scalability._record_fingerprint_hashes(bundle),
        )
        self.assertEqual(pending, [attempt])

    def test_attempt_runner_retries_only_infrastructure_outcomes(self):
        infrastructure = terminal_record("analysis_error")
        infrastructure["status"] = "infrastructure_error"
        success = terminal_record("success")
        with patch.object(
            scalability,
            "_run_attempt_once_with_timeout",
            side_effect=[infrastructure, success],
        ) as run_once, patch.object(scalability.time, "sleep") as sleep:
            result = scalability.run_attempt_with_timeout(
                self.extension_attempt(),
                self.bundle(),
                timeout_seconds=900,
                run_mode="test",
                shard_index=0,
                shard_count=1,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["run"]["infrastructure_retries_before_outcome"], 1)
        self.assertEqual(run_once.call_count, 2)
        sleep.assert_called_once_with(30.0)

    def test_qmax_outcomes_do_not_overwrite_successful_analysis(self):
        too_wide = QuantumCircuit(25)
        too_wide.h(0)
        wide_record = self.evaluate_with_selection(too_wide, {}, {})
        self.assertEqual(wide_record["status"], "success")
        self.assertEqual(
            wide_record["deployment_status"], "source_width_exceeds_qmax"
        )
        self.assertIsNotNone(wide_record["selection"])
        self.assertIsNone(wide_record["error"])
        self.assertEqual(scalability.validate_record(wide_record), [])

        partial = QuantumCircuit(24)
        partial.h(1)
        partial.cx(1, 0)
        partial_record = self.evaluate_with_selection(
            partial,
            {1: [0]},
            {(1, 0): {"a2": 0.0, "determinant": 0.0}},
        )
        self.assertEqual(partial_record["status"], "success")
        self.assertEqual(partial_record["deployment_status"], "partially_deployable")
        self.assertEqual(
            partial_record["instrumentation"][
                "individually_infeasible_gate_node_count"
            ],
            1,
        )
        self.assertEqual(scalability.validate_record(partial_record), [])

        no_candidates = QuantumCircuit(2)
        no_candidates.h(0)
        no_candidate_record = self.evaluate_with_selection(no_candidates, {}, {})
        self.assertEqual(no_candidate_record["deployment_status"], "no_candidates")

        fully_deployable = QuantumCircuit(2)
        fully_deployable.h(0)
        full_record = self.evaluate_with_selection(
            fully_deployable,
            {0: [0]},
            {(0, 0): {"a2": 0.0, "determinant": 0.0}},
        )
        self.assertEqual(full_record["deployment_status"], "fully_deployable")


class BatchFeasibilityTests(unittest.TestCase):
    @staticmethod
    def two_full_capacity_cones() -> tuple[QuantumCircuit, dict[int, list[int]]]:
        circuit = QuantumCircuit(20)
        nodes = {}
        for base in (0, 5):
            circuit.h(base + 4)
            circuit.cx(base + 4, base + 3)
            circuit.cx(base + 3, base + 2)
            circuit.cx(base + 2, base + 1)
            gate = len(circuit.data)
            circuit.cx(base + 1, base)
            nodes[gate] = [base]
            circuit.x(base)  # candidate above is not the free final read
        return circuit, nodes

    def test_actual_partitioner_emits_two_feasible_qmax24_batches(self):
        circuit, nodes = self.two_full_capacity_cones()
        metrics = scalability.build_instrumentation_metrics(circuit, nodes)

        self.assertEqual(metrics["qmax"], 24)
        self.assertEqual(metrics["partition_solver"]["backend"], "cbc")
        self.assertEqual(
            metrics["partition_solver_sha256"],
            scalability.stable_hash(metrics["partition_solver"]),
        )
        self.assertEqual(
            metrics["partition_metadata"]["solver"],
            metrics["partition_solver"],
        )
        self.assertEqual(metrics["candidate_gate_node_count"], 2)
        self.assertEqual(metrics["candidate_monitor_event_count"], 2)
        self.assertEqual(metrics["individually_feasible_gate_node_count"], 2)
        self.assertEqual(metrics["individually_feasible_monitor_event_count"], 2)
        self.assertEqual(metrics["deployed_gate_node_count"], 2)
        self.assertEqual(metrics["deployed_monitor_event_count"], 2)
        self.assertEqual(metrics["individually_infeasible_gate_node_count"], 0)
        self.assertEqual(metrics["infeasible_gate_nodes"], [])
        self.assertEqual(metrics["batch_count"], 2)
        self.assertEqual(metrics["maximum_run_qubits"], 24)
        self.assertEqual([batch["cost"] for batch in metrics["batches"]], [4, 4])
        self.assertEqual(metrics["planned_peak_extra_ancillas"], 4)
        self.assertEqual(
            metrics["planned_cumulative_extra_ancilla_allocations"], 8
        )
        expected_replay = sum(
            row["cone_gate_union_count"] for row in metrics["gate_node_costs"]
        )
        self.assertEqual(metrics["planned_replay_gate_executions"], expected_replay)
        self.assertEqual(
            metrics["planned_original_gate_executions"],
            metrics["batch_count"] * metrics["original_unitary_gate_count_m"],
        )
        self.assertEqual(metrics["planned_monitor_measurement_operations"], 2)
        self.assertEqual(metrics["planned_monitor_reset_operations"], 2)
        self.assertEqual(
            metrics["planned_total_counted_operations"],
            metrics["planned_original_gate_executions"] + expected_replay + 4,
        )
        self.assertTrue(
            all(batch["run_qubits"] <= 24 for batch in metrics["batches"])
        )

        record = terminal_record(
            "success", instrumentation=metrics, original_qubits=20
        )
        self.assertEqual(scalability.validate_record(record), [])
        exported = scalability.figure_data_rows([record])[0]
        for field in (
            "planned_peak_extra_ancillas",
            "planned_cumulative_extra_ancilla_allocations",
            "planned_original_gate_executions",
            "planned_replay_gate_executions",
            "planned_monitor_measurement_operations",
            "planned_monitor_reset_operations",
            "planned_total_counted_operations",
        ):
            self.assertEqual(exported[field], metrics[field])
        self.assertEqual(exported["partition_solver_backend"], "cbc")

    def test_solver_provenance_tampering_is_rejected(self):
        circuit, nodes = self.two_full_capacity_cones()
        record = terminal_record(
            "success",
            instrumentation=scalability.build_instrumentation_metrics(circuit, nodes),
            original_qubits=20,
        )
        recorded_solver = copy.deepcopy(
            record["instrumentation"]["partition_solver"]
        )
        record["instrumentation"]["partition_metadata"]["solver"] = recorded_solver
        record["instrumentation"]["partition_solver"] = {
            **recorded_solver,
            "backend": "gurobi",
        }
        scalability._add_record_checksum(record)

        errors = scalability.validate_record(record)
        self.assertTrue(
            any("solver provenance hash mismatch" in error for error in errors)
        )
        self.assertTrue(
            any("metadata solver differs" in error for error in errors)
        )

    def test_partition_override_rejects_feasible_repartition(self):
        circuit = QuantumCircuit(20)
        nodes = {}
        for control, target in ((1, 0), (3, 2)):
            circuit.h(control)
            gate = len(circuit.data)
            circuit.cx(control, target)
            nodes[gate] = [target]
            circuit.x(target)  # keep the selected event out of the free final read

        metrics = scalability.build_instrumentation_metrics(circuit, nodes)
        self.assertEqual(metrics["batch_count"], 1)
        self.assertEqual(metrics["partition_metadata"]["ffd_batch_count"], 1)

        forged = copy.deepcopy(metrics)
        forged["batches"] = [
            {
                "batch_index": index,
                "nodes": [row["gate"]],
                "cost": row["extra_qubit_demand"],
                "run_qubits": circuit.num_qubits + row["extra_qubit_demand"],
            }
            for index, row in enumerate(metrics["gate_node_costs"])
        ]
        forged["partition_metadata"]["final_batch_count"] = len(
            forged["batches"]
        )

        with self.assertRaisesRegex(
            scalability.ProtocolError, "partition certificate"
        ):
            scalability.build_instrumentation_metrics(
                circuit, nodes, partition_override=forged
            )

    def test_validator_recomputes_batch_feasibility_from_stored_costs(self):
        circuit, nodes = self.two_full_capacity_cones()
        metrics = scalability.build_instrumentation_metrics(circuit, nodes)
        record = terminal_record(
            "success", instrumentation=metrics, original_qubits=20
        )
        record["instrumentation"]["batches"][0]["run_qubits"] = 25

        errors = scalability.validate_record(record)
        self.assertTrue(any("expected 24" in error for error in errors))
        self.assertTrue(any("uses 25>24" in error for error in errors))

        record = terminal_record(
            "success",
            instrumentation=scalability.build_instrumentation_metrics(circuit, nodes),
            original_qubits=20,
        )
        record["instrumentation"]["batches"][0]["cost"] = 3
        record["instrumentation"]["batches"][0]["run_qubits"] = 23
        errors = scalability.validate_record(record)
        self.assertTrue(
            any("expected 4 from gate_node_costs" in error for error in errors)
        )

    def test_partitioner_refuses_any_other_qmax(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        with self.assertRaisesRegex(scalability.ProtocolError, "requires Qmax=24"):
            scalability.build_instrumentation_metrics(circuit, {0: [0]}, qmax=20)

    def test_refinement_and_license_budgets_fit_the_attempt_contract(self):
        with patch.object(scalability.time, "perf_counter", return_value=0.0):
            node_limit, time_limit = scalability._instrumentation_partition_budget(
                attempt_started=0.0,
                attempt_timeout_seconds=scalability.DEFAULT_TIMEOUT_SECONDS,
            )
        self.assertEqual(node_limit, scalability.REFINEMENT_NODE_LIMIT)
        self.assertEqual(
            time_limit, scalability.MAX_REFINEMENT_TIME_LIMIT_SECONDS
        )
        self.assertLessEqual(
            time_limit
            + sum(scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
            + scalability.INSTRUMENTATION_COMPLETION_RESERVE_SECONDS,
            scalability.DEFAULT_TIMEOUT_SECONDS,
        )

        with patch.object(scalability.time, "perf_counter", return_value=850.0):
            node_limit, time_limit = scalability._instrumentation_partition_budget(
                attempt_started=0.0,
                attempt_timeout_seconds=scalability.DEFAULT_TIMEOUT_SECONDS,
            )
        self.assertEqual((node_limit, time_limit), (0, 1.0))

        circuit = QuantumCircuit(1)
        circuit.h(0)
        metrics = scalability.build_instrumentation_metrics(
            circuit,
            {0: [0]},
            refinement_node_limit=0,
            refinement_time_limit_seconds=1.0,
        )
        self.assertEqual(
            metrics["partition_metadata"]["refinement"]["node_limit"], 0
        )
        self.assertFalse(metrics["partition_metadata"]["refinement"]["attempted"])
        self.assertEqual(
            metrics["partition_budget"]["license_retry_backoff_budget_seconds"],
            float(sum(scalability.GUROBI_LICENSE_RETRY_DELAYS_SECONDS)),
        )

    def test_final_read_is_free_only_for_a_terminally_measured_qubit(self):
        unmeasured = QuantumCircuit(1)
        unmeasured.h(0)
        unmeasured_metrics = scalability.build_instrumentation_metrics(
            unmeasured, {0: [0]}
        )
        self.assertEqual(unmeasured_metrics["candidate_monitor_event_count"], 1)
        self.assertEqual(unmeasured_metrics["deployed_monitor_event_count"], 1)
        self.assertEqual(unmeasured_metrics["free_final_read_event_count"], 0)

        measured = QuantumCircuit(1, 1)
        measured.h(0)
        measured.measure(0, 0)
        measured_metrics = scalability.build_instrumentation_metrics(
            measured, {0: [0]}
        )
        self.assertEqual(measured_metrics["candidate_monitor_event_count"], 0)
        self.assertEqual(measured_metrics["deployed_monitor_event_count"], 0)
        self.assertEqual(measured_metrics["free_final_read_event_count"], 1)
        self.assertEqual(measured_metrics["terminally_measured_qubits"], [0])
        self.assertEqual(measured_metrics["batch_count"], 0)
        self.assertEqual(measured_metrics["planned_peak_extra_ancillas"], 0)
        self.assertEqual(
            measured_metrics["planned_cumulative_extra_ancilla_allocations"], 0
        )
        self.assertEqual(measured_metrics["planned_replay_gate_executions"], 0)
        self.assertEqual(measured_metrics["planned_total_counted_operations"], 0)
        self.assertGreater(
            measured_metrics[
                "diagnostic_selected_per_event_cone_gate_occurrences"
            ],
            0,
        )

    def test_same_gate_monitor_events_use_one_union_cone_for_planned_replay(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        metrics = scalability.build_instrumentation_metrics(circuit, {1: [0, 1]})

        self.assertEqual(metrics["candidate_gate_node_count"], 1)
        self.assertEqual(metrics["deployed_gate_node_count"], 1)
        self.assertEqual(metrics["deployed_monitor_event_count"], 2)
        gate_group = metrics["gate_node_costs"][0]
        self.assertEqual(gate_group["monitor_event_count"], 2)
        self.assertEqual(
            metrics["planned_replay_gate_executions"],
            gate_group["cone_gate_union_count"],
        )
        self.assertGreater(
            metrics["diagnostic_candidate_per_event_cone_gate_occurrences"],
            metrics["planned_replay_gate_executions"],
        )
        self.assertEqual(metrics["planned_monitor_measurement_operations"], 2)
        self.assertEqual(metrics["planned_monitor_reset_operations"], 2)
        self.assertEqual(
            metrics["planned_total_counted_operations"],
            metrics["original_unitary_gate_count_m"]
            + metrics["planned_replay_gate_executions"]
            + 4,
        )

    def test_individually_infeasible_group_is_not_counted_as_deployed_work(self):
        circuit = QuantumCircuit(24)
        circuit.h(1)
        circuit.cx(1, 0)

        metrics = scalability.build_instrumentation_metrics(circuit, {1: [0]})

        self.assertEqual(metrics["candidate_gate_node_count"], 1)
        self.assertEqual(metrics["individually_infeasible_gate_node_count"], 1)
        self.assertEqual(metrics["infeasible_gate_nodes"], [1])
        self.assertEqual(metrics["deployed_gate_node_count"], 0)
        self.assertEqual(metrics["batch_count"], 0)
        self.assertEqual(metrics["planned_peak_extra_ancillas"], 0)
        self.assertEqual(
            metrics["planned_cumulative_extra_ancilla_allocations"], 0
        )
        self.assertEqual(metrics["planned_replay_gate_executions"], 0)
        self.assertEqual(metrics["planned_monitor_measurement_operations"], 0)
        self.assertEqual(metrics["planned_monitor_reset_operations"], 0)
        self.assertEqual(metrics["planned_total_counted_operations"], 0)
        self.assertGreater(
            metrics["diagnostic_candidate_per_event_cone_gate_occurrences"], 0
        )

    def test_planned_accounting_tampering_fails_checksum_and_semantics(self):
        circuit, nodes = self.two_full_capacity_cones()
        metrics = scalability.build_instrumentation_metrics(circuit, nodes)
        record = terminal_record(
            "success", instrumentation=metrics, original_qubits=20
        )
        self.assertEqual(scalability.validate_record(record), [])

        record["instrumentation"]["planned_replay_gate_executions"] += 1
        errors = scalability.validate_record(record)
        self.assertTrue(any("integrity checksum mismatch" in error for error in errors))
        self.assertTrue(
            any("planned_replay_gate_executions" in error for error in errors)
        )

        checksummed = terminal_record(
            "success",
            instrumentation=scalability.build_instrumentation_metrics(circuit, nodes),
            original_qubits=20,
        )
        checksummed["instrumentation"]["planned_replay_gate_executions"] += 1
        scalability._add_record_checksum(checksummed)
        errors = scalability.validate_record(checksummed)
        self.assertFalse(any("integrity checksum mismatch" in error for error in errors))
        self.assertTrue(
            any("planned_replay_gate_executions" in error for error in errors)
        )


class EligibilityAndDensityTests(unittest.TestCase):
    def test_terminal_directives_are_excluded_and_original_indices_are_preserved(self):
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.barrier(0)
        circuit.measure(0, 0)

        eligibility = scalability.validate_circuit_eligibility(circuit)
        nodes, diagnostics = scalability.select_monitorable_nodes_mps(circuit)

        self.assertTrue(eligibility["eligible"])
        self.assertEqual(eligibility["unitary_operation_count"], 1)
        self.assertEqual(nodes, {0: [0]})
        self.assertEqual(diagnostics["point_count"], 1)

    def test_rejects_mid_measurement_reset_control_flow_nonunitary_and_three_qubit(self):
        cases = []

        mid_measurement = QuantumCircuit(1, 1)
        mid_measurement.h(0)
        mid_measurement.measure(0, 0)
        mid_measurement.x(0)
        cases.append((mid_measurement, "mid_circuit_measurement"))

        reset = QuantumCircuit(1)
        reset.reset(0)
        cases.append((reset, "reset_operation"))

        control_flow = QuantumCircuit(1, 1)
        with control_flow.if_test((control_flow.clbits[0], True)):
            control_flow.x(0)
        cases.append((control_flow, "control_flow_operation"))

        conditioned = SimpleNamespace(
            data=[
                SimpleNamespace(
                    operation=SimpleNamespace(
                        name="x",
                        condition=("classical-register", 1),
                        condition_bits=(),
                        num_qubits=1,
                    )
                )
            ]
        )
        cases.append((conditioned, "conditioned_operation"))

        nonunitary = QuantumCircuit(1)
        nonunitary.append(Initialize([1, 0]), [0])
        cases.append((nonunitary, "nonunitary_operation"))

        three_qubit = QuantumCircuit(3)
        three_qubit.ccx(0, 1, 2)
        cases.append((three_qubit, "operation_arity_exceeds_two"))

        for circuit, category in cases:
            with self.subTest(category=category):
                with self.assertRaises(scalability.CircuitEligibilityError) as caught:
                    scalability.validate_circuit_eligibility(circuit)
                self.assertEqual(caught.exception.category, category)

    def test_density_validation_rejects_nonphysical_inputs_and_preserves_positive_values(self):
        positive = np.diag([1.0 - 5e-13, 5e-13])
        scores = scalability._validated_density_scores(positive)
        self.assertGreater(scores["determinant"], 0.0)
        self.assertGreater(
            scores["a2"], scalability.SCHMIDT_AMPLITUDE_TOLERANCE
        )

        negative_residue = np.diag([1.0 + 1e-14, -1e-14])
        residue_scores = scalability._validated_density_scores(negative_residue)
        self.assertEqual(residue_scores["a2"], 0.0)
        self.assertEqual(residue_scores["determinant"], 0.0)

        invalid = (
            np.diag([1.0001, -0.0001]),
            np.array([[1.0, 0.1], [0.0, 0.0]], dtype=complex),
            np.array([[np.nan, 0.0], [0.0, np.nan]], dtype=complex),
            np.eye(3, dtype=complex),
            np.diag([0.4, 0.4]),
        )
        for density in invalid:
            with self.subTest(density=repr(density)):
                with self.assertRaises(scalability.AnalysisFailure):
                    scalability._validated_density_scores(density)

    def test_mps_statevector_crosscheck_uses_the_same_a2_rule(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        nodes, _, scores = scalability._select_monitorable_nodes_mps_details(
            circuit
        )

        result = scalability.crosscheck_mps_against_statevector(
            circuit, nodes, scores
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["selection_mismatch_count"], 0)
        self.assertEqual(result["missing_mps_point_count"], 0)
        self.assertEqual(result["missing_statevector_point_count"], 0)
        self.assertGreaterEqual(result["maximum_absolute_a2_difference"], 0.0)

    def test_native_mps_schmidt_extraction_fixes_known_density_eigen_failures(self):
        frame = scalability.resolve_canonical_frame()
        attempts = {
            attempt.basename: attempt
            for attempt in scalability.build_attempt_grid(frame)[
                : scalability.EXPECTED_BASE_COUNT
            ]
        }
        known_failures = [
            "ae_indep_qiskit_12",
            "bmw_quark_cardinality_indep_qiskit_10",
            "bmw_quark_cardinality_indep_qiskit_11",
            "bmw_quark_cardinality_indep_qiskit_12",
            "bmw_quark_cardinality_indep_qiskit_13",
            "bmw_quark_cardinality_indep_qiskit_14",
            "bmw_quark_cardinality_indep_qiskit_2",
            "bmw_quark_cardinality_indep_qiskit_3",
            "bmw_quark_cardinality_indep_qiskit_4",
        ]
        for basename in known_failures:
            with self.subTest(basename=basename):
                circuit = scalability._build_circuit(attempts[basename])
                nodes, _, scores = (
                    scalability._select_monitorable_nodes_mps_details(circuit)
                )
                result = scalability.crosscheck_mps_against_statevector(
                    circuit, nodes, scores
                )
                self.assertTrue(result["passed"])
                self.assertEqual(result["selection_mismatch_count"], 0)
                self.assertEqual(
                    result["mps_selected_events_sha256"],
                    result["statevector_selected_events_sha256"],
                )
                self.assertGreaterEqual(
                    result["maximum_absolute_a2_difference"], 0.0
                )

    def test_native_mps_snapshot_rejects_negative_schmidt_bonds(self):
        tensors = [
            np.ones((2, 1, 1), dtype=complex) / np.sqrt(2.0),
            np.ones((2, 1, 1), dtype=complex) / np.sqrt(2.0),
        ]
        with self.assertRaises(scalability.AnalysisFailure):
            scalability._mps_snapshot_single_qubit_scores(
                (tensors, [np.array([-1.0])]), 0, 2
            )

    def test_lapack_mps_regression_for_qwalk_selection_floor(self):
        frame = scalability.resolve_canonical_frame()
        attempt = next(
            attempt
            for attempt in scalability.build_attempt_grid(frame)[
                : scalability.EXPECTED_BASE_COUNT
            ]
            if attempt.basename == "qwalk-noancilla_indep_qiskit_5"
        )
        circuit = scalability._build_circuit(attempt)
        nodes, diagnostics, scores = (
            scalability._select_monitorable_nodes_mps_details(circuit)
        )
        result = scalability.crosscheck_mps_against_statevector(
            circuit, nodes, scores
        )

        self.assertIs(diagnostics["configured_mps_lapack"], True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["selection_mismatch_count"], 0)
        self.assertEqual(result["mps_selected_event_count"], 181)
        self.assertEqual(result["statevector_selected_event_count"], 181)

    def test_determinant_is_diagnostic_and_does_not_select_a_weakly_entangled_point(self):
        circuit = QuantumCircuit(2)
        circuit.ry(1e-6, 0)
        circuit.cx(0, 1)

        nodes, scores = scalability.select_monitorable_nodes_statevector(circuit)
        point = scores[(1, 0)]
        self.assertLessEqual(point["determinant"], 1e-12)
        self.assertGreater(point["a2"], scalability.SCHMIDT_AMPLITUDE_TOLERANCE)
        self.assertNotIn(0, nodes.get(1, []))

    def test_repository_frame_has_exact_a2_event_set_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "criterion-comparison.jsonl"
            arguments = SimpleNamespace(
                manifest=scalability.DEFAULT_MANIFEST,
                qasm_dir=scalability.DEFAULT_QASM_DIR,
                output=output,
                overwrite=False,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    scalability.compare_base_criteria_protocol(arguments), 0
                )
            records = scalability.read_jsonl(output)

        summary = scalability.criterion_comparison_summary(records)
        expected_summary = scalability.BASE_CRITERION_REFERENCE
        self.assertEqual(expected_summary["circuit_count"], 310)
        for field in (
            "circuit_count",
            "total_acted_points",
            "v2_selected_event_count",
            "reference_selected_event_count",
            "selection_mismatch_count",
            "v2_only_count",
            "reference_only_count",
            "circuits_with_mismatches",
        ):
            self.assertEqual(summary[field], expected_summary[field])
        self.assertEqual(expected_summary["selector"], "second_schmidt_amplitude")
        self.assertEqual(expected_summary["schmidt_amplitude_tolerance"], 1e-12)
        self.assertTrue(
            scalability._is_sha256(expected_summary["scientific_records_sha256"])
        )
        self.assertEqual(
            scalability.criterion_comparison_scientific_sha256(records),
            expected_summary["scientific_records_sha256"],
        )

        platform_diagnostics_changed = copy.deepcopy(records)
        platform_diagnostics_changed[0]["elapsed_seconds"] += 1000.0
        platform_diagnostics_changed[0]["maximum_a2"] += 0.125
        platform_diagnostics_changed[0]["minimum_determinant"] -= 0.125
        self.assertEqual(
            scalability.criterion_comparison_scientific_sha256(
                platform_diagnostics_changed
            ),
            expected_summary["scientific_records_sha256"],
        )

        event_identity_changed = copy.deepcopy(records)
        event_identity_changed[0]["v2_selected_events"][0][0] += 1
        self.assertNotEqual(
            scalability.criterion_comparison_scientific_sha256(event_identity_changed),
            expected_summary["scientific_records_sha256"],
        )

    def test_repository_frame_has_native_mps_statevector_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mps-comparison.jsonl"
            arguments = SimpleNamespace(
                manifest=scalability.DEFAULT_MANIFEST,
                qasm_dir=scalability.DEFAULT_QASM_DIR,
                output=output,
                overwrite=False,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    scalability.compare_base_mps_protocol(arguments), 0
                )
            records = scalability.read_jsonl(output)

        summary = scalability.mps_crosscheck_summary(records)
        expected_summary = scalability.BASE_MPS_CROSSCHECK_REFERENCE
        for field in (
            "circuit_count",
            "point_count_mps",
            "point_count_statevector",
            "mps_selected_event_count",
            "statevector_selected_event_count",
            "selection_mismatch_count",
            "missing_mps_point_count",
            "missing_statevector_point_count",
            "circuits_with_selection_mismatches",
            "maximum_observed_chi",
        ):
            self.assertEqual(summary[field], expected_summary[field])
        self.assertEqual(
            scalability.mps_crosscheck_scientific_sha256(records),
            expected_summary["scientific_records_sha256"],
        )

    def test_mps_event_certificate_binds_selection_configuration(self):
        record = {
            "attempt_id": "base:synthetic",
            "attempt_index": 0,
            "basename": "synthetic",
            "qasm_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "corpus_sha256": "c" * 64,
            "selector": "second_schmidt_amplitude",
            "schmidt_amplitude_tolerance": 1e-12,
            "mps_a2_extraction": "aer_mps_local_tensor_svd",
            "mps_lapack": True,
            "mps_max_bond_dimension": None,
            "mps_truncation_threshold": 1e-16,
            "operational_eligibility_category": None,
            "point_count_mps": 1,
            "point_count_statevector": 1,
            "mps_selected_events": [[0, 0]],
            "statevector_selected_events": [[0, 0]],
            "selection_mismatch_count": 0,
            "selection_mismatch_points": [],
            "missing_mps_point_count": 0,
            "missing_statevector_point_count": 0,
        }
        baseline = scalability.mps_crosscheck_scientific_sha256([record])
        for field, value in (
            ("mps_lapack", False),
            ("mps_max_bond_dimension", 64),
            ("mps_truncation_threshold", 1e-12),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(record)
                mutated[field] = value
                self.assertNotEqual(
                    scalability.mps_crosscheck_scientific_sha256([mutated]),
                    baseline,
                )

        selection_rule = scalability._protocol_config()[
            "base_statevector_crosscheck"
        ]["selection_rule"]
        self.assertIn("direct SVD", selection_rule)
        self.assertNotIn("lambda_min", selection_rule)


class DeepValidationCertificateTests(unittest.TestCase):
    def test_one_deep_run_certificate_is_reused_without_science_recomputation(self):
        attempt = scalability.AttemptSpec(
            "base:synthetic", 0, "canonical_qasm", "synthetic"
        )
        record = bind_record(terminal_record("timeout"), attempt)
        expected_attempts = {attempt.attempt_id: attempt}

        with patch.object(
            scalability, "_recompute_scientific_errors", return_value=[]
        ) as recompute:
            certificate = scalability.create_deep_validation_certificate(
                [record],
                expected_attempts=expected_attempts,
                expected_fingerprints=fingerprints(),
                require_complete_grid=True,
                expected_run_mode="test",
            )
        self.assertEqual(recompute.call_count, 1)
        self.assertTrue(certificate["recompute_science_performed"])
        self.assertTrue(scalability._is_sha256(certificate["source_checksums"]["sha256"]))
        self.assertTrue(scalability._is_sha256(certificate["result_checksums"]["sha256"]))

        with patch.object(
            scalability,
            "_recompute_scientific_errors",
            side_effect=AssertionError("certificate verification reran science"),
        ) as recompute:
            summary = scalability.verify_deep_validation_certificate(
                certificate,
                [record],
                expected_attempts=expected_attempts,
                expected_fingerprints=fingerprints(),
                require_complete_grid=True,
                expected_run_mode="test",
            )
        self.assertEqual(summary, {"timeout": 1})
        recompute.assert_not_called()

        changed = copy.deepcopy(record)
        changed["elapsed_seconds"] = 2.0
        scalability._add_record_checksum(changed)
        with self.assertRaisesRegex(
            scalability.ProtocolError, "artifact checksum mismatch"
        ):
            scalability.verify_deep_validation_certificate(
                certificate,
                [changed],
                expected_attempts=expected_attempts,
                expected_fingerprints=fingerprints(),
                require_complete_grid=True,
                expected_run_mode="test",
            )

        changed_certificate = copy.deepcopy(certificate)
        changed_certificate["source_checksums"]["attempt_count"] = 2
        changed_certificate["certificate_sha256"] = (
            scalability._deep_validation_certificate_sha256(changed_certificate)
        )
        with self.assertRaisesRegex(
            scalability.ProtocolError, "source checksum mismatch"
        ):
            scalability.verify_deep_validation_certificate(
                changed_certificate,
                [record],
                expected_attempts=expected_attempts,
                expected_fingerprints=fingerprints(),
                require_complete_grid=True,
                expected_run_mode="test",
            )

    def test_finalization_cli_requires_deep_output_or_existing_certificate(self):
        parser = scalability.build_parser()
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["validate", "artifact.jsonl"])
        parsed = parser.parse_args(
            [
                "validate",
                "artifact.jsonl",
                "--complete-grid",
                "--deep-certificate-out",
                "certificate.json",
            ]
        )
        self.assertEqual(parsed.deep_certificate_out, Path("certificate.json"))


class MpsAndCheckpointTests(unittest.TestCase):
    def test_mps_snapshots_report_observed_maximum_chi(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        nodes, diagnostics = scalability.select_monitorable_nodes_mps(circuit)

        self.assertEqual(nodes, {0: [0]})
        self.assertIsNone(diagnostics["configured_max_bond_dimension"])
        self.assertEqual(
            diagnostics["configured_max_bond_dimension_mode"], "unbounded"
        )
        self.assertEqual(diagnostics["observed_max_chi"], 2)
        self.assertTrue(diagnostics["observed_max_chi_available"])
        self.assertIn(
            "save_matrix_product_state",
            diagnostics["observed_max_chi_method"],
        )
        self.assertIsNone(diagnostics["observed_max_chi_unavailable_reason"])

    def test_atomic_jsonl_round_trip(self):
        records = [
            terminal_record("success", attempt_id="base:a"),
            terminal_record("timeout", attempt_id="base:b"),
        ]
        records[1]["attempt_index"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "records.jsonl"
            scalability.atomic_write_jsonl(records, path)
            self.assertEqual(scalability.read_jsonl(path), records)
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp.*")), [])

    def test_figure_rows_retain_failures_and_explicit_denominators(self):
        success = terminal_record("success", attempt_id="base:a")
        success["selection"]["mps_runtime_seconds"] = 1.25
        success["selection"]["mps_peak_rss_bytes"] = 2048
        success["selection"]["validation_runtime_seconds"] = 9.5
        success["selection"]["total_selection_certificate_runtime_seconds"] = 10.75
        success["selection"]["total_selection_certificate_peak_rss_bytes"] = 2048
        success["phase_elapsed_seconds"]["selection"] = 10.75
        scalability._add_record_checksum(success)
        timeout = terminal_record("timeout", attempt_id="base:b")
        timeout["attempt_index"] = 1
        scalability._add_record_checksum(timeout)
        unsupported = terminal_record("unsupported", attempt_id="base:c")
        unsupported["attempt_index"] = 2
        scalability._add_record_checksum(unsupported)

        rows = scalability.figure_data_rows([success, timeout, unsupported])

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["attempted"] for row in rows], [1, 1, 1])
        self.assertEqual([row["fingerprinted"] for row in rows], [1, 0, 0])
        self.assertEqual(
            [row["analysis_status"] for row in rows],
            ["success", "timeout", "unsupported"],
        )
        self.assertEqual(
            [row["selection_outcome"] for row in rows],
            ["selector_complete", "timeout", "unsupported"],
        )
        self.assertEqual(
            [row["deployment_status"] for row in rows],
            ["no_candidates", "not_evaluated", "not_evaluated"],
        )
        self.assertTrue(
            all(set(row) == set(scalability.FIGURE_DATA_FIELDS) for row in rows)
        )
        self.assertTrue(all(row["attempt_denominator"] == 3 for row in rows))
        self.assertTrue(
            all(row["fingerprinted_attempt_denominator"] == 1 for row in rows)
        )
        self.assertTrue(
            all(row["hash_distinct_circuit_denominator"] == 1 for row in rows)
        )
        for status in scalability.STATUS_ORDER:
            expected = 1 if status in {"success", "timeout", "unsupported"} else 0
            field = f"analysis_status_{status}_attempt_denominator"
            self.assertTrue(all(row[field] == expected for row in rows))
        for outcome in scalability.SELECTION_OUTCOME_ORDER:
            expected = 1 if outcome in {
                "selector_complete",
                "timeout",
                "unsupported",
            } else 0
            field = f"selection_outcome_{outcome}_attempt_denominator"
            self.assertTrue(all(row[field] == expected for row in rows))
        for status in scalability.DEPLOYMENT_STATUS_ORDER:
            expected = 1 if status == "no_candidates" else 2 if status == "not_evaluated" else 0
            field = f"deployment_status_{status}_attempt_denominator"
            self.assertTrue(all(row[field] == expected for row in rows))
        self.assertEqual(rows[0]["candidate_gate_node_count"], 0)
        self.assertEqual(rows[0]["individually_feasible_gate_node_count"], 0)
        self.assertEqual(rows[0]["deployed_gate_node_count"], 0)
        self.assertEqual(rows[0]["mps_runtime_seconds"], 1.25)
        self.assertEqual(rows[0]["mps_peak_rss_bytes"], 2048)
        self.assertEqual(rows[0]["observed_max_chi"], 1)
        self.assertEqual(rows[0]["validation_runtime_seconds"], 9.5)
        self.assertNotIn("total_selection_certificate_runtime_seconds", rows[0])
        self.assertEqual(rows[1]["selection_available"], 0)
        self.assertEqual(rows[1]["error_phase"], "selection")
        self.assertEqual(rows[2]["selection_available"], 0)
        self.assertEqual(rows[2]["analysis_status"], "unsupported")
        self.assertEqual(rows[2]["error_phase"], "build")
        self.assertIsNone(rows[2]["circuit_sha256"])

        with self.assertRaisesRegex(scalability.ProtocolError, "missing"):
            first = scalability.AttemptSpec("base:a", 0, "canonical_qasm", "a")
            second = scalability.AttemptSpec("base:b", 1, "canonical_qasm", "b")
            bind_record(success, first)
            scalability.validate_record_set(
                [success],
                expected_attempts={"base:a": first, "base:b": second},
                require_complete_grid=True,
            )

    def test_exact_hash_clusters_report_attempt_and_hash_distinct_denominators(self):
        first = terminal_record("success", attempt_id="extension:one:q18")
        second = terminal_record("success", attempt_id="extension:two:q18")
        second["attempt_index"] = 1
        scalability._add_record_checksum(second)

        denominators = scalability.artifact_denominators([first, second])
        rows = scalability.figure_data_rows([first, second])

        self.assertEqual(denominators["attempts"], 2)
        self.assertEqual(denominators["fingerprinted_attempts"], 2)
        self.assertEqual(denominators["hash_distinct_circuits"], 1)
        self.assertEqual(denominators["analysis_status_success_attempts"], 2)
        self.assertEqual(denominators["analysis_status_unsupported_attempts"], 0)
        self.assertEqual(
            denominators["selection_outcome_selector_complete_attempts"], 2
        )
        self.assertEqual(
            denominators["deployment_status_no_candidates_attempts"], 2
        )
        self.assertEqual(denominators["exact_duplicate_clusters"], 1)
        self.assertEqual(denominators["duplicate_attempt_excess"], 1)
        self.assertEqual([row["duplicate_cluster_size"] for row in rows], [2, 2])
        self.assertEqual(
            [row["hash_distinct_circuit_representative"] for row in rows],
            [1, 0],
        )
        self.assertEqual(
            [row["inverse_exact_hash_cluster_weight"] for row in rows],
            [0.5, 0.5],
        )


if __name__ == "__main__":
    unittest.main()
