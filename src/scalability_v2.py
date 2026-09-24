"""Scalability protocol for Appendix C.

This module deliberately does not reuse ``scalability_study.py`` artifacts.
It defines a new, exhaustive attempt frame, records every terminal outcome,
and treats Qmax=24 as an invariant of the operational experiment.

Run from the project root:

    PYTHONPATH=src python src/scalability_v2.py run --smoke \
        --output results/scalability_v2_smoke.jsonl
    PYTHONPATH=src python src/scalability_v2.py run --shard-index 0 --shard-count 16 \
        --timeout 900 --output results/scalability_v2_shards/part-0.jsonl
    PYTHONPATH=src python src/scalability_v2.py validate \
        results/scalability_v2_shards/part-*.jsonl \
        --complete-grid --deep-certificate-out results/scalability_v2.deep.json
    PYTHONPATH=src python src/scalability_v2.py merge \
        results/scalability_v2_shards/part-*.jsonl \
        --certificate results/scalability_v2.deep.json \
        --output results/scalability_v2.jsonl

JSONL is rewritten through fsync + os.replace after every attempt. Resume
retains every valid complete terminal outcome and runs only missing or
explicitly incomplete attempts. Final runs never retry terminal records.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_partition import (
    partition_nodes,
    solver_provenance,
    validate_partition_certificate,
)
from fast_analysis import SKIP_OPS
from mps_analysis import causal_cone


SCHEMA_VERSION = "qmon.scalability.v2.record/8"
DEEP_VALIDATION_CERTIFICATE_SCHEMA = (
    "qmon.scalability.v2.deep-validation-certificate/1"
)
EXPECTED_BASE_COUNT = 310
EXPECTED_MANIFEST_SHA256 = (
    "fae150dbacc70b25efbc14c719b04e6e20d8e306d2fb703927f000b4bc6014c1"
)
EXPECTED_CORPUS_SHA256 = (
    "7347d0f3b43b2d6a28ae846d48ff5e677bf88af624261a7d505bca3eb804a5f4"
)
CANONICAL_QMAX = 24
SCHMIDT_AMPLITUDE_TOLERANCE = 1e-12
DENSITY_MATRIX_TOLERANCE = 1e-12
MPS_MAX_BOND_DIMENSION = None
MPS_TRUNCATION_THRESHOLD = 1e-16
MPS_LAPACK = True
TRANSPILER_SEED = 2025
DEFAULT_TIMEOUT_SECONDS = 900.0
MAX_REFINEMENT_TIME_LIMIT_SECONDS = 120.0
REFINEMENT_NODE_LIMIT = 120
INSTRUMENTATION_COMPLETION_RESERVE_SECONDS = 30.0
GUROBI_LICENSE_RETRY_DELAYS_SECONDS = (5, 10, 20, 40)
IMMEDIATE_INFRASTRUCTURE_RETRY_DELAYS_SECONDS = (30.0,)

STATUS_ORDER = (
    "success",
    "timeout",
    "build_error",
    "analysis_error",
    "resource_limit",
    "unsupported",
)
STATUSES = frozenset(STATUS_ORDER)
RETRYABLE_STATUSES = frozenset({"infrastructure_error"})
OUTCOME_STATUSES = STATUSES | RETRYABLE_STATUSES

SELECTION_OUTCOME_ORDER = (
    "selector_complete",
    "unsupported",
    "timeout",
    "resource_limit",
    "analysis_error",
    "build_error",
)
SELECTION_OUTCOMES = frozenset(SELECTION_OUTCOME_ORDER)

DEPLOYMENT_STATUS_ORDER = (
    "fully_deployable",
    "partially_deployable",
    "source_width_exceeds_qmax",
    "no_candidates",
    "not_evaluated",
)
DEPLOYMENT_STATUSES = frozenset(DEPLOYMENT_STATUS_ORDER)
PUBLICATION_PARTITION_SOLVER_BACKEND = "gurobi"

INSTRUMENTATION_OPERATION_ACCOUNTING = {
    "scope": "deployed gate groups in stored batches only",
    "grouped_replay": (
        "for each deployed source-gate group, count its causal-cone gate union "
        "once across all qubits monitored at that gate"
    ),
    "original_gate_executions": (
        "B*m, where B is stored batch_count and m is the eligible original "
        "unitary-gate count excluding terminal measurements and barriers"
    ),
    "monitor_operations": (
        "one measurement and one reset per deployed gate-qubit monitor event"
    ),
    "excluded_operations": "terminal standard reads and barriers",
    "total_formula": (
        "B*m + planned_replay_gate_executions + "
        "planned_monitor_measurement_operations + "
        "planned_monitor_reset_operations"
    ),
}

# This is the complete public MQT Bench 2.2.2 generator catalog, declared in
# source rather than discovered at runtime. No family is removed because a
# previous run was slow, failed, or produced no successful data.
EXTENSION_FAMILIES = (
    "ae",
    "bmw_quark_cardinality",
    "bmw_quark_copula",
    "bv",
    "cdkm_ripple_carry_adder",
    "dj",
    "draper_qft_adder",
    "full_adder",
    "ghz",
    "ghz_dynamic",
    "graphstate",
    "grover",
    "half_adder",
    "hhl",
    "hrs_cumulative_multiplier",
    "modular_adder",
    "multiplier",
    "qaoa",
    "qft",
    "qftentangled",
    "qnn",
    "qpeexact",
    "qpeinexact",
    "qwalk",
    "randomcircuit",
    "rg_qft_multiplier",
    "seven_qubit_steane_code",
    "shor",
    "shors_nine_qubit_code",
    "vbe_ripple_carry_adder",
    "vqe_real_amp",
    "vqe_su2",
    "vqe_two_local",
    "wstate",
)
EXTENSION_SIZES = (15, 18, 20, 24, 30, 40, 50)

# MQT Bench uses deterministic internal seeds for most stochastic generators.
# These two expose their seed and are pinned explicitly.
GENERATOR_KWARGS = {
    "graphstate": {"seed": TRANSPILER_SEED},
    "qaoa": {"seed": TRANSPILER_SEED},
}

# Reference from the first of two fresh-process q15 checks on the fixed
# environment. Tests rebuild all 34 families and compare a second pass against
# these hashes/statuses. This checks q15 only, not all 238 extension attempts.
Q15_GENERATOR_REFERENCE = {
    "ae": {"status": "eligible", "sha256": "d6ae9d679c3c8ecfbfcc4f9a9c5b45911da005cc16ba0a3855a2120453a46ad4"},
    "bmw_quark_cardinality": {"status": "unsupported", "category": "nonterminal_barrier"},
    "bmw_quark_copula": {"status": "unsupported", "category": "generator_rejected"},
    "bv": {"status": "eligible", "sha256": "40b9980629cd44e26d33637943609e2dc454f488faa28431d03b293a4b044597"},
    "cdkm_ripple_carry_adder": {"status": "unsupported", "category": "generator_rejected"},
    "dj": {"status": "eligible", "sha256": "c8215547856dd0f086f461e28e9e3757f9d5b8d1084495273a63c1ea936827ac"},
    "draper_qft_adder": {"status": "unsupported", "category": "generator_rejected"},
    "full_adder": {"status": "unsupported", "category": "generator_rejected"},
    "ghz": {"status": "eligible", "sha256": "2a8fbf1564f17a81ba4d8a772cec977658194677007fc66e26a476a53136fa01"},
    "ghz_dynamic": {"status": "unsupported", "category": "control_flow_operation"},
    "graphstate": {"status": "eligible", "sha256": "329ec5b909e5c6538b47ad8f08125ff7e28b15c9f68abe93eee4c97b0fa59ee4"},
    "grover": {"status": "eligible", "sha256": "537d82f6ca5ccbb8ec22b7ff4ad260c0c44bb48c7df716708002b82aefa505c5"},
    "half_adder": {"status": "eligible", "sha256": "ba94044d1f394d292c79611a0f957e484dbb232c4a74bac90406a1d61c0449be"},
    "hhl": {"status": "eligible", "sha256": "cca31de25b2a75c3f12a51bdbc6d4a4412c7c8a82394477d16fa56c5b8e47b98"},
    "hrs_cumulative_multiplier": {"status": "unsupported", "category": "generator_rejected"},
    "modular_adder": {"status": "unsupported", "category": "generator_rejected"},
    "multiplier": {"status": "unsupported", "category": "generator_rejected"},
    "qaoa": {"status": "eligible", "sha256": "4d8e1e18a2c06d8891ff02f482125f652dc9b6caef2260c3b7c3f886bf07ce00"},
    "qft": {"status": "eligible", "sha256": "f348cecf32f702470560832b2b69937d96a5e0d8c065114d547dfdff94f2bfe3"},
    "qftentangled": {"status": "eligible", "sha256": "b4416a2799e370b040066915820e2ebfd8f6fd577df003072903c6bd794a5daf"},
    "qnn": {"status": "eligible", "sha256": "c103cf42dc0d2ee5d50b9113136d69a665bfcb675d00c77a8ae45f95b0701284"},
    "qpeexact": {"status": "eligible", "sha256": "0d3cd7418ada969236f60055330edb354166949a1946a3a436064103ad926e46"},
    "qpeinexact": {"status": "eligible", "sha256": "4ef70634bcf7ff8fe378ec4f556a7a17a9c689e21eb80a0088186f02e9ace8f7"},
    "qwalk": {"status": "eligible", "sha256": "3fbd3cb23aea9aa35115fa3ac172d4992893775b23884d1987935f65abb0a695"},
    "randomcircuit": {"status": "eligible", "sha256": "fbe544cca9d60bc35ba4413a109e61c80a35bb9c152588f26705e491184d972e"},
    "rg_qft_multiplier": {"status": "unsupported", "category": "generator_rejected"},
    "seven_qubit_steane_code": {"status": "unsupported", "category": "generator_rejected"},
    "shor": {"status": "unsupported", "category": "generator_rejected"},
    "shors_nine_qubit_code": {"status": "unsupported", "category": "generator_rejected"},
    "vbe_ripple_carry_adder": {"status": "unsupported", "category": "generator_rejected"},
    "vqe_real_amp": {"status": "eligible", "sha256": "de2fd557937b621fa3dc89d906f7dd7e46ed51cfac9d5656d278b1847d2a245b"},
    "vqe_su2": {"status": "eligible", "sha256": "376e785caf16f402c335671ab651eb41a175852959d914b02a920442bc964889"},
    "vqe_two_local": {"status": "eligible", "sha256": "77fa525511b222e17a5019b2821f925a3860c4a7bca68fa971c017ccfbea3ec9"},
    "wstate": {"status": "eligible", "sha256": "de2a79b2bb59d80cd565301b4950c1b4863161014b4f79c266c16072eb79b22b"},
}

KNOWN_DUPLICATE_EXTENSION_HASHES = {
    18: "a544e16a10d136b01e8eddb1723d9f1c648a0c07f6ee526ecc7d3fc5a7aca8c4",
    20: "4c229bf80fc828425bc7c141e8b11ab1ba7f55aec2bab1bf0ae5184795d2ceea",
    24: "6147ff208c76b1b896b8bc511d58a1b123bb626808b05fc9f237a6bab893f9a5",
    30: "ccbb8a806d726c4c45596378f42c107decf02f45bfe1f5e0a87dc06d7678f8f3",
    40: "2854449d16dcdd13fc9c76d1f19f941a7461c591bc7d1f5f051c86bbf992a138",
    50: "be6290c052723d74c2a5ff786665e65534d7b76e8700e6d3a4da9d80b224cc56",
}

BASE_CRITERION_REFERENCE = {
    "corpus_sha256": EXPECTED_CORPUS_SHA256,
    "circuit_count": 310,
    "total_acted_points": 56386,
    "selector": "second_schmidt_amplitude",
    "schmidt_amplitude_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
    "reference": "fast_analysis.analyze_per_gate_nolock",
    "v2_selected_event_count": 22332,
    "reference_selected_event_count": 22332,
    "selection_mismatch_count": 0,
    "v2_only_count": 0,
    "reference_only_count": 0,
    "circuits_with_mismatches": 0,
    "operationally_eligible_circuits": 290,
    "nonterminal_barrier_circuits": 20,
    "scientific_records_sha256": (
        "17a9c948ca5648ee47063ada6adc44e154575a254c8e2f03252742f9338ac749"
    ),
}

# Only fixed corpus/config labels and discrete event identities/counts belong
# in the criterion science hash. Runtime and floating diagnostics intentionally
# remain in the diagnostic output but cannot perturb this hash across platforms.
CRITERION_SCIENTIFIC_FIELDS = (
    "schema_version",
    "attempt_id",
    "attempt_index",
    "basename",
    "qasm_sha256",
    "manifest_sha256",
    "corpus_sha256",
    "selector",
    "schmidt_amplitude_tolerance",
    "reference",
    "operational_eligibility_category",
    "total_acted_points",
    "v2_selected_event_count",
    "reference_selected_event_count",
    "selection_mismatch_count",
    "v2_only_count",
    "reference_only_count",
    "v2_selected_events",
    "reference_selected_events",
    "v2_selected_events_sha256",
    "reference_selected_events_sha256",
    "mismatch_examples",
)

# Reference counts for the native-MPS/statevector comparison.
BASE_MPS_CROSSCHECK_REFERENCE = {
    "corpus_sha256": EXPECTED_CORPUS_SHA256,
    "circuit_count": 310,
    "point_count_mps": 56386,
    "point_count_statevector": 56386,
    "selector": "second_schmidt_amplitude",
    "schmidt_amplitude_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
    "mps_a2_extraction": "aer_mps_local_tensor_svd",
    "mps_lapack": MPS_LAPACK,
    "mps_selected_event_count": 22332,
    "statevector_selected_event_count": 22332,
    "selection_mismatch_count": 0,
    "missing_mps_point_count": 0,
    "missing_statevector_point_count": 0,
    "circuits_with_selection_mismatches": 0,
    "maximum_observed_chi": 128,
    "scientific_records_sha256": (
        "a3cc316fa09906b71c5e2fbdc1ce2102b6948ba26e9b76447027a6de71545ca5"
    ),
}

TRANSPILE_BASIS = (
    "u",
    "p",
    "cx",
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "sx",
    "sxdg",
    "cz",
    "cy",
    "swap",
    "ch",
    "cp",
    "cu",
    "crx",
    "cry",
    "crz",
    "rxx",
    "rzz",
)

DEFAULT_MANIFEST = PROJECT_ROOT / "inputs" / "certified_circuits.txt"
DEFAULT_QASM_DIR = PROJECT_ROOT / "quantum_circuits"
SOURCE_FILES = (
    SCRIPT_DIR / "scalability_v2.py",
    SCRIPT_DIR / "batch_partition.py",
    SCRIPT_DIR / "mps_analysis.py",
    SCRIPT_DIR / "fast_analysis.py",
)

SMOKE_ATTEMPT_IDS = (
    "base:bv_indep_qiskit_3",
    "base:ghz_indep_qiskit_3",
    "extension:ghz:q15",
)


class ProtocolError(ValueError):
    """The declared experiment frame or an artifact violates the protocol."""


class UnsupportedAttempt(RuntimeError):
    """A declared extension combination is rejected by its generator."""


class CircuitEligibilityError(UnsupportedAttempt):
    """The generated circuit falls outside the closed-unitary theorem."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class AnalysisFailure(RuntimeError):
    """A deterministic analysis step failed."""


class MPSBackendFailure(AnalysisFailure):
    """Aer returned a non-success result without a narrower classification."""


class MPSBackendNumericalFailure(MPSBackendFailure):
    """Aer reported a numerical-linear-algebra failure."""


class MPSBackendResourceLimit(RuntimeError):
    """Aer reported that the requested MPS execution exhausted memory."""


ELIGIBILITY_ERROR_CATEGORIES = frozenset(
    {
        "control_flow_operation",
        "conditioned_operation",
        "reset_operation",
        "mid_circuit_measurement",
        "nonterminal_barrier",
        "operation_arity_exceeds_two",
        "operation_arity_not_one_or_two",
        "nonunitary_operation",
    }
)

RUNNING_PHASES = frozenset(
    {
        "worker_start",
        "build",
        "eligibility",
        "selection",
        "selection_complete",
        "instrumentation",
    }
)

STATUS_PHASE_CATEGORIES = {
    "build_error": {"build": frozenset({"circuit_build_failed"})},
    "unsupported": {
        "build": frozenset({"generator_rejected"}),
        "eligibility": ELIGIBILITY_ERROR_CATEGORIES,
    },
    "analysis_error": {
        "eligibility": frozenset({"eligibility_or_fingerprint_failed"}),
        "selection": frozenset(
            {
                "mps_backend_numerical_failure",
                "mps_backend_failure",
                "mps_native_schmidt_validation_failure",
                "statevector_crosscheck_failure",
                "mps_analysis_failed",
                "statevector_crosscheck_mismatch",
            }
        ),
        "instrumentation": frozenset({"instrumentation_analysis_failed"}),
        "worker_boundary": frozenset({"uncaught_worker_exception"}),
    },
    "resource_limit": {
        "build": frozenset({"memory_error"}),
        "eligibility": frozenset({"memory_error"}),
        "selection": frozenset({"memory_error", "mps_backend_out_of_memory"}),
        "instrumentation": frozenset({"memory_error"}),
    },
    "infrastructure_error": {
        "instrumentation": frozenset(
            {"gurobi_license_infrastructure_failure"}
        )
    },
}

MPS_OUT_OF_MEMORY_MARKERS = (
    "insufficient memory",
    "out of memory",
    "cannot allocate memory",
    "memory allocation",
    "std::bad_alloc",
    "bad_alloc",
)
MPS_NUMERICAL_FAILURE_MARKERS = (
    "wrong svd calculations",
    "svd calculation",
    "lapack",
    "singular value decomposition",
    "numerical failure",
    "non-finite",
)
GUROBI_INFRASTRUCTURE_MARKERS = (
    "too many sessions",
    "active sessions",
    "web license service",
    "wls",
    "license server",
    "token server",
    "failed to acquire a license",
    "unable to acquire a license",
    "license checkout",
    "license expired",
    "license not found",
    "no gurobi license",
    "gurobi license",
    "token.gurobi.com",
)


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: str
    attempt_index: int
    source_kind: str
    label: str
    basename: str | None = None
    qasm_path: str | None = None
    qasm_sha256: str | None = None
    family: str | None = None
    requested_qubits: int | None = None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _exception_chain_text(exc: BaseException) -> str:
    return "\n".join(
        f"{type(item).__module__}.{type(item).__name__}: {item}"
        for item in _exception_chain(exc)
    ).lower()


def _has_any_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _is_mps_out_of_memory_error(exc: BaseException) -> bool:
    return isinstance(exc, (MemoryError, MPSBackendResourceLimit)) or _has_any_marker(
        _exception_chain_text(exc), MPS_OUT_OF_MEMORY_MARKERS
    )


def _is_mps_numerical_error(exc: BaseException) -> bool:
    return isinstance(exc, MPSBackendNumericalFailure) or _has_any_marker(
        _exception_chain_text(exc), MPS_NUMERICAL_FAILURE_MARKERS
    )


def _is_gurobi_infrastructure_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        qualified_type = f"{type(item).__module__}.{type(item).__name__}".lower()
        message = str(item).lower()
        if _has_any_marker(message, GUROBI_INFRASTRUCTURE_MARKERS) and (
            "gurobi" in qualified_type
            or "gurobi" in message
            or "wls" in message
            or "managed gurobi" in message
        ):
            return True
    return False


def _mps_backend_failure_message(result: Any) -> str:
    statuses = [f"result_status={getattr(result, 'status', None)!r}"]
    for index, experiment in enumerate(getattr(result, "results", ()) or ()):
        statuses.append(
            f"experiment_{index}_status={getattr(experiment, 'status', None)!r}"
        )
    return "MPS backend failed: " + "; ".join(statuses)


def _raise_mps_backend_failure(result: Any) -> None:
    message = _mps_backend_failure_message(result)
    if _has_any_marker(message, MPS_OUT_OF_MEMORY_MARKERS):
        raise MPSBackendResourceLimit(message)
    if _has_any_marker(message, MPS_NUMERICAL_FAILURE_MARKERS):
        raise MPSBackendNumericalFailure(message)
    raise MPSBackendFailure(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_basename(value: str) -> str:
    basename = value.strip()
    if basename.endswith(".qasm"):
        basename = basename[:-5]
    if not basename or Path(basename).name != basename:
        raise ProtocolError(f"invalid circuit basename {value!r}")
    return basename


def resolve_canonical_frame(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    qasm_dir: str | Path = DEFAULT_QASM_DIR,
    *,
    expected_count: int = EXPECTED_BASE_COUNT,
    requested_basenames: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve and hash the exact base frame defined by the circuit list.

    Extra QASM files in the directory are detected and reported but excluded:
    the circuit list, not a directory glob, defines the base frame. If a
    caller supplies ``requested_basenames``, that set must equal the circuit
    list exactly; missing or extra requested entries are errors.
    """

    manifest = Path(manifest_path).resolve()
    directory = Path(qasm_dir).resolve()
    if not manifest.is_file():
        raise ProtocolError(f"circuit-list manifest missing: {manifest}")
    if not directory.is_dir():
        raise ProtocolError(f"QASM directory missing: {directory}")
    manifest_sha256 = file_sha256(manifest)
    enforce_fixed_identity = expected_count == EXPECTED_BASE_COUNT
    if (
        enforce_fixed_identity
        and manifest_sha256 != EXPECTED_MANIFEST_SHA256
    ):
        raise ProtocolError(
            "circuit-list manifest checksum substitution: "
            f"{manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )

    basenames = [
        _normalize_basename(line.split("#", 1)[0])
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    duplicate_names = sorted(
        name for name, count in Counter(basenames).items() if count > 1
    )
    if duplicate_names:
        raise ProtocolError(
            f"circuit list contains duplicate basenames: {duplicate_names}"
        )
    if len(basenames) != expected_count:
        raise ProtocolError(
            f"circuit list has {len(basenames)} entries; "
            f"expected exactly {expected_count}"
        )

    canonical_set = set(basenames)
    if requested_basenames is not None:
        requested = [_normalize_basename(name) for name in requested_basenames]
        requested_duplicates = sorted(
            name for name, count in Counter(requested).items() if count > 1
        )
        if requested_duplicates:
            raise ProtocolError(
                f"requested base frame contains duplicates: {requested_duplicates}"
            )
        missing_requested = sorted(canonical_set - set(requested))
        extra_requested = sorted(set(requested) - canonical_set)
        if missing_requested or extra_requested:
            raise ProtocolError(
                "requested base frame differs from the circuit list: "
                f"missing={missing_requested}, extra={extra_requested}"
            )

    entries = []
    missing_files = []
    for basename in basenames:
        path = directory / f"{basename}.qasm"
        if not path.is_file():
            missing_files.append(basename)
            continue
        entries.append(
            {
                "basename": basename,
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    if missing_files:
        raise ProtocolError(f"QASM files missing: {missing_files}")

    available = {path.stem for path in directory.glob("*.qasm") if path.is_file()}
    excluded_extras = sorted(available - canonical_set)
    corpus_value = [
        {"basename": entry["basename"], "sha256": entry["sha256"]}
        for entry in entries
    ]
    corpus_sha256 = stable_hash(corpus_value)
    if enforce_fixed_identity and corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise ProtocolError(
            "QASM corpus checksum substitution: "
            f"{corpus_sha256} != {EXPECTED_CORPUS_SHA256}"
        )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "qasm_dir": str(directory),
        "circuit_count": len(entries),
        "basenames": basenames,
        "entries": entries,
        "excluded_extra_qasm_basenames": excluded_extras,
        "excluded_extra_qasm_count": len(excluded_extras),
        "corpus_sha256": corpus_sha256,
        "corpus_fingerprint_algorithm": (
            "sha256(canonical-json([{basename,sha256}], manifest order))"
        ),
    }


def build_attempt_grid(frame: Mapping[str, Any]) -> list[AttemptSpec]:
    if frame.get("circuit_count") != EXPECTED_BASE_COUNT:
        raise ProtocolError(
            f"base frame must contain exactly {EXPECTED_BASE_COUNT} circuits"
        )
    if frame.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ProtocolError("base frame does not match the pinned manifest SHA-256")
    if frame.get("corpus_sha256") != EXPECTED_CORPUS_SHA256:
        raise ProtocolError("base frame does not match the pinned corpus SHA-256")
    attempts: list[AttemptSpec] = []
    for entry in frame["entries"]:
        basename = entry["basename"]
        attempts.append(
            AttemptSpec(
                attempt_id=f"base:{basename}",
                attempt_index=len(attempts),
                source_kind="canonical_qasm",
                label=basename,
                basename=basename,
                qasm_path=entry["path"],
                qasm_sha256=entry["sha256"],
            )
        )
    for family in EXTENSION_FAMILIES:
        for requested_qubits in EXTENSION_SIZES:
            attempts.append(
                AttemptSpec(
                    attempt_id=f"extension:{family}:q{requested_qubits}",
                    attempt_index=len(attempts),
                    source_kind="mqt_bench_extension",
                    label=f"{family}_q{requested_qubits}",
                    family=family,
                    requested_qubits=requested_qubits,
                )
            )
    identifiers = [attempt.attempt_id for attempt in attempts]
    if len(identifiers) != len(set(identifiers)):
        raise ProtocolError("declared attempt grid contains duplicate identities")
    return attempts


def select_attempts(
    attempts: Sequence[AttemptSpec],
    *,
    smoke: bool,
    shard_index: int,
    shard_count: int,
) -> list[AttemptSpec]:
    if shard_count <= 0:
        raise ProtocolError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ProtocolError(
            f"shard_index must be in [0, {shard_count}); got {shard_index}"
        )
    selected = list(attempts)
    if smoke:
        by_id = {attempt.attempt_id: attempt for attempt in attempts}
        missing = [identifier for identifier in SMOKE_ATTEMPT_IDS if identifier not in by_id]
        if missing:
            raise ProtocolError(f"smoke attempt identities missing from grid: {missing}")
        selected = [by_id[identifier] for identifier in SMOKE_ATTEMPT_IDS]
    return [
        attempt
        for position, attempt in enumerate(selected)
        if position % shard_count == shard_index
    ]


def _dependency_versions() -> dict[str, str]:
    packages = (
        "qiskit",
        "qiskit-aer",
        "mqt.bench",
        "pulp",
        "gurobipy",
        "numpy",
        "scipy",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _runtime_value() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "system": platform.system(),
        "libc": list(platform.libc_ver()),
        "machine": platform.machine(),
    }


def _protocol_config() -> dict[str, Any]:
    partition_solver = solver_provenance(
        gurobi_license_retry_delays_seconds=(
            GUROBI_LICENSE_RETRY_DELAYS_SECONDS
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "qmax": CANONICAL_QMAX,
        "base_count": EXPECTED_BASE_COUNT,
        "extension_families": list(EXTENSION_FAMILIES),
        "extension_sizes": list(EXTENSION_SIZES),
        "generator_kwargs": GENERATOR_KWARGS,
        "generator_level": "INDEP",
        "generator_random_parameters": True,
        "generator_optimization_level": 0,
        "q15_generator_reference_sha256": stable_hash(Q15_GENERATOR_REFERENCE),
        "q15_generator_reference_scope": "34 families at requested_qubits=15 only",
        "duplicate_policy": (
            "exact canonical circuit SHA-256 clusters; report attempt and "
            "hash-distinct-circuit denominators; attempts are not independent "
            "samples"
        ),
        "known_duplicate_extension_hashes": KNOWN_DUPLICATE_EXTENSION_HASHES,
        "base_criterion_reference": BASE_CRITERION_REFERENCE,
        "base_mps_crosscheck_reference": BASE_MPS_CROSSCHECK_REFERENCE,
        "transpiler_seed": TRANSPILER_SEED,
        "transpile_basis": list(TRANSPILE_BASIS),
        "selection": (
            "second Schmidt amplitude a2 after each acted gate; select "
            "a2 <= tolerance"
        ),
        "mps_a2_extraction": (
            "SVD of the local MPS site tensor weighted by its adjacent "
            "Schmidt bond vectors from save_matrix_product_state"
        ),
        "schmidt_amplitude_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
        "density_matrix_validation_tolerance": DENSITY_MATRIX_TOLERANCE,
        "mps_max_bond_dimension": MPS_MAX_BOND_DIMENSION,
        "mps_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
        "mps_lapack": MPS_LAPACK,
        "base_statevector_crosscheck": {
            "required": True,
            "selection_rule": (
                "both MPS and pure-state paths obtain the second Schmidt "
                "amplitude a2 by direct SVD and select a2 <= tolerance"
            ),
            "determinant_role": "reported diagnostic only; never selection",
            "agreement_gate": "exact acted-point identities and selected event set",
        },
        "circuit_eligibility": (
            "only unconditioned 1/2-qubit unitary operations; terminal "
            "measurements/barriers allowed and excluded from evaluated locations"
        ),
        "generated_circuit_fingerprint": (
            "canonical JSON instruction stream plus unitary matrix SHA-256"
        ),
        "partitioner": "batch_partition.partition_nodes",
        "partitioner_qmax": CANONICAL_QMAX,
        "attempt_timeout_default_seconds": DEFAULT_TIMEOUT_SECONDS,
        "refinement_node_limit": REFINEMENT_NODE_LIMIT,
        "refinement_max_time_limit_seconds": (
            MAX_REFINEMENT_TIME_LIMIT_SECONDS
        ),
        "instrumentation_completion_reserve_seconds": (
            INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
        ),
        "gurobi_license_retry_delays_seconds": list(
            GUROBI_LICENSE_RETRY_DELAYS_SECONDS
        ),
        "infrastructure_outcome_policy": (
            "checkpoint as incomplete, retry once immediately, and exclude from "
            "terminal publication artifacts"
        ),
        "partition_solver": partition_solver,
        "partition_solver_sha256": stable_hash(partition_solver),
        "publication_partition_solver_backend": (
            PUBLICATION_PARTITION_SOLVER_BACKEND
        ),
        "status_semantics": (
            "status is the analysis terminal status; deployment_status is the "
            "separate operational Qmax outcome"
        ),
        "instrumentation_operation_accounting": dict(
            INSTRUMENTATION_OPERATION_ACCOUNTING
        ),
        "final_read_rule": (
            "exclude a qubit's final non-skip gate only when a terminal "
            "standard measurement reads that qubit"
        ),
    }


def build_fingerprint_bundle(frame: Mapping[str, Any]) -> dict[str, Any]:
    dependencies = _dependency_versions()
    runtime = _runtime_value()
    config = _protocol_config()
    source_entries = [
        {"path": path.name, "sha256": file_sha256(path)} for path in SOURCE_FILES
    ]
    return {
        "corpus": {
            "sha256": frame["corpus_sha256"],
            "manifest_sha256": frame["manifest_sha256"],
            "circuit_count": frame["circuit_count"],
            "excluded_extra_qasm_basenames": frame[
                "excluded_extra_qasm_basenames"
            ],
        },
        "dependencies": {
            "sha256": stable_hash(dependencies),
            "versions": dependencies,
        },
        "runtime": {"sha256": stable_hash(runtime), "value": runtime},
        "sources": {"sha256": stable_hash(source_entries), "files": source_entries},
        "config": {"sha256": stable_hash(config), "value": config},
    }


def _record_fingerprint_hashes(bundle: Mapping[str, Any]) -> dict[str, str]:
    return {
        "corpus_sha256": bundle["corpus"]["sha256"],
        "manifest_sha256": bundle["corpus"]["manifest_sha256"],
        "dependencies_sha256": bundle["dependencies"]["sha256"],
        "runtime_sha256": bundle["runtime"]["sha256"],
        "sources_sha256": bundle["sources"]["sha256"],
        "config_sha256": bundle["config"]["sha256"],
    }


def _record_skeleton(
    spec: AttemptSpec,
    bundle: Mapping[str, Any],
    *,
    run_mode: str,
    shard_index: int,
    shard_count: int,
    attempt_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": spec.attempt_id,
        "attempt_index": spec.attempt_index,
        "source": asdict(spec),
        "qmax": CANONICAL_QMAX,
        "status": None,
        "status_semantics": "analysis_terminal_status",
        "deployment_status": "not_evaluated",
        "record_complete": False,
        "record_sha256": None,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "phase_elapsed_seconds": {},
        "last_phase": "pending",
        "error": None,
        "error_category": None,
        "circuit": None,
        "selection": None,
        "instrumentation": None,
        "fingerprints": _record_fingerprint_hashes(bundle),
        "provenance": {
            "dependencies": bundle["dependencies"]["versions"],
            "runtime": bundle["runtime"]["value"],
            "protocol_config": bundle["config"]["value"],
        },
        "run": {
            "mode": run_mode,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "attempt_timeout_seconds": float(attempt_timeout_seconds),
            "infrastructure_retries_before_outcome": 0,
        },
    }


def _plain_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "device",
        "num_qubits",
        "time_taken",
        "matrix_product_state_max_bond_dimension",
        "matrix_product_state_truncation_threshold",
        "matrix_product_state_lapack",
        "mps_lapack",
        "parallel_state_update",
        "max_memory_mb",
    )
    output: dict[str, Any] = {}
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
    return output


def _validated_density_scores(
    value: Any, *, tolerance: float = DENSITY_MATRIX_TOLERANCE
) -> dict[str, float]:
    """Validate and normalize one-qubit rho, then report a2 and det(rho).

    The returned eigenspectrum quantities are diagnostics only; selection uses
    the separate direct-SVD ``a2`` path. The determinant is retained as a
    physical diagnostic. A negative minimum eigenvalue is clipped only when its
    magnitude is within the declared numerical validation tolerance.
    """

    import numpy as np

    try:
        density_matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise AnalysisFailure(f"density matrix cannot be converted: {exc}") from exc
    if density_matrix.shape != (2, 2):
        raise AnalysisFailure(
            f"one-qubit density matrix has shape {density_matrix.shape}, expected (2, 2)"
        )
    if not np.all(np.isfinite(density_matrix.real)) or not np.all(
        np.isfinite(density_matrix.imag)
    ):
        raise AnalysisFailure("one-qubit density matrix contains non-finite values")
    if not np.allclose(
        density_matrix, density_matrix.conj().T, atol=tolerance, rtol=0.0
    ):
        raise AnalysisFailure("one-qubit density matrix is not Hermitian")
    trace = complex(np.trace(density_matrix))
    trace_tolerance = max(10.0 * tolerance, 1e-12)
    if abs(trace.imag) > trace_tolerance or abs(trace.real - 1.0) > trace_tolerance:
        raise AnalysisFailure(
            f"one-qubit density matrix has non-unit trace {trace!r}"
        )
    normalized = (density_matrix + density_matrix.conj().T) / (2.0 * trace.real)
    eigenvalues = np.linalg.eigvalsh(normalized)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if minimum < -tolerance:
        raise AnalysisFailure(
            f"one-qubit density matrix is not positive semidefinite: "
            f"min_eigenvalue={minimum}"
        )
    if maximum > 1.0 + tolerance:
        raise AnalysisFailure(
            f"one-qubit density matrix has eigenvalue above one: {maximum}"
        )
    lambda_min = 0.0 if minimum < 0.0 else minimum
    lambda_max = 0.0 if maximum < 0.0 else maximum
    determinant_complex = complex(np.linalg.det(normalized))
    if not np.isfinite(determinant_complex.real) or not np.isfinite(
        determinant_complex.imag
    ):
        raise AnalysisFailure("one-qubit determinant is non-finite")
    if abs(determinant_complex.imag) > trace_tolerance:
        raise AnalysisFailure(
            f"one-qubit determinant has material imaginary part "
            f"{determinant_complex.imag}"
        )
    determinant = float(determinant_complex.real)
    if determinant < -tolerance:
        raise AnalysisFailure(
            f"one-qubit determinant {determinant} is below the physical range"
        )
    if determinant > 0.25 + tolerance:
        raise AnalysisFailure(
            f"one-qubit determinant {determinant} exceeds the physical maximum"
        )
    if determinant < 0.0:
        determinant = 0.0
    return {
        "a2": math.sqrt(lambda_min),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "determinant": determinant,
        "trace_before_normalization": float(trace.real),
    }


def _validated_density_determinant(value: Any) -> float:
    """Compatibility helper for tests and determinant-only diagnostics."""

    return _validated_density_scores(value)["determinant"]


def _validated_schmidt_scores(values: Any) -> dict[str, float]:
    """Normalize a direct Schmidt spectrum and derive physical diagnostics."""

    import numpy as np

    singular_values = np.asarray(values, dtype=float).reshape(-1)
    if not 1 <= len(singular_values) <= 2:
        raise AnalysisFailure(
            f"single-qubit Schmidt spectrum has length {len(singular_values)}"
        )
    if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
        raise AnalysisFailure("single-qubit Schmidt spectrum is nonphysical")
    singular_values = np.sort(singular_values)[::-1]
    norm_squared = float(np.dot(singular_values, singular_values))
    normalization_tolerance = 1e-10
    if (
        not math.isfinite(norm_squared)
        or norm_squared <= 0.0
        or abs(norm_squared - 1.0) > normalization_tolerance
    ):
        raise AnalysisFailure(
            "single-qubit Schmidt spectrum is not normalized: "
            f"sum_squares={norm_squared}"
        )
    singular_values = singular_values / math.sqrt(norm_squared)
    a1 = float(singular_values[0])
    a2 = float(singular_values[1]) if len(singular_values) == 2 else 0.0
    lambda_max = a1 * a1
    lambda_min = a2 * a2
    return {
        "a1": a1,
        "a2": a2,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "determinant": lambda_min * lambda_max,
        "trace_before_normalization": norm_squared,
    }


def _mps_snapshot_single_qubit_scores(
    snapshot: Any, qubit: int, num_qubits: int
) -> dict[str, float]:
    """Recover q|rest Schmidt amplitudes from the Aer MPS site tensors."""

    import numpy as np

    if not isinstance(snapshot, tuple) or len(snapshot) != 2:
        raise AnalysisFailure("Aer MPS snapshot is not a (tensors, bonds) tuple")
    tensors, bonds = snapshot
    if len(tensors) != num_qubits or len(bonds) != max(0, num_qubits - 1):
        raise AnalysisFailure(
            "Aer MPS snapshot has inconsistent tensor/bond counts"
        )
    if not 0 <= qubit < num_qubits:
        raise AnalysisFailure(f"MPS snapshot qubit {qubit} is out of range")
    site = np.asarray(tensors[qubit], dtype=np.complex128)
    if site.ndim != 3 or site.shape[0] != 2:
        raise AnalysisFailure(
            f"Aer MPS site tensor has shape {site.shape}, expected (2, chiL, chiR)"
        )
    weighted = site.copy()
    if qubit > 0:
        left = np.asarray(bonds[qubit - 1], dtype=float).reshape(-1)
        if (
            len(left) != site.shape[1]
            or not np.all(np.isfinite(left))
            or np.any(left < 0.0)
        ):
            raise AnalysisFailure("Aer MPS left bond is incompatible with site tensor")
        weighted *= left[None, :, None]
    if qubit < num_qubits - 1:
        right = np.asarray(bonds[qubit], dtype=float).reshape(-1)
        if (
            len(right) != site.shape[2]
            or not np.all(np.isfinite(right))
            or np.any(right < 0.0)
        ):
            raise AnalysisFailure("Aer MPS right bond is incompatible with site tensor")
        weighted *= right[None, None, :]
    singular_values = np.linalg.svd(
        weighted.reshape(2, -1), compute_uv=False, full_matrices=False
    )
    score = _validated_schmidt_scores(singular_values)
    score["left_chi"] = float(site.shape[1])
    score["right_chi"] = float(site.shape[2])
    return score


def _serialize_point_scores(
    scores: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "gate": gate,
            "qubit": qubit,
            "a2": float(score["a2"]),
            "determinant": float(score["determinant"]),
        }
        for (gate, qubit), score in sorted(scores.items())
    ]


def _select_monitorable_nodes_mps_details(
    qc: Any,
) -> tuple[
    dict[int, list[int]],
    dict[str, Any],
    dict[tuple[int, int], dict[str, float]],
]:
    """Apply the exact a2 rule through Aer-native saved MPS tensors."""

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    snapshot_circuit = qc.copy_empty_like()
    labels: dict[str, tuple[int, list[int]]] = {}
    for gate_index, instruction in enumerate(qc.data):
        operation, qargs = instruction.operation, instruction.qubits
        if operation.name in {"measure", "barrier"}:
            continue
        snapshot_circuit.append(operation, qargs, instruction.clbits)
        acted_qubits = [qc.find_bit(qubit).index for qubit in qargs]
        label = f"mps_{len(labels)}"
        snapshot_circuit.save_matrix_product_state(label=label)
        labels[label] = (gate_index, acted_qubits)

    simulator_options = {
        "method": "matrix_product_state",
        "matrix_product_state_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
        "mps_lapack": MPS_LAPACK,
    }
    if MPS_MAX_BOND_DIMENSION is not None:
        simulator_options["matrix_product_state_max_bond_dimension"] = (
            MPS_MAX_BOND_DIMENSION
        )
    simulator = AerSimulator(**simulator_options)
    compiled = transpile(
        snapshot_circuit,
        simulator,
        optimization_level=0,
        seed_transpiler=TRANSPILER_SEED,
    )
    result = simulator.run(compiled).result()
    if not result.success:
        _raise_mps_backend_failure(result)
    data = result.data(0)
    nodes: dict[int, list[int]] = {}
    point_scores: dict[tuple[int, int], dict[str, float]] = {}
    observed_max_chi = 1
    for label, (gate_index, acted_qubits) in labels.items():
        if label not in data:
            raise AnalysisFailure(f"MPS result omitted tensor snapshot {label}")
        snapshot = data[label]
        if not isinstance(snapshot, tuple) or len(snapshot) != 2:
            raise AnalysisFailure(f"MPS result has malformed snapshot {label}")
        _, bonds = snapshot
        observed_max_chi = max(
            observed_max_chi,
            max((len(bond) for bond in bonds), default=1),
        )
        for qubit_index in acted_qubits:
            score = _mps_snapshot_single_qubit_scores(
                snapshot, qubit_index, qc.num_qubits
            )
            point_scores[(gate_index, qubit_index)] = score
            if score["a2"] <= SCHMIDT_AMPLITUDE_TOLERANCE:
                nodes.setdefault(gate_index, []).append(qubit_index)
    for qubits in nodes.values():
        qubits.sort()

    metadata = result.results[0].metadata or {}
    diagnostics = {
        "configured_max_bond_dimension": MPS_MAX_BOND_DIMENSION,
        "configured_max_bond_dimension_mode": "unbounded",
        "configured_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
        "configured_mps_lapack": MPS_LAPACK,
        "observed_max_chi": observed_max_chi,
        "observed_max_chi_available": True,
        "observed_max_chi_method": (
            "maximum Aer MPS bond-vector length across every analyzed-gate "
            "save_matrix_product_state snapshot"
        ),
        "observed_max_chi_unavailable_reason": None,
        "backend_metadata": _plain_metadata(metadata),
        "snapshot_count": len(labels),
        "point_count": len(point_scores),
    }
    return nodes, diagnostics, point_scores


def select_monitorable_nodes_mps(qc: Any) -> tuple[dict[int, list[int]], dict[str, Any]]:
    nodes, diagnostics, _ = _select_monitorable_nodes_mps_details(qc)
    return nodes, diagnostics


def select_monitorable_nodes_statevector(
    qc: Any,
) -> tuple[
    dict[int, list[int]], dict[tuple[int, int], dict[str, float]]
]:
    """Replay the established a2 selector and validate every reduced state.

    A direct pure-state SVD evaluates sqrt(lambda_min(rho)) more stably than
    forming a nearly rank-one rho and diagonalizing it. The independent
    continuity check compares these selected events to the established
    ``analyze_per_gate_nolock`` path. We still validate normalized rho and
    retain det(rho) as a diagnostic.
    """

    import numpy as np
    from qiskit.quantum_info import Statevector, partial_trace

    state = Statevector.from_int(0, dims=(2,) * qc.num_qubits)
    nodes: dict[int, list[int]] = {}
    point_scores: dict[tuple[int, int], dict[str, float]] = {}
    for gate_index, instruction in enumerate(qc.data):
        operation = instruction.operation
        if operation.name in {"measure", "barrier"}:
            continue
        qubits = [qc.find_bit(bit).index for bit in instruction.qubits]
        state = state.evolve(operation, qargs=qubits)
        tensor = np.asarray(state.data, dtype=np.complex128).reshape(
            (2,) * qc.num_qubits
        )
        for qubit in qubits:
            reduced = partial_trace(
                state,
                [candidate for candidate in range(qc.num_qubits) if candidate != qubit],
            ).data
            density_score = _validated_density_scores(reduced)
            axis = qc.num_qubits - 1 - qubit
            amplitude_matrix = np.moveaxis(tensor, axis, 0).reshape(2, -1)
            singular_values = np.linalg.svd(
                amplitude_matrix, compute_uv=False, full_matrices=False
            )
            score = _validated_schmidt_scores(singular_values)
            score["density_eigen_a2_diagnostic"] = density_score["a2"]
            score["density_determinant_diagnostic"] = density_score[
                "determinant"
            ]
            a2 = score["a2"]
            point_scores[(gate_index, qubit)] = score
            if a2 <= SCHMIDT_AMPLITUDE_TOLERANCE:
                nodes.setdefault(gate_index, []).append(qubit)
    for qubits in nodes.values():
        qubits.sort()
    return nodes, point_scores


def compare_statevector_selection_to_reference(qc: Any) -> dict[str, Any]:
    """Compare exact v2 selected events to analyze_per_gate_nolock."""

    from fast_analysis import analyze_per_gate_nolock

    v2_nodes, scores = select_monitorable_nodes_statevector(qc)
    _, _, reference_paths = analyze_per_gate_nolock(
        qc, tol=SCHMIDT_AMPLITUDE_TOLERANCE
    )
    v2_events = sorted(_selection_node_set(v2_nodes))
    reference_events = sorted(
        (gate, qubit)
        for qubit, gates in reference_paths.items()
        for gate in gates
    )
    v2_set = set(v2_events)
    reference_set = set(reference_events)
    v2_only = sorted(v2_set - reference_set)
    reference_only = sorted(reference_set - v2_set)
    determinants = [float(score["determinant"]) for score in scores.values()]
    amplitudes = [float(score["a2"]) for score in scores.values()]
    return {
        "total_acted_points": len(scores),
        "v2_selected_event_count": len(v2_events),
        "reference_selected_event_count": len(reference_events),
        "selection_mismatch_count": len(v2_only) + len(reference_only),
        "v2_only_count": len(v2_only),
        "reference_only_count": len(reference_only),
        "v2_selected_events": [list(event) for event in v2_events],
        "reference_selected_events": [list(event) for event in reference_events],
        "v2_selected_events_sha256": stable_hash(v2_events),
        "reference_selected_events_sha256": stable_hash(reference_events),
        "mismatch_examples": [
            {"gate": gate, "qubit": qubit, "side": "v2_only"}
            for gate, qubit in v2_only[:10]
        ]
        + [
            {"gate": gate, "qubit": qubit, "side": "reference_only"}
            for gate, qubit in reference_only[:10]
        ],
        "maximum_a2": max(amplitudes, default=0.0),
        "minimum_determinant": min(determinants, default=0.0),
        "maximum_determinant": max(determinants, default=0.0),
    }


def criterion_comparison_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "total_acted_points",
        "v2_selected_event_count",
        "reference_selected_event_count",
        "selection_mismatch_count",
        "v2_only_count",
        "reference_only_count",
    )
    return {
        "circuit_count": len(records),
        **{
            field: sum(int(record[field]) for record in records)
            for field in numeric_fields
        },
        "circuits_with_mismatches": sum(
            int(record["selection_mismatch_count"] > 0) for record in records
        ),
    }


def criterion_comparison_scientific_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash only source/config labels and discrete scientific events/counts."""

    ordered = sorted(records, key=lambda record: int(record["attempt_index"]))
    return stable_hash(
        [
            {field: record[field] for field in CRITERION_SCIENTIFIC_FIELDS}
            for record in ordered
        ]
    )


def mps_crosscheck_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summed_fields = (
        "point_count_mps",
        "point_count_statevector",
        "mps_selected_event_count",
        "statevector_selected_event_count",
        "selection_mismatch_count",
        "missing_mps_point_count",
        "missing_statevector_point_count",
    )
    return {
        "circuit_count": len(records),
        **{
            field: sum(int(record[field]) for record in records)
            for field in summed_fields
        },
        "circuits_with_selection_mismatches": sum(
            int(record["selection_mismatch_count"] > 0) for record in records
        ),
        "maximum_absolute_a2_difference": max(
            (float(record["maximum_absolute_a2_difference"]) for record in records),
            default=0.0,
        ),
        "maximum_absolute_determinant_difference": max(
            (
                float(record["maximum_absolute_determinant_difference"])
                for record in records
            ),
            default=0.0,
        ),
        "maximum_observed_chi": max(
            (int(record["observed_max_chi"]) for record in records),
            default=0,
        ),
    }


def mps_crosscheck_scientific_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash source-bound event evidence, excluding timings and float drift."""

    fields = (
        "attempt_id",
        "attempt_index",
        "basename",
        "qasm_sha256",
        "manifest_sha256",
        "corpus_sha256",
        "selector",
        "schmidt_amplitude_tolerance",
        "mps_a2_extraction",
        "mps_lapack",
        "mps_max_bond_dimension",
        "mps_truncation_threshold",
        "operational_eligibility_category",
        "point_count_mps",
        "point_count_statevector",
        "mps_selected_events",
        "statevector_selected_events",
        "selection_mismatch_count",
        "selection_mismatch_points",
        "missing_mps_point_count",
        "missing_statevector_point_count",
    )
    return stable_hash(
        [{field: record[field] for field in fields} for record in records]
    )


def _selection_node_set(nodes: Mapping[int, Sequence[int]]) -> set[tuple[int, int]]:
    return {
        (int(gate_index), int(qubit))
        for gate_index, qubits in nodes.items()
        for qubit in qubits
    }


def _deserialize_event_rows(
    rows: Any, *, label: str
) -> tuple[list[tuple[int, int]], list[str]]:
    if not isinstance(rows, list):
        return [], [f"{label} is not a list"]
    events: list[tuple[int, int]] = []
    for position, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not int
            or row[0] < 0
            or row[1] < 0
        ):
            return [], [f"{label} row {position} is invalid"]
        events.append((row[0], row[1]))
    if events != sorted(set(events)):
        return events, [f"{label} is not sorted and unique"]
    return events, []


def _serialize_monitorable_nodes(
    nodes: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    return [
        {
            "gate": int(gate_index),
            "qubits": sorted({int(qubit) for qubit in qubits}),
        }
        for gate_index, qubits in sorted(nodes.items())
    ]


def _deserialize_monitorable_nodes(
    rows: Any,
) -> tuple[dict[int, list[int]], list[str]]:
    errors: list[str] = []
    nodes: dict[int, list[int]] = {}
    if not isinstance(rows, list):
        return {}, ["selection monitorable_nodes is not a list"]
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"selection node row {position} is not an object")
            continue
        gate = row.get("gate")
        qubits = row.get("qubits")
        if type(gate) is not int or gate < 0:
            errors.append(f"selection node row {position} has invalid gate")
            continue
        if gate in nodes:
            errors.append(f"selection monitorable_nodes repeats gate {gate}")
            continue
        if (
            not isinstance(qubits, list)
            or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
            or len(qubits) != len(set(qubits))
            or qubits != sorted(qubits)
        ):
            errors.append(f"selection node row {position} has invalid qubits")
            continue
        nodes[gate] = list(qubits)
    if list(nodes) != sorted(nodes):
        errors.append("selection monitorable_nodes gates are not sorted")
    return nodes, errors


def _deserialize_point_scores(
    rows: Any,
) -> tuple[dict[tuple[int, int], dict[str, float]], list[str]]:
    errors: list[str] = []
    scores: dict[tuple[int, int], dict[str, float]] = {}
    if not isinstance(rows, list):
        return {}, ["selection point_scores is not a list"]
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"selection point score {position} is not an object")
            continue
        gate = row.get("gate")
        qubit = row.get("qubit")
        a2 = row.get("a2")
        determinant = row.get("determinant")
        point = (gate, qubit)
        if type(gate) is not int or gate < 0 or type(qubit) is not int or qubit < 0:
            errors.append(f"selection point score {position} has invalid identity")
            continue
        if point in scores:
            errors.append(f"selection point_scores repeats point {point}")
            continue
        if (
            not isinstance(a2, (int, float))
            or isinstance(a2, bool)
            or not math.isfinite(float(a2))
            or not 0.0 <= float(a2) <= 1.0
        ):
            errors.append(f"selection point score {position} has invalid a2")
            continue
        if (
            not isinstance(determinant, (int, float))
            or isinstance(determinant, bool)
            or not math.isfinite(float(determinant))
            or not 0.0 <= float(determinant) <= 0.25 + DENSITY_MATRIX_TOLERANCE
        ):
            errors.append(
                f"selection point score {position} has invalid determinant"
            )
            continue
        scores[(gate, qubit)] = {
            "a2": float(a2),
            "determinant": float(determinant),
        }
    if list(scores) != sorted(scores):
        errors.append("selection point_scores are not sorted")
    return scores, errors


def crosscheck_mps_against_statevector(
    qc: Any,
    mps_nodes: Mapping[int, Sequence[int]],
    mps_scores: Mapping[tuple[int, int], Mapping[str, float]],
) -> dict[str, Any]:
    started = time.perf_counter()
    statevector_nodes, statevector_scores = (
        select_monitorable_nodes_statevector(qc)
    )
    mps_points = set(mps_scores)
    statevector_points = set(statevector_scores)
    common_points = mps_points & statevector_points
    maximum_determinant_difference = max(
        (
            abs(
                mps_scores[point]["determinant"]
                - statevector_scores[point]["determinant"]
            )
            for point in common_points
        ),
        default=0.0,
    )
    maximum_a2_difference = max(
        (
            abs(mps_scores[point]["a2"] - statevector_scores[point]["a2"])
            for point in common_points
        ),
        default=0.0,
    )
    mps_selected = _selection_node_set(mps_nodes)
    statevector_selected = _selection_node_set(statevector_nodes)
    mps_selected_events = sorted(mps_selected)
    statevector_selected_events = sorted(statevector_selected)
    selection_mismatches = sorted(mps_selected ^ statevector_selected)
    missing_mps_points = sorted(statevector_points - mps_points)
    missing_statevector_points = sorted(mps_points - statevector_points)
    passed = (
        not selection_mismatches
        and not missing_mps_points
        and not missing_statevector_points
    )
    return {
        "required": True,
        "performed": True,
        "passed": passed,
        "rule": (
            "both backends classify second Schmidt amplitude a2 <= "
            f"{SCHMIDT_AMPLITUDE_TOLERANCE}; det(rho) is diagnostic only"
        ),
        "point_count_mps": len(mps_points),
        "point_count_statevector": len(statevector_points),
        "mps_selected_event_count": len(mps_selected_events),
        "statevector_selected_event_count": len(statevector_selected_events),
        "mps_selected_events": [list(point) for point in mps_selected_events],
        "statevector_selected_events": [
            list(point) for point in statevector_selected_events
        ],
        "mps_selected_events_sha256": stable_hash(mps_selected_events),
        "statevector_selected_events_sha256": stable_hash(
            statevector_selected_events
        ),
        "selection_mismatch_count": len(selection_mismatches),
        "selection_mismatch_points": [list(point) for point in selection_mismatches],
        "missing_mps_point_count": len(missing_mps_points),
        "missing_statevector_point_count": len(missing_statevector_points),
        "maximum_absolute_a2_difference": maximum_a2_difference,
        "maximum_absolute_determinant_difference": (
            maximum_determinant_difference
        ),
        "runtime_seconds": round(time.perf_counter() - started, 9),
    }


def _last_non_skip_gate_by_qubit(qc: Any) -> list[int | None]:
    last: list[int | None] = [None] * qc.num_qubits
    for gate_index, instruction in enumerate(qc.data):
        if instruction.operation.name in SKIP_OPS:
            continue
        for qubit in instruction.qubits:
            last[qc.find_bit(qubit).index] = gate_index
    return last


def _instrumentation_partition_budget(
    *, attempt_started: float, attempt_timeout_seconds: float
) -> tuple[int, float]:
    """Fit optional refinement and license backoff inside the attempt deadline."""

    if (
        not isinstance(attempt_timeout_seconds, (int, float))
        or isinstance(attempt_timeout_seconds, bool)
        or not math.isfinite(float(attempt_timeout_seconds))
        or attempt_timeout_seconds <= 0
    ):
        raise ProtocolError("attempt_timeout_seconds must be finite and positive")
    remaining = attempt_timeout_seconds - (time.perf_counter() - attempt_started)
    usable_for_solver = (
        remaining
        - sum(GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
        - INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
    )
    if usable_for_solver < 1.0:
        # FFD is already a complete feasible plan.  Skip only the optional
        # one-fewer-bin query when the wall-clock contract cannot contain it.
        return 0, 1.0
    return REFINEMENT_NODE_LIMIT, min(
        MAX_REFINEMENT_TIME_LIMIT_SECONDS, float(usable_for_solver)
    )


def build_instrumentation_metrics(
    qc: Any,
    nodes: Mapping[int, Sequence[int]],
    *,
    qmax: int = CANONICAL_QMAX,
    partition_override: Mapping[str, Any] | None = None,
    refinement_node_limit: int = REFINEMENT_NODE_LIMIT,
    refinement_time_limit_seconds: float = MAX_REFINEMENT_TIME_LIMIT_SECONDS,
    attempt_started: float | None = None,
    attempt_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Compute structural cones and the actual RQ3 batch partition.

    Gate indices are the partitioner's item identities, matching RQ3. If a
    gate has multiple monitorable qubits, its extra-qubit demand is the union
    of the causal-cone qubits not already represented by those monitored
    qubits. Each qubit's final read is excluded from batching, as in RQ3.
    """

    if type(qmax) is not int or qmax != CANONICAL_QMAX:
        raise ProtocolError(
            f"operational scalability protocol requires Qmax={CANONICAL_QMAX}; "
            f"got {qmax!r}"
        )
    if type(refinement_node_limit) is not int or refinement_node_limit < 0:
        raise ProtocolError("refinement_node_limit must be a non-negative integer")
    if (
        not isinstance(refinement_time_limit_seconds, (int, float))
        or isinstance(refinement_time_limit_seconds, bool)
        or not math.isfinite(float(refinement_time_limit_seconds))
        or not 0 < float(refinement_time_limit_seconds) <= (
            MAX_REFINEMENT_TIME_LIMIT_SECONDS
        )
    ):
        raise ProtocolError(
            "refinement_time_limit_seconds must be in "
            f"(0, {MAX_REFINEMENT_TIME_LIMIT_SECONDS}]"
        )

    original_unitary_gate_count_m = sum(
        1
        for instruction in qc.data
        if instruction.operation.name not in {"measure", "barrier"}
    )

    selected_events: list[tuple[int, int]] = []
    per_event_cones: dict[tuple[int, int], tuple[set[int], set[int]]] = {}
    for gate_index in sorted(nodes):
        for qubit in sorted(set(int(value) for value in nodes[gate_index])):
            event = (int(gate_index), qubit)
            selected_events.append(event)
            per_event_cones[event] = causal_cone(qc, event[0], event[1])

    last_by_qubit = _last_non_skip_gate_by_qubit(qc)
    terminally_measured_qubits = {
        qc.find_bit(instruction.qubits[0]).index
        for instruction in qc.data
        if instruction.operation.name == "measure" and instruction.qubits
    }
    candidate_events = [
        event
        for event in selected_events
        if not (
            event[1] in terminally_measured_qubits
            and event[0] == last_by_qubit[event[1]]
        )
    ]
    events_by_gate: dict[int, list[int]] = {}
    for gate_index, qubit in candidate_events:
        events_by_gate.setdefault(gate_index, []).append(qubit)

    constraint: dict[Any, Any] = {"num_oq": int(qc.num_qubits)}
    gate_node_costs: list[dict[str, Any]] = []
    for gate_index in sorted(events_by_gate):
        monitored_qubits = sorted(set(events_by_gate[gate_index]))
        cone_qubits: set[int] = set()
        cone_gates: set[int] = set()
        for qubit in monitored_qubits:
            gates, qubits = per_event_cones[(gate_index, qubit)]
            cone_gates.update(gates)
            cone_qubits.update(qubits)
        extra_qubits = sorted(cone_qubits - set(monitored_qubits))
        constraint[gate_index] = {
            "num_eq": len(extra_qubits),
            "qubits": monitored_qubits,
        }
        gate_node_costs.append(
            {
                "gate": gate_index,
                "monitor_qubits": monitored_qubits,
                "monitor_event_count": len(monitored_qubits),
                "extra_qubit_demand": len(extra_qubits),
                "cone_qubit_union_count": len(cone_qubits),
                "cone_gate_union_count": len(cone_gates),
            }
        )

    if partition_override is None:
        if attempt_started is not None:
            refinement_node_limit, refinement_time_limit_seconds = (
                _instrumentation_partition_budget(
                    attempt_started=attempt_started,
                    attempt_timeout_seconds=attempt_timeout_seconds,
                )
            )
        batches, infeasible, partition_metadata = partition_nodes(
            constraint,
            qmax,
            ilp_node_limit=refinement_node_limit,
            return_metadata=True,
            refinement_time_limit_seconds=float(
                refinement_time_limit_seconds
            ),
            gurobi_license_retry_delays_seconds=(
                GUROBI_LICENSE_RETRY_DELAYS_SECONDS
            ),
        )
    else:
        raw_batches = partition_override.get("batches")
        raw_infeasible = partition_override.get("infeasible_gate_nodes")
        partition_metadata = partition_override.get("partition_metadata")
        if (
            not isinstance(raw_batches, list)
            or not isinstance(raw_infeasible, list)
            or not isinstance(partition_metadata, Mapping)
        ):
            raise ProtocolError("stored scalability partition evidence is malformed")
        batches = []
        for raw in raw_batches:
            if not isinstance(raw, Mapping):
                raise ProtocolError("stored scalability batch is malformed")
            batches.append({
                "nodes": list(raw.get("nodes", ())),
                "cost": raw.get("cost"),
            })
        infeasible = list(raw_infeasible)
        partition_metadata = dict(partition_metadata)
        refinement = partition_metadata.get("refinement")
        if not isinstance(refinement, Mapping):
            raise ProtocolError(
                "stored scalability partition refinement evidence is malformed"
            )
        refinement_node_limit = refinement.get("node_limit")
        refinement_time_limit_seconds = refinement.get("time_limit_seconds")
        if type(refinement_node_limit) is not int or refinement_node_limit < 0:
            raise ProtocolError("stored refinement node limit is invalid")
        if (
            not isinstance(refinement_time_limit_seconds, (int, float))
            or isinstance(refinement_time_limit_seconds, bool)
            or not math.isfinite(float(refinement_time_limit_seconds))
            or not 0 < float(refinement_time_limit_seconds) <= (
                MAX_REFINEMENT_TIME_LIMIT_SECONDS
            )
        ):
            raise ProtocolError("stored refinement time limit is invalid")
    try:
        validate_partition_certificate(
            constraint,
            qmax,
            batches,
            infeasible,
            partition_metadata,
            ilp_node_limit=refinement_node_limit,
            refinement_time_limit_seconds=float(
                refinement_time_limit_seconds
            ),
            gurobi_license_retry_delays_seconds=(
                GUROBI_LICENSE_RETRY_DELAYS_SECONDS
            ),
        )
    except ValueError as exc:
        raise ProtocolError(f"invalid instrumentation partition certificate: {exc}") from exc
    serialized_batches = [
        {
            "batch_index": batch_index,
            "nodes": [int(node) for node in batch["nodes"]],
            "cost": int(batch["cost"]),
            "run_qubits": int(qc.num_qubits) + int(batch["cost"]),
        }
        for batch_index, batch in enumerate(batches)
    ]
    max_run_qubits = max(
        (batch["run_qubits"] for batch in serialized_batches), default=None
    )
    infeasible_set = {int(node) for node in infeasible}
    individually_feasible_nodes = set(events_by_gate) - infeasible_set
    deployed_nodes = {
        int(node) for batch in serialized_batches for node in batch["nodes"]
    }
    individually_feasible_monitor_events = sum(
        len(events_by_gate[node]) for node in individually_feasible_nodes
    )
    deployed_monitor_events = sum(
        len(events_by_gate[node]) for node in deployed_nodes
    )

    gate_cost_by_node = {row["gate"]: row for row in gate_node_costs}
    planned_replay_gate_executions = sum(
        gate_cost_by_node[node]["cone_gate_union_count"] for node in deployed_nodes
    )
    planned_monitor_measurement_operations = deployed_monitor_events
    planned_monitor_reset_operations = deployed_monitor_events
    planned_original_gate_executions = (
        len(serialized_batches) * original_unitary_gate_count_m
    )
    planned_total_counted_operations = (
        planned_original_gate_executions
        + planned_replay_gate_executions
        + planned_monitor_measurement_operations
        + planned_monitor_reset_operations
    )
    batch_costs = [batch["cost"] for batch in serialized_batches]
    planned_peak_extra_ancillas = max(batch_costs, default=0)
    planned_cumulative_extra_ancilla_allocations = sum(batch_costs)

    selected_total_cone_gates = sum(
        len(per_event_cones[event][0]) for event in selected_events
    )
    selected_total_cone_qubits = sum(
        len(per_event_cones[event][1]) for event in selected_events
    )
    candidate_total_cone_gates = sum(
        len(per_event_cones[event][0]) for event in candidate_events
    )
    candidate_total_cone_qubits = sum(
        len(per_event_cones[event][1]) for event in candidate_events
    )
    selected_total_extra_cone_qubits = sum(
        len(per_event_cones[event][1] - {event[1]}) for event in selected_events
    )
    return {
        "qmax": qmax,
        "partitioner": "batch_partition.partition_nodes",
        "partition_solver": partition_metadata["solver"],
        "partition_solver_sha256": stable_hash(partition_metadata["solver"]),
        "partition_metadata": partition_metadata,
        "partition_budget": {
            "refinement_node_limit": refinement_node_limit,
            "refinement_time_limit_seconds": float(
                refinement_time_limit_seconds
            ),
            "license_retry_backoff_budget_seconds": float(
                sum(GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
            ),
            "completion_reserve_seconds": (
                INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
            ),
        },
        "partition_certificate_kind": "structural_ancilla_width",
        "emitted_circuits_executed_or_width_checked": False,
        "operation_accounting": dict(INSTRUMENTATION_OPERATION_ACCOUNTING),
        "original_qubits": int(qc.num_qubits),
        "original_unitary_gate_count_m": original_unitary_gate_count_m,
        "monitorable_node_count": len(selected_events),
        "monitorable_gate_node_count": len(nodes),
        "candidate_monitor_event_count": len(candidate_events),
        "candidate_gate_node_count": len(events_by_gate),
        "individually_feasible_monitor_event_count": (
            individually_feasible_monitor_events
        ),
        "individually_feasible_gate_node_count": len(individually_feasible_nodes),
        "deployed_monitor_event_count": deployed_monitor_events,
        "deployed_gate_node_count": len(deployed_nodes),
        "terminally_measured_qubits": sorted(terminally_measured_qubits),
        "free_final_read_event_count": len(selected_events) - len(candidate_events),
        "planned_peak_extra_ancillas": planned_peak_extra_ancillas,
        "planned_cumulative_extra_ancilla_allocations": (
            planned_cumulative_extra_ancilla_allocations
        ),
        "planned_original_gate_executions": planned_original_gate_executions,
        "planned_replay_gate_executions": planned_replay_gate_executions,
        "planned_monitor_measurement_operations": (
            planned_monitor_measurement_operations
        ),
        "planned_monitor_reset_operations": planned_monitor_reset_operations,
        "planned_total_counted_operations": planned_total_counted_operations,
        "diagnostic_selected_per_event_cone_gate_occurrences": (
            selected_total_cone_gates
        ),
        "diagnostic_selected_per_event_cone_qubit_occurrences": (
            selected_total_cone_qubits
        ),
        "diagnostic_candidate_per_event_cone_gate_occurrences": (
            candidate_total_cone_gates
        ),
        "diagnostic_candidate_per_event_cone_qubit_occurrences": (
            candidate_total_cone_qubits
        ),
        "diagnostic_selected_per_event_extra_cone_qubit_occurrences": (
            selected_total_extra_cone_qubits
        ),
        "gate_node_costs": gate_node_costs,
        "individually_infeasible_gate_node_count": len(infeasible_set),
        "infeasible_gate_nodes": [int(node) for node in infeasible],
        "batch_count": len(serialized_batches),
        "maximum_run_qubits": max_run_qubits,
        "batches": serialized_batches,
    }


def _peak_rss_bytes() -> tuple[int | None, dict[str, Any]]:
    """Return the process high-water RSS with documented platform units."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = usage.ru_maxrss
    if not isinstance(raw, (int, float)) or raw < 0:
        return None, {
            "available": False,
            "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "reason": f"invalid ru_maxrss={raw!r}",
        }
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return int(raw * multiplier), {
        "available": True,
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "raw_value": raw,
        "raw_unit": "bytes" if multiplier == 1 else "KiB",
        "scope": (
            "isolated spawned attempt process high-water RSS through the end "
            "of selection; includes interpreter, circuit build, and MPS selection"
        ),
    }


def validate_circuit_eligibility(qc: Any) -> dict[str, Any]:
    """Enforce the closed, unconditioned 1/2-qubit unitary theorem domain."""

    from qiskit.circuit import ControlFlowOp
    from qiskit.quantum_info import Operator

    terminal_started = False
    terminal_seen: set[str] = set()
    unitary_count = 0
    terminal_count = 0
    for gate_index, instruction in enumerate(qc.data):
        operation = instruction.operation
        name = operation.name
        if isinstance(operation, ControlFlowOp):
            raise CircuitEligibilityError(
                "control_flow_operation",
                f"gate {gate_index} uses control-flow operation {name!r}",
            )
        condition = getattr(operation, "condition", None)
        condition_bits = tuple(getattr(operation, "condition_bits", ()) or ())
        if condition is not None or condition_bits:
            raise CircuitEligibilityError(
                "conditioned_operation",
                f"gate {gate_index} operation {name!r} is classically conditioned",
            )
        if name == "reset":
            raise CircuitEligibilityError(
                "reset_operation", f"gate {gate_index} is a reset operation"
            )
        if name in {"measure", "barrier"}:
            terminal_started = True
            terminal_seen.add(name)
            terminal_count += 1
            continue
        if terminal_started:
            category = (
                "mid_circuit_measurement"
                if "measure" in terminal_seen
                else "nonterminal_barrier"
            )
            raise CircuitEligibilityError(
                category,
                f"unitary candidate {name!r} at gate {gate_index} follows a "
                "terminal-only measurement/barrier",
            )
        if operation.num_qubits not in (1, 2):
            raise CircuitEligibilityError(
                (
                    "operation_arity_exceeds_two"
                    if operation.num_qubits > 2
                    else "operation_arity_not_one_or_two"
                ),
                f"gate {gate_index} operation {name!r} acts on "
                f"{operation.num_qubits} qubits",
            )
        try:
            Operator(operation)
        except Exception as exc:
            raise CircuitEligibilityError(
                "nonunitary_operation",
                f"gate {gate_index} operation {name!r} is not representable as "
                "a unitary",
            ) from exc
        unitary_count += 1
    return {
        "eligible": True,
        "rule": (
            "unconditioned 1/2-qubit unitaries followed only by terminal "
            "measurements/barriers"
        ),
        "unitary_operation_count": unitary_count,
        "terminal_measurement_or_barrier_count": terminal_count,
    }


def _canonical_complex(value: Any) -> dict[str, str]:
    number = complex(value)
    return {"real_hex": float(number.real).hex(), "imag_hex": float(number.imag).hex()}


def _circuit_canonical_sha256(qc: Any) -> str:
    """Hash eligible circuits without relying on a QASM exporter."""

    import numpy as np
    from qiskit.quantum_info import Operator

    instructions = []
    for instruction in qc.data:
        operation = instruction.operation
        item = {
            "name": operation.name,
            "operation_class": (
                f"{type(operation).__module__}.{type(operation).__qualname__}"
            ),
            "qubits": [qc.find_bit(bit).index for bit in instruction.qubits],
            "clbits": [qc.find_bit(bit).index for bit in instruction.clbits],
            "params": [_canonical_complex(value) for value in operation.params],
        }
        if operation.name not in {"measure", "barrier"}:
            matrix = np.asarray(Operator(operation).data, dtype="<c16", order="C")
            item["unitary_shape"] = list(matrix.shape)
            item["unitary_sha256"] = hashlib.sha256(matrix.tobytes()).hexdigest()
        instructions.append(item)
    value = {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "global_phase": _canonical_complex(qc.global_phase),
        "instructions": instructions,
    }
    return stable_hash(value)


def _build_circuit(spec: AttemptSpec) -> Any:
    from qiskit import QuantumCircuit, transpile

    if spec.source_kind == "canonical_qasm":
        if not spec.qasm_path or not spec.qasm_sha256:
            raise ProtocolError(f"base attempt lacks QASM provenance: {spec}")
        if file_sha256(spec.qasm_path) != spec.qasm_sha256:
            raise ProtocolError(f"QASM checksum mismatch for {spec.attempt_id}")
        return QuantumCircuit.from_qasm_file(spec.qasm_path)

    if spec.source_kind != "mqt_bench_extension":
        raise ProtocolError(f"unknown source kind {spec.source_kind!r}")
    if spec.family is None or spec.requested_qubits is None:
        raise ProtocolError(f"extension attempt lacks family/size: {spec}")

    from mqt.bench import BenchmarkLevel, get_benchmark

    try:
        circuit = get_benchmark(
            spec.family,
            BenchmarkLevel.INDEP,
            spec.requested_qubits,
            opt_level=0,
            random_parameters=True,
            **GENERATOR_KWARGS.get(spec.family, {}),
        )
    except (TypeError, ValueError) as exc:
        raise UnsupportedAttempt(str(exc)) from exc
    return transpile(
        circuit,
        basis_gates=list(TRANSPILE_BASIS),
        optimization_level=0,
        seed_transpiler=TRANSPILER_SEED,
    )


def check_q15_generator(family: str) -> dict[str, str]:
    """Build and hash one family for the narrow q15 reproducibility check."""

    if family not in EXTENSION_FAMILIES:
        raise ProtocolError(f"unknown extension family {family!r}")
    spec = AttemptSpec(
        attempt_id=f"extension:{family}:q15",
        attempt_index=EXTENSION_FAMILIES.index(family),
        source_kind="mqt_bench_extension",
        label=f"{family}_q15",
        family=family,
        requested_qubits=15,
    )
    try:
        circuit = _build_circuit(spec)
    except UnsupportedAttempt:
        return {"status": "unsupported", "category": "generator_rejected"}
    try:
        validate_circuit_eligibility(circuit)
    except CircuitEligibilityError as exc:
        return {"status": "unsupported", "category": exc.category}
    return {
        "status": "eligible",
        "sha256": _circuit_canonical_sha256(circuit),
    }


def _circuit_build_evidence(qc: Any, spec: AttemptSpec) -> dict[str, Any]:
    """Return actual post-build circuit facts safe to stream to the parent."""

    return {
        "evidence_stage": "build_complete",
        "original_qubits": int(qc.num_qubits),
        "original_m_operations": sum(
            1
            for instruction in qc.data
            if instruction.operation.name not in {"measure", "barrier"}
        ),
        "original_depth": int(qc.depth()),
        "total_instructions": len(qc.data),
        "canonical_circuit_sha256": None,
        "fingerprint_method": None,
        "requested_qubits": spec.requested_qubits,
        "eligibility": None,
    }


def _exception_evidence_sha256(exception_type: str, message: str) -> str:
    return stable_hash(
        {"exception_type": str(exception_type), "message": str(message)[:2000]}
    )


def _error_payload(
    phase: str,
    category: str,
    exc: BaseException,
) -> dict[str, Any]:
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    stored_trace = trace[-8000:]
    exception_type = type(exc).__name__
    message = str(exc)[:2000]
    return {
        "phase": phase,
        "category": category,
        "exception_type": exception_type,
        "message": message,
        "exception_evidence_sha256": _exception_evidence_sha256(
            exception_type, message
        ),
        "traceback": stored_trace,
        "traceback_sha256": hashlib.sha256(
            stored_trace.encode("utf-8")
        ).hexdigest(),
    }


def _deployment_status(
    circuit: Mapping[str, Any], instrumentation: Mapping[str, Any]
) -> str:
    if not isinstance(circuit, Mapping) or not isinstance(
        instrumentation, Mapping
    ):
        return "not_evaluated"
    original_qubits = circuit.get("original_qubits")
    if type(original_qubits) is not int or original_qubits < 0:
        return "not_evaluated"
    if original_qubits > CANONICAL_QMAX:
        return "source_width_exceeds_qmax"
    if instrumentation.get("candidate_gate_node_count") == 0:
        return "no_candidates"
    if instrumentation.get("individually_infeasible_gate_node_count", 0) > 0:
        return "partially_deployable"
    return "fully_deployable"


def selection_outcome(record: Mapping[str, Any]) -> str:
    """Derive the selector-stage outcome independently of instrumentation.

    A completed selection remains completed when the later instrumentation
    phase times out, exhausts resources, fails analysis, or encounters retryable
    infrastructure.  A selection-stage cross-check failure is not complete.
    """

    status = record.get("status")
    last_phase = record.get("last_phase")
    selection = record.get("selection")
    selection_finished = status == "success" or last_phase in {
        "selection_complete",
        "instrumentation",
        "complete",
    }
    if selection_finished:
        if isinstance(selection, Mapping):
            return "selector_complete"
        raise ProtocolError(
            "cannot derive selection outcome: post-selection phase lacks "
            "selection evidence"
        )
    if status in SELECTION_OUTCOMES - {"selector_complete"}:
        return str(status)
    raise ProtocolError(
        "cannot derive a final selection outcome from "
        f"status={status!r}, phase={last_phase!r}"
    )


def _outcome_semantic_errors(record: Mapping[str, Any]) -> list[str]:
    """Validate the status, phase, category, exception, and evidence contract."""

    errors: list[str] = []
    status = record.get("status")
    last_phase = record.get("last_phase")
    error = record.get("error")
    if status == "success":
        if last_phase != "complete":
            errors.append("success outcome must end in complete phase")
        return errors
    if not isinstance(error, Mapping):
        return errors

    phase = error.get("phase")
    category = error.get("category")
    exception_type = error.get("exception_type")
    message = error.get("message")
    if phase != last_phase:
        errors.append("error phase does not match last_phase")

    if status == "timeout":
        if phase not in RUNNING_PHASES or category != "wall_clock_timeout":
            errors.append("timeout outcome violates the phase/category contract")
        if exception_type != "ParentProcessOutcome":
            errors.append("wall-clock timeout lacks parent-process evidence")
    elif category == "worker_killed_resource_limit":
        if status != "resource_limit" or phase not in RUNNING_PHASES:
            errors.append("worker SIGKILL outcome violates the status/phase contract")
        if exception_type != "ParentProcessOutcome":
            errors.append("worker SIGKILL lacks parent-process evidence")
    elif category in {"worker_segmentation_fault", "worker_exit_without_record"}:
        if status != "analysis_error" or phase not in RUNNING_PHASES:
            errors.append("worker exit outcome violates the status/phase contract")
        if exception_type != "ParentProcessOutcome":
            errors.append("worker exit lacks parent-process evidence")
    else:
        allowed_by_phase = STATUS_PHASE_CATEGORIES.get(str(status), {})
        if category not in allowed_by_phase.get(str(phase), frozenset()):
            errors.append(
                "outcome violates status x phase x category semantics: "
                f"{status!r} x {phase!r} x {category!r}"
            )

    message_text = message if isinstance(message, str) else ""
    if category == "memory_error" and exception_type != "MemoryError":
        errors.append("memory_error category lacks MemoryError evidence")
    if category == "mps_backend_out_of_memory" and not (
        exception_type == "MPSBackendResourceLimit"
        or _has_any_marker(message_text, MPS_OUT_OF_MEMORY_MARKERS)
    ):
        errors.append("MPS out-of-memory category lacks backend memory evidence")
    if category == "mps_backend_numerical_failure" and not (
        exception_type == "MPSBackendNumericalFailure"
        or _has_any_marker(message_text, MPS_NUMERICAL_FAILURE_MARKERS)
    ):
        errors.append("MPS numerical category lacks numerical backend evidence")
    if category == "gurobi_license_infrastructure_failure" and not _has_any_marker(
        message_text, GUROBI_INFRASTRUCTURE_MARKERS
    ):
        errors.append("Gurobi infrastructure category lacks license evidence")
    if category == "gurobi_license_infrastructure_failure" and exception_type not in {
        "GurobiError",
        "PulpSolverError",
        "RuntimeError",
    }:
        errors.append("Gurobi infrastructure category has an invalid exception type")

    circuit = record.get("circuit")
    selection = record.get("selection")
    instrumentation = record.get("instrumentation")
    if isinstance(selection, Mapping):
        if not isinstance(circuit, Mapping) or circuit.get("evidence_stage") != (
            "eligibility_fingerprint_complete"
        ):
            errors.append("selection evidence lacks completed circuit checksum")
        elif (circuit.get("eligibility") or {}).get("eligible") is not True:
            errors.append("selection evidence lacks passing eligibility")
    if instrumentation is not None and status != "success":
        errors.append("non-success outcome unexpectedly contains instrumentation")
    if status == "unsupported" and phase == "eligibility":
        eligibility = circuit.get("eligibility") if isinstance(circuit, Mapping) else None
        if (
            not isinstance(circuit, Mapping)
            or circuit.get("evidence_stage") != "eligibility_rejected"
            or not isinstance(eligibility, Mapping)
            or eligibility.get("eligible") is not False
            or eligibility.get("category") != category
        ):
            errors.append("eligibility rejection lacks matching circuit evidence")
    if phase in {"selection_complete", "instrumentation"} and not isinstance(
        selection, Mapping
    ):
        errors.append(f"{phase} outcome lacks completed selection evidence")
    if status == "infrastructure_error":
        errors.append(
            "retryable infrastructure outcome cannot be a final terminal"
        )
    return errors


def _completion_errors(record: Mapping[str, Any], *, require_marker: bool) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "attempt_id",
        "attempt_index",
        "source",
        "qmax",
        "status",
        "status_semantics",
        "deployment_status",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "phase_elapsed_seconds",
        "last_phase",
        "fingerprints",
        "provenance",
        "run",
    )
    for field in required:
        if field not in record or record[field] is None:
            errors.append(f"missing required field {field}")
    if "error_category" not in record:
        errors.append("missing required field error_category")
    if require_marker and record.get("record_complete") is not True:
        errors.append("record_complete is not true")
    if record.get("status") not in OUTCOME_STATUSES:
        errors.append(f"invalid status {record.get('status')!r}")
    if record.get("status_semantics") != "analysis_terminal_status":
        errors.append("status_semantics is not analysis_terminal_status")
    if record.get("deployment_status") not in DEPLOYMENT_STATUSES:
        errors.append(
            f"invalid deployment_status {record.get('deployment_status')!r}"
        )
    if record.get("status") == "success":
        if record.get("error") is not None:
            errors.append("success record has an error")
        if record.get("error_category") is not None:
            errors.append("success record has an error_category")
        for field in ("circuit", "selection", "instrumentation"):
            if not isinstance(record.get(field), dict):
                errors.append(f"success record lacks {field}")
        if record.get("deployment_status") == "not_evaluated":
            errors.append("success record has no deployment outcome")
    else:
        if record.get("deployment_status") != "not_evaluated":
            errors.append("non-success analysis record has a deployment outcome")
        error = record.get("error")
        if not isinstance(error, dict):
            errors.append("non-success record lacks structured error")
        else:
            for field in (
                "phase",
                "category",
                "exception_type",
                "message",
                "exception_evidence_sha256",
                "traceback",
                "traceback_sha256",
            ):
                if field not in error:
                    errors.append(f"error lacks {field}")
            if record.get("error_category") != error.get("category"):
                errors.append("error_category does not match error.category")
    errors.extend(_outcome_semantic_errors(record))
    return errors


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return stable_hash(payload)


def _add_record_checksum(record: dict[str, Any]) -> dict[str, Any]:
    record["record_sha256"] = _record_sha256(record)
    return record


def _finish_record(record: dict[str, Any], started: float) -> dict[str, Any]:
    record["completed_at_utc"] = _utc_now()
    record["elapsed_seconds"] = round(time.perf_counter() - started, 9)
    record["error_category"] = (record.get("error") or {}).get("category")
    record["record_complete"] = not _completion_errors(record, require_marker=False)
    return _add_record_checksum(record)


def evaluate_attempt(
    spec: AttemptSpec,
    bundle: Mapping[str, Any],
    *,
    run_mode: str,
    shard_index: int,
    shard_count: int,
    phase_callback: Callable[[str], None] | None = None,
    circuit_evidence_callback: Callable[[Mapping[str, Any]], None] | None = None,
    selection_evidence_callback: Callable[[Mapping[str, Any]], None] | None = None,
    attempt_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Evaluate one attempt inside an isolated worker process."""

    started = time.perf_counter()
    record = _record_skeleton(
        spec,
        bundle,
        run_mode=run_mode,
        shard_index=shard_index,
        shard_count=shard_count,
        attempt_timeout_seconds=attempt_timeout_seconds,
    )

    def start_phase(phase: str) -> float:
        record["last_phase"] = phase
        if phase_callback is not None:
            phase_callback(phase)
        return time.perf_counter()

    phase_started = start_phase("build")
    try:
        qc = _build_circuit(spec)
    except UnsupportedAttempt as exc:
        record["phase_elapsed_seconds"]["build"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "unsupported"
        record["error"] = _error_payload("build", "generator_rejected", exc)
        return _finish_record(record, started)
    except MemoryError as exc:
        record["phase_elapsed_seconds"]["build"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "resource_limit"
        record["error"] = _error_payload("build", "memory_error", exc)
        return _finish_record(record, started)
    except Exception as exc:
        record["phase_elapsed_seconds"]["build"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "build_error"
        record["error"] = _error_payload("build", "circuit_build_failed", exc)
        return _finish_record(record, started)

    record["circuit"] = _circuit_build_evidence(qc, spec)
    if circuit_evidence_callback is not None:
        circuit_evidence_callback(dict(record["circuit"]))
    record["phase_elapsed_seconds"]["build"] = round(
        time.perf_counter() - phase_started, 9
    )

    phase_started = start_phase("eligibility")
    try:
        eligibility = validate_circuit_eligibility(qc)
        record["circuit"]["eligibility"] = eligibility
        record["circuit"]["evidence_stage"] = "eligibility_complete"
        if circuit_evidence_callback is not None:
            circuit_evidence_callback(dict(record["circuit"]))
        record["circuit"]["canonical_circuit_sha256"] = (
            _circuit_canonical_sha256(qc)
        )
        record["circuit"]["fingerprint_method"] = (
            "canonical instruction stream plus per-unitary matrix SHA-256"
        )
        record["circuit"]["evidence_stage"] = "eligibility_fingerprint_complete"
        if circuit_evidence_callback is not None:
            circuit_evidence_callback(dict(record["circuit"]))
    except CircuitEligibilityError as exc:
        record["phase_elapsed_seconds"]["eligibility"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["circuit"]["eligibility"] = {
            "eligible": False,
            "category": exc.category,
            "message": str(exc),
        }
        record["circuit"]["evidence_stage"] = "eligibility_rejected"
        if circuit_evidence_callback is not None:
            circuit_evidence_callback(dict(record["circuit"]))
        record["status"] = "unsupported"
        record["error"] = _error_payload("eligibility", exc.category, exc)
        return _finish_record(record, started)
    except MemoryError as exc:
        record["phase_elapsed_seconds"]["eligibility"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "resource_limit"
        record["error"] = _error_payload("eligibility", "memory_error", exc)
        return _finish_record(record, started)
    except Exception as exc:
        record["phase_elapsed_seconds"]["eligibility"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "analysis_error"
        record["error"] = _error_payload(
            "eligibility", "eligibility_or_fingerprint_failed", exc
        )
        return _finish_record(record, started)
    record["phase_elapsed_seconds"]["eligibility"] = round(
        time.perf_counter() - phase_started, 9
    )

    rss_before, rss_method = _peak_rss_bytes()
    phase_started = start_phase("selection")
    selection_subphase = "mps"
    mps_started = time.perf_counter()
    crosscheck = {
        "required": False,
        "performed": False,
        "passed": None,
        "reason": "extension attempts are outside the <=14q canonical base anchor",
    }
    try:
        nodes, mps_diagnostics, mps_scores = (
            _select_monitorable_nodes_mps_details(qc)
        )
        mps_runtime = time.perf_counter() - mps_started
        mps_peak_rss, mps_rss_method = _peak_rss_bytes()
        if spec.source_kind == "canonical_qasm":
            selection_subphase = "statevector_crosscheck"
            crosscheck = crosscheck_mps_against_statevector(
                qc, nodes, mps_scores
            )
    except MPSBackendResourceLimit as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "resource_limit"
        record["error"] = _error_payload(
            "selection", "mps_backend_out_of_memory", exc
        )
        return _finish_record(record, started)
    except MemoryError as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "resource_limit"
        record["error"] = _error_payload("selection", "memory_error", exc)
        return _finish_record(record, started)
    except MPSBackendNumericalFailure as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "analysis_error"
        record["error"] = _error_payload(
            "selection", "mps_backend_numerical_failure", exc
        )
        return _finish_record(record, started)
    except MPSBackendFailure as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "analysis_error"
        record["error"] = _error_payload(
            "selection", "mps_backend_failure", exc
        )
        return _finish_record(record, started)
    except AnalysisFailure as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "analysis_error"
        category = (
            "mps_native_schmidt_validation_failure"
            if selection_subphase == "mps"
            else "statevector_crosscheck_failure"
        )
        record["error"] = _error_payload("selection", category, exc)
        return _finish_record(record, started)
    except Exception as exc:
        record["phase_elapsed_seconds"]["selection"] = round(
            time.perf_counter() - phase_started, 9
        )
        if selection_subphase == "mps" and _is_mps_out_of_memory_error(exc):
            record["status"] = "resource_limit"
            category = "mps_backend_out_of_memory"
        elif selection_subphase == "mps" and _is_mps_numerical_error(exc):
            record["status"] = "analysis_error"
            category = "mps_backend_numerical_failure"
        else:
            record["status"] = "analysis_error"
            category = "mps_analysis_failed"
        record["error"] = _error_payload("selection", category, exc)
        return _finish_record(record, started)

    selection_elapsed = time.perf_counter() - phase_started
    total_peak_rss, total_rss_method = _peak_rss_bytes()
    mps_rss_details = mps_rss_method if mps_peak_rss is not None else rss_method
    serialized_nodes = _serialize_monitorable_nodes(nodes)
    serialized_scores = _serialize_point_scores(mps_scores)
    record["phase_elapsed_seconds"]["selection"] = round(selection_elapsed, 9)
    record["selection"] = {
        "mps_runtime_seconds": round(mps_runtime, 9),
        "mps_peak_rss_bytes": mps_peak_rss,
        "mps_rss_measurement": mps_rss_details,
        "pre_selection_high_water_rss_bytes": rss_before,
        "validation_runtime_seconds": (
            crosscheck.get("runtime_seconds") if crosscheck["required"] else 0.0
        ),
        "total_selection_certificate_runtime_seconds": round(
            selection_elapsed, 9
        ),
        "total_selection_certificate_peak_rss_bytes": total_peak_rss,
        "total_rss_measurement": total_rss_method,
        "monitorable_node_count": sum(len(qubits) for qubits in nodes.values()),
        "monitorable_gate_node_count": len(nodes),
        "monitorable_nodes": serialized_nodes,
        "monitorable_nodes_sha256": stable_hash(serialized_nodes),
        "point_scores": serialized_scores,
        "point_scores_sha256": stable_hash(serialized_scores),
        "point_score_definition": {
            "selection": "second_schmidt_amplitude_a2",
            "selection_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
            "a2_extraction": "aer_mps_local_tensor_svd",
            "determinant_role": "physical_diagnostic_only",
        },
        "mps": mps_diagnostics,
        "statevector_crosscheck": crosscheck,
    }
    if selection_evidence_callback is not None:
        selection_evidence_callback(
            {
                "selection": record["selection"],
                "phase_elapsed_seconds": dict(record["phase_elapsed_seconds"]),
            }
        )

    if crosscheck["required"] and crosscheck["passed"] is not True:
        record["status"] = "analysis_error"
        exc = AnalysisFailure(
            "MPS/statevector a2 selection cross-check did not pass: "
            f"selection_mismatches={crosscheck['selection_mismatch_count']}, "
            f"missing_mps_points={crosscheck['missing_mps_point_count']}, "
            "missing_statevector_points="
            f"{crosscheck['missing_statevector_point_count']}, "
            "diagnostic_max_abs_a2_difference="
            f"{crosscheck['maximum_absolute_a2_difference']}"
        )
        record["error"] = _error_payload(
            "selection", "statevector_crosscheck_mismatch", exc
        )
        return _finish_record(record, started)

    phase_started = start_phase("instrumentation")
    try:
        instrumentation = build_instrumentation_metrics(
            qc,
            nodes,
            attempt_started=started,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
    except MemoryError as exc:
        record["phase_elapsed_seconds"]["instrumentation"] = round(
            time.perf_counter() - phase_started, 9
        )
        record["status"] = "resource_limit"
        record["error"] = _error_payload("instrumentation", "memory_error", exc)
        return _finish_record(record, started)
    except Exception as exc:
        record["phase_elapsed_seconds"]["instrumentation"] = round(
            time.perf_counter() - phase_started, 9
        )
        if _is_gurobi_infrastructure_error(exc):
            record["status"] = "infrastructure_error"
            category = "gurobi_license_infrastructure_failure"
        else:
            record["status"] = "analysis_error"
            category = "instrumentation_analysis_failed"
        record["error"] = _error_payload("instrumentation", category, exc)
        return _finish_record(record, started)

    instrumentation_elapsed = time.perf_counter() - phase_started
    record["phase_elapsed_seconds"]["instrumentation"] = round(
        instrumentation_elapsed, 9
    )
    instrumentation["runtime_seconds"] = round(instrumentation_elapsed, 9)
    record["instrumentation"] = instrumentation

    record["status"] = "success"
    record["deployment_status"] = _deployment_status(
        record["circuit"], instrumentation
    )
    record["error"] = None
    record["last_phase"] = "complete"
    return _finish_record(record, started)


def _worker_entry(
    spec_value: Mapping[str, Any],
    bundle: Mapping[str, Any],
    run_mode: str,
    shard_index: int,
    shard_count: int,
    connection: Any,
    attempt_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    spec = AttemptSpec(**dict(spec_value))
    latest_circuit_evidence: dict[str, Any] | None = None
    latest_selection_evidence: dict[str, Any] | None = None
    latest_phase_elapsed_seconds: dict[str, float] = {}

    def report_phase(phase: str) -> None:
        connection.send({"kind": "phase", "phase": phase})

    def report_circuit_evidence(evidence: Mapping[str, Any]) -> None:
        nonlocal latest_circuit_evidence
        latest_circuit_evidence = dict(evidence)
        connection.send({"kind": "circuit_evidence", "circuit": dict(evidence)})

    def report_selection_evidence(evidence: Mapping[str, Any]) -> None:
        nonlocal latest_selection_evidence, latest_phase_elapsed_seconds
        latest_selection_evidence = dict(evidence["selection"])
        latest_phase_elapsed_seconds = dict(evidence["phase_elapsed_seconds"])
        connection.send(
            {
                "kind": "selection_evidence",
                "selection": latest_selection_evidence,
                "phase_elapsed_seconds": latest_phase_elapsed_seconds,
            }
        )

    try:
        record = evaluate_attempt(
            spec,
            bundle,
            run_mode=run_mode,
            shard_index=shard_index,
            shard_count=shard_count,
            phase_callback=report_phase,
            circuit_evidence_callback=report_circuit_evidence,
            selection_evidence_callback=report_selection_evidence,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
        connection.send({"kind": "record", "record": record})
    except BaseException as exc:
        started = time.perf_counter()
        record = _record_skeleton(
            spec,
            bundle,
            run_mode=run_mode,
            shard_index=shard_index,
            shard_count=shard_count,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
        if _is_gurobi_infrastructure_error(exc):
            record["status"] = "infrastructure_error"
            category = "gurobi_license_infrastructure_failure"
            phase = "instrumentation"
        else:
            record["status"] = "analysis_error"
            category = "uncaught_worker_exception"
            phase = "worker_boundary"
        record["last_phase"] = phase
        if latest_circuit_evidence is not None:
            record["circuit"] = latest_circuit_evidence
        if latest_selection_evidence is not None:
            record["selection"] = latest_selection_evidence
        record["phase_elapsed_seconds"] = latest_phase_elapsed_seconds
        record["error"] = _error_payload(
            phase, category, exc
        )
        connection.send({"kind": "record", "record": _finish_record(record, started)})
    finally:
        connection.close()


def _parent_failure_record(
    spec: AttemptSpec,
    bundle: Mapping[str, Any],
    *,
    run_mode: str,
    shard_index: int,
    shard_count: int,
    status: str,
    phase: str,
    category: str,
    message: str,
    elapsed: float,
    circuit_evidence: Mapping[str, Any] | None = None,
    selection_evidence: Mapping[str, Any] | None = None,
    phase_elapsed_seconds: Mapping[str, float] | None = None,
    attempt_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    record = _record_skeleton(
        spec,
        bundle,
        run_mode=run_mode,
        shard_index=shard_index,
        shard_count=shard_count,
        attempt_timeout_seconds=attempt_timeout_seconds,
    )
    record["status"] = status
    record["last_phase"] = phase
    if circuit_evidence is not None:
        record["circuit"] = dict(circuit_evidence)
    if selection_evidence is not None:
        record["selection"] = dict(selection_evidence)
    if phase_elapsed_seconds is not None:
        record["phase_elapsed_seconds"] = dict(phase_elapsed_seconds)
    exception_type = "ParentProcessOutcome"
    record["error"] = {
        "phase": phase,
        "category": category,
        "exception_type": exception_type,
        "message": message,
        "exception_evidence_sha256": _exception_evidence_sha256(
            exception_type, message
        ),
        "traceback": "",
        "traceback_sha256": hashlib.sha256(b"").hexdigest(),
    }
    record["error_category"] = category
    record["completed_at_utc"] = _utc_now()
    record["elapsed_seconds"] = round(elapsed, 9)
    record["record_complete"] = not _completion_errors(record, require_marker=False)
    return _add_record_checksum(record)


def _worker_exit_outcome(exitcode: int | None) -> tuple[str, str]:
    if exitcode == -signal.SIGKILL:
        return "resource_limit", "worker_killed_resource_limit"
    if exitcode == -signal.SIGSEGV:
        return "analysis_error", "worker_segmentation_fault"
    return "analysis_error", "worker_exit_without_record"


def _run_attempt_once_with_timeout(
    spec: AttemptSpec,
    bundle: Mapping[str, Any],
    *,
    timeout_seconds: float,
    run_mode: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ProtocolError("timeout_seconds must be positive")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            asdict(spec),
            dict(bundle),
            run_mode,
            shard_index,
            shard_count,
            child,
            timeout_seconds,
        ),
    )
    started = time.perf_counter()
    last_phase = "worker_start"
    circuit_evidence = None
    selection_evidence = None
    phase_elapsed_seconds: dict[str, float] = {}
    record = None
    process.start()
    child.close()
    deadline = started + timeout_seconds
    try:
        while time.perf_counter() < deadline:
            remaining = max(0.0, deadline - time.perf_counter())
            if parent.poll(min(0.2, remaining)):
                try:
                    message = parent.recv()
                except EOFError:
                    break
                if message.get("kind") == "phase":
                    last_phase = message["phase"]
                elif message.get("kind") == "circuit_evidence":
                    circuit_evidence = message.get("circuit")
                elif message.get("kind") == "selection_evidence":
                    selection_evidence = message.get("selection")
                    phase_elapsed_seconds = dict(
                        message.get("phase_elapsed_seconds") or {}
                    )
                    if last_phase == "selection":
                        last_phase = "selection_complete"
                elif message.get("kind") == "record":
                    record = message["record"]
                    break
            if not process.is_alive() and not parent.poll():
                break
    finally:
        parent.close()

    if record is not None:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        return record

    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return _parent_failure_record(
            spec,
            bundle,
            run_mode=run_mode,
            shard_index=shard_index,
            shard_count=shard_count,
            status="timeout",
            phase=last_phase,
            category="wall_clock_timeout",
            message=f"attempt exceeded {timeout_seconds} seconds",
            elapsed=elapsed,
            circuit_evidence=circuit_evidence,
            selection_evidence=selection_evidence,
            phase_elapsed_seconds=phase_elapsed_seconds,
            attempt_timeout_seconds=timeout_seconds,
        )

    exitcode = process.exitcode
    status, category = _worker_exit_outcome(exitcode)
    return _parent_failure_record(
        spec,
        bundle,
        run_mode=run_mode,
        shard_index=shard_index,
        shard_count=shard_count,
        status=status,
        phase=last_phase,
        category=category,
        message=f"worker exited with code {exitcode} without a record",
        elapsed=elapsed,
        circuit_evidence=circuit_evidence,
        selection_evidence=selection_evidence,
        phase_elapsed_seconds=phase_elapsed_seconds,
        attempt_timeout_seconds=timeout_seconds,
    )


def run_attempt_with_timeout(
    spec: AttemptSpec,
    bundle: Mapping[str, Any],
    *,
    timeout_seconds: float,
    run_mode: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Run an attempt, retrying only explicitly transient infrastructure errors."""

    record: dict[str, Any] | None = None
    retry_delays = (0.0,) + IMMEDIATE_INFRASTRUCTURE_RETRY_DELAYS_SECONDS
    for retry_index, delay in enumerate(retry_delays):
        if delay:
            time.sleep(delay)
        record = _run_attempt_once_with_timeout(
            spec,
            bundle,
            timeout_seconds=timeout_seconds,
            run_mode=run_mode,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        record["run"]["infrastructure_retries_before_outcome"] = retry_index
        _add_record_checksum(record)
        if record.get("status") != "infrastructure_error":
            return record
    assert record is not None
    return record


def atomic_write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    f"{source}:{line_number}: invalid JSONL record: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ProtocolError(f"{source}:{line_number}: record is not an object")
            records.append(value)
    return records


def atomic_write_json_object(value: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{source}: invalid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{source}: expected one JSON object")
    return value


FIGURE_ANALYSIS_STATUS_DENOMINATOR_FIELDS = tuple(
    f"analysis_status_{status}_attempt_denominator" for status in STATUS_ORDER
)
FIGURE_SELECTION_OUTCOME_DENOMINATOR_FIELDS = tuple(
    f"selection_outcome_{outcome}_attempt_denominator"
    for outcome in SELECTION_OUTCOME_ORDER
)
FIGURE_DEPLOYMENT_STATUS_DENOMINATOR_FIELDS = tuple(
    f"deployment_status_{status}_attempt_denominator"
    for status in DEPLOYMENT_STATUS_ORDER
)

FIGURE_DATA_FIELDS = (
    "schema_version",
    "attempt_id",
    "attempt_index",
    "attempted",
    "fingerprinted",
    "attempt_denominator",
    "fingerprinted_attempt_denominator",
    "hash_distinct_circuit_denominator",
) + FIGURE_SELECTION_OUTCOME_DENOMINATOR_FIELDS + (
    *FIGURE_ANALYSIS_STATUS_DENOMINATOR_FIELDS,
    *FIGURE_DEPLOYMENT_STATUS_DENOMINATOR_FIELDS,
    "circuit_sha256",
    "duplicate_cluster_id",
    "duplicate_cluster_size",
    "hash_distinct_circuit_representative",
    "inverse_exact_hash_cluster_weight",
    "source_kind",
    "family",
    "requested_qubits",
    "original_qubits",
    "original_m_operations",
    "original_depth",
    "selection_outcome",
    "analysis_status",
    "deployment_status",
    "error_category",
    "error_phase",
    "qmax",
    "partition_solver_backend",
    "partition_solver_provenance_sha256",
    "selection_available",
    "mps_runtime_seconds",
    "mps_peak_rss_bytes",
    "observed_max_chi",
    "validation_runtime_seconds",
    "monitorable_node_count",
    "monitorable_gate_node_count",
    "instrumentation_available",
    "candidate_monitor_event_count",
    "candidate_gate_node_count",
    "individually_feasible_monitor_event_count",
    "individually_feasible_gate_node_count",
    "deployed_monitor_event_count",
    "deployed_gate_node_count",
    "individually_infeasible_gate_node_count",
    "batch_count",
    "maximum_run_qubits",
    "planned_peak_extra_ancillas",
    "planned_cumulative_extra_ancilla_allocations",
    "planned_original_gate_executions",
    "planned_replay_gate_executions",
    "planned_monitor_measurement_operations",
    "planned_monitor_reset_operations",
    "planned_total_counted_operations",
    "diagnostic_selected_per_event_cone_gate_occurrences",
    "diagnostic_selected_per_event_cone_qubit_occurrences",
    "diagnostic_candidate_per_event_cone_gate_occurrences",
    "diagnostic_candidate_per_event_cone_qubit_occurrences",
    "diagnostic_selected_per_event_extra_cone_qubit_occurrences",
)


def circuit_duplicate_clusters(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        circuit = record.get("circuit") or {}
        circuit_hash = circuit.get("canonical_circuit_sha256")
        if _is_sha256(circuit_hash):
            by_hash.setdefault(circuit_hash, []).append(record)

    output: dict[str, dict[str, Any]] = {}
    for circuit_hash, members in sorted(by_hash.items()):
        ordered = sorted(members, key=lambda member: int(member["attempt_index"]))
        size = len(ordered)
        representative = str(ordered[0]["attempt_id"])
        for member in ordered:
            output[str(member["attempt_id"])] = {
                "circuit_sha256": circuit_hash,
                "duplicate_cluster_id": circuit_hash,
                "duplicate_cluster_size": size,
                "hash_distinct_circuit_representative": int(
                    str(member["attempt_id"]) == representative
                ),
                "inverse_exact_hash_cluster_weight": 1.0 / size,
            }
    return output


def artifact_denominators(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    clusters = circuit_duplicate_clusters(records)
    selection_outcome_counts = Counter(selection_outcome(record) for record in records)
    analysis_status_counts = Counter(record.get("status") for record in records)
    deployment_status_counts = Counter(
        record.get("deployment_status") for record in records
    )
    sizes_by_hash: dict[str, int] = {}
    for metadata in clusters.values():
        sizes_by_hash[metadata["circuit_sha256"]] = metadata[
            "duplicate_cluster_size"
        ]
    denominators = {
        "attempts": len(records),
        "fingerprinted_attempts": len(clusters),
        "hash_distinct_circuits": len(sizes_by_hash),
        "exact_duplicate_clusters": sum(
            1 for size in sizes_by_hash.values() if size > 1
        ),
        "duplicate_attempt_excess": sum(
            size - 1 for size in sizes_by_hash.values() if size > 1
        ),
    }
    denominators.update(
        {
            f"selection_outcome_{outcome}_attempts": (
                selection_outcome_counts.get(outcome, 0)
            )
            for outcome in SELECTION_OUTCOME_ORDER
        }
    )
    denominators.update(
        {
            f"analysis_status_{status}_attempts": analysis_status_counts.get(
                status, 0
            )
            for status in STATUS_ORDER
        }
    )
    denominators.update(
        {
            f"deployment_status_{status}_attempts": (
                deployment_status_counts.get(status, 0)
            )
            for status in DEPLOYMENT_STATUS_ORDER
        }
    )
    return denominators


def figure_data_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one figure-data row per attempt, including every non-success."""

    rows = []
    duplicate_metadata = circuit_duplicate_clusters(records)
    denominators = artifact_denominators(records)
    exported_denominators = {
        "attempt_denominator": denominators["attempts"],
        "fingerprinted_attempt_denominator": denominators[
            "fingerprinted_attempts"
        ],
        "hash_distinct_circuit_denominator": denominators[
            "hash_distinct_circuits"
        ],
        **{
            f"selection_outcome_{outcome}_attempt_denominator": denominators[
                f"selection_outcome_{outcome}_attempts"
            ]
            for outcome in SELECTION_OUTCOME_ORDER
        },
        **{
            f"analysis_status_{status}_attempt_denominator": denominators[
                f"analysis_status_{status}_attempts"
            ]
            for status in STATUS_ORDER
        },
        **{
            f"deployment_status_{status}_attempt_denominator": denominators[
                f"deployment_status_{status}_attempts"
            ]
            for status in DEPLOYMENT_STATUS_ORDER
        },
    }
    for record in sorted(records, key=lambda value: int(value["attempt_index"])):
        source = record.get("source") or {}
        circuit = record.get("circuit") or {}
        selection = record.get("selection") or {}
        mps = selection.get("mps") or {}
        instrumentation = record.get("instrumentation") or {}
        protocol_config = (record.get("provenance") or {}).get(
            "protocol_config"
        ) or {}
        partition_solver = protocol_config.get("partition_solver") or {}
        basename = source.get("basename")
        family = source.get("family")
        if family is None and isinstance(basename, str):
            family = basename.rsplit("_indep_qiskit_", 1)[0]
        duplicate = duplicate_metadata.get(str(record.get("attempt_id")), {})
        rows.append(
            {
                "schema_version": record.get("schema_version"),
                "attempt_id": record.get("attempt_id"),
                "attempt_index": record.get("attempt_index"),
                "attempted": 1,
                "fingerprinted": int(bool(duplicate)),
                **exported_denominators,
                "circuit_sha256": duplicate.get("circuit_sha256"),
                "duplicate_cluster_id": duplicate.get("duplicate_cluster_id"),
                "duplicate_cluster_size": duplicate.get("duplicate_cluster_size"),
                "hash_distinct_circuit_representative": duplicate.get(
                    "hash_distinct_circuit_representative"
                ),
                "inverse_exact_hash_cluster_weight": duplicate.get(
                    "inverse_exact_hash_cluster_weight"
                ),
                "source_kind": source.get("source_kind"),
                "family": family,
                "requested_qubits": source.get("requested_qubits"),
                # This is the only circuit-size x-axis for measured metrics.
                "original_qubits": circuit.get("original_qubits"),
                "original_m_operations": circuit.get("original_m_operations"),
                "original_depth": circuit.get("original_depth"),
                "selection_outcome": selection_outcome(record),
                "analysis_status": record.get("status"),
                "deployment_status": record.get("deployment_status"),
                "error_category": record.get("error_category"),
                "error_phase": (record.get("error") or {}).get("phase"),
                "qmax": record.get("qmax"),
                "partition_solver_backend": partition_solver.get("backend"),
                "partition_solver_provenance_sha256": protocol_config.get(
                    "partition_solver_sha256"
                ),
                "selection_available": int(bool(selection)),
                "mps_runtime_seconds": selection.get("mps_runtime_seconds"),
                "mps_peak_rss_bytes": selection.get("mps_peak_rss_bytes"),
                "observed_max_chi": mps.get("observed_max_chi"),
                "validation_runtime_seconds": selection.get(
                    "validation_runtime_seconds"
                ),
                "monitorable_node_count": selection.get("monitorable_node_count"),
                "monitorable_gate_node_count": selection.get(
                    "monitorable_gate_node_count"
                ),
                "instrumentation_available": int(bool(instrumentation)),
                "candidate_monitor_event_count": instrumentation.get(
                    "candidate_monitor_event_count"
                ),
                "candidate_gate_node_count": instrumentation.get(
                    "candidate_gate_node_count"
                ),
                "individually_feasible_monitor_event_count": instrumentation.get(
                    "individually_feasible_monitor_event_count"
                ),
                "individually_feasible_gate_node_count": instrumentation.get(
                    "individually_feasible_gate_node_count"
                ),
                "deployed_monitor_event_count": instrumentation.get(
                    "deployed_monitor_event_count"
                ),
                "deployed_gate_node_count": instrumentation.get(
                    "deployed_gate_node_count"
                ),
                "individually_infeasible_gate_node_count": instrumentation.get(
                    "individually_infeasible_gate_node_count"
                ),
                "batch_count": instrumentation.get("batch_count"),
                "maximum_run_qubits": instrumentation.get("maximum_run_qubits"),
                "planned_peak_extra_ancillas": instrumentation.get(
                    "planned_peak_extra_ancillas"
                ),
                "planned_cumulative_extra_ancilla_allocations": (
                    instrumentation.get(
                        "planned_cumulative_extra_ancilla_allocations"
                    )
                ),
                "planned_original_gate_executions": instrumentation.get(
                    "planned_original_gate_executions"
                ),
                "planned_replay_gate_executions": instrumentation.get(
                    "planned_replay_gate_executions"
                ),
                "planned_monitor_measurement_operations": instrumentation.get(
                    "planned_monitor_measurement_operations"
                ),
                "planned_monitor_reset_operations": instrumentation.get(
                    "planned_monitor_reset_operations"
                ),
                "planned_total_counted_operations": instrumentation.get(
                    "planned_total_counted_operations"
                ),
                "diagnostic_selected_per_event_cone_gate_occurrences": (
                    instrumentation.get(
                        "diagnostic_selected_per_event_cone_gate_occurrences"
                    )
                ),
                "diagnostic_selected_per_event_cone_qubit_occurrences": (
                    instrumentation.get(
                        "diagnostic_selected_per_event_cone_qubit_occurrences"
                    )
                ),
                "diagnostic_candidate_per_event_cone_gate_occurrences": (
                    instrumentation.get(
                        "diagnostic_candidate_per_event_cone_gate_occurrences"
                    )
                ),
                "diagnostic_candidate_per_event_cone_qubit_occurrences": (
                    instrumentation.get(
                        "diagnostic_candidate_per_event_cone_qubit_occurrences"
                    )
                ),
                "diagnostic_selected_per_event_extra_cone_qubit_occurrences": (
                    instrumentation.get(
                        "diagnostic_selected_per_event_extra_cone_qubit_occurrences"
                    )
                ),
            }
        )
    return rows


def atomic_write_figure_csv(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIGURE_DATA_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_batch_feasibility(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    instrumentation = record.get("instrumentation")
    if instrumentation is None:
        return errors
    if not isinstance(instrumentation, dict):
        return ["instrumentation is not an object"]
    if instrumentation.get("qmax") != CANONICAL_QMAX:
        errors.append(
            f"instrumentation qmax={instrumentation.get('qmax')!r}, expected 24"
        )
    if instrumentation.get("partitioner") != "batch_partition.partition_nodes":
        errors.append("instrumentation names an unknown partitioner")
    partition_solver = instrumentation.get("partition_solver")
    if not isinstance(partition_solver, dict):
        errors.append("instrumentation lacks partition solver provenance")
        partition_solver = {}
    elif partition_solver.get("backend") not in {"gurobi", "cbc"}:
        errors.append("instrumentation records an invalid partition solver backend")
    if instrumentation.get("partition_solver_sha256") != stable_hash(
        partition_solver
    ):
        errors.append("instrumentation partition solver provenance hash mismatch")
    partition_metadata = instrumentation.get("partition_metadata")
    if not isinstance(partition_metadata, dict):
        errors.append("instrumentation lacks partition metadata")
        partition_metadata = {}
    elif partition_metadata.get("solver") != partition_solver:
        errors.append("partition metadata solver differs from solver provenance")
    partition_budget = instrumentation.get("partition_budget")
    if not isinstance(partition_budget, dict):
        errors.append("instrumentation lacks partition budget evidence")
        partition_budget = {}
    if partition_budget.get("license_retry_backoff_budget_seconds") != float(
        sum(GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
    ):
        errors.append("partition license-retry budget differs from protocol")
    if partition_budget.get("completion_reserve_seconds") != (
        INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
    ):
        errors.append("partition completion reserve differs from protocol")
    if instrumentation.get("partition_certificate_kind") != (
        "structural_ancilla_width"
    ):
        errors.append("instrumentation lacks the structural partition certificate kind")
    if instrumentation.get("emitted_circuits_executed_or_width_checked") is not False:
        errors.append("instrumentation misstates generated-circuit execution validation")
    if instrumentation.get("operation_accounting") != (
        INSTRUMENTATION_OPERATION_ACCOUNTING
    ):
        errors.append("instrumentation operation_accounting is missing or invalid")
    circuit = record.get("circuit") or {}
    original = circuit.get("original_qubits")
    if type(original) is not int or original < 0:
        errors.append("circuit original_qubits is missing or invalid")
        return errors
    if instrumentation.get("original_qubits") != original:
        errors.append("instrumentation original_qubits differs from circuit metadata")
    if partition_metadata.get("capacity") != CANONICAL_QMAX - original:
        errors.append("partition metadata capacity differs from Qmax-source width")
    original_gate_count_m = circuit.get("original_m_operations")
    if type(original_gate_count_m) is not int or original_gate_count_m < 0:
        errors.append("circuit original_m_operations is missing or invalid")
        original_gate_count_m = None
    elif instrumentation.get("original_unitary_gate_count_m") != (
        original_gate_count_m
    ):
        errors.append(
            "instrumentation original_unitary_gate_count_m differs from circuit "
            "original_m_operations"
        )
    batches = instrumentation.get("batches")
    if not isinstance(batches, list):
        return errors + ["instrumentation batches is not a list"]
    if instrumentation.get("batch_count") != len(batches):
        errors.append("batch_count does not equal the stored batch list length")

    count_fields = (
        "candidate_monitor_event_count",
        "candidate_gate_node_count",
        "individually_feasible_monitor_event_count",
        "individually_feasible_gate_node_count",
        "deployed_monitor_event_count",
        "deployed_gate_node_count",
        "individually_infeasible_gate_node_count",
    )
    counts = {}
    for field in count_fields:
        value = instrumentation.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"{field} is missing or invalid")
        else:
            counts[field] = value

    cost_rows = instrumentation.get("gate_node_costs")
    cost_by_node: dict[int, int] = {}
    event_count_by_node: dict[int, int] = {}
    cone_gate_union_by_node: dict[int, int] = {}
    if not isinstance(cost_rows, list):
        errors.append("gate_node_costs is not a list")
    else:
        for position, row in enumerate(cost_rows):
            if not isinstance(row, dict):
                errors.append(f"gate_node_costs row {position} is not an object")
                continue
            node = row.get("gate")
            demand = row.get("extra_qubit_demand")
            monitor_qubits = row.get("monitor_qubits")
            monitor_event_count = row.get("monitor_event_count")
            cone_qubit_union_count = row.get("cone_qubit_union_count")
            cone_gate_union_count = row.get("cone_gate_union_count")
            if type(node) is not int or node < 0:
                errors.append(f"gate_node_costs row {position} has invalid gate")
                continue
            if node in cost_by_node:
                errors.append(f"gate_node_costs repeats gate {node}")
                continue
            if type(demand) is not int or demand < 0:
                errors.append(
                    f"gate_node_costs row {position} has invalid extra demand"
                )
                continue
            if (
                not isinstance(monitor_qubits, list)
                or any(type(qubit) is not int for qubit in monitor_qubits)
                or len(monitor_qubits) != len(set(monitor_qubits))
            ):
                errors.append(
                    f"gate_node_costs row {position} has invalid monitor_qubits"
                )
                continue
            if monitor_event_count != len(monitor_qubits):
                errors.append(
                    f"gate_node_costs row {position} has invalid monitor_event_count"
                )
                continue
            if (
                type(cone_qubit_union_count) is not int
                or cone_qubit_union_count < len(monitor_qubits)
            ):
                errors.append(
                    f"gate_node_costs row {position} has invalid cone-qubit union"
                )
                continue
            if type(cone_gate_union_count) is not int or cone_gate_union_count < 0:
                errors.append(
                    f"gate_node_costs row {position} has invalid cone-gate union"
                )
                continue
            cost_by_node[node] = demand
            event_count_by_node[node] = len(monitor_qubits)
            cone_gate_union_by_node[node] = cone_gate_union_count

    seen_nodes: set[int] = set()
    run_qubits_values = []
    batch_cost_values = []
    for position, batch in enumerate(batches):
        if not isinstance(batch, dict):
            errors.append(f"batch {position} is not an object")
            continue
        cost = batch.get("cost")
        run_qubits = batch.get("run_qubits")
        nodes = batch.get("nodes")
        if type(cost) is not int or cost < 0:
            errors.append(f"batch {position} has invalid cost {cost!r}")
            continue
        batch_cost_values.append(cost)
        if run_qubits != original + cost:
            errors.append(
                f"batch {position} run_qubits={run_qubits!r}, expected {original + cost}"
            )
        if type(run_qubits) is int:
            run_qubits_values.append(run_qubits)
            if run_qubits > CANONICAL_QMAX:
                errors.append(f"batch {position} uses {run_qubits}>24 qubits")
        if (
            not isinstance(nodes, list)
            or any(type(node) is not int for node in nodes)
            or len(nodes) != len(set(nodes))
        ):
            errors.append(f"batch {position} nodes are invalid")
        else:
            overlap = seen_nodes.intersection(nodes)
            if overlap:
                errors.append(f"batch {position} repeats nodes {sorted(overlap)}")
            seen_nodes.update(nodes)
            unknown = set(nodes) - set(cost_by_node)
            if unknown:
                errors.append(f"batch {position} contains unknown nodes {sorted(unknown)}")
            else:
                expected_cost = sum(cost_by_node[node] for node in nodes)
                if cost != expected_cost:
                    errors.append(
                        f"batch {position} cost={cost}, expected {expected_cost} "
                        "from gate_node_costs"
                    )
    expected_max = max(run_qubits_values, default=None)
    if instrumentation.get("maximum_run_qubits") != expected_max:
        errors.append(
            "maximum_run_qubits does not equal the maximum stored batch run size"
        )
    infeasible = instrumentation.get("infeasible_gate_nodes")
    if (
        not isinstance(infeasible, list)
        or any(type(node) is not int for node in infeasible)
        or len(infeasible) != len(set(infeasible))
    ):
        errors.append("infeasible_gate_nodes is invalid")
        infeasible_set: set[int] = set()
    else:
        infeasible_set = set(infeasible)
    candidate_nodes = set(cost_by_node)
    if infeasible_set - candidate_nodes:
        errors.append("infeasible_gate_nodes contains unknown candidates")
    expected_infeasible_set = {
        node
        for node, demand in cost_by_node.items()
        if original + demand > CANONICAL_QMAX
    }
    if infeasible_set != expected_infeasible_set:
        errors.append(
            "infeasible_gate_nodes differs from source-derived Qmax feasibility"
        )
    individually_feasible_nodes = candidate_nodes - infeasible_set
    if seen_nodes != individually_feasible_nodes:
        errors.append(
            "deployed batch nodes do not equal all individually feasible candidates"
        )
    expected_counts = {
        "candidate_gate_node_count": len(candidate_nodes),
        "candidate_monitor_event_count": sum(event_count_by_node.values()),
        "individually_infeasible_gate_node_count": len(infeasible_set),
        "individually_feasible_gate_node_count": len(individually_feasible_nodes),
        "individually_feasible_monitor_event_count": sum(
            event_count_by_node.get(node, 0) for node in individually_feasible_nodes
        ),
        "deployed_gate_node_count": len(seen_nodes),
        "deployed_monitor_event_count": sum(
            event_count_by_node.get(node, 0) for node in seen_nodes
        ),
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            errors.append(
                f"{field}={instrumentation.get(field)!r}, expected {expected}"
            )

    partition_expectations = {
        "item_count": len(candidate_nodes),
        "feasible_item_count": len(individually_feasible_nodes),
        "infeasible_item_count": len(infeasible_set),
        "final_batch_count": len(batches),
    }
    for field, expected in partition_expectations.items():
        if partition_metadata.get(field) != expected:
            errors.append(
                f"partition metadata {field}={partition_metadata.get(field)!r}, "
                f"expected {expected}"
            )
    ffd_batch_count = partition_metadata.get("ffd_batch_count")
    if type(ffd_batch_count) is not int or ffd_batch_count < len(batches):
        errors.append("partition metadata has an invalid FFD batch count")
    refinement = partition_metadata.get("refinement")
    if not isinstance(refinement, dict):
        errors.append("partition metadata lacks refinement evidence")
    else:
        node_limit = refinement.get("node_limit")
        time_limit = refinement.get("time_limit_seconds")
        if node_limit not in {0, REFINEMENT_NODE_LIMIT}:
            errors.append("partition refinement node limit violates the budget policy")
        if (
            not isinstance(time_limit, (int, float))
            or isinstance(time_limit, bool)
            or not math.isfinite(float(time_limit))
            or not 0 < float(time_limit) <= MAX_REFINEMENT_TIME_LIMIT_SECONDS
        ):
            errors.append("partition refinement time limit violates the budget policy")
        if partition_budget.get("refinement_node_limit") != node_limit:
            errors.append("partition budget node limit differs from refinement evidence")
        if partition_budget.get("refinement_time_limit_seconds") != time_limit:
            errors.append("partition budget time limit differs from refinement evidence")
        attempted = refinement.get("attempted")
        if type(attempted) is not bool:
            errors.append("partition refinement attempted flag is invalid")
        if attempted is False and refinement.get("solver_status") != "not-run":
            errors.append("unattempted partition refinement claims a solver result")
        resolved = refinement.get("resolved")
        fallback_used = refinement.get("fallback_used")
        if attempted is False:
            if resolved is not None or fallback_used is not False:
                errors.append("unattempted partition refinement has invalid outcome flags")
            if node_limit == 0 and refinement.get("solver_status") != "not-run":
                errors.append("budget-skipped refinement claims a solver result")
        elif resolved is True:
            if refinement.get("solver_status") not in {"Optimal", "Infeasible"}:
                errors.append("resolved partition refinement lacks a conclusive status")
            if fallback_used is not False:
                errors.append("resolved partition refinement claims FFD fallback")
        elif resolved is False:
            if refinement.get("solver_status") in {"Optimal", "Infeasible", "not-run"}:
                errors.append("unresolved partition refinement has an invalid status")
            if fallback_used is not True or refinement.get("improved") is not False:
                errors.append("unresolved partition refinement did not retain FFD")
            if partition_metadata.get("final_batch_count") != ffd_batch_count:
                errors.append("unresolved partition refinement changed the FFD packing")
        else:
            errors.append("attempted partition refinement lacks a resolved flag")

    instrumentation_runtime = instrumentation.get("runtime_seconds")
    if (
        not isinstance(instrumentation_runtime, (int, float))
        or isinstance(instrumentation_runtime, bool)
        or not math.isfinite(float(instrumentation_runtime))
        or instrumentation_runtime < 0
    ):
        errors.append("instrumentation runtime_seconds is missing or invalid")

    expected_replay_gate_executions = sum(
        cone_gate_union_by_node.get(node, 0) for node in seen_nodes
    )
    expected_monitor_operations = expected_counts["deployed_monitor_event_count"]
    expected_original_gate_executions = (
        len(batches) * original_gate_count_m
        if original_gate_count_m is not None
        else None
    )
    expected_total_counted_operations = (
        expected_original_gate_executions
        + expected_replay_gate_executions
        + 2 * expected_monitor_operations
        if expected_original_gate_executions is not None
        else None
    )
    planned_expectations = {
        "planned_peak_extra_ancillas": max(batch_cost_values, default=0),
        "planned_cumulative_extra_ancilla_allocations": sum(batch_cost_values),
        "planned_original_gate_executions": expected_original_gate_executions,
        "planned_replay_gate_executions": expected_replay_gate_executions,
        "planned_monitor_measurement_operations": expected_monitor_operations,
        "planned_monitor_reset_operations": expected_monitor_operations,
        "planned_total_counted_operations": expected_total_counted_operations,
    }
    for field, expected in planned_expectations.items():
        if expected is not None and instrumentation.get(field) != expected:
            errors.append(f"{field}={instrumentation.get(field)!r}, expected {expected}")

    diagnostic_fields = (
        "diagnostic_selected_per_event_cone_gate_occurrences",
        "diagnostic_selected_per_event_cone_qubit_occurrences",
        "diagnostic_candidate_per_event_cone_gate_occurrences",
        "diagnostic_candidate_per_event_cone_qubit_occurrences",
        "diagnostic_selected_per_event_extra_cone_qubit_occurrences",
    )
    diagnostic_values: dict[str, int] = {}
    for field in diagnostic_fields:
        value = instrumentation.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"{field} is missing or invalid")
        else:
            diagnostic_values[field] = value
    if diagnostic_values.get(
        "diagnostic_candidate_per_event_cone_gate_occurrences", 0
    ) > diagnostic_values.get(
        "diagnostic_selected_per_event_cone_gate_occurrences", 0
    ):
        errors.append("candidate per-event cone-gate diagnostic exceeds selected total")
    if diagnostic_values.get(
        "diagnostic_candidate_per_event_cone_qubit_occurrences", 0
    ) > diagnostic_values.get(
        "diagnostic_selected_per_event_cone_qubit_occurrences", 0
    ):
        errors.append("candidate per-event cone-qubit diagnostic exceeds selected total")
    return errors


def _validate_success_analysis(record: Mapping[str, Any]) -> list[str]:
    """Validate every completed selection certificate, including partial outcomes."""

    selection_value = record.get("selection")
    if not isinstance(selection_value, dict):
        return []
    errors: list[str] = []
    circuit = record.get("circuit") or {}
    if circuit.get("evidence_stage") != "eligibility_fingerprint_complete":
        errors.append("selection record lacks completed circuit evidence")
    eligibility = circuit.get("eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("eligible") is not True:
        errors.append("selection record lacks a passing circuit eligibility certificate")
    circuit_sha = circuit.get("canonical_circuit_sha256")
    if not _is_sha256(circuit_sha):
        errors.append("selection record lacks a circuit SHA-256")
    if circuit.get("fingerprint_method") != (
        "canonical instruction stream plus per-unitary matrix SHA-256"
    ):
        errors.append("selection record has an unknown circuit checksum method")

    selection = selection_value
    mps = selection.get("mps")
    if not isinstance(mps, dict):
        errors.append("selection record lacks MPS diagnostics")
    else:
        if mps.get("configured_max_bond_dimension") is not None:
            errors.append("selection record uses a finite MPS maximum bond dimension")
        if mps.get("configured_max_bond_dimension_mode") != "unbounded":
            errors.append("selection record does not declare unbounded MPS bond dimension")
        if mps.get("configured_truncation_threshold") != MPS_TRUNCATION_THRESHOLD:
            errors.append("selection record has the wrong MPS truncation threshold")
        if mps.get("configured_mps_lapack") is not MPS_LAPACK:
            errors.append("selection record does not use the fixed LAPACK MPS path")

    source = record.get("source") or {}
    crosscheck = selection.get("statevector_crosscheck")
    if source.get("source_kind") == "canonical_qasm":
        if not isinstance(crosscheck, dict):
            return errors + ["base-corpus selection lacks statevector cross-check"]
        for field in ("required", "performed", "passed"):
            if crosscheck.get(field) is not True:
                errors.append(
                    f"base-corpus statevector cross-check {field} is not true"
                )
        for field in (
            "selection_mismatch_count",
            "missing_mps_point_count",
            "missing_statevector_point_count",
        ):
            if crosscheck.get(field) != 0:
                errors.append(
                    f"base-corpus statevector cross-check {field} is not zero"
                )
        mps_count = crosscheck.get("point_count_mps")
        statevector_count = crosscheck.get("point_count_statevector")
        if type(mps_count) is not int or mps_count < 0:
            errors.append("base-corpus MPS cross-check point count is invalid")
        if type(statevector_count) is not int or statevector_count < 0:
            errors.append("base-corpus statevector point count is invalid")
        if mps_count != statevector_count:
            errors.append("base-corpus cross-check point counts differ")
        mps_events, event_errors = _deserialize_event_rows(
            crosscheck.get("mps_selected_events"),
            label="base-corpus MPS selected events",
        )
        errors.extend(event_errors)
        statevector_events, event_errors = _deserialize_event_rows(
            crosscheck.get("statevector_selected_events"),
            label="base-corpus statevector selected events",
        )
        errors.extend(event_errors)
        selected_nodes, node_errors = _deserialize_monitorable_nodes(
            selection.get("monitorable_nodes")
        )
        errors.extend(node_errors)
        if mps_events != sorted(_selection_node_set(selected_nodes)):
            errors.append(
                "base-corpus MPS cross-check events differ from selection nodes"
            )
        if statevector_events != mps_events:
            errors.append(
                "base-corpus statevector and MPS selected events differ"
            )
        for prefix, events in (
            ("mps", mps_events),
            ("statevector", statevector_events),
        ):
            if crosscheck.get(f"{prefix}_selected_event_count") != len(events):
                errors.append(
                    f"base-corpus {prefix} selected event count is invalid"
                )
            if crosscheck.get(f"{prefix}_selected_events_sha256") != stable_hash(
                events
            ):
                errors.append(
                    f"base-corpus {prefix} selected event hash is invalid"
                )
        maximum_a2_difference = crosscheck.get("maximum_absolute_a2_difference")
        if (
            not isinstance(maximum_a2_difference, (int, float))
            or isinstance(maximum_a2_difference, bool)
            or not math.isfinite(float(maximum_a2_difference))
            or maximum_a2_difference < 0
        ):
            errors.append("base-corpus maximum a2 difference is invalid")
        maximum_difference = crosscheck.get(
            "maximum_absolute_determinant_difference"
        )
        if (
            not isinstance(maximum_difference, (int, float))
            or isinstance(maximum_difference, bool)
            or not math.isfinite(float(maximum_difference))
            or maximum_difference < 0
        ):
            errors.append("base-corpus maximum determinant difference is invalid")
    elif isinstance(crosscheck, dict) and crosscheck.get("required") is not False:
        errors.append("extension success incorrectly requires the base cross-check")
    return errors


def _validate_selection_evidence(record: Mapping[str, Any]) -> list[str]:
    selection = record.get("selection")
    if selection is None:
        return []
    if not isinstance(selection, dict):
        return ["selection is not an object"]
    nodes, errors = _deserialize_monitorable_nodes(
        selection.get("monitorable_nodes")
    )
    serialized = _serialize_monitorable_nodes(nodes)
    if selection.get("monitorable_nodes") != serialized:
        errors.append("selection monitorable_nodes is not in normalized form")
    if selection.get("monitorable_nodes_sha256") != stable_hash(serialized):
        errors.append("selection monitorable_nodes_sha256 mismatch")
    if selection.get("monitorable_gate_node_count") != len(nodes):
        errors.append("selection monitorable_gate_node_count mismatch")
    if selection.get("monitorable_node_count") != sum(
        len(qubits) for qubits in nodes.values()
    ):
        errors.append("selection monitorable_node_count mismatch")
    scores, score_errors = _deserialize_point_scores(selection.get("point_scores"))
    errors.extend(score_errors)
    serialized_scores = _serialize_point_scores(scores)
    if selection.get("point_scores") != serialized_scores:
        errors.append("selection point_scores is not in normalized form")
    if selection.get("point_scores_sha256") != stable_hash(serialized_scores):
        errors.append("selection point_scores_sha256 mismatch")
    score_definition = selection.get("point_score_definition")
    if not isinstance(score_definition, dict):
        errors.append("selection lacks point_score_definition")
    else:
        if score_definition.get("selection") != "second_schmidt_amplitude_a2":
            errors.append("selection point scores name the wrong selector")
        if score_definition.get("selection_tolerance") != (
            SCHMIDT_AMPLITUDE_TOLERANCE
        ):
            errors.append("selection point scores use the wrong a2 tolerance")
        if score_definition.get("a2_extraction") != (
            "aer_mps_local_tensor_svd"
        ):
            errors.append("selection point scores use the wrong MPS extraction")
        if score_definition.get("determinant_role") != (
            "physical_diagnostic_only"
        ):
            errors.append("selection point scores misuse determinant")
    score_selected = {
        point
        for point, score in scores.items()
        if score["a2"] <= SCHMIDT_AMPLITUDE_TOLERANCE
    }
    if score_selected != _selection_node_set(nodes):
        errors.append("selection nodes do not follow the stored a2 scores")
    circuit = record.get("circuit") or {}
    original_qubits = circuit.get("original_qubits")
    total_instructions = circuit.get("total_instructions")
    if isinstance(mps := selection.get("mps"), dict):
        point_count = mps.get("point_count")
        if type(point_count) is int and point_count != len(scores):
            errors.append("selection point score count differs from MPS point count")
    for gate, qubits in nodes.items():
        if type(total_instructions) is int and gate >= total_instructions:
            errors.append(f"selection gate {gate} exceeds the circuit instruction range")
        if type(original_qubits) is int and any(
            qubit >= original_qubits for qubit in qubits
        ):
            errors.append(f"selection gate {gate} contains an out-of-range qubit")
    runtime_values: dict[str, float] = {}
    for field in (
        "mps_runtime_seconds",
        "validation_runtime_seconds",
        "total_selection_certificate_runtime_seconds",
    ):
        value = selection.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            errors.append(f"selection {field} is missing or invalid")
        else:
            runtime_values[field] = float(value)
    total_runtime = runtime_values.get("total_selection_certificate_runtime_seconds")
    for component in ("mps_runtime_seconds", "validation_runtime_seconds"):
        if (
            total_runtime is not None
            and component in runtime_values
            and runtime_values[component] > total_runtime + 1e-6
        ):
            errors.append(f"selection {component} exceeds total selection runtime")
    stored_phase_runtime = (record.get("phase_elapsed_seconds") or {}).get(
        "selection"
    )
    if (
        total_runtime is not None
        and isinstance(stored_phase_runtime, (int, float))
        and not isinstance(stored_phase_runtime, bool)
        and math.isfinite(float(stored_phase_runtime))
        and not math.isclose(
            total_runtime, float(stored_phase_runtime), rel_tol=0.0, abs_tol=1e-6
        )
    ):
        errors.append("selection total runtime differs from phase timing")

    rss_fields = (
        "pre_selection_high_water_rss_bytes",
        "mps_peak_rss_bytes",
        "total_selection_certificate_peak_rss_bytes",
    )
    rss_values: dict[str, int | None] = {}
    for field in rss_fields:
        value = selection.get(field)
        if value is not None and (type(value) is not int or value < 0):
            errors.append(f"selection {field} is invalid")
        else:
            rss_values[field] = value
    pre_selection_rss = rss_values.get("pre_selection_high_water_rss_bytes")
    mps_peak = rss_values.get("mps_peak_rss_bytes")
    total_peak = rss_values.get("total_selection_certificate_peak_rss_bytes")
    if (
        pre_selection_rss is not None
        and mps_peak is not None
        and mps_peak < pre_selection_rss
    ):
        errors.append("selection MPS peak RSS is below the pre-selection high water")
    if mps_peak is not None and total_peak is not None and total_peak < mps_peak:
        errors.append("selection total peak RSS is below the MPS peak RSS")
    for field, value_field in (
        ("mps_rss_measurement", "mps_peak_rss_bytes"),
        ("total_rss_measurement", "total_selection_certificate_peak_rss_bytes"),
    ):
        measurement = selection.get(field)
        if not isinstance(measurement, dict) or type(
            measurement.get("available")
        ) is not bool:
            errors.append(f"selection {field} is missing or invalid")
        elif measurement["available"] is True and rss_values.get(value_field) is None:
            errors.append(f"selection {field} claims unavailable RSS evidence")
        elif measurement["available"] is False and rss_values.get(value_field) is not None:
            errors.append(f"selection {field} contradicts stored RSS evidence")
    mps = selection.get("mps")
    if not isinstance(mps, dict):
        errors.append("selection lacks MPS diagnostics")
    else:
        if mps.get("configured_max_bond_dimension") is not None:
            errors.append("selection uses a finite MPS maximum bond dimension")
        if mps.get("configured_max_bond_dimension_mode") != "unbounded":
            errors.append("selection does not declare unbounded MPS bond dimension")
        if mps.get("configured_truncation_threshold") != MPS_TRUNCATION_THRESHOLD:
            errors.append("selection has the wrong MPS truncation threshold")
        if mps.get("configured_mps_lapack") is not MPS_LAPACK:
            errors.append("selection does not use the fixed LAPACK MPS path")
        if (
            type(mps.get("observed_max_chi")) is not int
            or mps.get("observed_max_chi") < 1
            or mps.get("observed_max_chi_available") is not True
        ):
            errors.append("selection lacks native observed MPS chi evidence")
        if "save_matrix_product_state" not in str(
            mps.get("observed_max_chi_method")
        ):
            errors.append("selection records an unknown observed MPS chi method")
        if mps.get("observed_max_chi_unavailable_reason") is not None:
            errors.append("selection incorrectly marks observed MPS chi unavailable")
    return errors


def _recompute_scientific_errors(
    record: Mapping[str, Any], expected_attempt: AttemptSpec
) -> list[str]:
    """Rebuild source evidence and recompute feasible scientific certificates."""

    errors: list[str] = []
    try:
        qc = _build_circuit(expected_attempt)
    except UnsupportedAttempt:
        if record.get("status") != "unsupported" or record.get(
            "error_category"
        ) != "generator_rejected":
            errors.append("source rebuild is unsupported but record says otherwise")
        return errors
    except Exception as exc:
        error = record.get("error") or {}
        expected_exception_type = type(exc).__name__
        expected_message = str(exc)[:2000]
        expected_evidence = _exception_evidence_sha256(
            expected_exception_type, expected_message
        )
        if record.get("status") != "build_error":
            errors.append(
                "source rebuild failed during validation but record is not a "
                f"build_error: {expected_exception_type}: {expected_message}"
            )
        else:
            if record.get("error_category") != "circuit_build_failed":
                errors.append(
                    "reproduced build_error has a different stored category"
                )
            if error.get("phase") != "build":
                errors.append("reproduced build_error does not name build phase")
            if error.get("exception_type") != expected_exception_type:
                errors.append(
                    "reproduced build_error exception type differs from stored "
                    "evidence"
                )
            if error.get("exception_evidence_sha256") != expected_evidence:
                errors.append(
                    "reproduced build_error exception evidence differs from stored "
                    "evidence"
                )
        return errors

    try:
        eligibility = validate_circuit_eligibility(qc)
    except CircuitEligibilityError as exc:
        if record.get("status") != "unsupported":
            errors.append("source circuit is theorem-ineligible but record is not unsupported")
        if record.get("error_category") != exc.category:
            errors.append(
                "source eligibility category differs from the stored outcome: "
                f"{exc.category!r} != {record.get('error_category')!r}"
            )
        return errors
    except Exception as exc:
        errors.append(
            "source eligibility recomputation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return errors

    circuit = record.get("circuit")
    if not isinstance(circuit, dict):
        # A parent timeout may occur before the child returns its built circuit.
        if record.get("status") not in {"timeout", "resource_limit", "analysis_error"}:
            errors.append("rebuilt source is eligible but record lacks circuit evidence")
        return errors
    expected_build_evidence = {
        "original_qubits": int(qc.num_qubits),
        "original_m_operations": sum(
            1
            for instruction in qc.data
            if instruction.operation.name not in {"measure", "barrier"}
        ),
        "original_depth": int(qc.depth()),
        "total_instructions": len(qc.data),
        "requested_qubits": expected_attempt.requested_qubits,
    }
    evidence_stage = circuit.get("evidence_stage")
    if evidence_stage not in {
        "build_complete",
        "eligibility_complete",
        "eligibility_rejected",
        "eligibility_fingerprint_complete",
    }:
        errors.append(f"stored circuit has invalid evidence_stage {evidence_stage!r}")
    for field, expected in expected_build_evidence.items():
        if circuit.get(field) != expected:
            errors.append(
                f"recomputed circuit {field}={expected!r}, stored {circuit.get(field)!r}"
            )

    if evidence_stage == "build_complete":
        if record.get("status") not in {"timeout", "resource_limit", "analysis_error"}:
            errors.append(
                "build-only circuit evidence is invalid for this analysis status"
            )
        for field in ("canonical_circuit_sha256", "fingerprint_method", "eligibility"):
            if circuit.get(field) is not None:
                errors.append(
                    f"build-only circuit evidence unexpectedly contains {field}"
                )
        return errors

    if evidence_stage == "eligibility_rejected":
        errors.append("stored circuit rejects eligibility but rebuilt source is eligible")
        return errors

    if evidence_stage == "eligibility_complete":
        if circuit.get("eligibility") != eligibility:
            errors.append("stored eligibility evidence differs from recomputation")
        for field in ("canonical_circuit_sha256", "fingerprint_method"):
            if circuit.get(field) is not None:
                errors.append(
                    f"eligibility-only circuit evidence unexpectedly contains {field}"
                )
        if record.get("status") not in {
            "timeout",
            "resource_limit",
            "analysis_error",
        }:
            errors.append("eligibility-only evidence is invalid for this status")
        return errors

    expected_completed_evidence = {
        "canonical_circuit_sha256": _circuit_canonical_sha256(qc),
        "fingerprint_method": (
            "canonical instruction stream plus per-unitary matrix SHA-256"
        ),
        "eligibility": eligibility,
    }
    for field, expected in expected_completed_evidence.items():
        if circuit.get(field) != expected:
            errors.append(
                f"recomputed circuit {field}={expected!r}, stored {circuit.get(field)!r}"
            )

    selection = record.get("selection")
    if not isinstance(selection, dict):
        return errors
    stored_nodes, node_errors = _deserialize_monitorable_nodes(
        selection.get("monitorable_nodes")
    )
    errors.extend(node_errors)
    stored_scores, score_errors = _deserialize_point_scores(
        selection.get("point_scores")
    )
    errors.extend(score_errors)

    try:
        recomputed_nodes, recomputed_mps, recomputed_scores = (
            _select_monitorable_nodes_mps_details(qc)
        )
    except Exception as exc:
        errors.append(
            "MPS selection recomputation failed: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        stored_mps = selection.get("mps") or {}
        if stored_mps.get("observed_max_chi") != recomputed_mps.get(
            "observed_max_chi"
        ):
            errors.append("stored observed maximum chi differs from MPS recomputation")
        if _serialize_monitorable_nodes(stored_nodes) != (
            _serialize_monitorable_nodes(recomputed_nodes)
        ):
            errors.append("stored monitorable nodes differ from MPS recomputation")
        if set(stored_scores) != set(recomputed_scores):
            errors.append("stored point score identities differ from MPS recomputation")
        else:
            for point in sorted(stored_scores):
                for field in ("a2", "determinant"):
                    if not math.isclose(
                        stored_scores[point][field],
                        recomputed_scores[point][field],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        errors.append(
                            f"stored point {point} {field} differs from MPS recomputation"
                        )
        if expected_attempt.source_kind == "canonical_qasm":
            try:
                recomputed_crosscheck = crosscheck_mps_against_statevector(
                    qc, recomputed_nodes, recomputed_scores
                )
            except Exception as exc:
                errors.append(
                    "base-corpus statevector recomputation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                stored_crosscheck = selection.get("statevector_crosscheck") or {}
                for field in (
                    "point_count_mps",
                    "point_count_statevector",
                    "mps_selected_event_count",
                    "statevector_selected_event_count",
                    "mps_selected_events",
                    "statevector_selected_events",
                    "mps_selected_events_sha256",
                    "statevector_selected_events_sha256",
                    "selection_mismatch_count",
                    "missing_mps_point_count",
                    "missing_statevector_point_count",
                    "passed",
                ):
                    if stored_crosscheck.get(field) != recomputed_crosscheck.get(field):
                        errors.append(
                            f"stored statevector cross-check {field} differs from recomputation"
                        )

    instrumentation = record.get("instrumentation")
    if isinstance(instrumentation, dict):
        # Recompute cones and every derived cost from source, while replaying
        # the stored feasible partition.  Optional refinement is wall-clock
        # bounded; solving it again is not a deterministic validation oracle.
        recomputed_instrumentation = build_instrumentation_metrics(
            qc,
            stored_nodes,
            partition_override=instrumentation,
        )
        for field, expected in recomputed_instrumentation.items():
            if instrumentation.get(field) != expected:
                errors.append(
                    f"recomputed instrumentation {field} differs from stored evidence"
                )
    return errors


def validate_record(
    record: Mapping[str, Any],
    *,
    expected_fingerprints: Mapping[str, str] | None = None,
    expected_attempt: AttemptSpec | None = None,
    recompute_science: bool = False,
) -> list[str]:
    errors = _completion_errors(record, require_marker=True)
    record_hash = record.get("record_sha256")
    if not _is_sha256(record_hash):
        errors.append("record lacks a valid record_sha256 integrity checksum")
    elif record_hash != _record_sha256(record):
        errors.append("record_sha256 integrity checksum mismatch")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"invalid schema_version {record.get('schema_version')!r}")
    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("record source is not an object")
    elif source.get("attempt_id") != record.get("attempt_id"):
        errors.append("source attempt_id does not match record attempt_id")
    if expected_attempt is not None:
        expected_source = asdict(expected_attempt)
        if record.get("attempt_id") != expected_attempt.attempt_id:
            errors.append(
                f"attempt_id={record.get('attempt_id')!r}, expected "
                f"{expected_attempt.attempt_id!r}"
            )
        if record.get("attempt_index") != expected_attempt.attempt_index:
            errors.append(
                f"attempt_index={record.get('attempt_index')!r}, expected "
                f"{expected_attempt.attempt_index}"
            )
        if source != expected_source:
            errors.append(
                f"fixed source metadata mismatch: {source!r} != "
                f"{expected_source!r}"
            )
    if type(record.get("qmax")) is not int or record.get("qmax") != CANONICAL_QMAX:
        errors.append(f"record qmax={record.get('qmax')!r}, expected exactly 24")
    run = record.get("run")
    if not isinstance(run, dict):
        errors.append("record run metadata is not an object")
    else:
        shard_index = run.get("shard_index")
        shard_count = run.get("shard_count")
        if type(shard_count) is not int or shard_count <= 0:
            errors.append("record shard_count must be a positive integer")
        if (
            type(shard_index) is not int
            or type(shard_count) is not int
            or shard_count <= 0
            or not 0 <= shard_index < shard_count
        ):
            errors.append("record shard_index is outside the declared shard_count")
        if (
            run.get("mode") == "full"
            and type(record.get("attempt_index")) is int
            and type(shard_index) is int
            and type(shard_count) is int
            and shard_count > 0
            and record["attempt_index"] % shard_count != shard_index
        ):
            errors.append("record attempt is assigned to the wrong full-run shard")
        attempt_timeout = run.get("attempt_timeout_seconds")
        if (
            not isinstance(attempt_timeout, (int, float))
            or isinstance(attempt_timeout, bool)
            or not math.isfinite(float(attempt_timeout))
            or attempt_timeout <= 0
        ):
            errors.append("record attempt_timeout_seconds is invalid")
        infrastructure_retries = run.get(
            "infrastructure_retries_before_outcome"
        )
        if (
            type(infrastructure_retries) is not int
            or not 0 <= infrastructure_retries <= len(
                IMMEDIATE_INFRASTRUCTURE_RETRY_DELAYS_SECONDS
            )
        ):
            errors.append("record infrastructure retry count is invalid")
    elapsed_seconds = record.get("elapsed_seconds")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0
    ):
        errors.append("record elapsed_seconds is invalid")
    phase_elapsed = record.get("phase_elapsed_seconds")
    if not isinstance(phase_elapsed, dict):
        errors.append("record phase_elapsed_seconds is not an object")
    else:
        for phase, value in phase_elapsed.items():
            if phase not in {"build", "eligibility", "selection", "instrumentation"}:
                errors.append(f"record has unknown phase timing {phase!r}")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                errors.append(f"record phase timing {phase!r} is invalid")
    error = record.get("error")
    if isinstance(error, dict):
        exception_type = error.get("exception_type")
        message = error.get("message")
        exception_evidence = error.get("exception_evidence_sha256")
        if not isinstance(exception_type, str) or not isinstance(message, str):
            errors.append("error exception evidence is malformed")
        elif exception_evidence != _exception_evidence_sha256(
            exception_type, message
        ):
            errors.append("error exception_evidence_sha256 does not match")
        stored_trace = error.get("traceback")
        stored_trace_hash = error.get("traceback_sha256")
        if not isinstance(stored_trace, str):
            errors.append("error traceback is not a string")
        elif not _is_sha256(stored_trace_hash):
            errors.append("error traceback_sha256 is invalid")
        elif hashlib.sha256(stored_trace.encode("utf-8")).hexdigest() != (
            stored_trace_hash
        ):
            errors.append("error traceback_sha256 does not match traceback")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("record provenance is not an object")
    else:
        protocol_config = provenance.get("protocol_config")
        if not isinstance(protocol_config, dict):
            errors.append("record provenance lacks protocol_config")
        else:
            if protocol_config.get("qmax") != CANONICAL_QMAX:
                errors.append(
                    "record protocol_config qmax is missing or not exactly 24"
                )
            if protocol_config.get("partitioner_qmax") != CANONICAL_QMAX:
                errors.append(
                    "record protocol_config partitioner_qmax is missing or not 24"
                )
            partition_solver = protocol_config.get("partition_solver")
            if not isinstance(partition_solver, dict) or partition_solver.get(
                "backend"
            ) not in {"gurobi", "cbc"}:
                errors.append(
                    "record protocol_config lacks valid partition solver provenance"
                )
            elif protocol_config.get("partition_solver_sha256") != stable_hash(
                partition_solver
            ):
                errors.append(
                    "record protocol_config partition solver provenance hash mismatch"
                )
            elif partition_solver.get(
                "transient_license_retry_delays_seconds"
            ) != (
                list(GUROBI_LICENSE_RETRY_DELAYS_SECONDS)
                if partition_solver.get("backend") == "gurobi"
                else []
            ):
                errors.append("record partition solver has the wrong license retry budget")
            budget_expectations = {
                "attempt_timeout_default_seconds": DEFAULT_TIMEOUT_SECONDS,
                "refinement_node_limit": REFINEMENT_NODE_LIMIT,
                "refinement_max_time_limit_seconds": (
                    MAX_REFINEMENT_TIME_LIMIT_SECONDS
                ),
                "instrumentation_completion_reserve_seconds": (
                    INSTRUMENTATION_COMPLETION_RESERVE_SECONDS
                ),
                "gurobi_license_retry_delays_seconds": list(
                    GUROBI_LICENSE_RETRY_DELAYS_SECONDS
                ),
                "infrastructure_outcome_policy": (
                    "checkpoint as incomplete, retry once immediately, and exclude "
                    "from terminal publication artifacts"
                ),
            }
            for field, expected in budget_expectations.items():
                if protocol_config.get(field) != expected:
                    errors.append(f"record protocol_config has invalid {field}")
            if protocol_config.get(
                "publication_partition_solver_backend"
            ) != PUBLICATION_PARTITION_SOLVER_BACKEND:
                errors.append(
                    "record protocol_config has the wrong final-run solver policy"
                )
            if protocol_config.get("status_semantics") != (
                "status is the analysis terminal status; deployment_status is the "
                "separate operational Qmax outcome"
            ):
                errors.append("record protocol_config has invalid status semantics")
            if protocol_config.get("instrumentation_operation_accounting") != (
                INSTRUMENTATION_OPERATION_ACCOUNTING
            ):
                errors.append(
                    "record protocol_config has invalid instrumentation operation "
                    "accounting"
                )
            if protocol_config.get("mps_max_bond_dimension") is not None:
                errors.append("record protocol_config uses a finite MPS chi cap")
            if protocol_config.get("mps_lapack") is not MPS_LAPACK:
                errors.append("record protocol_config does not freeze LAPACK MPS")
            if protocol_config.get("schmidt_amplitude_tolerance") != (
                SCHMIDT_AMPLITUDE_TOLERANCE
            ):
                errors.append("record protocol_config has the wrong a2 rule")
            if protocol_config.get("density_matrix_validation_tolerance") != (
                DENSITY_MATRIX_TOLERANCE
            ):
                errors.append(
                    "record protocol_config has the wrong density validation tolerance"
                )
    fingerprints = record.get("fingerprints")
    required_fingerprints = (
        "corpus_sha256",
        "manifest_sha256",
        "dependencies_sha256",
        "runtime_sha256",
        "sources_sha256",
        "config_sha256",
    )
    if not isinstance(fingerprints, dict):
        errors.append("record lacks checksums")
    else:
        for field in required_fingerprints:
            value = fingerprints.get(field)
            if not _is_sha256(value):
                errors.append(f"missing or invalid checksum {field}")
        if expected_fingerprints is not None:
            # Runtime may vary across otherwise homogeneous compute nodes. The
            # runtime checksum remains recorded, but merge invariants are
            # corpus, dependencies, sources, and protocol configuration.
            for field in (
                "corpus_sha256",
                "manifest_sha256",
                "dependencies_sha256",
                "sources_sha256",
                "config_sha256",
            ):
                if fingerprints.get(field) != expected_fingerprints.get(field):
                    errors.append(
                        f"checksum mismatch for {field}: "
                        f"{fingerprints.get(field)!r} != {expected_fingerprints.get(field)!r}"
                    )
    errors.extend(_validate_batch_feasibility(record))
    errors.extend(_validate_selection_evidence(record))
    errors.extend(_validate_success_analysis(record))
    try:
        selection_outcome(record)
    except ProtocolError as exc:
        errors.append(str(exc))
    if recompute_science and expected_attempt is not None:
        errors.extend(_recompute_scientific_errors(record, expected_attempt))
    if record.get("status") == "success":
        instrumentation = record.get("instrumentation") or {}
        circuit = record.get("circuit") or {}
        expected_deployment_status = _deployment_status(circuit, instrumentation)
        if record.get("deployment_status") != expected_deployment_status:
            errors.append(
                "deployment_status differs from the stored Qmax evidence: "
                f"{record.get('deployment_status')!r} != "
                f"{expected_deployment_status!r}"
            )
        expected_solver = (
            ((record.get("provenance") or {}).get("protocol_config") or {}).get(
                "partition_solver"
            )
        )
        if instrumentation.get("partition_solver") != expected_solver:
            errors.append(
                "instrumentation solver provenance differs from protocol config"
            )
    return errors


def validate_record_set(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_attempts: Mapping[str, AttemptSpec] | None = None,
    expected_fingerprints: Mapping[str, str] | None = None,
    require_complete_grid: bool = False,
    recompute_science: bool = False,
    expected_run_mode: str | None = None,
) -> dict[str, int]:
    identifiers = [record.get("attempt_id") for record in records]
    duplicates = sorted(
        str(identifier)
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        raise ProtocolError(f"artifact contains duplicate attempts: {duplicates}")
    indices = [record.get("attempt_index") for record in records]
    invalid_indices = [index for index in indices if type(index) is not int or index < 0]
    if invalid_indices:
        raise ProtocolError(f"artifact contains invalid attempt indices: {invalid_indices}")
    duplicate_indices = sorted(
        index for index, count in Counter(indices).items() if count > 1
    )
    if duplicate_indices:
        raise ProtocolError(
            f"artifact contains duplicate attempt indices: {duplicate_indices}"
        )

    if expected_run_mode is not None:
        wrong_modes = sorted(
            str(record.get("attempt_id"))
            for record in records
            if (record.get("run") or {}).get("mode") != expected_run_mode
        )
        if wrong_modes:
            raise ProtocolError(
                f"artifact contains records outside run mode {expected_run_mode!r}: "
                f"{wrong_modes}"
            )
    if expected_run_mode == "full":
        shard_counts = {
            (record.get("run") or {}).get("shard_count") for record in records
        }
        if len(shard_counts) != 1:
            raise ProtocolError(
                f"full artifact mixes shard_count values: {sorted(shard_counts)}"
            )
        solver_backends = {
            (
                ((record.get("provenance") or {}).get("protocol_config") or {}).get(
                    "partition_solver"
                )
                or {}
            ).get("backend")
            for record in records
        }
        if solver_backends != {PUBLICATION_PARTITION_SOLVER_BACKEND}:
            raise ProtocolError(
                "the full final artifact must be produced with "
                f"QMON_MILP_SOLVER={PUBLICATION_PARTITION_SOLVER_BACKEND}; "
                f"recorded backends={sorted(str(value) for value in solver_backends)}"
            )

    failures = []
    for record in records:
        expected_attempt = None
        if expected_attempts is not None:
            expected_attempt = expected_attempts.get(record.get("attempt_id"))
        errors = validate_record(
            record,
            expected_fingerprints=expected_fingerprints,
            expected_attempt=expected_attempt,
            recompute_science=recompute_science,
        )
        if errors:
            failures.append(f"{record.get('attempt_id')}: {errors}")
    if failures:
        raise ProtocolError("artifact validation failed:\n" + "\n".join(failures))

    if expected_attempts is not None:
        expected_attempt_ids = set(expected_attempts)
        actual = set(identifiers)
        extras = sorted(str(value) for value in actual - expected_attempt_ids)
        missing = sorted(expected_attempt_ids - actual)
        if extras or (require_complete_grid and missing):
            raise ProtocolError(
                f"artifact attempt frame mismatch: missing={missing}, extra={extras}"
            )
    return dict(sorted(Counter(record["status"] for record in records).items()))


_CERTIFICATE_FINGERPRINT_FIELDS = (
    "corpus_sha256",
    "manifest_sha256",
    "dependencies_sha256",
    "sources_sha256",
    "config_sha256",
)


def _deep_validation_source_checksums(
    expected_attempts: Mapping[str, AttemptSpec],
    expected_fingerprints: Mapping[str, str],
    *,
    expected_run_mode: str,
    require_complete_grid: bool,
) -> dict[str, Any]:
    attempts = [
        asdict(attempt)
        for attempt in sorted(
            expected_attempts.values(), key=lambda value: value.attempt_index
        )
    ]
    bound_fingerprints = {
        field: expected_fingerprints.get(field)
        for field in _CERTIFICATE_FINGERPRINT_FIELDS
    }
    payload = {
        "record_schema_version": SCHEMA_VERSION,
        "qmax": CANONICAL_QMAX,
        "expected_run_mode": expected_run_mode,
        "require_complete_grid": require_complete_grid,
        "attempt_frame_sha256": stable_hash(attempts),
        "attempt_count": len(attempts),
        "fingerprints": bound_fingerprints,
    }
    return {**payload, "sha256": stable_hash(payload)}


def _deep_validation_result_checksums(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record_checksums = [
        {
            "attempt_id": record.get("attempt_id"),
            "attempt_index": record.get("attempt_index"),
            "record_sha256": record.get("record_sha256"),
        }
        for record in sorted(
            records, key=lambda value: int(value.get("attempt_index", -1))
        )
    ]
    payload = {
        "record_count": len(records),
        "record_checksums_sha256": stable_hash(record_checksums),
    }
    return {**payload, "sha256": stable_hash(payload)}


def _deep_validation_certificate_sha256(
    certificate: Mapping[str, Any],
) -> str:
    payload = dict(certificate)
    payload.pop("certificate_sha256", None)
    return stable_hash(payload)


def create_deep_validation_certificate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_attempts: Mapping[str, AttemptSpec],
    expected_fingerprints: Mapping[str, str],
    require_complete_grid: bool,
    expected_run_mode: str,
) -> dict[str, Any]:
    """Run the one full science recomputation and record checksums of its exact inputs."""

    summary = validate_record_set(
        records,
        expected_attempts=expected_attempts,
        expected_fingerprints=expected_fingerprints,
        require_complete_grid=require_complete_grid,
        recompute_science=True,
        expected_run_mode=expected_run_mode,
    )
    certificate: dict[str, Any] = {
        "schema_version": DEEP_VALIDATION_CERTIFICATE_SCHEMA,
        "record_schema_version": SCHEMA_VERSION,
        "validation_kind": "full_source_and_science_recomputation",
        "recompute_science_performed": True,
        "created_at_utc": _utc_now(),
        "source_checksums": _deep_validation_source_checksums(
            expected_attempts,
            expected_fingerprints,
            expected_run_mode=expected_run_mode,
            require_complete_grid=require_complete_grid,
        ),
        "result_checksums": _deep_validation_result_checksums(records),
        "analysis_status_counts": summary,
        "deployment_status_counts": dict(
            sorted(Counter(record["deployment_status"] for record in records).items())
        ),
        "certificate_sha256": None,
    }
    certificate["certificate_sha256"] = _deep_validation_certificate_sha256(
        certificate
    )
    return certificate


def verify_deep_validation_certificate(
    certificate: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_attempts: Mapping[str, AttemptSpec],
    expected_fingerprints: Mapping[str, str],
    require_complete_grid: bool,
    expected_run_mode: str,
) -> dict[str, int]:
    """Verify a prior deep run without invoking MPS or statevector again."""

    summary = validate_record_set(
        records,
        expected_attempts=expected_attempts,
        expected_fingerprints=expected_fingerprints,
        require_complete_grid=require_complete_grid,
        recompute_science=False,
        expected_run_mode=expected_run_mode,
    )
    errors: list[str] = []
    if certificate.get("schema_version") != DEEP_VALIDATION_CERTIFICATE_SCHEMA:
        errors.append("invalid deep-validation certificate schema")
    if certificate.get("record_schema_version") != SCHEMA_VERSION:
        errors.append("deep-validation certificate binds another record schema")
    if certificate.get("validation_kind") != (
        "full_source_and_science_recomputation"
    ) or certificate.get("recompute_science_performed") is not True:
        errors.append("certificate does not attest full science recomputation")
    certificate_hash = certificate.get("certificate_sha256")
    if not _is_sha256(certificate_hash):
        errors.append("certificate lacks a valid integrity checksum")
    elif certificate_hash != _deep_validation_certificate_sha256(certificate):
        errors.append("deep-validation certificate integrity checksum mismatch")
    expected_source_checksums = _deep_validation_source_checksums(
        expected_attempts,
        expected_fingerprints,
        expected_run_mode=expected_run_mode,
        require_complete_grid=require_complete_grid,
    )
    if certificate.get("source_checksums") != expected_source_checksums:
        errors.append("deep-validation certificate source checksum mismatch")
    expected_result_checksums = _deep_validation_result_checksums(records)
    if certificate.get("result_checksums") != expected_result_checksums:
        errors.append("deep-validation certificate artifact checksum mismatch")
    if certificate.get("analysis_status_counts") != summary:
        errors.append("certificate analysis-status counts mismatch")
    deployment_counts = dict(
        sorted(Counter(record["deployment_status"] for record in records).items())
    )
    if certificate.get("deployment_status_counts") != deployment_counts:
        errors.append("certificate deployment-status counts mismatch")
    if errors:
        raise ProtocolError(
            "deep-validation certificate verification failed: " + "; ".join(errors)
        )
    return summary


def pending_attempts(
    attempts: Sequence[AttemptSpec],
    existing_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_fingerprints: Mapping[str, str],
) -> list[AttemptSpec]:
    pending = []
    for attempt in attempts:
        record = existing_records.get(attempt.attempt_id)
        if record is None:
            pending.append(attempt)
            continue
        compatibility_errors = _resume_compatibility_errors(
            record,
            expected_fingerprints=expected_fingerprints,
            expected_attempt=attempt,
        )
        if compatibility_errors:
            raise ProtocolError(
                f"resume record {attempt.attempt_id} is incompatible: "
                f"{compatibility_errors}"
            )
        if record.get("record_complete") is not True:
            pending.append(attempt)
            continue
        errors = validate_record(
            record,
            expected_fingerprints=expected_fingerprints,
            expected_attempt=attempt,
        )
        if errors:
            # A record that claims completion must itself validate. Only an
            # explicitly incomplete current-protocol record is rerunnable.
            raise ProtocolError(
                f"resume record {attempt.attempt_id} claims completion but is "
                f"invalid: {errors}"
            )
    return pending


def _resume_compatibility_errors(
    record: Mapping[str, Any],
    *,
    expected_fingerprints: Mapping[str, str],
    expected_attempt: AttemptSpec,
) -> list[str]:
    """Checks that apply even when a checkpoint entry is incomplete."""

    errors = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"invalid schema_version {record.get('schema_version')!r}")
    if record.get("attempt_id") != expected_attempt.attempt_id:
        errors.append(
            f"attempt_id={record.get('attempt_id')!r}, expected "
            f"{expected_attempt.attempt_id!r}"
        )
    if record.get("attempt_index") != expected_attempt.attempt_index:
        errors.append(
            f"attempt_index={record.get('attempt_index')!r}, expected "
            f"{expected_attempt.attempt_index}"
        )
    expected_source = asdict(expected_attempt)
    if record.get("source") != expected_source:
        errors.append(
            f"fixed source metadata mismatch: {record.get('source')!r} != "
            f"{expected_source!r}"
        )
    if type(record.get("qmax")) is not int or record.get("qmax") != CANONICAL_QMAX:
        errors.append(f"record qmax={record.get('qmax')!r}, expected exactly 24")
    fingerprints = record.get("fingerprints")
    if not isinstance(fingerprints, dict):
        errors.append("record lacks checksums")
    else:
        for field in (
            "corpus_sha256",
            "manifest_sha256",
            "dependencies_sha256",
            "sources_sha256",
            "config_sha256",
        ):
            if fingerprints.get(field) != expected_fingerprints.get(field):
                errors.append(
                    f"checksum mismatch for {field}: "
                    f"{fingerprints.get(field)!r} != {expected_fingerprints.get(field)!r}"
                )
    return errors


def _load_resume_records(
    path: Path,
    attempts: Sequence[AttemptSpec],
    expected_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    identifiers = [record.get("attempt_id") for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ProtocolError(f"resume artifact {path} contains duplicate attempts")
    expected = {attempt.attempt_id for attempt in attempts}
    extras = sorted(str(value) for value in set(identifiers) - expected)
    if extras:
        raise ProtocolError(f"resume artifact contains foreign attempts: {extras}")
    result = {record["attempt_id"]: record for record in records}
    # This call performs the strict old/mixed-Qmax and checksum checks while
    # permitting explicitly incomplete entries to be rerun below.
    pending_attempts(
        attempts, result, expected_fingerprints=expected_fingerprints
    )
    return result


def run_protocol(args: argparse.Namespace) -> int:
    if not args.smoke:
        backend = solver_provenance().get("backend")
        if backend != PUBLICATION_PARTITION_SOLVER_BACKEND:
            raise ProtocolError(
                "the full final run requires the launcher to set "
                f"QMON_MILP_SOLVER={PUBLICATION_PARTITION_SOLVER_BACKEND}; "
                f"actual backend={backend!r}"
            )
    frame = resolve_canonical_frame(args.manifest, args.qasm_dir)
    grid = build_attempt_grid(frame)
    attempts = select_attempts(
        grid,
        smoke=args.smoke,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    bundle = build_fingerprint_bundle(frame)
    expected_fingerprints = _record_fingerprint_hashes(bundle)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise ProtocolError(
            f"output already exists: {output}; pass --resume to retain complete "
            "terminal outcomes and fill only missing/incomplete attempts"
        )
    existing = (
        _load_resume_records(output, attempts, expected_fingerprints)
        if output.exists()
        else {}
    )
    pending = pending_attempts(
        attempts, existing, expected_fingerprints=expected_fingerprints
    )
    print(
        f"base={frame['circuit_count']} corpus={frame['corpus_sha256']} "
        f"excluded_qasm_extras={frame['excluded_extra_qasm_count']}"
    )
    print(
        f"attempts in shard={len(attempts)} pending={len(pending)} "
        f"qmax={CANONICAL_QMAX} timeout={args.timeout}s"
    )

    run_mode = "smoke" if args.smoke else "full"
    records_by_id = dict(existing)
    for position, attempt in enumerate(pending, start=1):
        print(f"[{position}/{len(pending)}] {attempt.attempt_id}", flush=True)
        record = run_attempt_with_timeout(
            attempt,
            bundle,
            timeout_seconds=args.timeout,
            run_mode=run_mode,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        records_by_id[attempt.attempt_id] = record
        ordered = [
            records_by_id[candidate.attempt_id]
            for candidate in attempts
            if candidate.attempt_id in records_by_id
        ]
        atomic_write_jsonl(ordered, output)
        category = (record.get("error") or {}).get("category")
        print(
            f"  analysis_status={record['status']} "
            f"deployment_status={record['deployment_status']} "
            f"elapsed={record['elapsed_seconds']:.3f}s category={category}",
            flush=True,
        )

    ordered = [records_by_id[attempt.attempt_id] for attempt in attempts]
    summary = validate_record_set(
        ordered,
        expected_attempts={attempt.attempt_id: attempt for attempt in attempts},
        expected_fingerprints=expected_fingerprints,
        require_complete_grid=True,
        expected_run_mode=run_mode,
    )
    atomic_write_jsonl(ordered, output)
    print(f"wrote {output}: analysis_statuses={summary}")
    return 0


def _current_context(args: argparse.Namespace) -> tuple[
    dict[str, Any], list[AttemptSpec], dict[str, str]
]:
    frame = resolve_canonical_frame(args.manifest, args.qasm_dir)
    grid = build_attempt_grid(frame)
    bundle = build_fingerprint_bundle(frame)
    return frame, grid, _record_fingerprint_hashes(bundle)


def validate_protocol(args: argparse.Namespace) -> int:
    _, grid, fingerprints = _current_context(args)
    attempts = select_attempts(
        grid, smoke=args.smoke, shard_index=0, shard_count=1
    )
    records = []
    for path in args.inputs:
        records.extend(read_jsonl(path))
    expected_attempts = {attempt.attempt_id: attempt for attempt in attempts}
    expected_run_mode = "smoke" if args.smoke else "full"
    if args.deep_certificate_out is not None:
        certificate_path = Path(args.deep_certificate_out).resolve()
        if certificate_path.exists() and not args.overwrite_certificate:
            raise ProtocolError(
                f"deep-validation certificate already exists: {certificate_path}; "
                "pass --overwrite-certificate"
            )
        certificate = create_deep_validation_certificate(
            records,
            expected_attempts=expected_attempts,
            expected_fingerprints=fingerprints,
            require_complete_grid=args.complete_grid,
            expected_run_mode=expected_run_mode,
        )
        atomic_write_json_object(certificate, certificate_path)
        summary = certificate["analysis_status_counts"]
        certificate_message = f"deep certificate wrote {certificate_path}"
    else:
        certificate = read_json_object(args.certificate)
        summary = verify_deep_validation_certificate(
            certificate,
            records,
            expected_attempts=expected_attempts,
            expected_fingerprints=fingerprints,
            require_complete_grid=args.complete_grid,
            expected_run_mode=expected_run_mode,
        )
        certificate_message = f"certificate verified {Path(args.certificate).resolve()}"
    print(
        f"valid records={len(records)} qmax=24 analysis_statuses={summary} "
        f"denominators={artifact_denominators(records)}; {certificate_message}"
    )
    return 0


def merge_protocol(args: argparse.Namespace) -> int:
    _, grid, fingerprints = _current_context(args)
    attempts = select_attempts(
        grid, smoke=args.smoke, shard_index=0, shard_count=1
    )
    records = []
    for path in args.inputs:
        records.extend(read_jsonl(path))
    summary = verify_deep_validation_certificate(
        read_json_object(args.certificate),
        records,
        expected_attempts={attempt.attempt_id: attempt for attempt in attempts},
        expected_fingerprints=fingerprints,
        require_complete_grid=True,
        expected_run_mode="smoke" if args.smoke else "full",
    )
    by_index = sorted(records, key=lambda record: int(record["attempt_index"]))
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise ProtocolError(f"merge output already exists: {output}; pass --overwrite")
    atomic_write_jsonl(by_index, output)
    print(
        f"merged {len(records)} records into {output}: "
        f"analysis_statuses={summary}; "
        f"denominators={artifact_denominators(records)}"
    )
    return 0


def export_figure_data(args: argparse.Namespace) -> int:
    """Export a complete, status-inclusive v2 frame for future figures."""

    _, grid, fingerprints = _current_context(args)
    attempts = select_attempts(grid, smoke=False, shard_index=0, shard_count=1)
    records = read_jsonl(args.input)
    summary = verify_deep_validation_certificate(
        read_json_object(args.certificate),
        records,
        expected_attempts={attempt.attempt_id: attempt for attempt in attempts},
        expected_fingerprints=fingerprints,
        require_complete_grid=True,
        expected_run_mode="full",
    )
    rows = figure_data_rows(records)
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise ProtocolError(
            f"figure-data output already exists: {output}; pass --overwrite"
        )
    atomic_write_figure_csv(rows, output)
    print(
        f"exported v2 figure data to {output}: attempted_denominator={len(rows)} "
        f"analysis_statuses={summary}; "
        f"denominators={artifact_denominators(records)}; "
        "x_axis=original_qubits qmax=24"
    )
    return 0


def compare_base_criteria_protocol(args: argparse.Namespace) -> int:
    """Prove exact 310-circuit continuity with analyze_per_gate_nolock."""

    frame = resolve_canonical_frame(args.manifest, args.qasm_dir)
    base_attempts = build_attempt_grid(frame)[:EXPECTED_BASE_COUNT]
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise ProtocolError(f"criterion-comparison output already exists: {output}")
    records: list[dict[str, Any]] = []
    for position, attempt in enumerate(base_attempts, start=1):
        started = time.perf_counter()
        qc = _build_circuit(attempt)
        eligibility_category = None
        try:
            validate_circuit_eligibility(qc)
        except CircuitEligibilityError as exc:
            if exc.category != "nonterminal_barrier":
                raise
            # This compares numerical selectors, not operational theorem
            # eligibility. Barriers are state-preserving directives and both
            # selectors omit them, so retain the circuit and record the reason
            # it would be unsupported by the final run.
            eligibility_category = exc.category
        comparison = compare_statevector_selection_to_reference(qc)
        record = {
            "schema_version": "qmon.scalability.v2.criterion-comparison/4",
            "attempt_id": attempt.attempt_id,
            "attempt_index": attempt.attempt_index,
            "basename": attempt.basename,
            "qasm_sha256": attempt.qasm_sha256,
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "selector": "second_schmidt_amplitude",
            "schmidt_amplitude_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
            "reference": "fast_analysis.analyze_per_gate_nolock",
            "operational_eligibility_category": eligibility_category,
            "elapsed_seconds": round(time.perf_counter() - started, 9),
            **comparison,
        }
        records.append(record)
        atomic_write_jsonl(records, output)
        if comparison["selection_mismatch_count"]:
            raise ProtocolError(
                "semantic continuity failure for "
                f"{attempt.attempt_id}: {comparison['mismatch_examples']}"
            )
        print(
            f"[{position}/{len(base_attempts)}] {attempt.attempt_id} "
            f"points={comparison['total_acted_points']} "
            f"selected={comparison['v2_selected_event_count']} "
            f"mismatches={comparison['selection_mismatch_count']}",
            flush=True,
        )
    summary = criterion_comparison_summary(records)
    scientific_hash = criterion_comparison_scientific_sha256(records)
    expected_hash = BASE_CRITERION_REFERENCE["scientific_records_sha256"]
    if scientific_hash != expected_hash:
        raise ProtocolError(
            "criterion continuity comparison changed from the fixed scientific "
            f"result: {scientific_hash} != {expected_hash}"
        )
    print(
        f"criterion comparison wrote {output}: {summary}; "
        f"scientific_sha256={scientific_hash}"
    )
    return 0


def compare_base_mps_protocol(args: argparse.Namespace) -> int:
    """Cross-check native MPS and direct statevector a2 on all 310 bases."""

    frame = resolve_canonical_frame(args.manifest, args.qasm_dir)
    base_attempts = build_attempt_grid(frame)[:EXPECTED_BASE_COUNT]
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise ProtocolError(f"MPS cross-check output already exists: {output}")
    records: list[dict[str, Any]] = []
    for position, attempt in enumerate(base_attempts, start=1):
        started = time.perf_counter()
        qc = _build_circuit(attempt)
        eligibility_category = None
        try:
            validate_circuit_eligibility(qc)
        except CircuitEligibilityError as exc:
            if exc.category != "nonterminal_barrier":
                raise
            eligibility_category = exc.category

        mps_started = time.perf_counter()
        mps_nodes, mps_diagnostics, mps_scores = (
            _select_monitorable_nodes_mps_details(qc)
        )
        mps_runtime = time.perf_counter() - mps_started
        comparison = crosscheck_mps_against_statevector(
            qc, mps_nodes, mps_scores
        )
        record = {
            "schema_version": "qmon.scalability.v2.mps-crosscheck/2",
            "attempt_id": attempt.attempt_id,
            "attempt_index": attempt.attempt_index,
            "basename": attempt.basename,
            "qasm_sha256": attempt.qasm_sha256,
            "manifest_sha256": frame["manifest_sha256"],
            "corpus_sha256": frame["corpus_sha256"],
            "qmax": CANONICAL_QMAX,
            "selector": "second_schmidt_amplitude",
            "schmidt_amplitude_tolerance": SCHMIDT_AMPLITUDE_TOLERANCE,
            "mps_a2_extraction": "aer_mps_local_tensor_svd",
            "mps_max_bond_dimension": MPS_MAX_BOND_DIMENSION,
            "mps_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
            "mps_lapack": MPS_LAPACK,
            "operational_eligibility_category": eligibility_category,
            "observed_max_chi": mps_diagnostics["observed_max_chi"],
            "mps_runtime_seconds": round(mps_runtime, 9),
            "elapsed_seconds": round(time.perf_counter() - started, 9),
            **comparison,
        }
        records.append(record)
        atomic_write_jsonl(records, output)
        print(
            f"[{position}/{len(base_attempts)}] {attempt.attempt_id} "
            f"points={comparison['point_count_mps']} "
            f"mps_selected={comparison['mps_selected_event_count']} "
            f"statevector_selected={comparison['statevector_selected_event_count']} "
            f"mismatches={comparison['selection_mismatch_count']} "
            f"max_a2_diff={comparison['maximum_absolute_a2_difference']:.3e}",
            flush=True,
        )

    summary = mps_crosscheck_summary(records)
    scientific_hash = mps_crosscheck_scientific_sha256(records)
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
        if summary[field] != BASE_MPS_CROSSCHECK_REFERENCE[field]:
            raise ProtocolError(
                "native MPS/statevector comparison changed from the fixed result: "
                f"{field}={summary[field]!r} != "
                f"{BASE_MPS_CROSSCHECK_REFERENCE[field]!r}"
            )
    expected_hash = BASE_MPS_CROSSCHECK_REFERENCE["scientific_records_sha256"]
    if scientific_hash != expected_hash:
        raise ProtocolError(
            "native MPS/statevector scientific event evidence changed: "
            f"{scientific_hash} != {expected_hash}"
        )
    if (
        summary["selection_mismatch_count"]
        or summary["missing_mps_point_count"]
        or summary["missing_statevector_point_count"]
    ):
        raise ProtocolError(
            "native MPS/statevector full-base cross-check failed: "
            f"{summary}; scientific_sha256={scientific_hash}"
        )
    print(
        f"MPS cross-check wrote {output}: {summary}; "
        f"scientific_sha256={scientific_hash}"
    )
    return 0


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the declared three-attempt smoke frame",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run = subparsers.add_parser("run", help="run one deterministic attempt shard")
    _add_frame_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument(
        "--resume",
        action="store_true",
        help="retain every complete terminal record; run only missing/incomplete",
    )
    run.set_defaults(handler=run_protocol)

    validate = subparsers.add_parser(
        "validate", help="validate Qmax, checksums, records, and batches"
    )
    _add_frame_arguments(validate)
    validate.add_argument("inputs", nargs="+", type=Path)
    validate.add_argument("--complete-grid", action="store_true")
    validation_mode = validate.add_mutually_exclusive_group(required=True)
    validation_mode.add_argument(
        "--deep-certificate-out",
        type=Path,
        help="run the one full science recomputation and write its certificate",
    )
    validation_mode.add_argument(
        "--certificate",
        type=Path,
        help="verify a prior deep-validation certificate without recomputing MPS",
    )
    validate.add_argument("--overwrite-certificate", action="store_true")
    validate.set_defaults(handler=validate_protocol)

    merge = subparsers.add_parser(
        "merge", help="validate and atomically merge a complete shard set"
    )
    _add_frame_arguments(merge)
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--certificate", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(handler=merge_protocol)

    export = subparsers.add_parser(
        "export-figure-data",
        help="export all 548 validated v2 attempts; partial/survivor inputs fail",
    )
    export.add_argument("input", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--certificate", type=Path, required=True)
    export.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    export.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=export_figure_data)

    criteria_comparison = subparsers.add_parser(
        "compare-base-criteria",
        help=(
            "prove exact a2<=1e-12 event-set continuity with "
            "analyze_per_gate_nolock on all 310 base circuits"
        ),
    )
    criteria_comparison.add_argument("--output", type=Path, required=True)
    criteria_comparison.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    criteria_comparison.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    criteria_comparison.add_argument("--overwrite", action="store_true")
    criteria_comparison.set_defaults(handler=compare_base_criteria_protocol)

    mps_comparison = subparsers.add_parser(
        "compare-base-mps",
        help=(
            "cross-check native MPS tensor-SVD a2 against direct "
            "statevector SVD on all 310 base circuits"
        ),
    )
    mps_comparison.add_argument("--output", type=Path, required=True)
    mps_comparison.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mps_comparison.add_argument("--qasm-dir", type=Path, default=DEFAULT_QASM_DIR)
    mps_comparison.add_argument("--overwrite", action="store_true")
    mps_comparison.set_defaults(handler=compare_base_mps_protocol)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS) <= 0:
            raise ProtocolError("--timeout must be positive")
        return int(args.handler(args))
    except ProtocolError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
