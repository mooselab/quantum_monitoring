"""Fixed single-error mutation list for the RQ3 experiment.

The builder in this module is the only place that draws the single-error RQ3
mutation stream.  It reproduces the RNG protocol of the main RQ3 run:

* seed Python's global ``random`` module once per ``(circuit, seed)``;
* draw three gate indices with ``random.sample``;
* call the original ``mutate_circuit`` three times, in sampled-gate order,
  without reseeding between calls.

Consumers replay the serialized replacement operation with
``apply_manifest_slot``.  They never redraw a replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2


SCRIPT_DIR = Path(__file__).resolve().parent
RECONSTRUCTION_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = RECONSTRUCTION_DIR.parent
CIRCUIT_DIR = WORKSPACE_DIR / "quantum_circuits"
CANONICAL_KEYFILE = RECONSTRUCTION_DIR / "inputs" / "rq3_circuits.txt"
DEFAULT_MANIFEST = (
    RECONSTRUCTION_DIR / "inputs" / "rq3_single_error_manifest.json"
)

SCHEMA = "qmon-rq3-single-error-manifest-v1"
SEEDS = (1, 2, 3, 4, 5)
SLOTS_PER_SEED = 3
EXPECTED_CIRCUITS = 238
EXPECTED_SLOTS = 3570
EXPECTED_EQUIVALENT = 710
EXPECTED_NON_EQUIVALENT = 2860
OUTPUT_EQ_L1 = 1e-9
_REPLAY_VERIFIED_HASHES = set()

PROTOCOL = {
    "rng": (
        "per circuit/seed: random.seed(seed), random.sample(candidates, 3), "
        "then legacy mutate_circuit in sampled order without reseeding"
    ),
    "candidate_rule": (
        "one/two-qubit unitary data indices not later than the final "
        "no-lock monitorable node"
    ),
    "seeds": list(SEEDS),
    "slots_per_seed": SLOTS_PER_SEED,
    "output_equivalence_l1": OUTPUT_EQ_L1,
}


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def stable_hash(value):
    """SHA-256 of a JSON value serialized with sorted keys and compact
    separators."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_key(key):
    key = str(key).strip()
    return key if key.endswith(".qasm") else key + ".qasm"


def canonical_keys(path=CANONICAL_KEYFILE):
    with Path(path).open() as handle:
        keys = [
            normalize_key(line.split("#", 1)[0].strip())
            for line in handle
            if line.split("#", 1)[0].strip()
        ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate circuit key in {path}")
    return keys


def _plain_param(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return repr(value)


def operation_signature(qc, gate):
    """Stable identity of one circuit instruction."""
    ci = qc.data[int(gate)]
    return {
        "name": ci.operation.name,
        "params": [_plain_param(value) for value in ci.operation.params],
        "qubits": [qc.find_bit(bit).index for bit in ci.qubits],
        "clbits": [qc.find_bit(bit).index for bit in ci.clbits],
        "num_qubits": int(ci.operation.num_qubits),
        "num_clbits": int(ci.operation.num_clbits),
    }


def circuit_qasm_sha256(qc):
    return hashlib.sha256(qasm2.dumps(qc).encode("utf-8")).hexdigest()


def dependency_versions():
    packages = (
        "qiskit",
        "qiskit-aer",
        "numpy",
        "scipy",
        "pytket",
        "pytket-qiskit",
        "mqt.debugger",
        "pulp",
        "gurobipy",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def qasm_corpus_fingerprint(keys=None):
    keys = canonical_keys() if keys is None else [normalize_key(k) for k in keys]
    entries = []
    for key in keys:
        path = CIRCUIT_DIR / key
        if not path.is_file():
            raise FileNotFoundError(f"circuit QASM file missing: {path}")
        entries.append({"circuit": key, "sha256": file_sha256(path)})
    return {
        "keyfile_sha256": file_sha256(CANONICAL_KEYFILE),
        "circuit_count": len(entries),
        "sha256": stable_hash(entries),
        "circuits": entries,
    }


def source_fingerprint(paths=None):
    if paths is None:
        paths = (
            SCRIPT_DIR / "rq3_manifest.py",
            SCRIPT_DIR / "mutation.py",
            SCRIPT_DIR / "rq3_eval.py",
            SCRIPT_DIR / "fast_analysis.py",
            SCRIPT_DIR / "mps_analysis.py",
            SCRIPT_DIR / "utilities.py",
        )
    entries = [
        {"path": Path(path).name, "sha256": file_sha256(path)}
        for path in paths
    ]
    return {"sha256": stable_hash(entries), "files": entries}


def runtime_fingerprints(protocol, *, source_paths=None, keys=None):
    """Checksums stored in experiment artifacts and checked on resume."""
    dependencies = dependency_versions()
    corpus = qasm_corpus_fingerprint(keys)
    sources = source_fingerprint(source_paths)
    return {
        "sources": sources,
        "qasm_corpus": corpus,
        "dependencies": {
            "versions": dependencies,
            "sha256": stable_hash(dependencies),
        },
        "protocol": {
            "value": protocol,
            "sha256": stable_hash(protocol),
        },
    }


def _replay_cache_key(manifest, current=None):
    """Bind a replay check to both the fixed mutation list and the runtime
    environment."""
    if current is None:
        current = runtime_fingerprints(PROTOCOL)
    return (
        manifest["manifest_sha256"],
        current["sources"]["sha256"],
        current["qasm_corpus"]["sha256"],
        current["dependencies"]["sha256"],
        current["protocol"]["sha256"],
    )


def _empty_like(qc):
    out = QuantumCircuit()
    for register in qc.qregs:
        out.add_register(QuantumRegister(register.size, register.name))
    for register in qc.cregs:
        out.add_register(ClassicalRegister(register.size, register.name))
    out.global_phase = qc.global_phase
    return out


def apply_replacement(qc, gate, replacement, *, expected_original=None):
    """Return ``qc`` with exactly one serialized replacement applied."""
    gate = int(gate)
    if not 0 <= gate < len(qc.data):
        raise IndexError(f"mutation gate {gate} outside circuit of size {len(qc.data)}")
    actual = operation_signature(qc, gate)
    if expected_original is not None and actual != expected_original:
        raise ValueError(
            f"source operation drift at gate {gate}: {actual!r} != "
            f"{expected_original!r}"
        )
    out = _empty_like(qc)
    for index, ci in enumerate(qc.data):
        if index != gate:
            out.append(
                ci.operation,
                [out.qubits[qc.find_bit(bit).index] for bit in ci.qubits],
                [out.clbits[qc.find_bit(bit).index] for bit in ci.clbits],
            )
            continue
        qubits = [out.qubits[int(q)] for q in replacement["qubits"]]
        params = list(replacement.get("params", ()))
        method = getattr(out, replacement["name"], None)
        if method is None:
            raise ValueError(f"unsupported replacement gate {replacement['name']!r}")
        method(*params, *qubits)
    return out


def apply_manifest_slot(qc, slot, *, verify_hash=True):
    """Replay one mutation-list slot without drawing any random value."""
    if circuit_qasm_sha256(qc) != slot["source_qasm_sha256"]:
        raise ValueError(f"source QASM drift for mutation-list slot {slot['slot_id']}")
    mutant = apply_replacement(
        qc,
        slot["gate"],
        slot["replacement"],
        expected_original=slot["original"],
    )
    if verify_hash and circuit_qasm_sha256(mutant) != slot["mutant_qasm_sha256"]:
        raise ValueError(f"mutant replay hash mismatch for {slot['slot_id']}")
    return mutant


def apply_manifest_slots(qc, slots):
    """Apply several single-error replacements from the mutation list as one
    nested multi-mutant."""
    mutant = qc
    seen = set()
    for slot in slots:
        gate = int(slot["gate"])
        if gate in seen:
            raise ValueError(f"duplicate gate in nested mutation-list slots: {gate}")
        seen.add(gate)
        if circuit_qasm_sha256(qc) != slot["source_qasm_sha256"]:
            raise ValueError(f"source QASM drift for mutation-list slot {slot['slot_id']}")
        mutant = apply_replacement(
            mutant,
            gate,
            slot["replacement"],
            expected_original=slot["original"],
        )
    return mutant


def _builder_imports():
    # Sibling imports are intentionally delayed so metadata-only consumers do
    # not pay the statevector/analysis import cost.
    import sys

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from mutation import mutate_circuit
    from rq3_eval import final_probs, monitored_nodes, unitary_gate_indices
    from utilities import quantum_gates

    return mutate_circuit, final_probs, monitored_nodes, unitary_gate_indices, quantum_gates


def build_manifest(keys=None):
    """Build the complete mutation list in memory."""
    keys = canonical_keys() if keys is None else [normalize_key(k) for k in keys]
    mutate_circuit, final_probs, monitored_nodes, unitary_gate_indices, quantum_gates = (
        _builder_imports()
    )
    slots = []
    circuit_entries = []
    for key in keys:
        path = CIRCUIT_DIR / key
        qc = QuantumCircuit.from_qasm_file(path)
        nodes = monitored_nodes(qc)
        if not nodes:
            raise ValueError(f"RQ3 circuit has no monitorable nodes: {key}")
        last_node = max(index for index, _ in nodes)
        candidates = [
            gate for gate in unitary_gate_indices(qc) if gate <= last_node
        ]
        if len(candidates) < SLOTS_PER_SEED:
            raise ValueError(f"RQ3 circuit has fewer than three candidates: {key}")
        measured = [
            qc.find_bit(ci.qubits[0]).index
            for ci in qc.data
            if ci.operation.name == "measure"
        ]
        base_probs = final_probs(qc, measured)
        source_hash = circuit_qasm_sha256(qc)
        circuit_entries.append(
            {
                "circuit": key,
                "source_qasm_sha256": source_hash,
                "candidate_count": len(candidates),
                "last_monitorable_gate": int(last_node),
            }
        )
        for seed in SEEDS:
            random.seed(seed)
            gates = random.sample(candidates, SLOTS_PER_SEED)
            for ordinal, gate in enumerate(gates):
                original = operation_signature(qc, gate)
                mutant = mutate_circuit(qc, gate, quantum_gates)
                replacement = operation_signature(mutant, gate)
                if replacement == original:
                    raise AssertionError(f"no-op mutation {key}/s{seed}/{ordinal}")
                mutant_probs = final_probs(mutant, measured)
                l1 = (
                    float(np.abs(base_probs - mutant_probs).sum())
                    if len(base_probs) == len(mutant_probs)
                    else None
                )
                equivalent = bool(l1 is not None and l1 < OUTPUT_EQ_L1)
                slot_id = f"{key}|s{seed}|m{ordinal}"
                slots.append(
                    {
                        "slot_id": slot_id,
                        "circuit": key,
                        "seed": int(seed),
                        "ordinal": int(ordinal),
                        "gate": int(gate),
                        "source_qasm_sha256": source_hash,
                        "original": original,
                        "replacement": replacement,
                        "mutant_qasm_sha256": circuit_qasm_sha256(mutant),
                        "output_equivalent": equivalent,
                        "output_l1": l1,
                    }
                )
    equivalent_count = sum(slot["output_equivalent"] for slot in slots)
    manifest = {
        "schema": SCHEMA,
        "protocol": dict(PROTOCOL),
        "fingerprints": runtime_fingerprints(PROTOCOL, keys=keys),
        "circuits": circuit_entries,
        "counts": {
            "circuits": len(keys),
            "slots": len(slots),
            "output_equivalent": equivalent_count,
            "non_output_equivalent": len(slots) - equivalent_count,
        },
        "slots": slots,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    validate_manifest(manifest, require_canonical_counts=(keys == canonical_keys()))
    return manifest


def validate_manifest(manifest, *, require_canonical_counts=True,
                      verify_runtime=False, verify_replay=None):
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} mutation list")
    expected_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if expected_hash != stable_hash(unhashed):
        raise ValueError("mutation list content hash mismatch")
    slots = manifest.get("slots")
    if not isinstance(slots, list):
        raise ValueError("mutation list slots must be a list")
    identities = [
        (slot.get("circuit"), slot.get("seed"), slot.get("ordinal"))
        for slot in slots
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate slot identity")
    actual_counts = {
        "circuits": len({slot["circuit"] for slot in slots}),
        "slots": len(slots),
        "output_equivalent": sum(bool(slot["output_equivalent"]) for slot in slots),
        "non_output_equivalent": sum(
            not bool(slot["output_equivalent"]) for slot in slots
        ),
    }
    if manifest.get("counts") != actual_counts:
        raise ValueError(
            f"mutation list count metadata mismatch: {manifest.get('counts')} != {actual_counts}"
        )
    if require_canonical_counts:
        expected_counts = {
            "circuits": EXPECTED_CIRCUITS,
            "slots": EXPECTED_SLOTS,
            "output_equivalent": EXPECTED_EQUIVALENT,
            "non_output_equivalent": EXPECTED_NON_EQUIVALENT,
        }
        if actual_counts != expected_counts:
            raise AssertionError(
                f"RQ3 count regression: {actual_counts} != {expected_counts}"
            )
    circuits = manifest.get("circuits")
    if not isinstance(circuits, list):
        raise ValueError("mutation list circuits must be a list")
    circuit_keys = [entry.get("circuit") for entry in circuits]
    if (len(circuit_keys) != len(set(circuit_keys))
            or set(circuit_keys) != {slot["circuit"] for slot in slots}):
        raise ValueError("mutation list circuit table is incomplete or duplicated")
    by_circuit_seed = {}
    for slot in slots:
        required = {
            "slot_id", "circuit", "seed", "ordinal", "gate", "original",
            "replacement", "source_qasm_sha256", "mutant_qasm_sha256",
            "output_equivalent", "output_l1",
        }
        if set(slot) != required:
            raise ValueError(
                f"mutation-list slot has unexpected fields: "
                f"missing={sorted(required - set(slot))}, "
                f"extra={sorted(set(slot) - required)}")
        identity = (slot["circuit"], int(slot["seed"]))
        by_circuit_seed.setdefault(identity, []).append(slot)
    for identity, group in by_circuit_seed.items():
        ordered = sorted(group, key=lambda slot: int(slot["ordinal"]))
        if (len(ordered) != SLOTS_PER_SEED
                or [int(slot["ordinal"]) for slot in ordered]
                != list(range(SLOTS_PER_SEED))):
            raise ValueError(f"malformed slot ordinals for {identity}")
    if verify_replay is None:
        verify_replay = verify_runtime
    current = None
    if verify_runtime:
        current = runtime_fingerprints(PROTOCOL)
        stored = manifest.get("fingerprints")
        if not isinstance(stored, dict):
            raise ValueError("mutation list missing runtime checksums")
        # QASM, dependencies, and the mutation protocol are fixed inputs.
        # Source hashes describe the code that generated the stored slots.
        # Later bug fixes may change those files without changing any slot;
        # such drift is accepted only after a deterministic replay of every
        # gate, replacement, mutant QASM hash, and equivalence label.
        for field in ("qasm_corpus", "dependencies", "protocol"):
            if stored.get(field, {}).get("sha256") != current[field]["sha256"]:
                raise ValueError(f"mutation list {field} checksum mismatch")
        source_drift = (
            stored.get("sources", {}).get("sha256")
            != current["sources"]["sha256"]
        )
        if source_drift and not verify_replay:
            raise ValueError(
                "mutation list source checksum mismatch requires "
                "complete deterministic replay"
            )
    replay_key = _replay_cache_key(manifest, current) if verify_replay else None
    if verify_replay and replay_key not in _REPLAY_VERIFIED_HASHES:
        assert_manifest_replay(manifest)
        _REPLAY_VERIFIED_HASHES.add(replay_key)
    return manifest


def save_manifest(manifest, path=DEFAULT_MANIFEST):
    validate_manifest(manifest, require_canonical_counts=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(manifest))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_manifest(path=DEFAULT_MANIFEST, *, verify_runtime=True):
    with Path(path).open("rb") as handle:
        manifest = json.load(handle)
    return validate_manifest(manifest, verify_runtime=verify_runtime)


def slots_for_circuit(manifest, key):
    key = normalize_key(key)
    slots = [slot for slot in manifest["slots"] if slot["circuit"] == key]
    return sorted(slots, key=lambda slot: (int(slot["seed"]), int(slot["ordinal"])))


def slot_lookup(manifest):
    return {
        (slot["circuit"], int(slot["seed"]), int(slot["ordinal"])): slot
        for slot in manifest["slots"]
    }


def assert_manifest_replay(manifest):
    """Rebuild and validate every mutation slot from the source QASM."""
    slots_by_identity = {}
    for slot in manifest["slots"]:
        identity = (slot["circuit"], int(slot["seed"]))
        slots_by_identity.setdefault(identity, []).append(slot)
    mutate_circuit, final_probs, monitored_nodes, unitary_gate_indices, quantum_gates = (
        _builder_imports())
    entries = {entry["circuit"]: entry for entry in manifest["circuits"]}
    random_state = random.getstate()
    try:
        for key in canonical_keys():
            qc = QuantumCircuit.from_qasm_file(CIRCUIT_DIR / key)
            source_hash = circuit_qasm_sha256(qc)
            nodes = monitored_nodes(qc)
            last_node = max(index for index, _ in nodes)
            candidates = [
                gate for gate in unitary_gate_indices(qc)
                if gate <= last_node
            ]
            measured = [
                qc.find_bit(ci.qubits[0]).index
                for ci in qc.data
                if ci.operation.name == "measure"
            ]
            base_probs = final_probs(qc, measured)
            expected_entry = {
                "circuit": key,
                "source_qasm_sha256": source_hash,
                "candidate_count": len(candidates),
                "last_monitorable_gate": int(last_node),
            }
            if entries.get(key) != expected_entry:
                raise AssertionError(
                    f"mutation list circuit metadata drift for {key}")
            for seed in SEEDS:
                identity = (key, seed)
                group = sorted(
                    slots_by_identity[identity],
                    key=lambda slot: int(slot["ordinal"]))
                random.seed(seed)
                expected_gates = random.sample(candidates, SLOTS_PER_SEED)
                for ordinal, (slot, gate) in enumerate(zip(group, expected_gates)):
                    if (slot["slot_id"] != f"{key}|s{seed}|m{ordinal}"
                            or int(slot["gate"]) != int(gate)
                            or slot["source_qasm_sha256"] != source_hash
                            or slot["original"] != operation_signature(qc, gate)):
                        raise AssertionError(
                            f"mutation list source/gate drift for {key}/s{seed}/m{ordinal}")
                    mutant = mutate_circuit(qc, gate, quantum_gates)
                    replacement = operation_signature(mutant, gate)
                    mutant_hash = circuit_qasm_sha256(mutant)
                    if slot["replacement"] != replacement:
                        raise AssertionError(
                            f"mutation list replacement drift for {slot['slot_id']}")
                    if slot["mutant_qasm_sha256"] != mutant_hash:
                        raise AssertionError(
                            f"mutation list mutant hash drift for {slot['slot_id']}")
                    replayed = apply_manifest_slot(qc, slot)
                    if circuit_qasm_sha256(replayed) != mutant_hash:
                        raise AssertionError(
                            f"serialized replacement replay drift for "
                            f"{slot['slot_id']}")
                    mutant_probs = final_probs(mutant, measured)
                    output_l1 = (
                        float(np.abs(base_probs - mutant_probs).sum())
                        if len(base_probs) == len(mutant_probs)
                        else None
                    )
                    stored_l1 = slot["output_l1"]
                    if output_l1 is None or stored_l1 is None:
                        l1_matches = output_l1 is stored_l1
                    else:
                        l1_matches = bool(np.isclose(
                            output_l1,
                            float(stored_l1),
                            rtol=0.0,
                            atol=1e-12,
                        ))
                    if not l1_matches:
                        raise AssertionError(
                            f"mutation list output L1 drift for {slot['slot_id']}: "
                            f"{stored_l1!r} != {output_l1!r}")
                    output_equivalent = bool(
                        output_l1 is not None and output_l1 < OUTPUT_EQ_L1)
                    if bool(slot["output_equivalent"]) != output_equivalent:
                        raise AssertionError(
                            f"mutation list output-equivalence drift for "
                            f"{slot['slot_id']}")
    finally:
        random.setstate(random_state)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="build the mutation list from the circuit set")
    parser.add_argument("--validate", action="store_true",
                        help="validate the mutation list and print its counts")
    parser.add_argument("--replay", action="store_true",
                        help="rebuild every slot from source QASM and compare")
    parser.add_argument("--out", default=str(DEFAULT_MANIFEST),
                        help="path of the mutation list JSON file")
    args = parser.parse_args(argv)
    if args.build:
        manifest = build_manifest()
        save_manifest(manifest, args.out)
    else:
        manifest = load_manifest(args.out)
    if args.validate or args.build:
        print(
            f"{manifest['schema']}: {manifest['counts']}; "
            f"sha256={manifest['manifest_sha256']}"
        )
    if args.replay:
        assert_manifest_replay(manifest)
        print(f"all {len(manifest['slots'])} slots replay from source QASM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
