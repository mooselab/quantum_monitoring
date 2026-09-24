"""Reference benchmark-family labels used by family-aware analyses."""

from __future__ import annotations


FAMILY_ALIASES = {
    # This file is a second Bernstein-Vazirani parameter instance.  The
    # ``alt`` suffix distinguishes the source filename, not the algorithm.
    "bv_alt": "bv",
}


def canonical_family_of(circuit: str) -> str:
    stem = circuit[:-5] if circuit.endswith(".qasm") else circuit
    raw_family = stem.rsplit("_indep_qiskit_", 1)[0]
    return FAMILY_ALIASES.get(raw_family, raw_family)
