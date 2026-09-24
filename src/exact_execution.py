"""Exact channel simulation for QMon instrumented circuits.

Mid-circuit ``measure; reset`` pairs are evaluated as a weighted mixture of
statevector branches.  A certified-separable read converges to one quantum
branch after reset.  An entangled read is retained as both physical branches;
it is never approximated by choosing the more likely branch.
"""

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace


SKIP_OPS = ("barrier", "delay", "snapshot")
DEFAULT_BRANCH_CAP = 64


class ExactExecutionIncomplete(RuntimeError):
    """The exact channel could not be represented without approximation."""


class ExactExecutionViolation(RuntimeError):
    """A reported QMon read violated its separability certificate."""


def _slices(state_data, n, q):
    """Return the q=0 and q=1 slices of a statevector (little-endian)."""
    arr = state_data.reshape((2,) * n)
    axis = n - 1 - q
    return np.take(arr, 0, axis=axis), np.take(arr, 1, axis=axis), arr, axis


def _embed_reset_zero(substate, n, axis):
    norm = float(np.linalg.norm(substate))
    if norm < 1e-15:
        return None
    out = np.zeros((2,) * n, dtype=complex)
    index = [slice(None)] * n
    index[axis] = 0
    out[tuple(index)] = (substate / norm).reshape((2,) * (n - 1))
    return out.reshape(-1)


def _reset_outcomes(state_data, n, q):
    s0, s1, _, axis = _slices(state_data, n, q)
    p0 = float(np.vdot(s0, s0).real)
    p1 = float(np.vdot(s1, s1).real)
    residual = max(0.0, p0 * p1 - abs(np.vdot(s0, s1)) ** 2)
    outcomes = []
    for bit, probability, substate in ((0, p0, s0), (1, p1, s1)):
        embedded = _embed_reset_zero(substate, n, axis)
        if embedded is not None and probability > 1e-15:
            outcomes.append((bit, probability, embedded))
    return p0, p1, residual, outcomes


def _measure_reset_pairs(qc):
    """Map each mid measurement to its paired reset and vice versa."""
    measure_to_reset = {}
    reset_to_measure = {}
    for index, instruction in enumerate(qc.data):
        if instruction.operation.name != "measure":
            continue
        q = qc.find_bit(instruction.qubits[0]).index
        for later in range(index + 1, len(qc.data)):
            candidate = qc.data[later]
            if candidate.operation.name in SKIP_OPS:
                continue
            if not any(qc.find_bit(bit).index == q
                       for bit in candidate.qubits):
                continue
            if candidate.operation.name == "reset":
                measure_to_reset[index] = later
                reset_to_measure[later] = index
            break
    return measure_to_reset, reset_to_measure


def _enforce_branch_cap(branches, branch_cap, location):
    if branch_cap is not None and len(branches) > branch_cap:
        raise ExactExecutionIncomplete(
            f"exact branch cap exceeded at {location}: "
            f"{len(branches)}>{branch_cap}; no distribution was reported"
        )


def simulate_reconstructed_channel(qc, *, sep_tol=1e-9,
                                   branch_cap=DEFAULT_BRANCH_CAP):
    """Return the exact post-channel branch mixture and read diagnostics.

    The returned branches are ``(weight, normalized_statevector)`` pairs.
    Branches are split only when a reset acts on an entangled qubit.  No branch
    is discarded or renormalized after a cap event: cap overflow raises
    :class:`ExactExecutionIncomplete`.
    """
    n = qc.num_qubits
    initial = np.zeros(2 ** n, dtype=complex)
    initial[0] = 1.0
    branches = [(1.0, initial)]
    measure_to_reset, reset_to_measure = _measure_reset_pairs(qc)
    mid_records = []
    violations = []
    clbit_writer = {}
    final_qubit_of_clbit = {}
    latent_mixture = False

    for index, instruction in enumerate(qc.data):
        operation = instruction.operation
        name = operation.name
        if name in SKIP_OPS:
            continue
        if getattr(operation, "condition", None) is not None:
            raise ExactExecutionIncomplete(
                f"classically conditioned operation at data index {index} "
                "is unsupported; no distribution was reported"
            )

        if name == "measure":
            q = qc.find_bit(instruction.qubits[0]).index
            c = qc.find_bit(instruction.clbits[0]).index
            if index not in measure_to_reset:
                clbit_writer[c] = ("final", q)
                final_qubit_of_clbit[c] = q
                continue

            p0_mix = 0.0
            p1_mix = 0.0
            max_residual = 0.0
            violating_branches = 0
            next_branches = []
            for weight, state_data in branches:
                p0, p1, residual, outcomes = _reset_outcomes(
                    state_data, n, q)
                p0_mix += weight * p0
                p1_mix += weight * p1
                max_residual = max(max_residual, residual)
                if residual <= sep_tol:
                    # For a product state, both nonzero outcomes leave the same
                    # rest state up to global phase.  Keeping either normalized
                    # reset branch is the exact quantum channel.
                    chosen = max(outcomes, key=lambda value: value[1])
                    next_branches.append((weight, chosen[2]))
                else:
                    violating_branches += 1
                    for _, probability, reset_state in outcomes:
                        next_branches.append(
                            (weight * probability, reset_state))
            _enforce_branch_cap(next_branches, branch_cap, f"measure {index}")
            branches = next_branches
            record = {
                "index": int(index),
                "qubit": int(q),
                "clbit": int(c),
                "p0": min(max(float(p0_mix), 0.0), 1.0),
                "p1": min(max(float(p1_mix), 0.0), 1.0),
                "residual": float(max_residual),
                "violating_branches": int(violating_branches),
            }
            mid_records.append(record)
            if violating_branches:
                violations.append(record)
            clbit_writer[c] = ("mid", record["p1"])
            final_qubit_of_clbit.pop(c, None)
            continue

        if name == "reset":
            if index in reset_to_measure:
                continue
            next_branches = []
            for weight, state_data in branches:
                _, _, residual, outcomes = _reset_outcomes(state_data, n,
                                                            qc.find_bit(
                                                                instruction.qubits[0]
                                                            ).index)
                latent_mixture = latent_mixture or residual > sep_tol
                for _, probability, reset_state in outcomes:
                    next_branches.append((weight * probability, reset_state))
            _enforce_branch_cap(next_branches, branch_cap, f"reset {index}")
            branches = next_branches
            continue

        qargs = [qc.find_bit(bit).index for bit in instruction.qubits]
        branches = [
            (weight, Statevector(state_data).evolve(
                operation, qargs=qargs).data)
            for weight, state_data in branches
        ]

    total_weight = sum(weight for weight, _ in branches)
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ExactExecutionIncomplete(
            "exact channel produced invalid branch weight"
        )
    branches = [(weight / total_weight, state)
                for weight, state in branches]
    return {
        "complete": True,
        "branches": branches,
        "mid_records": mid_records,
        "violations": violations,
        "clbit_writer": clbit_writer,
        "final_qubit_of_clbit": final_qubit_of_clbit,
        "latent_mixture": bool(latent_mixture),
        "branch_count": len(branches),
        "branch_cap": branch_cap,
        "channel_semantics": "exact_weighted_measure_reset_branch_mixture",
    }


def _branch_marginal(branches, n, qubits_asc):
    """Probability vector with the smallest listed qubit as the MSB."""
    qubits_asc = tuple(sorted(set(int(q) for q in qubits_asc)))
    if not qubits_asc:
        return np.array([1.0])
    output = np.zeros(2 ** len(qubits_asc), dtype=float)
    for weight, state_data in branches:
        probabilities = np.abs(state_data) ** 2
        array = probabilities.reshape((2,) * n)
        keep_axes = sorted(n - 1 - q for q in qubits_asc)
        sum_axes = tuple(axis for axis in range(n) if axis not in keep_axes)
        marginal = array.sum(axis=sum_axes) if sum_axes else array
        marginal = np.transpose(
            marginal, axes=tuple(range(marginal.ndim))[::-1])
        output += weight * marginal.reshape(-1)
    output = np.clip(output, 0.0, None)
    return output / output.sum()


def reconstructed_final_marginal(qc, measured_qubits, *, sep_tol=1e-9,
                                 branch_cap=DEFAULT_BRANCH_CAP,
                                 require_separable=False):
    """Return an exact final marginal plus mid-read diagnostics."""
    channel = simulate_reconstructed_channel(
        qc, sep_tol=sep_tol, branch_cap=branch_cap)
    if require_separable and channel["violations"]:
        worst = max(record["residual"] for record in channel["violations"])
        raise ExactExecutionViolation(
            f"{len(channel['violations'])} mid-read certificate violations; "
            f"max residual={worst:.6g}"
        )
    probabilities = _branch_marginal(
        channel["branches"], qc.num_qubits, measured_qubits)
    return probabilities, channel["mid_records"], channel["violations"]


def reconstructed_reduced_density(qc, keep_qubits, *, sep_tol=1e-9,
                                  branch_cap=DEFAULT_BRANCH_CAP,
                                  require_separable=False):
    """Return an exact reduced density matrix of the branch-mixture channel."""
    channel = simulate_reconstructed_channel(
        qc, sep_tol=sep_tol, branch_cap=branch_cap)
    if require_separable and channel["violations"]:
        worst = max(record["residual"] for record in channel["violations"])
        raise ExactExecutionViolation(
            f"{len(channel['violations'])} mid-read certificate violations; "
            f"max residual={worst:.6g}"
        )
    keep = tuple(sorted(set(int(qubit) for qubit in keep_qubits)))
    traced = [qubit for qubit in range(qc.num_qubits) if qubit not in keep]
    dimension = 2 ** len(keep)
    density = np.zeros((dimension, dimension), dtype=complex)
    for weight, state_data in channel["branches"]:
        state = Statevector(state_data)
        reduced = (partial_trace(state, traced).data if traced
                   else np.outer(state_data, np.conj(state_data)))
        density += weight * reduced
    return density, channel["mid_records"], channel["violations"]


def run_reconstructed_exact(qc, shots=8192, seed=2025, sep_tol=1e-9,
                            branch_cap=DEFAULT_BRANCH_CAP):
    """Sample an exact reconstructed-circuit distribution.

    This fast sampler is defined only for certificate-valid QMon circuits.  It
    hard-fails if a mid read is entangled, if an unpaired reset creates a latent
    mixture, or if exact branching exceeds ``branch_cap``.  Under those
    conditions, independent mid-bit sampling would not represent one physical
    trajectory and is therefore never returned as an exact result.
    """
    channel = simulate_reconstructed_channel(
        qc, sep_tol=sep_tol, branch_cap=branch_cap)
    if channel["violations"]:
        worst = max(record["residual"] for record in channel["violations"])
        raise ExactExecutionViolation(
            f"{len(channel['violations'])} mid-read certificate violations; "
            f"max residual={worst:.6g}"
        )
    if channel["latent_mixture"] or channel["branch_count"] != 1:
        raise ExactExecutionIncomplete(
            "unpaired reset created a latent branch mixture; exact aligned "
            "classical sampling is unavailable"
        )

    final_qubit_of_clbit = channel["final_qubit_of_clbit"]
    final_qubits = tuple(sorted(set(final_qubit_of_clbit.values())))
    final_probabilities = _branch_marginal(
        channel["branches"], qc.num_qubits, final_qubits)
    rng = np.random.default_rng(seed)
    final_outcomes = rng.choice(
        len(final_probabilities), size=shots, p=final_probabilities)

    clbit_writer = channel["clbit_writer"]
    mid_clbits = [c for c, writer in clbit_writer.items()
                  if writer[0] == "mid"]
    mid_bits = {
        c: rng.binomial(1, clbit_writer[c][1], size=shots)
        for c in mid_clbits
    }
    final_position = {q: position for position, q in enumerate(final_qubits)}
    final_width = len(final_qubits)

    clbit_global = {}
    for register in qc.cregs:
        for position in range(register.size):
            clbit_global[(register.name, position)] = qc.find_bit(
                register[position]).index

    counts = {}
    for shot in range(shots):
        bits = {c: int(mid_bits[c][shot]) for c in mid_clbits}
        outcome = int(final_outcomes[shot])
        for c, q in final_qubit_of_clbit.items():
            position = final_position[q]
            bits[c] = (outcome >> (final_width - 1 - position)) & 1
        parts = []
        for register in reversed(qc.cregs):
            parts.append("".join(
                str(bits.get(clbit_global[(register.name, position)], 0))
                for position in range(register.size - 1, -1, -1)
            ))
        key = " ".join(parts)
        counts[key] = counts.get(key, 0) + 1

    return counts, channel["mid_records"], []
