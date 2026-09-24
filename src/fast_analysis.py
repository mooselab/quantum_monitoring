import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, concurrence, schmidt_decomposition

from circuits_selection import _single_vs_rest_entangled


SKIP_OPS = ("measure", "barrier", "reset", "snapshot", "delay",
            "save_state", "save_statevector")


def analyze_per_gate_nolock(qc, tol: float = 1e-12):
    """
    Memoryless monitorable-node selection: gate index i is appended to
    qubit q's list iff its directly computed second Schmidt amplitude is at
    most ``tol`` immediately after gate i, with no lock state and no SWAP
    special case. This is a finite-precision implementation of the exact
    separability condition. It selects a strict superset of the lock-based
    rule (recovers entangle-then-uncompute windows the lock permanently
    closes) while applying the same declared numerical rule at every point.

    Returns (res, events, paths) shaped like analyze_per_gate_fast;
    events is empty (the pairwise double-check was log-only).
    """
    n = qc.num_qubits
    state = Statevector.from_int(0, dims=2 ** n)
    res = {}
    paths = {q: [] for q in range(n)}

    for i, ci in enumerate(qc.data):
        inst, qargs = ci.operation, ci.qubits
        if inst.name in SKIP_OPS:
            continue
        qidx = [qc.find_bit(q).index for q in qargs]
        state = state.evolve(inst, qargs=qidx)

        res[i] = []
        for qi in qidx:
            qubit_state = partial_trace(state, [q for q in range(n) if q != qi])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res[i].append([i, inst.name, qi, prob_zero, prob_one])

            ent, _, _ = _single_vs_rest_entangled(state, qi, tol)
            if not ent:
                paths[qi].append(i)

    return res, [], paths


def analyze_per_gate_fast(qc, tol: float = 1e-12, swap_unlock: bool = True):
    """
    Drop-in replacement for circuits_selection.analyze_per_gate.

    Instead of re-simulating the circuit prefix from scratch for every gate
    (O(m^2 * 2^n) total), it evolves a single statevector incrementally,
    one gate at a time (O(m * 2^n) total). The per-gate entanglement tests
    (partial trace, concurrence, Schmidt) and the lock/unlock rules are
    identical to the original implementation.

    Assumes no non-unitary op (measure/reset) appears before an analyzed
    gate, which holds for the MQTBench target-independent circuits where
    measurements are final.

    Returns the same (res, events, paths_until_first_ent).
    """
    n = qc.num_qubits
    state = Statevector.from_int(0, dims=2 ** n)

    res = {}
    events = []
    paths_until_first_ent = {q: [] for q in range(n)}
    locked = np.zeros(n, dtype=bool)

    for i, ci in enumerate(qc.data):
        inst, qargs = ci.operation, ci.qubits
        if inst.name in SKIP_OPS:
            continue

        qidx = [qc.find_bit(q).index if not isinstance(q, int) else q for q in qargs]
        state = state.evolve(inst, qargs=qidx)
        quantum_state = state

        res[i] = []
        for qi in qidx:
            qubit_state = partial_trace(quantum_state, [q for q in range(n) if q != qi])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res[i].append([i, inst.name, qi, prob_zero, prob_one])

        if len(qidx) == 1:
            qi = qidx[0]
            if not locked[qi]:
                paths_until_first_ent[qi].append(i)

        elif len(qidx) == 2:
            a, b = qidx

            rest = [k for k in range(n) if k not in (a, b)]
            if rest:
                rho_ab = partial_trace(quantum_state, rest)
                C = float(concurrence(rho_ab))
                lams_ab = [t[0] for t in schmidt_decomposition(quantum_state, qargs=[a, b])]
                s2_ab = float(lams_ab[1]) if len(lams_ab) > 1 else 0.0
            else:
                # n == 2: the pair IS the whole system; concurrence applies
                # to the pure state directly and the pair-vs-rest test is moot
                C = float(concurrence(quantum_state))
                s2_ab = 0.0

            ent_a, s2_a, lams_a = _single_vs_rest_entangled(quantum_state, a, tol)
            ent_b, s2_b, lams_b = _single_vs_rest_entangled(quantum_state, b, tol)

            events.append({
                'index': i, 'gate': inst.name, 'a': a, 'b': b,
                'pairwise_C': C, 'vs_rest_s2': s2_ab,
                'single': {
                    'a': {'ent_vs_rest': ent_a, 's2': s2_a, 'lams': lams_a},
                    'b': {'ent_vs_rest': ent_b, 's2': s2_b, 'lams': lams_b},
                }
            })

            if inst.name.lower() == 'swap' and swap_unlock:
                locked_a = bool(locked[a])
                locked_b = bool(locked[b])
                if not ent_a and not locked_b:
                    locked[a] = False
                if not ent_b and not locked_a:
                    locked[b] = False
                if ent_a:
                    locked[a] = True
                if ent_b:
                    locked[b] = True

                if not locked[a]:
                    paths_until_first_ent[a].append(i)
                if not locked[b]:
                    paths_until_first_ent[b].append(i)
            else:
                if ent_a and not locked[a]:
                    locked[a] = True
                if ent_b and not locked[b]:
                    locked[b] = True

                if not locked[a]:
                    paths_until_first_ent[a].append(i)
                if not locked[b]:
                    paths_until_first_ent[b].append(i)

        else:
            for qi in qidx:
                if not locked[qi]:
                    paths_until_first_ent[qi].append(i)

    return res, events, paths_until_first_ent
