"""
PROQ projection-based runtime assertions (Li, Zhou, Yu, Ding, Ying, Xie,
OOPSLA'20) as a QMon monitoring-node baseline, via the OFFICIAL pytket
implementation (the authors' method was adopted into t|ket>).

Mechanism (paper Sec. 3, Def. 3.2 + Sec. 3.1):
  An assertion at a program point is a projection P onto the EXPECTED
  subspace. PROQ constructs the projective measurement
        M = { M_true = P , M_false = I - P }.
  For the observed state rho the per-shot probability of the "true" outcome
  is  p_true = tr(P rho)  and the per-shot DETECTION (abort/"false")
  probability is  1 - tr(P rho).  If rho lies in the subspace of P (i.e. the
  state is correct) tr(P rho) = 1 and the assertion always passes; if rho
  deviates, tr(P rho) < 1 and the assertion fails with nonzero probability,
  approaching certainty as shots accumulate ( (tr P rho)^k -> 0 ).

Official runtime path used here:
  pytket.circuit.ProjectorAssertionBox(P)  builds the assertion sub-circuit,
  Circuit.add_assertion(box, [qubit], name=...) injects it at the monitoring
  node, and BackendResult.get_debug_info() returns, per assertion, the
  AVERAGE SUCCESS RATE across shots, which is pytket's own Monte-Carlo estimate of
  tr(P rho).  We declare the node FLAGGED when  1 - success_rate > tau.

QMon integration: the monitoring node and its separable qubit come from
fast_analysis.analyze_per_gate_nolock; the EXPECTED single-qubit pure
marginal at that node is taken from the ORIGINAL (unmutated) circuit (the
oracle) and turned into the rank-1 projector P = |psi><psi|.  The mutated
circuit (one gate replaced before the node) is then asserted against the same
P.  Detection = PROQ flags the mutated circuit at that node.
"""
import sys
import numpy as np

sys.setrecursionlimit(1_000_000)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace

from pytket.circuit import Circuit, ProjectorAssertionBox
from pytket.extensions.qiskit import qiskit_to_tk, AerBackend

SKIP_OPS = ("measure", "barrier", "reset", "snapshot", "delay",
            "save_state", "save_statevector")


# ==========================================================================
# Oracle: expected single-qubit pure marginal at the monitoring node, taken
# from the ORIGINAL circuit (statevector evolve up to the gate, partial_trace).
# Returns the 2x2 projector P = |psi><psi| onto that pure marginal.
# ==========================================================================
def expected_projector_at(circ, node_idx, qbit):
    sv = Statevector.from_int(0, dims=2 ** circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            continue
        sv = sv.evolve(ci.operation,
                       qargs=[circ.find_bit(b).index for b in ci.qubits])
        if j == node_idx:
            break
    rho = partial_trace(sv, [k for k in range(circ.num_qubits) if k != qbit])
    rho = np.asarray(rho.data, dtype=complex)
    # nearest pure state = leading eigenvector (the qubit is separable at a
    # monitoring node, so rho is pure to numerical tolerance and this is exact)
    w, v = np.linalg.eigh(rho)
    psi = v[:, int(np.argmax(w))]
    P = np.outer(psi, psi.conj())
    return P, float(w.max())


# ==========================================================================
# Build the pytket prefix circuit: the original/mutated circuit truncated to
# the gates up to and including `node_idx` (measures/barriers dropped), so the
# state of `qbit` at the assertion point is exactly the node's state.
# ==========================================================================
def prefix_circuit(circ, node_idx):
    pre = QuantumCircuit(circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            if j == node_idx:
                break
            continue
        pre.append(ci.operation,
                   [circ.find_bit(b).index for b in ci.qubits])
        if j == node_idx:
            break
    return pre


# ==========================================================================
# PROQ assertion at the node, via the official pytket runtime path.
# Returns (flagged, success_rate, detection_prob).
# ==========================================================================
def proq_assert_node(circ, node_idx, qbit, projector, backend,
                     shots=8192, tau=0.01, seed=1):
    pre = prefix_circuit(circ, node_idx)
    tkc = qiskit_to_tk(pre)
    tkc.add_assertion(ProjectorAssertionBox(projector), [qbit],
                      name=f"node{node_idx}_q{qbit}")
    compiled = backend.get_compiled_circuit(tkc)
    result = backend.run_circuit(compiled, n_shots=shots, seed=seed)
    info = result.get_debug_info()              # {assertion_name: success_rate}
    success = float(next(iter(info.values())))
    detection = 1.0 - success
    return (detection > tau), success, detection


if __name__ == "__main__":
    from fast_analysis import analyze_per_gate_nolock
    from mutation import mutate_circuit
    from utilities import quantum_path, quantum_gates

    key = "dj_indep_qiskit_5.qasm"
    SHOTS = 8192
    TAU = 0.01            # flag if per-shot detection prob > 1%

    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    res, _, paths = analyze_per_gate_nolock(qc)
    backend = AerBackend()

    # Choose a monitoring node that lies strictly AFTER some earlier gate on
    # the SAME qubit, so a mutation before the node actually reaches it.
    chosen = None
    for q in paths:
        for node in paths[q]:
            # need a single-qubit gate on q strictly before `node` to mutate
            gate_to_mutate = None
            for j, ci in enumerate(qc.data):
                if j >= node:
                    break
                if ci.operation.name in SKIP_OPS:
                    continue
                qidx = [qc.find_bit(b).index for b in ci.qubits]
                if len(qidx) == 1 and qidx[0] == q:
                    gate_to_mutate = j
            if gate_to_mutate is not None:
                chosen = (q, node, gate_to_mutate)
                break
        if chosen:
            break
    if chosen is None:
        # fall back to the template's choice
        node = next(i for q in paths for i in paths[q])
        q = next(qq for qq in paths if node in paths[qq])
        chosen = (q, node, max(0, node - 1))
    qbit, node, gmut = chosen

    print(f"circuit  : {key}  ({qc.num_qubits} qubits, {len(qc.data)} ops)")
    print(f"node     : gate index {node}, monitored qubit q{qbit}")
    print(f"mutating : gate index {gmut} (a gate on q{qbit} before the node)")
    print(f"shots    : {SHOTS}   flag threshold tau (detection prob) : {TAU}")
    print("-" * 64)

    # Oracle projector from the ORIGINAL circuit.
    P, purity = expected_projector_at(qc, node, qbit)
    print(f"expected pure marginal at node, leading eigval = {purity:.6f}")
    print(f"projector P = |psi><psi| (rank 1):\n{np.round(P, 4)}")
    print("-" * 64)

    # 1) ORIGINAL circuit -> assertion should PASS (success ~ 1, not flagged).
    f0, s0, d0 = proq_assert_node(qc, node, qbit, P, backend,
                                  shots=SHOTS, tau=TAU)
    print(f"ORIGINAL  : success_rate(tr P rho) = {s0:.4f}  "
          f"detection = {d0:.4f}  -> flagged = {f0}")

    # 2) MUTATED circuit -> assertion should FAIL (success < 1, flagged).
    mqc = mutate_circuit(qc, gmut, quantum_gates)
    f1, s1, d1 = proq_assert_node(mqc, node, qbit, P, backend,
                                  shots=SHOTS, tau=TAU)
    print(f"MUTATED   : success_rate(tr P rho) = {s1:.4f}  "
          f"detection = {d1:.4f}  -> flagged = {f1}")
    print("-" * 64)

    ok = (f1 is True) and (f0 is False)
    print(f"VERDICT   : PROQ flags mutant = {f1}, passes original = {not f0}")
    print(f"            smoke test {'PASSED' if ok else 'NEEDS REVIEW'}")
