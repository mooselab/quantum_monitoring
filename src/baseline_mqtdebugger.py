"""
MQT DEBUGGER (Rovara, Burgholzer, Wille; arXiv:2412.12269) simulation-level
baseline, driven through the OFFICIAL `mqt.debugger` package (1.2.0).

Mechanism (the tool's OWN code, not a reimplementation)
=======================================================
MQT DEBUGGER reads an extended OpenQASM that carries assertion statements and
simulates the program with its decision-diagram (DD) backend:

    state = mqt.debugger.create_ddsim_simulation_state()
    state.load_code(<extended-OpenQASM string>)
    n_failed = state.run_all()          # number of assertions that FAILED

The extended assertions (paper Sec. III; reference repo test/.../end-to-end):
  * assert-eq  [similarity,] q[...] { amplitudes }  : state-equality oracle
  * assert-sup q[...]                               : superposition assertion
  * assert-ent q[...], q[...] [, ...]               : entanglement assertion

We use `assert-eq` as the per-node oracle: at a QMon monitoring node the
monitored qubit is SEPARABLE (that is exactly what analyze_per_gate_nolock's
`paths` certifies), so its reduced state is a pure 1-qubit state with two
amplitudes (a0, a1). We embed that expected pure state as
`assert-eq <sim>, q[k] { a0, a1 }` right at the node and let MQT DEBUGGER's DD
backend decide. A similarity threshold (< 1.0) is required because the paper's
checker compares amplitudes for *exact* DD-equality otherwise, and floating
amplitudes never match to the last bit (cf. the reference repo's fail_eq.qasm,
where the 1.0 thresholds "should fail" and the 0.9 ones "should pass").

Oracle = the ORIGINAL (unmutated) circuit's reduced state at the node.
Detection = MQT DEBUGGER reports the mutated program as deviating at that node.
That happens in two ways, BOTH of which we count as a detection:
  (a) n_failed >= 1, i.e. the assert-eq state-equality check failed; or
  (b) load/run raises
      "Equality assertion on entangled sub-state is not allowed.",
      i.e. the mutation made the monitored qubit no longer separable, so the
      separable-pure oracle can no longer even be evaluated. The expected
      separable form is violated, which is itself a deviation from the
      original. (We additionally confirm/illustrate this case with an
      assert-sup / assert-ent fallback, which the tool CAN evaluate.)

This file mirrors baseline_statistical.py: pick a monitorable node and its
qubit, compute the expected reduced state from the original circuit, mutate a
gate before the node, and show the baseline FLAGS the mutant (and not the
original).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector, partial_trace

import mqt.debugger as mqtd

from fast_analysis import analyze_per_gate_nolock, SKIP_OPS
from mutation import mutate_circuit
from utilities import quantum_path, quantum_gates


# =========================================================================== #
# MQT DEBUGGER driver (the baseline's own mechanism)
# =========================================================================== #
def run_debugger(qasm_with_assert: str):
    """Load an extended-OpenQASM string into the OFFICIAL MQT DEBUGGER DD
    backend and run every assertion. Returns (flagged, status, detail):

      status "pass"        -> all assertions evaluated and passed (flagged False)
      status "fail"        -> an assertion evaluated and FAILED (flagged True)
      status "unevaluable" -> the DD backend could not evaluate the assertion
                              (flagged True under the declared protocol)

    The DD backend raises RuntimeError for the following two signatures, which
    the declared protocol groups as an unevaluable/refusal outcome:
      * "Equality assertion on entangled sub-state is not allowed."
      * the generic "An error occurred while executing the operation".
    The second signature does not retain enough subtype information to label
    every aggregated outcome specifically as an entangled-substate refusal.
    For a node where the ORIGINAL run is "pass" (the oracle qubit really IS
    separable there), the protocol treats an "unevaluable" mutant outcome as a
    conservative loss-of-applicability flag. The caller therefore requires the
    original to be a genuine "pass" before accepting a mutant's
    "fail"/"unevaluable" as a flag.
    """
    state = mqtd.create_ddsim_simulation_state()
    try:
        state.load_code(qasm_with_assert)
        n_failed = state.run_all()
        if n_failed >= 1:
            return True, "fail", f"{n_failed} failed assertion(s)"
        return False, "pass", "0 failed assertion(s)"
    except RuntimeError as e:
        msg = str(e).strip().splitlines()[0] if str(e) else repr(e)
        if ("entangled sub-state" in msg.lower()
                or "executing the operation" in msg.lower()):
            return True, "unevaluable", f"assert-eq could not be evaluated: {msg}"
        raise


# =========================================================================== #
# Helpers: reduced 1-qubit pure state of a separable qubit at a node, and
# building the extended-OpenQASM (prefix up to the node + an assert line).
# =========================================================================== #
def reduced_1q_amps(circ: QuantumCircuit, upto_idx: int, q: int):
    """Reduced 1-qubit density matrix of qubit q right after gate `upto_idx`
    (skipping non-unitary ops), and its dominant-eigenvector pure-state
    amplitudes (global phase fixed). Valid as a *pure* oracle only when q is
    separable there, which is guaranteed at a QMon node in the ORIGINAL run."""
    sv = Statevector.from_int(0, dims=2 ** circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            if j == upto_idx:
                break
            continue
        sv = sv.evolve(ci.operation,
                       qargs=[circ.find_bit(b).index for b in ci.qubits])
        if j == upto_idx:
            break
    rho = partial_trace(sv, [k for k in range(circ.num_qubits) if k != q])
    evals, evecs = np.linalg.eigh(rho.data)
    amps = evecs[:, int(np.argmax(evals.real))]
    k = int(np.argmax(np.abs(amps)))          # fix global phase
    amps = amps * np.exp(-1j * np.angle(amps[k]))
    return amps, rho


def prefix_circuit(circ: QuantumCircuit, upto_idx: int) -> QuantumCircuit:
    """A copy of `circ` truncated to the gates up to and including index
    `upto_idx` (non-unitary ops dropped), with no classical register, so it
    exports to clean OpenQASM the debugger can parse."""
    pre = QuantumCircuit(circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in SKIP_OPS:
            if j == upto_idx:
                break
            continue
        pre.append(ci.operation,
                   [circ.find_bit(b).index for b in ci.qubits])
        if j == upto_idx:
            break
    return pre


def _fmt(x: complex) -> str:
    """Format an amplitude for the debugger's `{ ... }` list (real, or re+imi).
    NO parentheses: the tool's parser rejects '(re+imi)' with a stod error
    (verified empirically); the bare 're+imi' form parses."""
    if abs(x.imag) < 1e-12:
        return f"{x.real:.10f}"
    sign = "+" if x.imag >= 0 else "-"
    return f"{x.real:.10f}{sign}{abs(x.imag):.10f}i"


def assert_eq_qasm(circ: QuantumCircuit, upto_idx: int, q: int,
                   amps, similarity: float = 0.99) -> str:
    """Extended-OpenQASM string = circuit prefix up to the node + a single
    `assert-eq <similarity>, q[q] { a0, a1 }` line embedding the expected
    1-qubit oracle state at the monitoring node."""
    body = qasm2.dumps(prefix_circuit(circ, upto_idx))
    line = f"assert-eq {similarity}, q[{q}] {{ {_fmt(amps[0])}, {_fmt(amps[1])} }}"
    return body + "\n" + line + "\n"


def assert_sup_qasm(circ: QuantumCircuit, upto_idx: int, q: int) -> str:
    """Prefix + `assert-sup q[q]` (a superposition assertion the DD backend can
    always evaluate, even when assert-eq's separability precondition is lost)."""
    body = qasm2.dumps(prefix_circuit(circ, upto_idx))
    return body + "\n" + f"assert-sup q[{q}];\n"


# =========================================================================== #
# Smoke test
# =========================================================================== #
if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(1_000_000)

    # dj_indep_qiskit_5 has many mid-circuit monitorable nodes where a query
    # qubit is prepared, entangled with the oracle ancilla, then un-entangled
    # back to a separable state, ideal for "mutate a prep gate, observe the
    # deviation at the later separable node". (GHZ-5 has only node 0, too early
    # to mutate before, so it is unsuitable for this baseline's assert-eq form.)
    key = "dj_indep_qiskit_5.qasm"
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    _, _, paths = analyze_per_gate_nolock(qc)

    # Enumerate (node, qubit, mutable-predecessor) candidates: a monitorable
    # node where the monitored qubit also has an EARLIER gate we can mutate so
    # the perturbation propagates to the node.
    candidates = []
    for q in paths:
        earlier = None
        for i, ci in enumerate(qc.data):
            if ci.operation.name in SKIP_OPS:
                continue
            if q in [qc.find_bit(b).index for b in ci.qubits]:
                if i in paths[q] and earlier is not None:
                    candidates.append((i, q, earlier))
                earlier = i

    # Require the ORIGINAL to be a GENUINE "pass" at the node: the DD backend
    # must be able to evaluate assert-eq on the separable oracle there. (On some
    # circuits, e.g. QPE, a numerically-separable point is still structurally
    # entangled for the DD check, so assert-eq is "unevaluable" even on the
    # original; such a node cannot serve as a reliable oracle and is skipped.)
    node = qbit = mut_idx = amps = rho = None
    status_orig = None
    for (n_, q_, m_) in candidates:
        a_, r_ = reduced_1q_amps(qc, n_, q_)
        fl, st, _ = run_debugger(assert_eq_qasm(qc, n_, q_, a_))
        if st == "pass":
            node, qbit, mut_idx, amps, rho, status_orig = n_, q_, m_, a_, r_, st
            break
    assert node is not None, (
        "no monitorable node where MQT DEBUGGER can evaluate the separable "
        "assert-eq oracle on the original circuit")

    op_node = qc.data[node].operation.name
    op_mut = qc.data[mut_idx].operation.name
    print(f"circuit             : {key} ({qc.num_qubits} qubits, "
          f"{len(qc.data)} ops)")
    print(f"monitoring node     : gate {node} ('{op_node}'), monitored qubit q{qbit}")
    print(f"mutated gate (<node) : gate {mut_idx} ('{op_mut}')")
    print(f"expected (oracle) q{qbit} amps = [{_fmt(amps[0])}, {_fmt(amps[1])}]"
          f"  (purity {rho.purity().real:.3f})")

    # 1) ORIGINAL circuit, oracle embedded -> MQT DEBUGGER should PASS.
    flagged_orig, status_orig, detail_orig = run_debugger(
        assert_eq_qasm(qc, node, qbit, amps))
    print(f"\n[ORIGINAL] assert-eq via MQT DEBUGGER -> flagged={flagged_orig}  "
          f"[{status_orig}] ({detail_orig})")

    # 2) MUTATED circuit (a gate before the node is replaced), SAME oracle ->
    #    MQT DEBUGGER should FLAG. Search seeds for a mutation that actually
    #    perturbs the monitored qubit (mutate_circuit may pick an equivalent
    #    gate); report the first detecting mutant deterministically.
    flagged_mut = False
    status_mut = "n/a"
    detail_mut = "no detecting mutant found"
    for seed in range(20):
        np.random.seed(seed)
        mqc = mutate_circuit(qc, mut_idx, quantum_gates)
        m_amps, m_rho = reduced_1q_amps(mqc, node, qbit)
        fid = float(np.real(np.trace(rho.data @ m_rho.data)))
        if fid > 0.999 and m_rho.purity().real > 0.999:
            continue                      # equivalent gate, not a real mutant
        flagged_mut, status_mut, detail_mut = run_debugger(
            assert_eq_qasm(mqc, node, qbit, amps))
        mut_op = mqc.data[mut_idx].operation.name
        print(f"\n[MUTANT seed={seed}] gate {mut_idx} '{op_mut}' -> '{mut_op}'"
              f"  (oracle fidelity {fid:.3f})")
        if m_rho.purity().real > 0.999:    # mutant still separable: show amps
            print(f"           observed q{qbit} amps = "
                  f"[{_fmt(m_amps[0])}, {_fmt(m_amps[1])}]")
        print(f"           assert-eq via MQT DEBUGGER -> flagged={flagged_mut}  "
              f"[{status_mut}] ({detail_mut})")
        # If assert-eq became unevaluable, illustrate the assert-sup assertion
        # the DD backend can still evaluate without assigning a finer subtype.
        if status_mut == "unevaluable":
            f_sup, st_sup, d_sup = run_debugger(assert_sup_qasm(mqc, node, qbit))
            print(f"           assert-sup fallback        -> flagged={f_sup}  "
                  f"[{st_sup}] ({d_sup})")
        if flagged_mut:
            break

    print("\n" + "=" * 64)
    print("VERDICT (MQT DEBUGGER baseline, official mqt.debugger 1.2.0)")
    print(f"  original circuit flagged : {flagged_orig}   "
          f"[{status_orig}]  (expected False / pass)")
    print(f"  mutated  circuit flagged : {flagged_mut}   "
          f"[{status_mut}]  (expected True)")
    ok = (flagged_orig is False and status_orig == "pass"
          and flagged_mut is True)
    print(f"  >>> baseline detects the mutation at the QMon node: {ok}")
