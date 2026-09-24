"""
Huang & Martonosi statistical assertions (ISCA'19), simulation-level baseline.

The decision rule is pure classical statistics on the measured distribution at
a monitoring point, so no quantum-circuit construction is needed (unlike the
ancilla- or projector-based baselines):
  - classical / superposition assertion -> chi-square goodness-of-fit of the
    observed distribution against the expected one;
  - product (independence) assertion    -> chi-square test of independence
    (contingency table) between two qubit subsets.
The expected distribution is the original (unmutated) circuit's distribution at
that point; the observed one is the mutated circuit's. A mutation is FLAGGED
when the test's p-value < alpha.
"""
import numpy as np
from scipy.stats import chisquare, chi2_contingency


def gof_flag(expected_probs, observed_probs, shots=8192, alpha=0.01):
    """Goodness-of-fit (classical / superposition assertion). Returns
    (flagged, p_value): flagged True if observed deviates from expected."""
    exp = np.asarray(expected_probs, float)
    obs = np.asarray(observed_probs, float)
    exp = exp / exp.sum()
    obs = obs / obs.sum()
    # expected/observed counts at the assertion's shot budget
    f_exp = exp * shots
    f_obs = obs * shots
    keep = f_exp > 1e-9          # chi-square needs nonzero expected bins
    if keep.sum() < 2:
        return (abs(exp - obs).sum() > 1e-2), 1.0
    # match totals so chisquare doesn't complain
    f_obs = f_obs[keep] * f_exp[keep].sum() / f_obs[keep].sum()
    chi2, p = chisquare(f_obs, f_exp[keep])
    return bool(p < alpha), float(p)


def independence_flag(joint_expected, joint_observed, shots=8192, alpha=0.01):
    """Product/independence assertion: chi-square test of independence on the
    2x2 (or kxk) joint distribution of two qubit subsets. Flags entanglement
    that should not be there (or its absence). joint_* are 2D arrays summing
    to 1. Returns (flagged, p_value) on the OBSERVED joint table."""
    table = np.asarray(joint_observed, float) * shots
    table = table[table.sum(1) > 1e-9][:, table.sum(0) > 1e-9]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return False, 1.0
    chi2, p, *_ = chi2_contingency(table)
    return bool(p < alpha), float(p)


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(1000000)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace
    from fast_analysis import analyze_per_gate_nolock
    from mutation import mutate_circuit
    from utilities import quantum_path, quantum_gates

    key = "qpeinexact_indep_qiskit_5.qasm"
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    res, _, paths = analyze_per_gate_nolock(qc)
    # a monitored node and its qubit
    node = next(i for q in paths for i in paths[q])
    qbit = next(q for q in paths if node in paths[q])

    def marginal_at(circ, i, q):
        sv = Statevector.from_int(0, dims=2 ** circ.num_qubits)
        for j, ci in enumerate(circ.data):
            if ci.operation.name in ("measure", "barrier", "reset"):
                continue
            sv = sv.evolve(ci.operation,
                           qargs=[circ.find_bit(b).index for b in ci.qubits])
            if j == i:
                break
        rho = partial_trace(sv, [k for k in range(circ.num_qubits) if k != q])
        return [rho.data[0, 0].real, rho.data[1, 1].real]

    expected = marginal_at(qc, node, qbit)
    # mutate one gate before the node, recompute the marginal
    mqc = mutate_circuit(qc, max(0, node - 1), quantum_gates)
    observed = marginal_at(mqc, node, qbit)

    flagged, p = gof_flag(expected, observed, shots=8192)
    print(f"{key}: node {node}, qubit q{qbit}")
    print(f"  expected p0/p1 = {expected[0]:.3f}/{expected[1]:.3f}")
    print(f"  observed p0/p1 = {observed[0]:.3f}/{observed[1]:.3f}  (mutated)")
    print(f"  statistical-assertion (chi-square gof): "
          f"flagged={flagged}, p={p:.2e}")
