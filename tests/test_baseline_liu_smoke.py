"""
Smoke test: Liu-Byrd-Zhou (ASPLOS'20) dynamic runtime assertions detecting a
mutation at a QMon monitoring node, using the authors' OWN assertion circuit
(ported in baseline_liu.py), not a statistical reimplementation.

QMon picks a monitoring node where a qubit is separable; at the near-classical
nodes the qubit's reduced state is a (near-)basis state |b>. The oracle for the
expected value `b` is the ORIGINAL (unmutated) circuit at that node.

Liu's classical_assertion is inserted on the data qubit at the node: it loads a
fresh ancilla with the expected bit `b`, CXs data->ancilla, and measures the
ancilla. The ancilla reads 0 iff the data qubit matches `b`. We then:

  * run the ORIGINAL circuit truncated at the node + assertion  -> ancilla 0
    (assertion passes, no bug);
  * run the MUTATED circuit truncated at the node + assertion   -> ancilla
    flips to nonzero (assertion fails => bug DETECTED).

A mutation is FLAGGED at the node iff P(ancilla != 0) is large under the
mutant. Run with:
  python -m unittest test_baseline_liu_smoke
or, for the printed report on both circuits:
  python test_baseline_liu_smoke.py
"""
import contextlib
import io
import sys
import unittest

sys.setrecursionlimit(1000000)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator

from fast_analysis import analyze_per_gate_nolock
from mutation import mutate_circuit
from utilities import quantum_path, quantum_gates, execute
from baseline_liu import classical_assertion


def truncate_to_node(circ, node_idx):
    """Return a copy of `circ` containing only the unitary gates up to and
    including data-gate index `node_idx` (drops final measure/barrier/reset).
    This realises the QMon monitoring point: the circuit state right after the
    monitored gate, where the asserted qubit is separable."""
    new = QuantumCircuit(*circ.qregs)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in ("measure", "barrier", "reset"):
            continue
        new.append(ci.operation, ci.qubits)
        if j == node_idx:
            break
    return new


def expected_bit(circ, node_idx, qbit):
    """Oracle: the expected classical value of `qbit` at the node, computed
    from the ORIGINAL circuit by statevector evolution + partial trace."""
    sv = Statevector.from_int(0, dims=2 ** circ.num_qubits)
    for j, ci in enumerate(circ.data):
        if ci.operation.name in ("measure", "barrier", "reset"):
            continue
        sv = sv.evolve(ci.operation,
                       qargs=[circ.find_bit(b).index for b in ci.qubits])
        if j == node_idx:
            break
    rho = partial_trace(sv, [k for k in range(circ.num_qubits) if k != qbit])
    p0, p1 = rho.data[0, 0].real, rho.data[1, 1].real
    return (1 if p1 >= p0 else 0), p0, p1


def assert_at_node(circ, node_idx, qbit, value, shots=8192):
    """Insert Liu's classical_assertion on `qbit` at the node and run it.
    Returns P(ancilla != 0): 0 => assertion passes, >0 => assertion fails."""
    trunc = truncate_to_node(circ, node_idx)
    # the data qubit object the assertion controls from
    data_qubit = trunc.qubits[qbit]
    ok = classical_assertion(trunc, [data_qubit], value)
    assert ok, "classical_assertion failed to build"

    sim = AerSimulator()
    counts = execute(trunc, sim, shots=shots).result().get_counts()

    # The assertion's classical register is the most-significant field of the
    # counts key (space-separated, multiple cregs). Find which field is the
    # assertion register and measure P(that field != all-zero).
    creg_names = [c.name for c in trunc.cregs]
    assert_pos = creg_names[::-1].index("cr_classical_assertion")
    fail = 0
    for key, n in counts.items():
        fields = key.split(" ")
        if int(fields[assert_pos], 2) != 0:
            fail += n
    return fail / shots


def run_one(key, seed=0):
    import random
    random.seed(seed)
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    res, _, paths = analyze_per_gate_nolock(qc)

    # pick a near-classical monitorable node: |p1-p0| ~ 1 so the expected state
    # is essentially a basis state (the regime Liu's classical_assertion targets)
    candidates = []
    for q in paths:
        for i in paths[q]:
            for e in res[i]:
                if e[2] == q and (abs(e[3] - 1) < 1e-6 or abs(e[4] - 1) < 1e-6):
                    candidates.append((i, q))
    if not candidates:
        print(f"{key}: no near-classical monitorable node found; skipping")
        return None
    node, qbit = candidates[0]

    b, p0, p1 = expected_bit(qc, node, qbit)
    print(f"\n=== {key} ===")
    print(f"QMon node: gate index {node}, monitored qubit q{qbit}")
    print(f"  expected reduced state p0/p1 = {p0:.4f}/{p1:.4f} "
          f"=> classical value b={b}")

    # baseline applied to the ORIGINAL circuit (control: should PASS)
    fail_orig = assert_at_node(qc, node, qbit, b)

    # mutate a gate strictly before the node, then assert at the node
    # try a few gate indices < node until the mutation actually perturbs q
    flagged_mut = False
    fail_mut = 0.0
    mut_idx_used = None
    for mut_idx in range(node, -1, -1):
        mqc = mutate_circuit(qc, mut_idx, quantum_gates)
        f = assert_at_node(mqc, node, qbit, b)
        if f > 0.05:                       # mutation visibly flips the ancilla
            flagged_mut, fail_mut, mut_idx_used = True, f, mut_idx
            break
        # remember the strongest even if below threshold
        if f > fail_mut:
            fail_mut, mut_idx_used = f, mut_idx

    print(f"  ORIGINAL  : P(ancilla != 0) = {fail_orig:.4f}  "
          f"-> flagged={fail_orig > 0.05}  (expect False)")
    print(f"  MUTATED   : gate {mut_idx_used} replaced, "
          f"P(ancilla != 0) = {fail_mut:.4f}  "
          f"-> flagged={fail_mut > 0.05}  (expect True)")

    verdict = (fail_orig <= 0.05) and (fail_mut > 0.05)
    print(f"  VERDICT   : Liu classical_assertion "
          f"{'DETECTS the mutation' if verdict else 'did NOT cleanly detect'} "
          f"at this QMon node")
    return fail_orig, fail_mut, verdict


class LiuAssertionSmokeTests(unittest.TestCase):
    """Discovery entry point: one circuit, one node, one mutation."""

    def test_assertion_passes_on_the_original_and_fails_on_the_mutant(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_one("dj_indep_qiskit_5.qasm")
        self.assertIsNotNone(result, "no near-classical monitorable node found")
        fail_original, fail_mutant, verdict = result
        self.assertLessEqual(fail_original, 0.05)
        self.assertGreater(fail_mutant, 0.05)
        self.assertTrue(verdict)


def report() -> None:
    keys = ["dj_indep_qiskit_5.qasm", "ghz_indep_qiskit_5.qasm"]
    results = {}
    for k in keys:
        r = run_one(k)
        if r is not None:
            results[k] = r

    print("\n================ SUMMARY ================")
    any_detect = False
    for k, (fo, fm, v) in results.items():
        print(f"{k}: orig_fail={fo:.3f}  mut_fail={fm:.3f}  detected={v}")
        any_detect = any_detect or v
    print(f"\nBaseline (Liu-Byrd-Zhou) detected a mutation at a QMon node: "
          f"{any_detect}")


if __name__ == "__main__":
    report()
