"""
Liu, Byrd & Zhou dynamic runtime assertions (ASPLOS'20), ported to qiskit 2.4.1.

This is a faithful port of the AUTHORS' code
(Quantum-Circuits-for-Dynamic-Runtime-Assertions-in-Quantum-Computation/
assertion.py); the assertion *mechanism* is identical, only the deprecated
qiskit-0.13-era API is updated:

  * `qProg += assertCircuit`  (circuit addition, removed) -> add the ancilla
    register(s) directly to the live circuit and append gates in place. The
    original `+=` only served to splice the freshly-built ancilla register and
    its classical register into the program; we do the same with
    `add_register`, which keeps the data qubits/clbits untouched.
  * `cu3(theta, phi, lam, ...)` (removed) -> `cu(theta, phi, lam, 0, ...)`,
    the qiskit-2 controlled-U with the extra global-phase argument set to 0,
    which reproduces the old `cu3` exactly.
  * register / measure API is otherwise unchanged.

Mechanism (unchanged from the paper):
  insert a fresh ancilla register, optionally pre-load it (X gates) to encode
  the expected classical value, entangle data -> ancilla with CX (or the
  H/CU/H sandwich for the superposition assertion), then MEASURE the ancilla.
  Reading the ancilla register as 0 => the assertion PASSES; any nonzero
  ancilla reading => the assertion FAILS => a bug is flagged at that node.

The functions mutate `qProg` in place and return True on success (matching the
authors' signatures), so existing call sites port over verbatim.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math

pi = math.pi


def classical_assertion(qProg, qubitList, value):
    """Assert the qubits in `qubitList` are in the classical basis state whose
    little-endian integer is `value`. Port of the authors' classical_assertion.

    For each ancilla i, pre-set it to the expected bit of `value` (X if bit
    set), then CX(data_i -> ancilla_i). If the data qubit matches the expected
    bit, the CX returns the ancilla to |0>; otherwise the ancilla ends in |1>.
    A nonzero ancilla reading therefore signals a deviation (bug)."""
    qr_assert = QuantumRegister(len(qubitList), name="classical_assertion")
    cr_assert = ClassicalRegister(len(qubitList), name="cr_classical_assertion")
    # qiskit-0.13:  qProg += QuantumCircuit(qr_assert, cr_assert)
    # qiskit-2.4 :  splice the same registers into the live program in place
    qProg.add_register(qr_assert)
    qProg.add_register(cr_assert)

    if value >= pow(2, len(qubitList)):
        print("Value larger than 2^number of qubits")
        return False
    for i in range(len(qubitList)):
        if value & (1 << i):
            qProg.x(qr_assert[i])
    for j in range(len(qubitList)):
        qProg.cx(qubitList[j], qr_assert[j])
    qProg.measure(qr_assert, cr_assert)
    return True


def superposition_assertion(qProg, qubitList, phaseDict, flag):
    """Port of the authors' superposition_assertion.

    flag == 0: assert a uniform (|+>-like) superposition on each qubit via the
    CX / H / H / CX sandwich; the ancilla stays 0 iff the data qubit is in the
    asserted superposition.
    flag == 1: assert an arbitrary single-qubit superposition parameterised by
    (theta, phi) per qubit through the H / controlled-U / H sandwich. The old
    `cu3(2*theta, phi, pi-phi, ...)` becomes `cu(2*theta, phi, pi-phi, 0, ...)`.
    """
    qr_assert = QuantumRegister(len(qubitList), name="superposition_assertion")
    cr_assert = ClassicalRegister(len(qubitList),
                                  name="cr_superposition_assertion")
    qProg.add_register(qr_assert)
    qProg.add_register(cr_assert)

    if flag == 0:
        for i in range(len(qubitList)):
            qProg.cx(qubitList[i], qr_assert[i])
            qProg.h(qubitList[i])
            qProg.h(qr_assert[i])
            qProg.cx(qubitList[i], qr_assert[i])
        qProg.measure(qr_assert, cr_assert)
    elif flag == 1:
        if len(qubitList) != len(phaseDict):
            print("qubitList length and phaseDict length does not match")
            return False
        for i in range(len(qubitList)):
            theta = phaseDict[qubitList[i]][0]
            phi = phaseDict[qubitList[i]][1]
            qProg.h(qr_assert[i])
            # qiskit-0.13:  cu3(2*theta, phi, pi - phi, ctrl, tgt)
            # qiskit-2.4 :  cu(theta, phi, lam, gamma=0, ctrl, tgt) == old cu3
            qProg.cu(2 * theta, phi, pi - phi, 0, qr_assert[i], qubitList[i])
            qProg.h(qr_assert[i])
        qProg.measure(qr_assert, cr_assert)
    else:
        print("Invalid flag")
        return False
    return True


def entanglement_assertion(qProg, qubitList, flag):
    """Port of the authors' entanglement_assertion: a single ancilla collects
    the parity of the data qubits via CX so that a Bell/GHZ-type entangled
    pattern leaves the ancilla at 0 (pass) and a broken pattern flips it."""
    qr_assert = QuantumRegister(1, name="entanglement_assertion")
    cr_assert = ClassicalRegister(1, name="cr_entanglement_assertion")
    qProg.add_register(qr_assert)
    qProg.add_register(cr_assert)

    # pattern "a|00> + b|11>"
    if flag == 0:
        if len(qubitList) % 2 == 0:
            for i in range(len(qubitList)):
                qProg.cx(qubitList[i], qr_assert[0])
        elif len(qubitList) % 2 == 1:
            for i in range(len(qubitList)):
                qProg.cx(qubitList[i], qr_assert[0])
            for i in range(len(qubitList)):
                if i % 2 == 1:
                    qProg.cx(qubitList[i], qr_assert[0])
    # pattern "a|01> + b|10>"
    elif flag == 1:
        qProg.x(qr_assert[0])
        if len(qubitList) % 2 == 0:
            for i in range(len(qubitList)):
                qProg.cx(qubitList[i], qr_assert[0])
        elif len(qubitList) % 2 == 1:
            for i in range(len(qubitList)):
                qProg.cx(qubitList[i], qr_assert[0])
            for i in range(len(qubitList)):
                if i % 2 == 1:
                    qProg.cx(qubitList[i], qr_assert[0])
    else:
        print("Invalid flag")
        return False
    qProg.measure(qr_assert, cr_assert)
    return True


def calcSuccessrate(result, correct_output, num_assertion):
    """Port of the authors' calcSuccessrate. The assertion bits are the most
    significant bits of the counts key; an all-zero assertion prefix means the
    run passed the assertion. Returns the success probability over the runs
    whose assertion register read 0."""
    total = 0
    success = 0
    assertionString = '0' * num_assertion
    for keys in result.keys():
        if num_assertion == 0:
            total += result[keys]
        elif keys.startswith(assertionString):
            total += result[keys]
    for outputs in correct_output:
        success += result.get(outputs, 0)
    probability = success / total if total else 0.0
    print("total_count = ", total, "success_count = ", success,
          "success_rate = ", probability * 100, "%")
    return probability
