from typing import List, Tuple, Dict, Any
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UGate, PhaseGate


# ===== Allowed single and two qubit primitive gates (from the gate list, removing all gates with ≥ 3 qubits) =====
BASIS_1Q_2Q: List[str] = [
    "u3","u2","u1","cx","id","u0","u","p",
    "x","y","z","h","s","sdg","t","tdg",
    "rx","ry","rz","sx","sxdg",
    "cz","cy","swap","ch",
    "cu1","cp","cu3","csx","cu",
    "rxx","rzz","xx_plus_yy","ecr"
]
# ============================================================================


def contains_three_plus_ops(qc: QuantumCircuit) -> Tuple[bool, List[Tuple[int, str, int]]]:
    """Check whether the circuit contains any gates acting on ≥ 3 qubits; return (has_such_gates, [(index, gate_name, num_qubits), ...])."""
    info = []
    for i, ci in enumerate(qc.data):
        nq = ci.operation.num_qubits
        if nq >= 3:
            info.append((i, ci.operation.name, nq))
    return (len(info) > 0), info


def rewrite_legacy_gates_inplace(qc: QuantumCircuit, copy_label: bool = True, tol: float = 1e-12) -> QuantumCircuit:
    """
    In-place replacement of u1/u2/u3/cu1/cu2/cu3 with their modern equivalent gates
    (without changing qubits/clbits/order/conditions).
    - u1(λ) -> PhaseGate(λ) ~ RZ(λ) (differs only by a global phase)
    - u2(φ, λ) -> UGate(pi/2, φ, λ)
    - u3(θ, φ, λ) -> UGate(θ, φ, λ)
    - cu1(λ) -> ControlledPhaseGate(λ) (implemented as PhaseGate(λ).control(1) here)
    - cu2(φ, λ) -> UGate(pi/2, φ, λ).control(1)
    - cu3(θ, φ, λ) -> UGate(θ, φ, λ).control(1)
    """
    new_data = []
    for ci in qc.data:
        op = ci.operation
        name = op.name.lower()
        new_op = None

        if name == "u1":
            (lam,) = op.params
            new_op = PhaseGate(lam)
        elif name == "u2":
            phi, lam = op.params
            new_op = UGate(np.pi/2, phi, lam)
        elif name == "u3":
            theta, phi, lam = op.params
            new_op = UGate(theta, phi, lam)
        elif name == "cu1":
            (lam,) = op.params
            new_op = PhaseGate(lam).control(1)
        elif name == "cu2":
            phi, lam = op.params
            new_op = UGate(np.pi/2, phi, lam).control(1)
        elif name == "cu3":
            theta, phi, lam = op.params
            new_op = UGate(theta, phi, lam).control(1)

        if new_op is None:
            new_data.append(ci)
            continue

        # Copy label (if any)
        if copy_label and getattr(op, "label", None) is not None:
            mut = new_op.to_mutable()
            mut.label = op.label
            new_op = mut

        # Copy the condition (stored on the operation in different versions)
        cond = getattr(op, "condition", None)
        if cond is not None:
            try:
                mut = new_op.to_mutable()
                mut.condition = cond
                new_op = mut
            except Exception:
                try:
                    new_op.condition = cond
                except Exception:
                    pass

        new_data.append(ci.replace(operation=new_op))

    qc.data = new_data
    return qc


# ---------- Internal utilities: getting/setting conditions, compatible with different Terra versions ----------
def _get_condition_from_ci(ci) -> Any:
    return getattr(ci.operation, "condition", None)

def _set_condition_on_op(op, cond):
    if cond is None:
        return op
    try:
        mut = op.to_mutable()
        mut.condition = cond
        return mut
    except Exception:
        try:
            op.condition = cond
        except Exception:
            pass
        return op
# ----------------------------------------------------------------


def _decompose_op_to_1q2q(ci, basis: List[str], seed: int = 2025) -> List:
    """
    Locally decompose this single ≥ 3-qubit instruction `ci` into a list of
    instructions containing only 1- and 2-qubit gates.
    - Preserve the original qubit mapping and order;
    - Copy the original condition to every expanded instruction;
    - Do not modify any other instructions.
    """

    op = ci.operation
    qargs = list(ci.qubits)
    k = op.num_qubits

    # Place this single gate into a k-qubit subcircuit
    sub = QuantumCircuit(k, name=f"__local_unroll_{op.name}")
    sub.append(op, list(range(k)))

    # Decompose only this subcircuit, restricted to the specified 1- and 2-qubit primitive gates
    sub_t = transpile(
        sub,
        basis_gates=basis,
        optimization_level=0,
        layout_method="trivial",
        seed_transpiler=seed,
    )

    # If there are still ≥ 3-qubit gates, try decomposing a few more layers and then retry
    if any(x.operation.num_qubits >= 3 for x in sub_t.data):
        sub_t = transpile(
            sub.decompose(reps=10),
            basis_gates=basis,
            optimization_level=0,
            layout_method="trivial",
            seed_transpiler=seed,
        )

    cond = _get_condition_from_ci(ci)
    out = []

    for ci2 in sub_t.data:
        # Map the subcircuit qubits back to the physical qubits of this gate in the original circuit
        local_idxs = [sub_t.find_bit(q).index for q in ci2.qubits]
        mapped_qubits = tuple(qargs[j] for j in local_idxs)

        # Set the condition on the new operation
        new_op = _set_condition_on_op(ci2.operation, cond)

        # Replace the operation / qubits (keep clbits etc. as in ci2 by default)
        out.append(ci2.replace(operation=new_op, qubits=mapped_qubits))

    return out


def replace_three_plus_ops_inplace(
    qc: QuantumCircuit,
    basis_1q_2q: List[str] = BASIS_1Q_2Q,
    seed: int = 2025,
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    In-place replacement of all ≥ 3-qubit gates in the circuit with equivalent
    sequences composed only of 1- and 2-qubit gates.
    All other instructions keep their original order and positions.
    Return (qc, stats).
    """
    new_data = []
    stats: Dict[str, Any] = {"replaced": 0, "kept": 0, "names": {}}

    for ci in qc.data:
        op = ci.operation
        stats["names"][op.name] = stats["names"].get(op.name, 0) + 1

        if op.num_qubits >= 3:
            repl = _decompose_op_to_1q2q(ci, basis_1q_2q, seed)
            # A very small number of custom gates may still not be fully decomposable: keep them as a fallback
            if any(x.operation.num_qubits >= 3 for x in repl):
                new_data.append(ci)
                stats["kept"] += 1
            else:
                new_data.extend(repl)
                stats["replaced"] += 1
        else:
            new_data.append(ci)
            stats["kept"] += 1

    qc.data = new_data
    return qc


def rewrite_legacy_gates_inplace(qc, copy_label=True, tol=1e-12):
    """
    In-place replacement of all u1/u2/u3/cu1/cu2/cu3 gates with their modern equivalents.
    - Only replace the operation; do not change qubits/clbits/conditions, so the circuit's quantum behavior stays the same.
    - If the original gate has a label, copy it to the new gate by default (using to_mutable()).
    """
    for i, ci in enumerate(qc.data):
        op = ci.operation
        name = op.name.lower()
        new_op = None

        if name == 'u1':
            (lam,) = op.params
            new_op = PhaseGate(lam)

        elif name == 'u2':
            phi, lam = op.params
            new_op = UGate(np.pi/2, phi, lam)

        elif name == 'u3':
            theta, phi, lam = op.params
            new_op = UGate(theta, phi, lam)

        elif name == 'cu1':
            (lam,) = op.params
            # Could also specialize: if ≈ π, use CZGate(); here we use the general controlled-phase form
            new_op = PhaseGate(lam).control(1)

        elif name == 'cu2':
            phi, lam = op.params
            new_op = UGate(np.pi/2, phi, lam).control(1)

        elif name == 'cu3':
            theta, phi, lam = op.params
            new_op = UGate(theta, phi, lam).control(1)

        if new_op is not None:
            if copy_label and getattr(op, "label", None) is not None:
                mut = new_op.to_mutable()
                mut.label = op.label
                new_op = mut
            qc.data[i] = ci.replace(operation=new_op)

    return qc