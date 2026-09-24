"""
MPS-based monitorable-node selection and structural reconstruction-overhead
counting, for scalability analysis on circuits too large for statevector
(>~20 qubits).

Two ideas make 30-50 qubit analysis feasible without executing anything:

1. Node finding (which gates are monitorable) needs only the single-qubit
   reduced density matrix after each gate. With a matrix-product-state
   backend that is O(n*chi^3) per save, so structured low-entanglement
   circuits (exactly where monitoring is useful) are tractable. q is
   separable from the rest iff det(rho_q) <= tol -- the same criterion the
   statevector path uses, since residual == det(rho_q).

2. Reconstruction overhead (extra qubits / extra gates to monitor a node)
   is the node's backward causal cone -- a pure DAG property, computed by
   backward reachability, no simulation at all.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from fast_analysis import SKIP_OPS

_MPS_SIM = AerSimulator(method="matrix_product_state")


def monitorable_nodes_mps(qc, tol: float = 1e-9, max_bond=None):
    """
    No-lock monitorable-node selection via MPS single-qubit RDMs.
    Returns {gate_index: [qubit, ...]} for every (gate, acted-qubit) where
    the qubit is separable from the rest immediately after the gate.

    One MPS pass: a save_density_matrix is inserted after each non-trivial
    gate for each acted qubit. Raises if the MPS backend cannot finish
    (bond dimension blow-up) -- caller treats that as "not MPS-tractable".
    """
    snapshot_circuit = qc.copy_empty_like()
    labels = {}  # label -> (gate_index, qubit)
    k = 0
    for i, ci in enumerate(qc.data):
        inst, qargs = ci.operation, ci.qubits
        snapshot_circuit.append(inst, qargs, ci.clbits)
        if inst.name in SKIP_OPS:
            continue
        for q in qargs:
            qi = qc.find_bit(q).index
            lbl = f"r{k}"
            snapshot_circuit.save_density_matrix(qubits=[qi], label=lbl)
            labels[lbl] = (i, qi)
            k += 1

    opts = {}
    if max_bond:
        # cap bond dimension; an exceeded cap (truncation that loses weight)
        # means the circuit is too entangled to analyze exactly -> we treat
        # it as "not MPS-tractable" rather than silently truncating
        opts["matrix_product_state_max_bond_dimension"] = max_bond
    sim = AerSimulator(method="matrix_product_state", **opts)
    res = sim.run(transpile(snapshot_circuit, sim)).result()
    if not res.success:
        raise RuntimeError("MPS backend failed (likely bond-dimension blow-up)")
    data = res.data()

    nodes = {}
    for lbl, (i, qi) in labels.items():
        rho = np.asarray(data[lbl])
        det = float(np.linalg.det(rho).real)  # == residual == entanglement
        if det <= tol:
            nodes.setdefault(i, []).append(qi)
    return nodes


def causal_cone(qc, gate_index, qubit):
    """
    Backward causal cone of qubit `qubit`'s state just after gate
    `gate_index`: the set of earlier gates it depends on and the set of
    qubits those gates touch. Pure DAG reachability, no simulation.

    Returns (cone_gate_indices: set, cone_qubits: set). The number of
    EXTRA qubits a reconstruction needs for this node is
    len(cone_qubits) - 1 (all cone qubits except the monitored one), and
    the number of EXTRA replayed gates is len(cone_gate_indices).
    """
    tracked = {qubit}
    cone_gates = set()
    cone_qubits = {qubit}
    for i in range(gate_index, -1, -1):
        ci = qc.data[i]
        if ci.operation.name in ("barrier", "measure", "reset"):
            continue
        idxs = [qc.find_bit(q).index for q in ci.qubits]
        if tracked.intersection(idxs):
            cone_gates.add(i)
            for q in idxs:
                tracked.add(q)
                cone_qubits.add(q)
    return cone_gates, cone_qubits


def overhead_for_nodes(qc, nodes):
    """
    Structural reconstruction overhead to monitor every (gate, qubit) in
    `nodes` (dict gate_index -> [qubits]); no execution.

    Returns dict with totals and per-node lists. Extra qubits are summed
    per node (each node uses fresh ancillas, matching generate_monitoring_
    circuit); a union variant is also reported for the shared-ancilla bound.
    """
    per_node = []
    total_extra_q = 0
    total_extra_g = 0
    union_gates = set()
    for i in sorted(nodes):
        for q in nodes[i]:
            cg, cq = causal_cone(qc, i, q)
            eq = len(cq) - 1
            per_node.append({"gate": i, "qubit": q,
                             "extra_qubits": eq, "extra_gates": len(cg)})
            total_extra_q += eq
            total_extra_g += len(cg)
            union_gates |= cg
    return {
        "num_nodes": len(per_node),
        "total_extra_qubits": total_extra_q,
        "total_extra_gates": total_extra_g,
        "union_replayed_gates": len(union_gates),
        "max_cone_qubits": max((p["extra_qubits"] for p in per_node), default=0),
        "max_cone_gates": max((p["extra_gates"] for p in per_node), default=0),
        "per_node": per_node,
    }
