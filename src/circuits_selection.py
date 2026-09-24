from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import passes
from qiskit.transpiler.passmanager import PassManager
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from math import pi
import copy

import pickle
from qiskit.quantum_info import partial_trace, Statevector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, concurrence, schmidt_decomposition

from utilities import *

simulator = Aer.get_backend('statevector_simulator')

_DIRECTIVE_OPS = frozenset({
    "barrier", "snapshot", "delay", "save_state", "save_statevector",
})


def standard_final_read_nodes(qc, nodes):
    """Return selected points observable through a terminal measurement.

    A source-gate point is a free final read only when the last
    non-directive operation on that same qubit is a measurement.  Merely
    being the last source gate on an unmeasured qubit is not observable.
    """
    last_source_gate = [None] * qc.num_qubits
    terminally_measured = [False] * qc.num_qubits
    for gate_index, instruction in enumerate(qc.data):
        name = instruction.operation.name
        qubits = [qc.find_bit(bit).index for bit in instruction.qubits]
        if name in _DIRECTIVE_OPS:
            continue
        if name == "measure":
            for qubit in qubits:
                terminally_measured[qubit] = True
            continue
        if name == "reset":
            raise ValueError("standard final-read classification does not support reset")
        for qubit in qubits:
            last_source_gate[qubit] = gate_index
            terminally_measured[qubit] = False

    return {
        (int(gate_index), int(qubit))
        for gate_index, qubit in nodes
        if terminally_measured[int(qubit)]
        and int(gate_index) == last_source_gate[int(qubit)]
    }


def _single_vs_rest_entangled(psi: Statevector, q: int, tol=1e-12):
    """Test q|rest with the direct second Schmidt amplitude.

    Qiskit's ``schmidt_decomposition`` applies its own absolute cutoff before
    returning coefficients, which can silently override a stricter caller
    tolerance.  A direct 2 x 2**(n-1) SVD preserves the declared ``tol`` rule.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be finite and nonnegative, got {tol!r}")
    n = psi.num_qubits
    if n is None:
        raise ValueError("single-vs-rest selection requires qubit dimensions")
    if not isinstance(q, (int, np.integer)) or not 0 <= int(q) < n:
        raise ValueError(f"qubit index {q!r} is outside [0, {n})")

    # Qiskit statevector axes are ordered q_(n-1), ..., q_0.
    tensor = np.asarray(psi.data, dtype=np.complex128).reshape((2,) * n)
    axis = n - 1 - int(q)
    amplitude_matrix = np.moveaxis(tensor, axis, 0).reshape(2, -1)
    amplitudes = np.linalg.svd(
        amplitude_matrix, compute_uv=False, full_matrices=False
    )
    values = [float(value) for value in amplitudes]
    s2 = values[1] if len(values) > 1 else 0.0
    return (s2 > tol), s2, values


def analyze_per_gate(qc, tol: float = 1e-12, swap_unlock: bool = True):
    """
    Evolve the circuit gate by gate (without transpiling).
    - Single-qubit gate: if the qubit is not locked, append gate_index to its path.
    - Two-qubit gate: perform a "double check" and record an event; for each qubit,
    use the q|rest test to decide whether it becomes locked.
    - If it is a SWAP and swap_unlock=True: re-evaluate a, b individually; if a qubit
        is "not entangled with the rest" → allow unlocking, and append this gate index
        to that qubit's path.
    - Non-SWAP gates: can only lock qubits, never unlock them.
    Returns:
    res: List[[gate_index, gate_name, qubit_index, p0, p1]]
    events: List[... information from the double checks and per-qubit results ...]
    paths_until_first_ent: Dict[int, List[int]] (allows "reopening the path" at SWAP
        gates and adding that step)
    """
    n = qc.num_qubits
    partial_circuit = QuantumCircuit()
    for reg in qc.qregs:
        new_reg = QuantumRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    for reg in qc.cregs:
        new_reg = ClassicalRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)

    res = {}
    events = []
    paths_until_first_ent = {q: [] for q in range(n)}
    locked = np.zeros(n, dtype=bool)

    for i, (inst, qargs, cargs) in enumerate(qc.data):
        if inst.name in ("measure", "barrier", "reset", "snapshot", "delay",
                         "save_state", "save_statevector"):
            continue

        partial_circuit.data = qc.data[:i+1]
        
        job = simulator.run(partial_circuit)
        result = job.result()
        statevector = result.get_statevector()

        quantum_state = Statevector(statevector)
        
        qidx = [qc.find_bit(q).index if not isinstance(q, int) else q for q in qargs]

        res[i] = []
        for qi in qidx:
            qubit_state = partial_trace(quantum_state, [q for q in range(qc.num_qubits) if q != qi])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res[i].append([i, inst.name, qi, prob_zero, prob_one])

        if len(qidx) == 1:
            qi = qidx[0]
            if not locked[qi]:
                paths_until_first_ent[qi].append(i)

        elif len(qidx) == 2:
            a, b = qidx

            # Double-check (for analysis/logging).
            rest = [k for k in range(n) if k not in (a, b)]
            if rest:
                rho_ab = partial_trace(quantum_state, rest)          # keep a,b
                C = float(concurrence(rho_ab))
                lams_ab = [t[0] for t in schmidt_decomposition(quantum_state, qargs=[a, b])]
                s2_ab = float(lams_ab[1]) if len(lams_ab) > 1 else 0.0
            else:
                # n == 2: the pair IS the whole system; concurrence applies
                # to the pure state directly and the pair-vs-rest test is moot
                C = float(concurrence(quantum_state))
                s2_ab = 0.0
            pairwise = (C > tol)
            vs_rest = (s2_ab > tol)

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

            # Locking / unlocking rules.
            if inst.name.lower() == 'swap' and swap_unlock:
                locked_a = copy.deepcopy(locked[a])
                locked_b = copy.deepcopy(locked[b])
                # print(a, locked_a)
                # print(b, locked_b)
                # After a SWAP: whichever qubit is not entangled with the rest at this step is allowed to be “unlocked”
                if not ent_a and not locked_b:
                    locked[a] = False
                if not ent_b and not locked_a:
                    locked[b] = False
                if ent_a:
                    locked[a] = True
                if ent_b:
                    locked[b] = True

                # Now record this step index in the paths of all currently unlocked qubits (including those just unlocked)
                if not locked[a]:
                    paths_until_first_ent[a].append(i)
                if not locked[b]:
                    paths_until_first_ent[b].append(i)

                # Note: if ent_a / ent_b is true, they remain locked as usual (do not extend their paths)
            else:
                # Non-SWAP gates: only “lock”; never unlock
                if ent_a and not locked[a]:
                    locked[a] = True
                if ent_b and not locked[b]:
                    locked[b] = True

                # For the side that is not yet locked, continue appending this gate index
                if not locked[a]:
                    paths_until_first_ent[a].append(i)
                if not locked[b]:
                    paths_until_first_ent[b].append(i)


        else:
            # > 2-qubit gates: do not check entanglement; append the index if the qubit is not locked
            for qi in qidx:
                if not locked[qi]:
                    paths_until_first_ent[qi].append(i)

        # print(paths_until_first_ent)
        # print(locked)

    return res, events, paths_until_first_ent


def filter_circuits_with_output_num(res_dict, paths_until_first_ent_dict):
    definited_node_dict = {}
    count = 0
    for key in res_dict:
        try:
            qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")

            selected_identities = {
                (int(node), int(qubit))
                for qubit, node_indices in paths_until_first_ent_dict[key].items()
                for node in node_indices
            }
            final_reads = standard_final_read_nodes(qc, selected_identities)
                        
            num = 0
            for qubit in paths_until_first_ent_dict[key]:
                for node in paths_until_first_ent_dict[key][qubit]:
                    if (node, qubit) in final_reads:
                        continue
                    if key not in definited_node_dict:
                        definited_node_dict[key] = []

                    for node_with_probability in res_dict[key][node]:
                        # only the qubit whose monitorable-node list contains
                        # this node is safe to measure here; the partner qubit
                        # of a two-qubit gate may still be entangled
                        if node_with_probability[2] != qubit:
                            continue
                        if node_with_probability not in definited_node_dict[key]:
                            definited_node_dict[key].append(node_with_probability)

                    num += 1
                        
            if num > 0:
                count += 1
        except EOFError:
                break
    
    print('circuits_count:', count)
    return definited_node_dict


# Function to perform Depth-First Search (DFS) on the quantum circuit
def dfs_quantum_circuit(qc, qubit_info, current_index, visited, path):
    # visited is a set of (index, qubit) tuples; list membership made large
    # circuits (grover/qwalk-noancilla >=8) blow past 6h, set is O(1)
    # Iterate backward through the circuit instructions
    for index in range(current_index, -1, -1):
        if (index, qubit_info) in visited:
            continue
        gate, qargs, cargs = qc.data[index]
        qubit_indices = [qc.find_bit(qubit).index for qubit in qargs]
        if qubit_info in qubit_indices:
            # Handle multi-qubit gates
            if len(qubit_indices) > 1:
                # # print(gate.name)
                # if gate.name == "swap":
                #     # Special handling for swap gate
                #     path.insert(0, [index, gate.name, gate.params, qubit_indices])
                #     other_qubit = qubit_indices[0] if qubit_indices[1] == qubit_info else qubit_indices[1]
                #     dfs_quantum_circuit(qc, other_qubit, index - 1, visited, path)
                #     # path.sort()
                #     break
                # elif gate.name == "iswap":
                #     print(gate.name)
                # # elif qubit_info == qubit_indices[0]:
                # #     # If the qubit is the control qubit, this gate does not affect the qubit's state
                # #     continue
                # else:
                #     # print(gate.name)
                #     # print(gate.name)
                #     # Handle other multi-qubit gates (e.g., cx, cz, etc.)
                for q in qubit_indices:
                    if q != qubit_info:
                        path.insert(0, [index, gate.name, gate.params, qubit_indices, dfs_quantum_circuit(qc, q, index - 1, visited, [])])
            # Record the current gate if it affects the qubit
            else:
                path.insert(0, [index, gate.name, gate.params, qubit_indices])
            # path.sort()
            visited.add((index, qubit_info))
    return path


# Retained corpus circuits whose backward-cone search exceeds the runtime
# limit and therefore has no monitoring plan.
DFS_EXCLUDED_CIRCUITS = frozenset(
    {
        "grover-noancilla_indep_qiskit_5.qasm",
        "grover-noancilla_indep_qiskit_6.qasm",
        "grover-noancilla_indep_qiskit_7.qasm",
        "qwalk-noancilla_indep_qiskit_6.qasm",
    }
)


def get_final_definited_node_dict(definited_node_dict, *, verbose=False):
    final_definited_node_dict = {}
    for key, value in definited_node_dict.items():
        if key in DFS_EXCLUDED_CIRCUITS:
            continue
        if verbose:
            print(key)
        qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")
        for node in value:
            index = node[0]
            gate_name = node[1]
            qubit_info = node[2]
            p0 = node[3]
            p1 = node[4]

            dfs_path = []
            visited = set()
            path = []
            # Perform DFS on the circuit
            dfs_path = dfs_quantum_circuit(qc, qubit_info, index, visited, path)

            if key not in final_definited_node_dict:
                final_definited_node_dict[key] = []
            final_definited_node_dict[key].append([index, gate_name, qubit_info, p0, p1, dfs_path])
            # print(index, "visited = ", visited)
    
    return final_definited_node_dict


def transformation(qc, qubit_info, gate_list, operation_list):
    if not gate_list:
        return operation_list
    # for gate in gate_list:
    #     if gate[1] == 'swap':
    #         for qubit in gate[3]:
    #             if qubit_info != qubit:
    #                 swap_recorder[qubit_info] = qubit
    #                 swap_recorder[qubit] = qubit_info
    #                 break
    
    for gate in gate_list:
        # print(f"gate_list = { gate_list}", '\n')
        # if not gate or gate[1] == 'swap':
        #     continue
        
        if len(gate) == 4:
            # flag = True
            # for qubit in gate[3]:
            #    if qubit_info == qubit:
            #        flag = False
            # if flag:
            #     operation_list.append([gate[0], gate[1], gate[2], [qubit_info if swap_qubit_info == qubit else qubit for qubit in gate[3]]])
            # else:
            #     operation_list.append([gate[0], gate[1], gate[2], [swap_qubit_info if qubit_info == qubit else qubit for qubit in gate[3]]])
            # operation_list.append([gate[0], gate[1], gate[2], [swap_recorder[qubit] if qubit in swap_recorder else qubit for qubit in gate[3]]])
            operation_list.append(gate)
        else:
            # flag = True
            # for qubit in gate[3]:
            #     if qubit_info == qubit:
            #         flag = False
            # if flag:
            #     target_gate = [gate[0], gate[1], gate[2], [qubit_info if swap_qubit_info == qubit else qubit for qubit in gate[3]]]
            # else:
            #     target_gate = [gate[0], gate[1], gate[2], [swap_qubit_info if qubit_info == qubit else qubit for qubit in gate[3]]]
            # target_gate = [gate[0], gate[1], gate[2], [swap_recorder[qubit] if qubit in swap_recorder else qubit for qubit in gate[3]]]
            target_gate = gate[:4]

            
            for qubit in gate[3]:
                other_qubit_info = None
                other_gate = []
                
                # if qubit == qubit_info or (qubit_info in swap_recorder and swap_recorder[qubit_info] == qubit):
                if qubit == qubit_info:
                    continue
                
                other_qubit_info = qubit
                other_other_gate = transformation(qc, other_qubit_info, gate[4], [])
                other_gate.extend(other_other_gate)
                
                target_gate.extend([other_gate])
                operation_list.append(target_gate)
    
    return operation_list


def bulid_partial_circuit(final_definited_node_dict, *, verbose=False):
    res = {}
    for key, value in final_definited_node_dict.items():
        if verbose:
            print(key, '\n')
        qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")
        for node in value:
            if not node[-1]:
                continue

            index = node[0]
            gate_name = node[1]
            qubit_info = node[2]
            p0 = node[3]
            p1 = node[4]

            operation_list = []
            
            swap_recorder = {}
            operation_list = transformation(qc, qubit_info, node[-1], operation_list)

            if key not in res:
                res[key] = []
            res[key].append([index, gate_name, qubit_info, p0, p1, operation_list])
    
    return res


def get_filtered_operation_list(operation_list, gate_length, num, unwanted_circuit_list):
    bad_list = []
    operation_list_new = {}
    circuit_node_constraint = {}
    for key, value in operation_list.items():
        # print(key)
        # if key != "qpeexact_indep_qiskit_4.qasm":
        #     continue
        extra_qubits_num, eq_container = filter_operation_list(key, operation_list, gate_length)
        if extra_qubits_num <= num and key not in [str(i) for i in unwanted_circuit_list]:
            operation_list_new[key] = value

            circuit_node_constraint[key] = {}
            circuit_node_constraint[key]['num_oq'] = int(key.split('.')[0].split('_')[-1])
        
            for node, eq in eq_container.items():
                circuit_node_constraint.setdefault(key, {}).setdefault(node, {})
                circuit_node_constraint[key][node]['num_eq'] = len(eq)
                circuit_node_constraint[key][node]['qubits'] = []
                
                for gate in operation_list_new[key]:
                    if gate[0] == node:
                        circuit_node_constraint[key][node]['qubits'].append(gate[2])

        # print(key)
    
    # print(operation_list_new)
    
    comparison_circuit = {}
    unblanced_circuit_list = []

    res = copy.deepcopy(operation_list_new)
    ans_dict = {}
    # print("operation_list_new", "\n")
    for k, v in operation_list_new.items():
        # print(k)
        # if k != "qpeexact_indep_qiskit_4.qasm":
        #     continue
        qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{k}")
        for l in v:
            if l[1] != qc.data[l[0]][0].name:
                print('unmatch:', k)
                del res[k] # delete unmatched circuits
                break

        ans = solve_qmon_for_circuit(k, circuit_node_constraint[k], Qmax=24, two_stage=True)
        ans_dict[k] = ans
        
        
        if k in res:
            # comparison_circuit[k] = generate_monitoring_circuit(qc, k, operation_list_new, gate_length)
            try:
                comparison_circuit[k] = generate_monitoring_circuit(qc, k, operation_list_new, gate_length, ans)
                result_qc = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=8192).result()
                if len(list(result_qc.get_counts().keys())[0]) < qc.num_qubits:
                    print('unblanced', k)
                    unblanced_circuit_list.append(k)
            except Exception as e:
                print(k, e)
                bad_list.append(k)
    
    # with open('../quantum_circuits/operation_list_new.pkl', 'wb') as file:
    #     pickle.dump(res, file)
    # print(bad_list)
    return res, comparison_circuit, unblanced_circuit_list, ans_dict
