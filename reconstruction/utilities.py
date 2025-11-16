from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.providers.aer.noise import NoiseModel

from qiskit.providers.fake_provider import FakeWashington
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.providers.aer.noise import NoiseModel
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

import argparse
from collections import defaultdict
import pulp

quantum_path = '../quantum_circuits/'


quantum_gates = {
    "single_qubit": {
        "parameterized": {
            "rx": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],  
                "probability_change": "high",
                "recommended_params": [np.pi, np.pi/2]
            },
            "ry": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],  
                "probability_change": "high",
                "recommended_params": [np.pi, np.pi/2]
            }
        },
        "non_parameterized": {
            "x": {
                "probability_change": "high"
            },
            "y": {
                "probability_change": "high"
            },
            "h": {
                "probability_change": "high"
            }
        }
    },
    "two_qubit": {
        "parameterized": {
            "cp": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],
                "probability_change": "high",
                "recommended_params": [np.pi, np.pi/2]
            },
            "rzx": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],
                "probability_change": "high",
                "recommended_params": [np.pi/2, np.pi/4]
            },
            "rzz": { 
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],
                "probability_change": "high",
                "recommended_params": [np.pi, np.pi/2]
            }
        },
        "non_parameterized": {
            "cx": {
                "probability_change": "high"
            },
            "cz": {
                "probability_change": "high"
            }
        }
    }
}


def solve_qmon_for_circuit(circ_name: str, entry: dict,
                           Qmax: int = 30,
                           two_stage: bool = True,
                           prefer_large_index: bool = True):
    Qbase = int(entry['num_oq'])
    nodes = sorted(k for k in entry.keys() if k != 'num_oq')
    anc = {n: int(entry[n]['num_eq']) for n in nodes}
    S   = {n: set(entry[n]['qubits']) for n in nodes}
    qubits = sorted(set().union(*S.values()) if nodes else [])

    # ---------- Stage 1: maximize the number of nodes ----------
    m1 = pulp.LpProblem(f"{circ_name}_obj1", pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', nodes, lowBound=0, upBound=1, cat='Binary')
    z = pulp.LpVariable.dicts('z', qubits, lowBound=0, upBound=1, cat='Binary')

    m1 += Qbase + pulp.lpSum(anc[n] * x[n] for n in nodes) <= Qmax

    for q in qubits:
        m1 += z[q] <= pulp.lpSum(x[n] for n in nodes if q in S[n])
        for n in nodes:
            if q in S[n]:
                m1 += z[q] >= x[n]

    m1 += pulp.lpSum(x[n] for n in nodes)
    m1.solve(pulp.PULP_CBC_CMD(msg=False))
    obj1_star = int(round(pulp.value(pulp.lpSum(x[n] for n in nodes)) or 0))

    if not two_stage:
        # Single-stage approximate lexicographic ordering + optional tie-breaker:
        # use a large weight to ensure the main objective is not affected
        W = 1 + len(qubits)
        T = 1 + sum(nodes) if prefer_large_index else 0
        mW = pulp.LpProblem(f"{circ_name}_weighted", pulp.LpMaximize)
        xW = pulp.LpVariable.dicts('x', nodes, 0, 1, cat='Binary')
        zW = pulp.LpVariable.dicts('z', qubits, 0, 1, cat='Binary')
        mW += Qbase + pulp.lpSum(anc[n] * xW[n] for n in nodes) <= Qmax
        for q in qubits:
            mW += zW[q] <= pulp.lpSum(xW[n] for n in nodes if q in S[n])
            for n in nodes:
                if q in S[n]:
                    mW += zW[q] >= xW[n]
        # Objective: W * Obj1 + Obj2 + (tiny secondary) preference on the index
        mW += W * pulp.lpSum(xW[n] for n in nodes) \
              + pulp.lpSum(zW[q] for q in qubits) \
              + (0 if not prefer_large_index else pulp.lpSum(n * xW[n] for n in nodes) / T)
        mW.solve(pulp.PULP_CBC_CMD(msg=False))
        picked_nodes  = [n for n in nodes if pulp.value(xW[n]) > 0.5]
        picked_qubits = sorted({q for q in qubits if pulp.value(zW[q]) > 0.5})
        return {
            'obj1': len(picked_nodes),
            'obj2': len(picked_qubits),
            'picked_nodes': picked_nodes,
            'picked_qubits': picked_qubits,
        }

    # ---------- Stage 2: maximize the number of distinct qubits after fixing Obj1 ----------
    m2 = pulp.LpProblem(f"{circ_name}_obj2", pulp.LpMaximize)
    x2 = pulp.LpVariable.dicts('x', nodes, 0, 1, cat='Binary')
    z2 = pulp.LpVariable.dicts('z', qubits, 0, 1, cat='Binary')
    m2 += Qbase + pulp.lpSum(anc[n] * x2[n] for n in nodes) <= Qmax
    m2 += pulp.lpSum(x2[n] for n in nodes) == obj1_star
    for q in qubits:
        m2 += z2[q] <= pulp.lpSum(x2[n] for n in nodes if q in S[n])
        for n in nodes:
            if q in S[n]:
                m2 += z2[q] >= x2[n]
    m2 += pulp.lpSum(z2[q] for q in qubits)
    m2.solve(pulp.PULP_CBC_CMD(msg=False))
    obj2_star = int(round(pulp.value(pulp.lpSum(z2[q] for q in qubits)) or 0))

    if not prefer_large_index:
        picked_nodes  = [n for n in nodes if pulp.value(x2[n]) > 0.5]
        picked_qubits = sorted({q for q in qubits if pulp.value(z2[q]) > 0.5})
        return {'obj1': obj1_star, 'obj2': len(picked_qubits),
                'picked_nodes': picked_nodes, 'picked_qubits': picked_qubits}

    # ---------- Stage 3 (tie-break): with Obj1/Obj2 fixed at optimum, prefer larger indices ----------
    m3 = pulp.LpProblem(f"{circ_name}_obj3", pulp.LpMaximize)
    x3 = pulp.LpVariable.dicts('x', nodes, 0, 1, cat='Binary')
    z3 = pulp.LpVariable.dicts('z', qubits, 0, 1, cat='Binary')
    m3 += Qbase + pulp.lpSum(anc[n] * x3[n] for n in nodes) <= Qmax
    m3 += pulp.lpSum(x3[n] for n in nodes) == obj1_star
    for q in qubits:
        m3 += z3[q] <= pulp.lpSum(x3[n] for n in nodes if q in S[n])
        for n in nodes:
            if q in S[n]:
                m3 += z3[q] >= x3[n]
    m3 += pulp.lpSum(z3[q] for q in qubits) == obj2_star
    # Third objective: maximize the weighted sum of node indices (the larger, the better)
    m3 += pulp.lpSum(n * x3[n] for n in nodes)
    m3.solve(pulp.PULP_CBC_CMD(msg=False))

    picked_nodes  = [n for n in nodes if pulp.value(x3[n]) > 0.5]
    picked_qubits = sorted({q for q in qubits if pulp.value(z3[q]) > 0.5})
    return {'obj1': obj1_star, 'obj2': len(picked_qubits),
            'picked_nodes': picked_nodes, 'picked_qubits': picked_qubits}



def calculate_num_of_extra_qubits(gate_list, qubit_index_list, eq_container, num_container, node_index):
    # print("qubit_index_list = ", qubit_index_list)
    if node_index not in eq_container:
        eq_container[node_index] = {}
    # print(eq_container[node_index])
    for gate in gate_list:
        # print(node_index, eq_container[node_index], num_container)
        ind = num_container[0]
        if not gate:
            continue
        for qubit in gate[3]:
            # print("calculate_num_of_extra_qubits", eq_container, gate[3])
            if qubit not in qubit_index_list:
                # print('qubit = ', qubit)
                if qubit not in eq_container[node_index]:
                    eq_container[node_index][qubit] = ind
                    num_container[0] += 1
                    ind = num_container[0]
        if len(gate) >= 5:
            for nested_g in gate[4:]:
                if not nested_g:
                    continue
                for n_g in nested_g:
                    for qubit in n_g[3]:
                        if qubit not in qubit_index_list:
                            if qubit not in eq_container[node_index]:
                                eq_container[node_index][qubit] = ind
                                num_container[0] += 1
                                ind = num_container[0]
                    if len(n_g) >= 5:
                        for g in n_g[4:]:
                            if not g:
                                continue
                            calculate_num_of_extra_qubits(g, qubit_index_list, eq_container, num_container, node_index)


def require_extra_qubit(gate_list):
    for gate in gate_list:
        if len(gate) >= 5:
            for i in range(4, len(gate)):
                if gate[i]:
                    return True
    return False


def filter_operation_list(file, operation_list, num=1000):
    eq_container = {}
    num_container = [0]
    operation_list_dict = {}
    
    for i, gate_list in enumerate(operation_list[file]):
        # print("gate_list = ", gate_list)
        if gate_list[0] not in operation_list_dict:
                operation_list_dict[gate_list[0]] = [gate_list[0], gate_list[1], {}, gate_list[-1]]
        operation_list_dict[gate_list[0]][2][gate_list[2]] = [gate_list[3], gate_list[4]]
        # if gate_list[1] == 'swap' and len(operation_list_dict[gate_list[0]][2]) >= 1:
        #     operation_list_dict[gate_list[0]][-1].extend(gate_list[-1])
        
    # for i, gate_list in enumerate(operation_list[file]):
    #     # print("gate_list = ", gate_list)
    #     if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
    #         if gate_list[0] not in node_list:
    #             node_list.append(gate_list[0])
    #             calculate_num_of_extra_qubits(gate_list[-1], gate_list[2], eq_container, num_container, gate_list[0])
    for key, gate_list in operation_list_dict.items():       
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            qubit_index_list = []
            for qubit_index in gate_list[2]:
                qubit_index_list.append(qubit_index)
            calculate_num_of_extra_qubits(gate_list[-1], qubit_index_list, eq_container, num_container, gate_list[0])
    
    print(file, f"eq_container = {eq_container}")
    
    return num_container[0], eq_container


def generate_new_operation_list(gate_list, node_index, gate_list_container):
    if node_index not in gate_list_container:
        gate_list_container[node_index] = []  # Create a separate mapping for each node
    for gate in gate_list:
        # print("gate = ", gate)
        if gate[:4] in gate_list_container[node_index]:
            if len(gate) == 4:
                continue
        else:
            gate_list_container[node_index].append(gate[:4])
        
        if len(gate) >= 5:
            for i in range(4, len(gate)):
                if gate[i]:
                    generate_new_operation_list(gate[i], node_index, gate_list_container)
        
    # print("gate_list_container = ", gate_list_container)
    return gate_list_container



def generate_gate(gate_list, new_qc, qubit_index_list, q_to_eq, ind_container, node_index, gate_list_container):
    # print(gate_list)
    new_operation_list = generate_new_operation_list(gate_list, node_index, gate_list_container)[node_index]
    new_operation_list.sort()

    # print(node_index, new_operation_list)

    # return new_operation_list
    eq = new_qc.qregs[-1]  # Get the extra ancilla qubit register
    if node_index not in q_to_eq:
        q_to_eq[node_index] = {}  # Create a separate mapping for each node
    
    i = 1
    for gate in new_operation_list:
                    
        ind = ind_container[0]
        # Determine the mapping of the target qubits
        for qubit in gate[3]:
            # if qubit != qubit_index:
            if qubit not in qubit_index_list:
                # print(qubit, qubit_index)
                if qubit not in q_to_eq[node_index]:
                    q_to_eq[node_index][qubit] = ind
                    ind_container[0] += 1   # update ind
                    ind = ind_container[0]
        
        # print("q_to_eq[node_index]", q_to_eq[node_index])
        if len(gate[3]) >= 2:
            try:
                qubit_list = []
                for qubit in gate[3]:
                    qubit_list.append(eq[q_to_eq[node_index][qubit]] if qubit in q_to_eq[node_index] else new_qc.qubits[qubit])
                # print(qubit_list)
                if not gate[2]:  # Parameterless single-qubit gate
                    getattr(new_qc, gate[1])(*qubit_list)

                else:  # Parameterized single-qubit gate
                    # if gate[1] == 'cu1':
                    #     gate[1] = 'cp'
                    para = gate[2]
                    getattr(new_qc, gate[1])(*para, *qubit_list)
            except Exception as e:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        else:
            # print(eq, q_to_eq, gate[3][0])
            target_qubit = eq[q_to_eq[node_index][gate[3][0]]] if gate[3][0] in q_to_eq[node_index] else new_qc.qubits[qubit]
            if not gate[2]:
                try:
                # print(new_qc.data)
                    getattr(new_qc, gate[1])(target_qubit)
                except Exception as e:
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            else:
                try:
                    # if gate[1] in ['u2', 'u3']:
                    #     theta = pi/2 if gate[1] == 'u2' else gate[2][0]
                    #     phi = gate[2][0] if gate[1] == 'u2' else gate[2][1]
                    #     lam = gate[2][1] if gate[1] == 'u2' else gate[2][2]
                    #     getattr(new_qc, 'u')(theta, phi, lam, target_qubit)
                    # elif gate[1] == 'u1':
                    #     getattr(new_qc, 'u')(0, 0, gate[2][0], target_qubit)
                    # else:
                    para = gate[2]
                    getattr(new_qc, gate[1])(*para, target_qubit)
                except Exception as e:
                    print("*****************************")
        
        if i == len(new_operation_list):
            # new_qc.data[-1][0].label = "test_" + str(node_index)
            ci = new_qc.data[-1]                 # CircuitInstruction（含 operation / qubits / clbits / condition）
            op = ci.operation.to_mutable()       # Obtain a mutable copy of the same gate (parameters, definition, etc. are copied along with it)
            op.label = f"test_{node_index}"      # Only modify metadata
            new_qc.data[-1] = ci.replace(operation=op)  # Replace only the operation; keep all other fields unchanged
            # print("reset")
        
        i += 1

    return new_qc


def get_unwanted_circuits_list(folder_path, simulator):
    unwanted_circuit_list = []
    with open(folder_path + '/statevector_list.pkl', 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                qc = QuantumCircuit.from_qasm_file(item[0])
                for i, gate in enumerate(transpile(qc, simulator).data):
                    # if gate[0].name in ['cswap', 'ccx', 'ccz']:
                    #     unwanted_circuit_list.append(str(item[0]))
                        break
            except EOFError:
                break
    
    return unwanted_circuit_list


def generate_monitoring_circuit(qc, file, operation_list, num=1000, ans=None):
    operation_list_dict = {}
    eq_container = {}
    num_container = [0]
    
    for i, gate_list in enumerate(operation_list[file]):
        # print("gate_list = ", gate_list)
        if ans and gate_list[0] not in ans["picked_nodes"]:
            continue

        if gate_list[0] not in operation_list_dict:
                operation_list_dict[gate_list[0]] = [gate_list[0], gate_list[1], {}, gate_list[-1]]
        operation_list_dict[gate_list[0]][2][gate_list[2]] = [gate_list[3], gate_list[4]]
        # if gate_list[1] == 'swap' and len(operation_list_dict[gate_list[0]][2]) >= 1:
        #     operation_list_dict[gate_list[0]][-1].extend(gate_list[-1])

    # for i, gate_list in enumerate(operation_list_new):
    for key, gate_list in operation_list_dict.items():
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            # if gate_list[0] not in operation_list_new:
                # operation_list_new[gate_list[0]] = [gate_list[0], gate_list[1], {}, gate_list[-1]]
                # operation_list_new.append(gate_list)
            qubit_index_list = []
            for qubit_index in gate_list[2]:
                qubit_index_list.append(qubit_index)
            calculate_num_of_extra_qubits(gate_list[-1], qubit_index_list, eq_container, num_container, gate_list[0])
            # print('\n')

            # operation_list_new[gate_list[0]][2][gate_list[2]] = [gate_list[3], gate_list[4]]
            
    
    print(file, num_container[0])
    # qc = QuantumCircuit.from_qasm_file(file)
    # for op in operation_list_dict:
    #     print(operation_list_dict[op])
    #     print('\n')
    # return
    new_qc = QuantumCircuit()
    for reg in qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    if new_qc.num_clbits < new_qc.num_qubits:
        new_qc.add_register(ClassicalRegister(new_qc.num_qubits - new_qc.num_clbits, 'ec'))
    
    # Add additional ancilla qubit registers
    if num_container[0] != 0:
        additional_qubits = QuantumRegister(num_container[0], 'eq')
        new_qc.add_register(additional_qubits)
    # return
    # Iterate through and insert all gates
    ind_container = [0]
    q_to_eq = {}
    gate_list_container = {}
    level = 0
    for i, (instr, qargs, cargs) in enumerate(qc.data):
        new_qc.append(instr, qargs, cargs)  # Add the original gate operations
        
        for _, op in operation_list_dict.items():
            if i == op[0]:
                # print(f"i={i}, op[0]={op[0]}")
                qubit_index_list = []
                for qubit in op[2]:
                    qubit_index_list.append(qubit)
                for qubit in qargs:
                    # if qc.find_bit(qubit).index == op[2]:
                    if qc.find_bit(qubit).index in qubit_index_list:
                        # print("Insert measurement and reset gate operations")
                        # new_qc.measure(new_qc.qubits[op[2]], new_qc.clbits[op[2]])
                        # new_qc.reset(new_qc.qubits[op[2]])
                        new_qc.measure(new_qc.qubits[qc.find_bit(qubit).index], new_qc.clbits[qc.find_bit(qubit).index])
                        new_qc.reset(new_qc.qubits[qc.find_bit(qubit).index])
                        # eq = new_qc.qregs[-1]
                        # continue
                        # new_qc = generate_gate(op[-1], new_qc, op[2], q_to_eq, ind_container, i, gate_list_container)
                eq = new_qc.qregs[-1]
                new_qc = generate_gate(op[-1], new_qc, qubit_index_list, q_to_eq, ind_container, i, gate_list_container)
    return new_qc