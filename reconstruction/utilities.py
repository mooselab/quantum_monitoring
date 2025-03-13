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
            "crx": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],  
                "probability_change": "high",
                "recommended_params": [np.pi]
            },
            "cry": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],  
                "probability_change": "high",
                "recommended_params": [np.pi]
            },
            "crz": {
                "params": ["theta"],
                "ranges": [[0, 2*np.pi]],  
                "probability_change": "high",
                "recommended_params": [np.pi]
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



def remove_unmeasured_qubit_and_merge(counts):
    new_counts = defaultdict(int)  # Using `defaultdict` makes it easier to accumulate duplicate results
    for key, value in counts.items():
        new_key = key.split(' ')[-1]  # Remove the qubits that are not being measured.
        new_counts[new_key] += value  # Merge identical keys' results
    return dict(new_counts)


def calculate_tvd(counts1, counts2, shots):
    """ Compute the total variation distance (TVD) between two measurement distributions """
    # Normalize to a probability distribution
    prob1 = {k: v / shots for k, v in counts1.items()}
    prob2 = {k: v / shots for k, v in counts2.items()}
    
    # Merge all keys from both distributions
    all_keys = set([key.split(' ')[-1] for key in list(prob1.keys())]).union(set([key.split(' ')[-1] for key in list(prob2.keys())]))
    
    # Compute the total variation distance (TVD)
    tvd = 0.5 * sum(abs(prob1.get(k, 0) - prob2.get(k, 0)) for k in all_keys)
    return tvd


def calculate_num_of_extra_qubits(gate_list, qubit_index, eq_container, num_container, node_index):
    if node_index not in eq_container:
        eq_container[node_index] = {}
    for gate in gate_list:
        ind = num_container[0]
        if not gate:
            continue
        for qubit in gate[3]:
            if qubit.index != qubit_index:
                if qubit.index not in eq_container[node_index]:
                    eq_container[node_index][qubit.index] = ind
                    num_container[0] += 1
        if len(gate) == 5 and gate[4]:
            for cg in gate[4]:
                for qubit in cg[3]:
                    if qubit.index != qubit_index:
                        if qubit.index not in eq_container[node_index]:
                            eq_container[node_index][qubit.index] = ind
                            num_container[0] += 1
                if len(cg) == 5 and cg[4]:
                    calculate_num_of_extra_qubits(cg[4], qubit_index, eq_container, num_container, node_index)


def require_extra_qubit(gate_list):
    for gate in gate_list:
        if len(gate) == 5 and gate[4]:
            return True

    return False


def filter_operation_list(file, operation_list, num=1000):
    eq_container = {}
    num_container = [0]
    for i, gate_list in enumerate(operation_list[file]):
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            calculate_num_of_extra_qubits(gate_list[-1], gate_list[2].index, eq_container, num_container, i)
    
    return num_container[0]


def generate_new_operation_list(gate_list, node_index, gate_list_container):
    if node_index not in gate_list_container:
        gate_list_container[node_index] = []  # Create a separate mapping for each node
    for gate in gate_list:
        if gate[:4] in gate_list_container[node_index]:
            if len(gate) == 4:
                continue
        else:
            gate_list_container[node_index].append(gate[:4])
        
        if len(gate) == 5 and gate[4]:
            generate_new_operation_list(gate[4], node_index, gate_list_container)
        
    return gate_list_container



def generate_gate(gate_list, new_qc, qubit_index, q_to_eq, ind_container, node_index, gate_list_container):
    new_operation_list = generate_new_operation_list(gate_list, node_index, gate_list_container)[node_index]
    new_operation_list.sort()

    # return new_operation_list
    eq = new_qc.qregs[-1]  # Get the extra ancilla qubit register
    if node_index not in q_to_eq:
        q_to_eq[node_index] = {}  # Create a separate mapping for each node
    
    i = 1
    for gate in new_operation_list:
                    
        ind = ind_container[0]
        # Determine the mapping of the target qubits
        for qubit in gate[3]:
            if qubit.index != qubit_index:
                if qubit.index not in q_to_eq[node_index]:
                    q_to_eq[node_index][qubit.index] = ind
                    ind_container[0] += 1   # update ind
        
        # Insert a single-qubit gate
        if len(gate) == 4 or (len(gate) == 5 and not gate[4]):
            if len(gate[3]) == 2:
                # Ensure that both the control and target qubits use the ancilla qubits within the current node
                control_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                target_qubit = eq[q_to_eq[node_index][gate[3][1].index]] if gate[3][1].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                
                if not gate[2]:  # Parameterless single-qubit gate
                    getattr(new_qc, gate[1])(control_qubit, target_qubit)
                else:  # Parameterized single-qubit gate
                    if gate[1] == 'cu1':
                        gate[1] = 'cp'
                    para = gate[2][0]
                    getattr(new_qc, gate[1])(para, control_qubit, target_qubit)
            else:
                target_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                if not gate[2]:
                    getattr(new_qc, gate[1])(target_qubit)
                else:
                    if gate[1] in ['u2', 'u3']:
                        theta = pi/2 if gate[1] == 'u2' else gate[2][0]
                        phi = gate[2][0] if gate[1] == 'u2' else gate[2][1]
                        lam = gate[2][1] if gate[1] == 'u2' else gate[2][2]
                        getattr(new_qc, 'u')(theta, phi, lam, target_qubit)
                    elif gate[1] == 'u1':
                        getattr(new_qc, 'u')(0, 0, gate[2][0], target_qubit)
                    else:
                        para = gate[2][0]
                        getattr(new_qc, gate[1])(para, target_qubit)
        
        # Insert a two-qubit gate
        else:
            # If nested operations exist, process them recursively and update `ind_container` to prevent reuse
            if len(gate[3]) == 2:
                # Ensure that both the control and target qubits use the ancilla qubits within the current node
                control_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                target_qubit = eq[q_to_eq[node_index][gate[3][1].index]] if gate[3][1].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                
                if not gate[2]:  # Parameterless two-qubit gate
                    getattr(new_qc, gate[1])(control_qubit, target_qubit)
                else:  # Parameterized two-qubit gate
                    if gate[1] == 'cu1':
                        gate[1] = 'cp'
                    para = gate[2][0]
                    getattr(new_qc, gate[1])(para, control_qubit, target_qubit)
            else:
                target_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                if not gate[1]:
                    getattr(new_qc, gate[1])(target_qubit)
                else:
                    if gate[1] in ['u2', 'u3']:
                        theta = pi/2 if gate[1] == 'u2' else gate[2][0]
                        phi = gate[2][0] if gate[1] == 'u2' else gate[2][1]
                        lam = gate[2][1] if gate[1] == 'u2' else gate[2][2]
                        getattr(new_qc, 'u')(theta, phi, lam, target_qubit)
                    elif gate[1] == 'u1':
                        getattr(new_qc, 'u')(0, 0, gate[2][0], target_qubit)
                    else:
                        para = gate[2][0]
                        getattr(new_qc, gate[1])(para, target_qubit)
        
        if i == len(new_operation_list):
            new_qc.data[-1][0].label = "test_" + str(node_index)
            # print("reset")
        
        i += 1


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


def generate_monitoring_circuit(transpiled_qc, file, operation_list, num=1000):
    operation_list_new = []
    eq_container = {}
    num_container = [0]
    for i, gate_list in enumerate(operation_list[file]):
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            operation_list_new.append(gate_list)
            
            calculate_num_of_extra_qubits(gate_list[-1], gate_list[2].index, eq_container, num_container, i)
    
    # qc = QuantumCircuit.from_qasm_file(file)
    # transpiled_qc = transpile(qc, simulator)
    
    new_qc = QuantumCircuit()
    for reg in transpiled_qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in transpiled_qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    if new_qc.num_clbits < new_qc.num_qubits:
        new_qc.add_register(ClassicalRegister(new_qc.num_qubits - new_qc.num_clbits, 'ec'))
    
    # Add additional ancilla qubit registers
    if num_container[0] != 0:
        additional_qubits = QuantumRegister(num_container[0], 'eq')
        new_qc.add_register(additional_qubits)
    
    # Iterate through and insert all gates
    ind_container = [0]
    q_to_eq = {}
    gate_list_container = {}
    level = 0
    for i, (instr, qargs, cargs) in enumerate(transpiled_qc.data):
        new_qc.append(instr, qargs, cargs)  # Add the original gate operations
        
        for op in operation_list_new:
            if i == op[0]:
                for qubit in qargs:
                    if transpiled_qc.find_bit(qubit) == op[2]:
                        # Insert measurement and reset gate operations
                        new_qc.measure(new_qc.qubits[op[2].index], new_qc.clbits[op[2].index])
                        new_qc.reset(new_qc.qubits[op[2].index])
                        eq = new_qc.qregs[-1]
                        generate_gate(op[-1], new_qc, op[2].index, q_to_eq, ind_container, i, gate_list_container)
    return new_qc