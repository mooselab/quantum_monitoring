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

simulator = Aer.get_backend('statevector_simulator')


def find_probablity(file, qc, simulator):
    res = []
    partial_circuit = QuantumCircuit()
    for reg in qc.qregs:
        new_reg = QuantumRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    transpile_qc = transpile(qc, simulator)
    for i, gate in enumerate(qc.data):
        # Create a new circuit and add all the previous gates to the new circuit
        partial_circuit.data = transpile_qc.data[:i+1]  # Only include operations up to step i
    
        # Simulate a partial circuit and obtain the state vector
        job = simulator.run(partial_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert the state vector into a quantum state object
        quantum_state = Statevector(statevector)
    
        # Extract the state of the qubits associated with the gate operation
        qubits_involved = [partial_circuit.find_bit(qubit) for qubit in gate[1]]  # Get the qubits associated with the gate
        for qubit in qubits_involved:
            # Use `partial_trace` to extract only the state of the specific qubit
            qubit_state = partial_trace(quantum_state, [q for q in range(qc.num_qubits) if q != qubit.index])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res.append([i, gate[0].name, qubit, prob_zero, prob_one])
    return res


def find_monitorable_nodes(simulator, folder_path):
    f = open(folder_path + '/statevector_list.pkl', 'ab+')

    # Recursively traverse all files in the folder (including subfolders)
    for file in Path(folder_path).rglob('*'):
        if file.is_file() and file.suffix == '.qasm': # Check if the file ends with `.qasm`
            print(file)
            try:
                qc = QuantumCircuit.from_qasm_file(file)
            except Exception as e:
                print(e)
                continue

            pm = PassManager(passes.RemoveFinalMeasurements())
            qc = pm.run(qc)

            res = find_probablity(file, qc, simulator)
    
            pickle.dump([file, res], f)

    f.close()


def get_circuits_with_monitorable_nodes(simulator, statevector_list):
    quantum_file = {}
    num = 0
    count = 0

    unwanted_circuit_set = set()
    with open(statevector_list, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                qc = QuantumCircuit.from_qasm_file(item[0])
                
                last_gate_index_before_measurement = [None] * qc.num_qubits
                
                for i, gate in enumerate(transpile(qc, simulator).data):
                    # if gate[0].name in ['cswap', 'ccx', 'ccz']:
                    #     unwanted_circuit_set.add(item[0])
                    if gate[0].name not in ['measure', 'barrier']:
                        for qubit in gate[1]:
                            last_gate_index_before_measurement[transpile(qc, simulator).find_bit(qubit).index] = i

                for l in item[1]:
                    if l[0] == last_gate_index_before_measurement[l[2].index]:
                        continue
                    if l[3] >= 0.99 or l[4] >= 0.99:
                        num += 1
                        if item[0] not in quantum_file:
                            quantum_file[item[0]] = 0
                        quantum_file[item[0]] += 1
                
                if item[0] in quantum_file:
                    count += 1
            except EOFError:
                break

    print('total circuits:', count)
    print('total nodes:', num)

    return quantum_file


def filter_circuits_with_output_num(quantum_file, output_num=10000000):
    filetered_quantum_file = {}
    for key, value in quantum_file.items():
        qc = QuantumCircuit.from_qasm_file(key)
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=8192).result()
        counts = result.get_counts()
        if len(counts) <= output_num:
            filetered_quantum_file[key] = counts
        filetered_quantum_file[key] = counts

    return filetered_quantum_file


def get_nodes_with_certain_probability(simulator, statevector_list, filetered_quantum_file):
    definited_node_dict = {}
    count = 0
    with open(statevector_list, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                if item[0] not in filetered_quantum_file:
                    continue
                
                qc = QuantumCircuit.from_qasm_file(item[0])
                
                last_gate_index_before_measurement = [None] * qc.num_qubits
                for i, gate in enumerate(transpile(qc, simulator).data):
                    if gate[0].name not in ['measure', 'barrier']:
                        for qubit in gate[1]:
                            last_gate_index_before_measurement[transpile(qc, simulator).find_bit(qubit).index] = i
                            
                num = 0
                for l in item[1]:
                    if l[0] == last_gate_index_before_measurement[l[2].index]:
                        continue
                    if l[3] >= 0.99 or l[4] >= 0.99:
                        gate = transpile(qc, simulator).data[l[0]]
                        print(l, gate)
                        num += 1
                        if str(item[0]) not in definited_node_dict:
                            definited_node_dict[str(item[0])] = []
                        definited_node_dict[str(item[0])].append([l[0], l[1], l[2], l[3], l[4], \
                                                                  gate[0].num_qubits, gate[0].params, l[2] == qc.find_bit(gate.qubits[0])])
                if num > 0:
                    count += 1
            except EOFError:
                break
    
    print('circuits_count:', count)
    return definited_node_dict


# Function to perform Depth-First Search (DFS) on the quantum circuit
def dfs_quantum_circuit(transpiled_qc, qubit_info, current_index, visited, path):
    # Iterate backward through the circuit instructions
    for index in range(current_index, -1, -1):
        if [index, qubit_info] in visited:
            continue
        gate, qargs, cargs = transpiled_qc.data[index]
        qubit_indices = [transpiled_qc.find_bit(qubit) for qubit in qargs]
        if qubit_info in qubit_indices:
            # Handle multi-qubit gates
            if len(qubit_indices) > 1:
                if gate.name == "swap":
                    # Special handling for swap gate
                    path.insert(0, [index, gate.name, gate.params, qubit_indices])
                    other_qubit = qubit_indices[0] if qubit_indices[1] == qubit_info else qubit_indices[1]
                    dfs_quantum_circuit(transpiled_qc, other_qubit, index - 1, visited, path)
                    # path.sort()
                    break
                elif gate.name == "iswap":
                    print(gate.name)
                elif qubit_info == qubit_indices[0]:
                    # If the qubit is the control qubit, this gate does not affect the qubit's state
                    continue
                else:
                    # Handle other multi-qubit gates (e.g., cx, cz, etc.)
                    for q in qubit_indices:
                        if q != qubit_info:
                            path.insert(0, [index, gate.name, gate.params, qubit_indices, dfs_quantum_circuit(transpiled_qc, q, index - 1, visited, [])])
            # Record the current gate if it affects the qubit
            else:
                path.insert(0, [index, gate.name, gate.params, qubit_indices])
            # path.sort()
            visited.append([index, qubit_info])
    return path


def get_final_definited_node_dict(definited_node_dict):
    final_definited_node_dict = {}
    for key, value in definited_node_dict.items():
        print(key)
        qc = QuantumCircuit.from_qasm_file(key)
        transpiled_qc = transpile(qc, simulator)
        for node in value:
            index = node[0]
            gate_name = node[1]
            qubit_info = node[2]
            p0 = node[3]
            p1 = node[4]
            num_qubits = node[5]
            parameters = node[6]

            dfs_path = []
            visited = list()
            path = []
            # Perform DFS on the circuit
            dfs_path = dfs_quantum_circuit(transpiled_qc, qubit_info, index, visited, path)

            if key not in final_definited_node_dict:
                final_definited_node_dict[key] = []
            final_definited_node_dict[key].append([index, gate_name, qubit_info, p0, p1, dfs_path])
    
    return final_definited_node_dict


def transformation(transpiled_qc, swap_qubit_info, qubit_info, gate_list, operation_list):
    if not gate_list:
        return operation_list
    for gate in gate_list:
        if gate[1] == 'swap':
            for qubit in gate[3]:
                if swap_qubit_info != qubit:
                    swap_qubit_info = qubit
                    break
    
    for gate in gate_list:
        if not gate or gate[1] == 'swap':
            continue
        
        if len(gate) == 4:
            flag = True
            for qubit in gate[3]:
               if qubit_info == qubit:
                   flag = False
            if flag:
                operation_list.append([gate[0], gate[1], gate[2], [qubit_info if swap_qubit_info == qubit else qubit for qubit in gate[3]]])
            else:
                operation_list.append([gate[0], gate[1], gate[2], [qubit for qubit in gate[3]]])
            
        else:
            control_qubit_info = None
            control_gate = []
            flag = True
            for qubit in gate[3]:
                if qubit_info == qubit:
                    flag = False
            if flag:
                target_gate = [gate[0], gate[1], gate[2], [qubit_info if swap_qubit_info == qubit else qubit for qubit in gate[3]]]
            else:
                target_gate = [gate[0], gate[1], gate[2], [qubit for qubit in gate[3]]]

            control_qubit_info = gate[3][0]
           
            control_control_gate = transformation(transpiled_qc, control_qubit_info, control_qubit_info, gate[4], [])
            control_gate.extend(control_control_gate)
            
            target_gate.extend([control_gate])
            operation_list.append(target_gate)
    return operation_list


def bulid_partial_circuit(final_definited_node_dict):
    res = {}
    for key, value in final_definited_node_dict.items():
        print(key, '\n')
        qc = QuantumCircuit.from_qasm_file(key)
        transpiled_qc = transpile(qc, simulator)
        for node in value:
            if not node[-1]:
                continue

            index = node[0]
            gate_name = node[1]
            qubit_info = node[2]
            p0 = node[3]
            p1 = node[4]

            operation_list = []
            
            swap_qubit_info = qubit_info = node[2]
            operation_list = transformation(transpiled_qc, swap_qubit_info, qubit_info, node[-1], operation_list)

            if key not in res:
                res[key] = []
            res[key].append([index, gate_name, qubit_info, p0, p1, operation_list])
    
    return res


