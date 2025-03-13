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

from utilities import *


def find_probablity_of_monitoring_nodes(qc, simulator):
    res = []
    partial_circuit = QuantumCircuit()
    for reg in qc.qregs:
        new_reg = QuantumRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    for reg in qc.cregs:
        new_reg = ClassicalRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    for i, gate in enumerate(qc.data):
        # Create a new circuit and add all the previous gates to the new circuit
        # partial_circuit = QuantumCircuit(qc.num_qubits)
        partial_circuit.data = qc.data[:i+1]  # Only include operations up to step i
        
    
        # Simulate a partial circuit and obtain the state vector
        job = simulator.run(partial_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert the state vector into a quantum state object
        quantum_state = Statevector(statevector)
        # quantum_state = Statevector.from_instruction(partial_circuit)
    
        # Extract the state of the qubits associated with the gate operation
        qubits_involved = [partial_circuit.find_bit(qubit) for qubit in gate[1]]  # Get the qubits associated with the gate
        for qubit in qubits_involved:
            # Use `partial_trace` to extract only the state of the specific qubit
            qubit_state = partial_trace(quantum_state, [q for q in range(qc.num_qubits) if q != qubit.index])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res.append([i, gate[0].name, qubit, prob_zero, prob_one, gate[0].params, gate[0].label])
    return res


def get_circuits_monitoring_node_probability(comparison_circuit, simulator):
    result = {}
    pm = PassManager(passes.RemoveFinalMeasurements())
    for key, value in comparison_circuit.items():
        print(key)
        new_qc_no_meas = pm.run(value)
        result[key] =  find_probablity_of_monitoring_nodes(new_qc_no_meas, simulator)
    with open('../quantum_circuits/circuits_monitoring_node_probability.pkl', 'wb') as file:
        pickle.dump(result, file)


def get_filtered_operation_list(operation_list, gate_length, num, unwanted_circuit_list, simulator):
    operation_list_new = {}
    for key, value in operation_list.items():
        if filter_operation_list(key, operation_list, gate_length) <= num and key not in [str(i) for i in unwanted_circuit_list]:
            operation_list_new[key] = value
    
    comparison_circuit = {}
    unblanced_circuit_list = []

    res = copy.deepcopy(operation_list_new)
        
    for k, v in operation_list_new.items():
        qc = QuantumCircuit.from_qasm_file(k)
        transpiled_qc = transpile(qc, simulator)
        for l in v:
            if l[1] != transpiled_qc.data[l[0]][0].name:
                print('unmatch:', k)
                del res[k] # delete unmatched circuits
                break
        
        if k in res:
            try:
                comparison_circuit[k] = generate_monitoring_circuit(transpiled_qc, k, operation_list, gate_length)
                result_qc = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=8192).result()
                if len(list(result_qc.get_counts().keys())[0]) < qc.num_qubits:
                    print('unblanced', k)
                    unblanced_circuit_list.append(k)
            except Exception as e:
                print(e)
    
    with open('../quantum_circuits/operation_list_new.pkl', 'wb') as file:
        pickle.dump(res, file)
    
    return res, comparison_circuit, unblanced_circuit_list


def execute_circuits(shots, operation_list, comparison_circuit):
    execution_list = []
    for key in operation_list:
        try:
            qc = QuantumCircuit.from_qasm_file(key)
            new_qc = comparison_circuit[key]
            result_qc = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
            result_qc = remove_unmeasured_qubit_and_merge(result_qc)

            result_new_qc = execute(new_qc, backend=Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
            result_new_qc = remove_unmeasured_qubit_and_merge(result_new_qc)
            
            execution_list.append([{key:[result_qc, result_new_qc, calculate_tvd(result_qc, result_new_qc, shots)]}])
        except Exception as e:
            print(e)
    with open('../quantum_circuits/execution_result.pkl', 'wb') as file:
        pickle.dump(execution_list, file)



def get_possible_outputs(file, unblanced_circuit_list, comparison_circuit, simulator):
    qc = QuantumCircuit.from_qasm_file(file)
    qc.remove_final_measurements()

    reconstructed_qc = comparison_circuit[file]
    reconstructed_qc.remove_final_measurements()
    
    result = execute(qc, backend=simulator).result()
    result_reconstructed_qc = execute(reconstructed_qc, backend=simulator).result()
    
    # Get the quantum state vector
    statevector = result.get_statevector()
    statevector_reconstructed_qc = result_reconstructed_qc.get_statevector()
    # return statevector_reconstructed_qc
    
    # Identify the nonzero states (the parts with nonzero probability amplitudes)
    num_qubits = qc.num_qubits
    num_qubits_reconstructed_qc = reconstructed_qc.num_qubits
    
    possible_states_original_qc = []
    possible_states_reconstructed_qc = []
    
    for idx, amplitude in enumerate(statevector):
        # Check if the magnitude of the probability amplitude is close to 0
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string (padded to the length of the number of qubits)
            binary_state = format(idx, f'0{num_qubits}b') if file not in unblanced_circuit_list else format(idx, f'0{num_qubits}b')[1:]
            possible_states_original_qc.append(binary_state)
    
    len_binary_state = len(possible_states_original_qc[0])
    # print(len_binary_state)
    

    for idx, amplitude in enumerate(statevector_reconstructed_qc):
        # Check if the magnitude of the probability amplitude is close to zero
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string, padded to match the number of qubits
            binary_state = format(idx, f'0{num_qubits_reconstructed_qc}b')[-len_binary_state:]
            possible_states_reconstructed_qc.append(binary_state)
    
    return list(set(possible_states_original_qc)), list(set(possible_states_reconstructed_qc))


def get_outputs(unblanced_circuit_list, comparison_circuit, simulator):
    circuits_outputs = {}
    for key in comparison_circuit:
        possible_states_original_qc, possible_states_reconstructed_qc = get_possible_outputs(key, unblanced_circuit_list, comparison_circuit, simulator)
        circuits_outputs[key] = [possible_states_original_qc, possible_states_reconstructed_qc]
    with open('../quantum_circuits/circuits_outputs.pkl', 'wb') as file:
        pickle.dump(circuits_outputs, file)



if __name__ == "__main__":   
    simulator = Aer.get_backend('statevector_simulator')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--gate_length', type=int, default=1000)
    parser.add_argument('--folder_path', type=str, default='../quantum_circuits')
    
    args = parser.parse_args()

    with open(args.folder_path + '/operation_list.pkl', 'rb') as pickle_file:
        operation_list = pickle.load(pickle_file)

    unwanted_circuit_list = get_unwanted_circuits_list(args.folder_path, simulator)
    operation_list_new, comparison_circuit, unblanced_circuit_list = get_filtered_operation_list(operation_list, args.gate_length, args.num, unwanted_circuit_list)
    
    print(len(operation_list))

    execute_circuits(args.shots, operation_list_new, comparison_circuit)

    get_circuits_monitoring_node_probability(comparison_circuit, simulator)

    get_outputs(unblanced_circuit_list, comparison_circuit, simulator)




    
    

