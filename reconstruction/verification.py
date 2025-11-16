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

from typing import Dict, Iterable, Optional, Tuple, Union
from scipy.stats import chi2 as chi2_dist

import time
from tqdm.auto import tqdm

from utilities import *
from circuits_selection import *

def tvd(obs: dict, ideal: dict, shots: int = 8192, eps: float = 1e-12, return_stats: bool = False):
    """
    Compute the TVD of two distributions (dict), comparing only on the common keys.
    - If the sum over the common keys <= 1, treat them as probabilities; otherwise divide by shots.
    - Do not renormalize p and q a second time (same behavior as the original wodf code).
    """
    if shots <= 0:
        raise ValueError("shots 必须 > 0")

    # common keys
    keys = sorted(set(obs) & set(ideal))
    if not keys:
        return (0.0, {"keys": []}) if return_stats else 0.0

    # Take the corresponding values
    pv = np.array([obs[k]   for k in keys], dtype=float)
    qv = np.array([ideal[k] for k in keys], dtype=float)

    # For obs: if the sum <= 1, treat as probabilities; otherwise normalize by shots
    p = pv if pv.sum() <= 1.0 + 1e-12 else pv / shots
    # For ideal: apply the same rule
    q = qv if qv.sum() <= 1.0 + 1e-12 else qv / shots

    # TVD on the common-key subset (no re-normalization)
    dist = 0.5 * float(np.abs(p - q).sum())

    if return_stats:
        return dist, {"keys": keys, "p_sum": float(p.sum()), "q_sum": float(q.sum())}
    return dist


def wodf(obs: dict, ideal: dict, shots: int = 8192, p_thresh: float = 0.01, eps: float = 1e-12, return_stats: bool = False):
    if shots <= 0:
        raise ValueError("shots 必须 > 0")

    # common keys
    keys = sorted(set(obs) & set(ideal))
    if not keys:
        return ("P", {"pval": 1.0, "chi2": 0.0, "df": 0, "keys": []}) if return_stats else "P"

    # Observed probabilities p and ideal probabilities q (no renormalization, directly use the filtered values as requested)
    # p = np.array([obs[k] / shots for k in keys], dtype=float)
    # q = np.array([ideal[k]        for k in keys], dtype=float)

    # Take the corresponding values
    pv = np.array([obs[k]   for k in keys], dtype=float)
    qv = np.array([ideal[k] for k in keys], dtype=float)

    sum_p = float(pv.sum())
    # print(f"sum_p = {sum_p}")
    p = pv / shots if sum_p > 1.0 + 1e-12 else pv
    
    sum_q = float(qv.sum())
    # print(f"sum_q = {sum_q}")
    q = qv / shots if sum_q > 1.0 + 1e-12 else qv


    
    # Prevent division by zero
    # q = np.where(q <= 0.0, eps, q)

    # Pearson χ² (written in terms of probabilities but multiplied by shots, equivalent to the count-based form)
    chi2_stat = float(shots * np.sum((p - q) ** 2 / q))
    df = max(len(keys) - 1, 1)
    pval = float(chi2_dist.sf(chi2_stat, df))
    result = "P" if pval >= p_thresh else "F"

    if return_stats:
        return result, {"pval": pval, "chi2": chi2_stat, "df": df, "keys": keys}
    return result



def first_part_remove_unmeasured_qubit_and_merge(counts):
    new_counts = defaultdict(int)  # Using `defaultdict` makes it easier to accumulate duplicate results
    for key, value in counts.items():
        new_key = key.split(' ')[-1]  # Remove the qubits that are not being measured.
        new_counts[new_key] += value  # Merge identical keys' results
    return dict(new_counts)

second_part_list = []


def second_part_remove_unmeasured_qubit_and_merge(counts):
    new_counts = defaultdict(int)  # Using `defaultdict` makes it easier to accumulate duplicate results
    for key, value in counts.items():
        new_key = key.split(' ')[0]  # Remove the qubits that are not being measured.
        new_counts[new_key] += value  # Merge identical keys' results
    return dict(new_counts)


def execute_circuits(shots, operation_list, comparison_circuit):
    execution_list = []
    for key in operation_list:
        # if key not in long_et_list:
        #     continue
        print(key)
        try:
            qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")
            new_qc = copy.deepcopy(comparison_circuit[key])
            
            t1 = time.perf_counter()
            
            result_qc = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=shots, seed_transpiler=2025, seed_simulator=2025).result().get_counts()
            t2 = time.perf_counter()
            
            result_new_qc = execute(new_qc, backend=Aer.get_backend('qasm_simulator'), shots=shots, seed_transpiler=2025, seed_simulator=2025).result().get_counts()
            t3 = time.perf_counter()
            
            if key in second_part_list:
                result_qc = second_part_remove_unmeasured_qubit_and_merge(result_qc)
                result_new_qc = second_part_remove_unmeasured_qubit_and_merge(result_new_qc)
            else:
                result_qc = first_part_remove_unmeasured_qubit_and_merge(result_qc)
                result_new_qc = first_part_remove_unmeasured_qubit_and_merge(result_new_qc)
            
            execution_list.append([{key:[result_qc, result_new_qc]}])

            print("orig:", t2 - t1, "recon:", t3 - t2, '\n')
        except Exception as e:
            print(e)

    return execution_list


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

        del job, result, statevector, quantum_state
    return res


def get_circuits_monitoring_node_probability(comparison_circuit, simulator):
    result = {}
    pm = PassManager(passes.RemoveFinalMeasurements())
    for key, value in comparison_circuit.items():
        # if key != 'qwalk-v-chain_indep_qiskit_5.qasm':
        #     continue
        print(key)
        new_qc_no_meas = pm.run(value)
        result[key] = find_probablity_of_monitoring_nodes(new_qc_no_meas, simulator)

    return result



def get_possible_outputs(key, unblanced_circuit_list, comparison_circuit, simulator):
    qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")
    qc.remove_final_measurements()

    reconstructed_qc = copy.deepcopy(comparison_circuit[key])
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

    probs_original_dict = {}
    
    for idx, amplitude in enumerate(statevector):
        # Check if the magnitude of the probability amplitude is close to 0
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string (padded to the length of the number of qubits)
            binary_state = format(idx, f'0{num_qubits}b') if key not in unblanced_circuit_list else format(idx, f'0{num_qubits}b')[1:]
            possible_states_original_qc.append(binary_state)

            probs_original_dict[binary_state] = probs_original_dict.get(binary_state, 0.0) + float(np.abs(amplitude)**2)
    
    len_binary_state = len(possible_states_original_qc[0])
    # print(len_binary_state)
    

    for idx, amplitude in enumerate(statevector_reconstructed_qc):
        # Check if the magnitude of the probability amplitude is close to zero
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string, padded to match the number of qubits
            binary_state = format(idx, f'0{num_qubits_reconstructed_qc}b')[-len_binary_state:]
            possible_states_reconstructed_qc.append(binary_state)
    
    return list(set(possible_states_original_qc)), list(set(possible_states_reconstructed_qc)), probs_original_dict


def get_outputs(unblanced_circuit_list, comparison_circuit, simulator):
    circuits_outputs = {}
    for key in tqdm(comparison_circuit):
        print(key)
        possible_states_original_qc, possible_states_reconstructed_qc, probs_original_dict = get_possible_outputs(key, unblanced_circuit_list, comparison_circuit, simulator)
        circuits_outputs[key] = [possible_states_original_qc, possible_states_reconstructed_qc, probs_original_dict]

    return circuits_outputs



# if __name__ == "__main__":   
#     simulator = Aer.get_backend('statevector_simulator')

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num', type=int, default=0)
#     parser.add_argument('--shots', type=int, default=8192)
#     parser.add_argument('--gate_length', type=int, default=1000)
#     parser.add_argument('--folder_path', type=str, default='../quantum_circuits')
    
#     args = parser.parse_args()

#     with open(args.folder_path + '/operation_list.pkl', 'rb') as pickle_file:
#         operation_list = pickle.load(pickle_file)

#     unwanted_circuit_list = get_unwanted_circuits_list(args.folder_path, simulator)
#     operation_list_new, comparison_circuit, unblanced_circuit_list = get_filtered_operation_list(operation_list, args.gate_length, args.num, unwanted_circuit_list)
    
#     print(len(operation_list))

#     execute_circuits(args.shots, operation_list_new, comparison_circuit)

#     get_circuits_monitoring_node_probability(comparison_circuit, simulator)

#     get_outputs(unblanced_circuit_list, comparison_circuit, simulator)




    
    

