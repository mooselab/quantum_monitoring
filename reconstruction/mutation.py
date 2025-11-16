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
import random

import copy

import pickle
from qiskit.quantum_info import partial_trace, Statevector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import argparse
from collections import defaultdict

from utilities import *
from circuits_selection import *
from verification import *


def generate_full_path(gate_list, full_gate_list):
    if not gate_list:
        return full_gate_list
    
    for gate in gate_list:
        if not gate:
            continue
        
        if len(gate) == 4:
            full_gate_list.append(gate)
            
        else:
            target_gate = gate
            for g in gate[4:]:
                another_gate = []
               
                another_another_gate = generate_full_path(g, [])
                another_gate.extend(another_another_gate)
                
                target_gate.extend([another_gate])
            full_gate_list.append(target_gate)

    # print('full_gate_list', full_gate_list)
    return full_gate_list


def generate_monitoring_mutated_circuit(qc, file, res, operation_list, num=1000, ans=None):
    operation_list_dict = {}
    eq_container = {}
    num_container = [0]
    
    for i, gate_list in enumerate(res[file]):
        if ans and gate_list[0] not in ans[res]["picked_nodes"]:
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
    
    new_qc = QuantumCircuit()
    for reg in qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    ec = ClassicalRegister(len(operation_list), 'ec')
    new_qc.add_register(ec)
    
    if num_container[0] != 0:
        additional_qubits = QuantumRegister(num_container[0], 'eq')
        new_qc.add_register(additional_qubits)
    
    ind_container = [0]
    ec_container = [0]
    q_to_eq = {}
    gate_list_container = {}
    for i, (instr, qargs, cargs) in enumerate(qc.data):
        new_qc.append(instr, qargs, cargs)
        
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



def mutate_circuit(qc, mutation_idx, quantum_gates) -> QuantumCircuit:
    """
    Mutate a quantum circuit by randomly selecting a gate from the index_list
    and replacing it with a compatible gate (single->single, two->two qubit).
    
    Args:
        circuit: Original quantum circuit
        index_list: List of indices indicating which gates can be mutated
        quantum_gates: Dictionary containing gate information
        
    Returns:
        QuantumCircuit: Mutated circuit
    """
    
    def get_random_params(param_info):
        """Generate random parameters for a gate based on its ranges"""
        return [random.uniform(range_[0], range_[1]) for range_ in param_info['ranges']]
    
    def get_random_gate(is_single_qubit: bool):
        """Get a random gate of specified type (single/two qubit)"""
        gate_type = 'single_qubit' if is_single_qubit else 'two_qubit'
        # Decide if we want parameterized or non-parameterized gate
        is_parameterized = random.choice([True, False])
        
        if is_parameterized:
            gate_name = random.choice(list(quantum_gates[gate_type]['parameterized'].keys()))
            params = get_random_params(quantum_gates[gate_type]['parameterized'][gate_name])
            return gate_name, params
        else:
            gate_name = random.choice(list(quantum_gates[gate_type]['non_parameterized'].keys()))
            return gate_name, None

    # Create a copy of the circuit
    new_qc = QuantumCircuit()
    for reg in qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    
    # Get all gates and their locations from the original circuit
    gates_list = []
    current_idx = 0
    
    for current_idx, (instr, qargs, cargs) in enumerate(qc.data):
        qubits = [qc.find_bit(qubit) for qubit in qargs]
        old_gate_params = instr.params if hasattr(instr, 'params') else []
        old_gate_name = instr.name
        
        if current_idx == mutation_idx:
            # Determine if it's a single or two qubit gate
            is_single = len(qubits) == 1
            
            # Get random replacement gate
            new_gate_name = copy.deepcopy(old_gate_name)
            new_gate_params = copy.deepcopy(old_gate_params)
            

            while new_gate_name == old_gate_name and new_gate_params == old_gate_params:
                new_gate_name, new_gate_params = get_random_gate(is_single)
                
            # print(new_gate_name, new_gate_params)
            
            # Add the new gate to the circuit
                        # For gates with parameters
            if new_gate_params:
                getattr(new_qc, new_gate_name)(*new_gate_params, *[new_qc.qubits[q.index] for q in qubits])
            else:
                getattr(new_qc, new_gate_name)(*[new_qc.qubits[q.index] for q in qubits])
        else:
            # Copy the original gate
            # if params:
            #     getattr(new_qc, instr.name)(*params, *[new_qc.qubits(q.index) for q in qubits])
            # else:
            #     getattr(new_qc, instr.name)(*[new_qc.qubits(q.index) for q in qubits])
            new_qc.append(instr, qargs, cargs)
    
    return new_qc


def compare_outputs_original_vs_mutation(original_qc, mutation_qc, simulator):
    original_qc.remove_final_measurements()
    mutation_qc.remove_final_measurements()
    
    result_original_qc = execute(original_qc, backend=simulator).result()
    result_mutation_qc = execute(mutation_qc, backend=simulator).result()
    
    # Get the quantum state vector
    statevector_original_qc = result_original_qc.get_statevector()
    statevector_mutation_qc = result_mutation_qc.get_statevector()
    # return statevector_reconstructed_qc
    
    # Identify the nonzero states (the parts with nonzero probability amplitudes)
    num_qubits_original_qc = original_qc.num_qubits
    num_qubits_mutation_qc = mutation_qc.num_qubits
    
    possible_states_original_qc = []
    possible_states_mutation_qc = []

    probs_original_dict = {}
    
    for idx, amplitude in enumerate(statevector_original_qc):
        # Check if the magnitude of the probability amplitude is close to 0
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string (padded to the length of the number of qubits)
            binary_state = format(idx, f'0{num_qubits_original_qc}b')
            possible_states_original_qc.append(binary_state)

            probs_original_dict[binary_state] = probs_original_dict.get(binary_state, 0.0) + float(np.abs(amplitude)**2)
    
    len_binary_state = len(possible_states_original_qc[0])
    # print(len_binary_state)
    

    probs_mutation_dict = {}
    for idx, amplitude in enumerate(statevector_mutation_qc):
        # Check if the magnitude of the probability amplitude is close to zero
        if not np.isclose(np.abs(amplitude), 0, atol=1e-5):
            # Convert the index to a binary string, padded to match the number of qubits
            binary_state = format(idx, f'0{num_qubits_mutation_qc}b')[-len_binary_state:]
            possible_states_mutation_qc.append(binary_state)

            probs_mutation_dict[binary_state] = probs_mutation_dict.get(binary_state, 0.0) + float(np.abs(amplitude)**2)
    
    # print(possible_states_mutation_qc)
    if not set(possible_states_mutation_qc) == (set(possible_states_original_qc)):
        return 'F'
    if tvd(probs_mutation_dict, probs_original_dict) > 0.15 or wodf(probs_mutation_dict, probs_original_dict) == 'F':
        return 'F'
    return 'P'

    
def execute_mutated_circuits(operation_list, ans_dict, simulator, seed):
    circuit_count = 0
    random.seed(seed)
    pm = PassManager(passes.RemoveFinalMeasurements())
    mutation_execution_result = {}
    for key, value in operation_list.items():
        # if key != 'qaoa_indep_qiskit_6.qasm':
        #     continue
        # print(key)
        index_list = []
        # monitor_qubit_gate_list = []
        assertion_value_dict = {}
        gate_list_container = {}
        gate_list_container_test = {}
        
        for gate_list in value:
            # print(gate_list)
            if gate_list[0] in ans_dict[key]['picked_nodes']:
                if gate_list[0] not in index_list:
                    index_list.append(gate_list[0])
                if gate_list[0] not in assertion_value_dict:
                    assertion_value_dict[gate_list[0]] = {}
                    
                # monitor_qubit_gate_list.append(gate_list[0])
                assertion_value_dict[gate_list[0]][gate_list[2]] = {0: gate_list[3], 1: gate_list[4]}
                
                gate_list_container = generate_new_operation_list(gate_list[-1], gate_list[0], gate_list_container)
                # print(gate_list_container)
                
                for node in operation_list[key]:
                    if node[0] == gate_list[0]:
                        # print('node[0] = ', node[0])
                        temp_gate_list = copy.deepcopy(node[-1])
                        full_path_gate_list = generate_full_path(temp_gate_list, [])
                        gate_list_container_test = generate_new_operation_list(full_path_gate_list, gate_list[0], gate_list_container)
                        break
                            
        index_list.sort()
        # print('index_list = ', index_list)
        qc = QuantumCircuit.from_qasm_file(f'{quantum_path}/{key}')
        if index_list[-1] <= qc.num_qubits - 1:
            # print('**********', index_list, '**********')
            continue
        # print(index_list)
        # continue
        mutation_execution_result[key] = [index_list, gate_list_container_test, gate_list_container, assertion_value_dict, {}]
        # print(mutation_execution_result[key])
        # continue

        num_gate = len(pm.run(qc).data)
        # print(index_list, num_gate)
        circuit_count += 1
        # print('******', circuit_count, '******')
        # continue
        # print(num_gate)
        # continue

        random.seed(seed)
        mutation_idx_list = random.sample(list(range(index_list[-1] + 1)), 3) # if num_gate >= 1000 else random.sample(list(range(num_gate)), num_gate //2)
        # print(mutation_idx_list)
        # continue
        
        for mutation_idx in mutation_idx_list:
            # print(mutation_idx)
            mutation_qc = mutate_circuit(qc, mutation_idx, quantum_gates)
            reconstructed_mutation_qc = generate_monitoring_circuit(mutation_qc, key, operation_list, num=10000000, ans=ans_dict[key])
            # result_reconstructed_mutation_qc = execute(reconstructed_mutation_qc, backend=Aer.get_backend('qasm_simulator'), shots=1000).result().get_counts()
            # reconstructed_mutation_qc = transpile(reconstructed_mutation_qc, backend=simulator)
            new_qc_no_meas = pm.run(mutation_qc)
            # print(new_qc_no_meas.data[mutation_idx])
            circuits_monitoring_node_probability = find_probablity_of_monitoring_nodes(new_qc_no_meas, simulator)

            # new_counts = defaultdict(int)
            # for k, v in result_reconstructed_mutation_qc.items():
            #     new_key = k.split(' ')[0][::-1]
            #     new_counts[new_key] += v
            # # print(new_counts)
            # mutation_execution_result[key][-1][mutation_idx] = [dict(new_counts)]
            
            # total = sum(new_counts.values())
            
            # bit_count = defaultdict(int)
            
            # # Calculate the frequency of 0s and 1s for each bit
            # for bitstring, stringcount in new_counts.items():
            #     for i, bit in enumerate(bitstring):
            #         bit_count[(i, bit)] += stringcount
            
            # # Calculate and output the percentage of 0s and 1s for each bit
            # bit_percentages = {}
            # for (position, bit), count in bit_count.items():
            #     bit_percentages[(position, bit)] = (count / total)

            # mutation_execution_result[key][-1][mutation_idx].append(dict(bit_percentages))
            mutation_execution_result[key][-1][mutation_idx] = [circuits_monitoring_node_probability, compare_outputs_original_vs_mutation(qc, mutation_qc, simulator)]
            del mutation_qc, reconstructed_mutation_qc, new_qc_no_meas, circuits_monitoring_node_probability
        
        # print('\n')
    
    return mutation_execution_result


def get_bug_detection_result(mutation_execution_result):
    bug_detection_count = 0
    detection_result = {}
    for key, value in mutation_execution_result.items():
        # print(key)
        # print(res_new[key])
        qc = QuantumCircuit.from_qasm_file(f'{quantum_path}/{key}')
        for k, v in mutation_execution_result[key][-1].items():
            # print({key: sorted([item[0] for item in value]) for key, value in mutation_execution_result[key][1].items()})
            # print(v[1])
            # print((i, str(list(dic.keys())[0])), v[1][(i, str(list(dic.keys())[0]))])
            
            for i in v[0]:
                # print(index_value_set)
                # print(i[0], mutation_execution_result[key][0])
                if i[0] in mutation_execution_result[key][0]:
                    # print('i = ', i, '\n')
                    index = int(i[0])
                    for qubit_info in mutation_execution_result[key][3][index]:
                        # print(qubit_info, i[2].index)
                        if i[2].index == qubit_info:
                            if abs(i[3] - mutation_execution_result[key][3][index][qubit_info][0]) >= 1e-2 \
                            or abs(i[4] - mutation_execution_result[key][3][index][qubit_info][1]) >= 1e-2:
                                bug_detection_count += 1
                            if key not in detection_result:
                                detection_result[key] = []
                            # print(k, [transpiled_qc.find_bit(q).index for q in transpiled_qc.data[k].qubits], {key: [item[0] for item in value] for key, value in mutation_execution_result[key][2].items()}, mutation_execution_result[key][1], mutation_execution_result[key][0][i], (i, str(list(dic.keys())[0])), list(dic.values())[0], v[1][(i, str(list(dic.keys())[0]))])
                            detection_result[key].append([k, v[1], {index: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[index]}, i[2].index, (i[3], mutation_execution_result[key][3][index][qubit_info][0]), (i[4], mutation_execution_result[key][3][index][qubit_info][1])])
    #                         print(k, v[1], {index: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[index]}, i[2].index, (i[3], mutation_execution_result[key][3][index][qubit_info][0]), (i[4], mutation_execution_result[key][3][index][qubit_info][1]))
    #                         print('\n')
    #     print('\n')
    #         # break
    #         # print(mutation_execution_result['../limited_quantum_circuits/su2random_indep_qiskit_8.qasm'][2][48][1][(i, str(list(dic.keys())[0]))])

    # print(bug_detection_count, len(mutation_execution_result))
    
    return detection_result



def find_bestseed(max_seed, operation_list_new, ans_dict, simulator):
    best_seed = -1
    best_detect_count = -1
    for seed in range(0, max_seed+1):
        
        mutation_execution_result = execute_mutated_circuits(operation_list_new, ans_dict, simulator, seed)
        detection_result = get_bug_detection_result(mutation_execution_result)
        
        right_detect_count = 0
        
        for key, value in detection_result.items():
            mutation_id_list = []
            for v in value:
                if v[0] in mutation_id_list:
                    continue
                if abs(v[4][0] - v[4][1]) >= 1e-2 or abs(v[5][0] - v[5][1]) >= 1e-2:
                    if v[0] in v[2] or v[0] in list(v[2].values())[0]:
                        right_detect_count += 1
                        mutation_id_list.append(v[0])
        if right_detect_count > best_detect_count:
            best_detect_count = right_detect_count
            best_seed = seed
    return best_seed


def filter_circuits_random(res_dict, key, index_list):
    definited_node_dict = {}
    count = 0
        # print(res_dict[key])
    try:
        qc = QuantumCircuit.from_qasm_file(f"{quantum_path}/{key}")
        
        last_gate_index_before_measurement = [None] * qc.num_qubits
        for i, gate in enumerate(qc.data):
            if gate[0].name not in ['measure', 'barrier']:
                for qubit in gate[1]:
                    last_gate_index_before_measurement[qc.find_bit(qubit).index] = i
                    
        num = 0
        for i, gate in enumerate(qc.data):
            if gate[0].name not in ['measure', 'barrier']:
                for qubit in gate[1]:
                    if i not in index_list or i == last_gate_index_before_measurement[qc.find_bit(qubit).index]:
                        continue
                    
                    if key not in definited_node_dict:
                        definited_node_dict[key] = []
                    
                    for node_with_probability in res_dict[key][i]:
                        if node_with_probability not in definited_node_dict[key]:
                            definited_node_dict[key].append(node_with_probability)

                    num += 1
                    
        if num > 0:
            count += 1
    except EOFError as e:
        print(e)
    
    print('circuits_count:', count)
    return definited_node_dict


def generate_monitoring_circuit_random(qc, file, operation_list, num=1000):
    operation_list_dict = {}
    eq_container = {}
    num_container = [0]
    
    # print(operation_list)
    for gate_list in operation_list[file]:
        # print(gate_list)
        if gate_list[0] not in operation_list_dict:
                operation_list_dict[gate_list[0]] = [gate_list[0], gate_list[1], {}, gate_list[-1]]
        operation_list_dict[gate_list[0]][2][gate_list[2]] = [gate_list[3], gate_list[4]]

    num_extra_classic = 0
    for key, gate_list in operation_list_dict.items():
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            qubit_index_list = []
            for qubit_index in gate_list[2]:
                qubit_index_list.append(qubit_index)
                num_extra_classic += 1
            calculate_num_of_extra_qubits(gate_list[-1], qubit_index_list, eq_container, num_container, gate_list[0])
    
    print(file, num_container[0])
    
    new_qc = QuantumCircuit()
    for reg in qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    # if new_qc.num_clbits < new_qc.num_qubits:
    #     new_qc.add_register(ClassicalRegister(new_qc.num_qubits - new_qc.num_clbits, 'ec'))
    extra_classic = ClassicalRegister(num_extra_classic, 'ec')
    new_qc.add_register(extra_classic)
    
    # Add additional ancilla qubit registers
    if num_container[0] != 0:
        additional_qubits = QuantumRegister(num_container[0], 'eq')
        new_qc.add_register(additional_qubits)
    # return
    # Iterate through and insert all gates
    ind_container = [0]
    q_to_eq = {}
    gate_list_container = {}
    ec_count = 0
    for i, (instr, qargs, cargs) in enumerate(qc.data):
        new_qc.append(instr, qargs, cargs)  # Add the original gate operations
        
        for _, op in operation_list_dict.items():
            if i == op[0]:
                qubit_index_list = []
                for qubit in op[2]:
                    qubit_index_list.append(qubit)
                for qubit in qargs:
                    if qc.find_bit(qubit).index in qubit_index_list:
                        new_qc.measure(new_qc.qubits[qc.find_bit(qubit).index], extra_classic[ec_count])
                        new_qc.reset(new_qc.qubits[qc.find_bit(qubit).index])
                        ec_count += 1
    return new_qc

def map_bit_percentages(bit_percentages, assertion_value_dict):
    # 1) Build a mapping from “bit-order index” to (gate_idx, qubit)
    order_pairs = []
    for gate_idx in assertion_value_dict.keys():
        for qubit in assertion_value_dict[gate_idx].keys():
            order_pairs.append((gate_idx, qubit))

    # 2) Fill the probabilities back into the target structure
    out = {}
    for order_idx, (gate_idx, qubit) in enumerate(order_pairs):
        p0 = bit_percentages.get((order_idx, '0'))
        p1 = bit_percentages.get((order_idx, '1'))

        # If only one side is given, fill in the other side with 1 - p
        if p0 is None and p1 is not None:
            p0 = 1.0 - p1
        if p1 is None and p0 is not None:
            p1 = 1.0 - p0
        if p0 is None and p1 is None:
            continue  # Skip this bit if there is no data

        out.setdefault(gate_idx, {})[qubit] = {0: p0, 1: p1}

    return out


def execute_mutated_circuits_random(operation_list, ans_dict, res_dict, simulator, seed):
    circuit_count = 0
    random.seed(seed)
    pm = PassManager(passes.RemoveFinalMeasurements())
    mutation_execution_result = {}
    for key, value in operation_list.items():
        # if key != 'qaoa_indep_qiskit_6.qasm':
        #     continue
        print(key)
        index_list = []
        # monitor_qubit_gate_list = []
        assertion_value_dict = {}
        gate_list_container = {}
        gate_list_container_test = {}
        
        for gate_list in value:
            # print(gate_list)
            if gate_list[0] in ans_dict[key]['picked_nodes']:
                if gate_list[0] not in index_list:
                    index_list.append(gate_list[0])
        
        index_list.sort()
        try:
            if key in ['grover-v-chain_indep_qiskit_7.qasm', 'grover-v-chain_indep_qiskit_8.qasm', 'grover-v-chain_indep_qiskit_9.qasm', 'qnn_indep_qiskit_10.qasm']:
                random_index_list = list(range(0, len(index_list)))
            else:
                random_index_list = random.sample(list(range(index_list[-1] + 1)), len(index_list))
        except Exception as e:
            print(e)
            continue
        random_index_list.sort()
        definited_node_dict_random = filter_circuits_random(res_dict, key, random_index_list)
        final_definited_node_dict_random = get_final_definited_node_dict(definited_node_dict_random)
        operation_list_random = bulid_partial_circuit(final_definited_node_dict_random)
        
        
        for gate_list in operation_list_random[key]:
            if gate_list[0] not in assertion_value_dict:
                assertion_value_dict[gate_list[0]] = {}
                
            # monitor_qubit_gate_list.append(gate_list[0])
            assertion_value_dict[gate_list[0]][gate_list[2]] = {0: gate_list[3], 1: gate_list[4]}
            
            gate_list_container = generate_new_operation_list(gate_list[-1], gate_list[0], gate_list_container)
            # print(gate_list_container)
            
            temp_gate_list = copy.deepcopy(gate_list[-1])
            full_path_gate_list = generate_full_path(temp_gate_list, [])
            gate_list_container_test = generate_new_operation_list(full_path_gate_list, gate_list[0], gate_list_container)
                            
        
        # print(assertion_value_dict)
        qc = QuantumCircuit.from_qasm_file(f'{quantum_path}/{key}')
        if index_list[-1] <= qc.num_qubits - 1:
            # print('**********', index_list, '**********')
            continue
        # print(index_list)
        # continue
        mutation_execution_result[key] = [random_index_list, gate_list_container_test, gate_list_container, assertion_value_dict, {}]
        # print(mutation_execution_result[key])
        # continue

        num_gate = len(pm.run(qc).data)
        print(index_list, num_gate)
        circuit_count += 1
        print('******', circuit_count, '******')
        # continue
        # print(num_gate)
        # continue

        random.seed(seed)
        mutation_idx_list = random.sample(list(range(index_list[-1] + 1)), 3) # if num_gate >= 1000 else random.sample(list(range(num_gate)), num_gate //2)
        # print(mutation_idx_list)
        # continue
        
        for mutation_idx in mutation_idx_list:
            print(mutation_idx)
            mutation_qc = mutate_circuit(qc, mutation_idx, quantum_gates)
            reconstructed_mutation_qc = generate_monitoring_circuit_random(mutation_qc, key, operation_list_random, num=10000000)
            result_reconstructed_mutation_qc = execute(reconstructed_mutation_qc, backend=Aer.get_backend('qasm_simulator'), shots=1000).result().get_counts()
            # print(result_reconstructed_mutation_qc)
            
            new_counts = defaultdict(int)
            for k, v in result_reconstructed_mutation_qc.items():
                new_key = k.split(' ')[0][::-1]
                new_counts[new_key] += v
            # # print(new_counts)
            # mutation_execution_result[key][-1][mutation_idx] = [dict(new_counts)]
            
            total = sum(new_counts.values())
            
            bit_count = defaultdict(int)
            
            # Calculate the frequency of 0s and 1s for each bit
            for bitstring, stringcount in new_counts.items():
                for i, bit in enumerate(bitstring):
                    bit_count[(i, bit)] += stringcount
            
            # Calculate and output the percentage of 0s and 1s for each bit
            bit_percentages = {}
            for (position, bit), count in bit_count.items():
                bit_percentages[(position, bit)] = (count / total)
            # print(bit_percentages)
            # print(map_bit_percentages(bit_percentages, assertion_value_dict))

            # mutation_execution_result[key][-1][mutation_idx].append(dict(bit_percentages))
            mutation_execution_result[key][-1][mutation_idx] = [map_bit_percentages(bit_percentages, assertion_value_dict), compare_outputs_original_vs_mutation(qc, mutation_qc, simulator)]
            # del mutation_qc, reconstructed_mutation_qc, new_qc_no_meas, circuits_monitoring_node_probability
        
        print('\n')
    
    return mutation_execution_result


def get_bug_detection_result_random(mutation_execution_result):
    bug_detection_count = 0
    detection_result = {}
    for key, value in mutation_execution_result.items():
        print(key)
        # print(res_new[key])
        qc = QuantumCircuit.from_qasm_file(f'{quantum_path}/{key}')
        for k, v in mutation_execution_result[key][-1].items():
            # print({key: sorted([item[0] for item in value]) for key, value in mutation_execution_result[key][1].items()})
            # print(v[1])
            # print((i, str(list(dic.keys())[0])), v[1][(i, str(list(dic.keys())[0]))])
            
            for index, qubit in v[0].items():
                for qubit_info in mutation_execution_result[key][3][index]:
                    if abs(qubit[qubit_info][0] - mutation_execution_result[key][3][index][qubit_info][0]) >= 1e-2 \
                    or abs(qubit[qubit_info][1] - mutation_execution_result[key][3][index][qubit_info][1]) >= 1e-2:
                        bug_detection_count += 1
                    if key not in detection_result:
                        detection_result[key] = []
                    # print(k, [transpiled_qc.find_bit(q).index for q in transpiled_qc.data[k].qubits], {key: [item[0] for item in value] for key, value in mutation_execution_result[key][2].items()}, mutation_execution_result[key][1], mutation_execution_result[key][0][i], (i, str(list(dic.keys())[0])), list(dic.values())[0], v[1][(i, str(list(dic.keys())[0]))])
                    detection_result[key].append([k, v[1], {index: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[index]}, qubit_info, (qubit[qubit_info][0], mutation_execution_result[key][3][index][qubit_info][0]), (qubit[qubit_info][1], mutation_execution_result[key][3][index][qubit_info][1])])
                    print(k, v[1], {index: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[index]}, qubit_info, (qubit[qubit_info][0], mutation_execution_result[key][3][index][qubit_info][0]), (qubit[qubit_info][1], mutation_execution_result[key][3][index][qubit_info][1]))
                    print('\n')
        print('\n')
            # break
            # print(mutation_execution_result['../limited_quantum_circuits/su2random_indep_qiskit_8.qasm'][2][48][1][(i, str(list(dic.keys())[0]))])

    print(bug_detection_count, len(mutation_execution_result))
    
    return detection_result