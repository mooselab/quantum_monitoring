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


def generate_full_path(gate_list, operation_list):
    if not gate_list:
        return operation_list
    
    for gate in gate_list:
        if not gate:
            continue
        
        if len(gate) == 4:
            operation_list.append(gate)
            
        else:
            target_gate = gate
            control_gate = []
           
            control_control_gate = generate_full_path(gate[4], [])
            control_gate.extend(control_control_gate)
            
            target_gate.extend([control_gate])
            operation_list.append(target_gate)

    return operation_list


def generate_gate_list_container(gate_list, node_index, gate_list_container):
    if node_index not in gate_list_container:
        gate_list_container[node_index] = []
    for gate in gate_list:
        if gate[:4] in gate_list_container[node_index]:
            if len(gate) == 4:
                continue
        else:
            gate_list_container[node_index].append(gate[:4])
        
        if len(gate) == 5 and gate[4]:
            generate_new_operation_list(gate[4], node_index, gate_list_container)
        
    return gate_list_container


def generate_monitoring_mutated_circuit(transpiled_qc, file, res, num=1000):
    operation_list = []
    eq_container = {}
    num_container = [0]
    for i, gate_list in enumerate(res[file]):
        if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, num):
            operation_list.append(gate_list)
            
            calculate_num_of_extra_qubits(gate_list[-1], gate_list[2].index, eq_container, num_container, i)
    
    new_qc = QuantumCircuit()
    for reg in transpiled_qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in transpiled_qc.cregs:
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
    for i, (instr, qargs, cargs) in enumerate(transpiled_qc.data):
        new_qc.append(instr, qargs, cargs)
        
        for op in operation_list:
            if i == op[0]:
                
                for qubit in qargs:
                    if transpiled_qc.find_bit(qubit) == op[2]:
                        new_qc.measure(new_qc.qubits[op[2].index], ec[ec_container[0]])
                        ec_container[0] += 1
                        new_qc.reset(new_qc.qubits[op[2].index])
                        eq = new_qc.qregs[-1]
                        generate_gate(op[-1], new_qc, op[2].index, q_to_eq, ind_container, i, gate_list_container)
    return new_qc



def mutate_circuit(transpiled_qc, mutation_idx, quantum_gates) -> QuantumCircuit:
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
    for reg in transpiled_qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in transpiled_qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    
    # Get all gates and their locations from the original circuit
    gates_list = []
    current_idx = 0
    
    for current_idx, (instr, qargs, cargs) in enumerate(transpiled_qc.data):
        qubits = [transpiled_qc.find_bit(qubit) for qubit in qargs]
        params = instr.params if hasattr(instr, 'params') else []
        
        if current_idx == mutation_idx:
            # Determine if it's a single or two qubit gate
            is_single = len(qubits) == 1
            
            # Get random replacement gate
            new_gate_name, new_params = get_random_gate(is_single)
            print(new_gate_name, new_params)
            
            # Add the new gate to the circuit
                        # For gates with parameters
            if new_params:
                getattr(new_qc, new_gate_name)(*new_params, *[new_qc.qubits[q.index] for q in qubits])
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


def execute_mutated_circuits(operation_list, bad_list, simulator, final_definited_node_dict, seed):
    circuit_count = 0
    random.seed(seed)
    pm = PassManager(passes.RemoveFinalMeasurements())
    mutation_execution_result = {}
    for key, value in operation_list.items():
        if key not in bad_list:
            print(key)
            index_list = []
            # monitor_qubit_gate_list = []
            assertion_value_list = []
            gate_list_container = {}
            gate_list_container_test = {}
            
            for gate_list in value:
                # print(gate_list)
                if not require_extra_qubit(gate_list[-1]) or len(gate_list[-1]) in range(0, 1000):
                    index_list.append(gate_list[0])
                    #monitor_qubit_gate_list.append(gate_list[0])
                    assertion_value_list.append({0 if gate_list[3] >= 0.99 else 1: gate_list[3] if gate_list[3] >= 0.99 else gate_list[4]})
                    gate_list_container = generate_new_operation_list(gate_list[-1], gate_list[0], gate_list_container)
                    
                    for node in final_definited_node_dict[key]:
                        if node[0] == gate_list[0]:
                            operation_list_new = generate_full_path(node[-1], [])
                            gate_list_container_test = generate_new_operation_list(operation_list_new, gate_list[0], gate_list_container)
                            break
                            
            qc = QuantumCircuit.from_qasm_file(key)
            if index_list[-1] <= qc.num_qubits - 1:
                # print('**********', index_list, '**********')
                continue
            # print(index_list)
            # continue
            mutation_execution_result[key] = [index_list, gate_list_container_test, gate_list_container, assertion_value_list, {}]
            # print(mutation_execution_result[key])
            # continue
            
            transpiled_qc = transpile(qc, simulator)
            num_gate = len(pm.run(transpiled_qc).data)
            print(index_list, num_gate)
            circuit_count += 1
            print('******', circuit_count, '******')
            # continue
            # print(num_gate)
            # continue

            mutation_idx_list = random.sample(list(range(index_list[-1] + 1)), 3) # if num_gate >= 1000 else random.sample(list(range(num_gate)), num_gate //2)
            # print(mutation_idx_list)
            # continue
            
            for mutation_idx in mutation_idx_list:
                mutation_qc = mutate_circuit(transpiled_qc, mutation_idx, quantum_gates)
                reconstructed_mutation_qc = generate_monitoring_mutated_circuit(mutation_qc, key, operation_list, num=1000)
                result_reconstructed_mutation_qc = execute(reconstructed_mutation_qc, backend=Aer.get_backend('qasm_simulator'), shots=100).result().get_counts()

                new_counts = defaultdict(int)
                for k, v in result_reconstructed_mutation_qc.items():
                    new_key = k.split(' ')[0][::-1]
                    new_counts[new_key] += v
                # print(new_counts)
                mutation_execution_result[key][-1][mutation_idx] = [dict(new_counts)]
                
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

                mutation_execution_result[key][-1][mutation_idx].append(dict(bit_percentages))
    
    return mutation_execution_result


def get_bug_detection_result(mutation_execution_result, simulator):
    bug_detection_count = 0
    detection_result = {}
    for key, value in mutation_execution_result.items():
        print(key)
        # print(res_new[key])
        qc = QuantumCircuit.from_qasm_file(key)
        transpiled_qc = transpile(qc, simulator)
        for i, dic in enumerate(mutation_execution_result[key][3]):
            # print((i, str(list(dic.keys())[0])), list(dic.values())[0])
            for k, v in mutation_execution_result[key][-1].items():
                # print({key: sorted([item[0] for item in value]) for key, value in mutation_execution_result[key][1].items()})
                # print(v[1])
                # print((i, str(list(dic.keys())[0])), v[1][(i, str(list(dic.keys())[0]))])
                if (i, str(list(dic.keys())[0])) not in v[1]:
                    bug_detection_count += 1
                    print('no key')
                    continue
                for index_value_set in v[1].keys():
                    # print(index_value_set)
                    if index_value_set == (i, str(list(dic.keys())[0])) and abs(list(dic.values())[0] - v[1][(i, str(list(dic.keys())[0]))]) >= 1e-2:
                        bug_detection_count += 1
                        if key not in detection_result:
                            detection_result[key] = []
                        # print(k, [transpiled_qc.find_bit(q).index for q in transpiled_qc.data[k].qubits], {key: [item[0] for item in value] for key, value in mutation_execution_result[key][2].items()}, mutation_execution_result[key][1], mutation_execution_result[key][0][i], (i, str(list(dic.keys())[0])), list(dic.values())[0], v[1][(i, str(list(dic.keys())[0]))])
                        detection_result[key].append([k, {mutation_execution_result[key][0][i]: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[mutation_execution_result[key][0][i]]}, (i, str(list(dic.keys())[0])), list(dic.values())[0], v[1][(i, str(list(dic.keys())[0]))]])
                        print(k, {mutation_execution_result[key][0][i]: {key: list(set(sorted([item[0] for item in value]))) for key, value in mutation_execution_result[key][2].items()}[mutation_execution_result[key][0][i]]}, (i, str(list(dic.keys())[0])), list(dic.values())[0], v[1][(i, str(list(dic.keys())[0]))])
                        print('\n')
        print('\n')
            # break
            # print(mutation_execution_result['../limited_quantum_circuits/su2random_indep_qiskit_8.qasm'][2][48][1][(i, str(list(dic.keys())[0]))])

    print(bug_detection_count, len(mutation_execution_result))
    
    return detection_result