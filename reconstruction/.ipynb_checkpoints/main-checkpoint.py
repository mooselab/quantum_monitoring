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


def calculate_tvd(counts1, counts2, shots):
    """ 计算两个测量分布的总变异距离 (TVD) """
    # 归一化为概率分布
    prob1 = {k: v / shots for k, v in counts1.items()}
    prob2 = {k: v / shots for k, v in counts2.items()}
    
    # 合并两个分布中的所有键
    all_keys = set([key.split(' ')[-1] for key in list(prob1.keys())]).union(set([key.split(' ')[-1] for key in list(prob2.keys())]))
    
    # 计算总变异距离
    tvd = 0.5 * sum(abs(prob1.get(k, 0) - prob2.get(k, 0)) for k in all_keys)
    return tvd


simulator = Aer.get_backend('statevector_simulator')


def calculate_num_of_extra_qubits(gate_list, qubit_index, eq_container, num_container, node_index):
    if node_index not in eq_container:
        eq_container[node_index] = {}
    for gate in gate_list:
        ind = num_container[0]
        if not gate:
            continue
        # print(gate)
        # print('********************')
        for qubit in gate[3]:
            if qubit.index != qubit_index:
                if qubit.index not in eq_container[node_index]:
                    eq_container[node_index][qubit.index] = ind
                    num_container[0] += 1
        if len(gate) == 5 and gate[4]:
            for cg in gate[4]:
                # print(cg)
                # print('++++++++++++++++++++')
                for qubit in cg[3]:
                    if qubit.index != qubit_index:
                        if qubit.index not in eq_container[node_index]:
                            eq_container[node_index][qubit.index] = ind
                            num_container[0] += 1
                if len(cg) == 5 and cg[4]:
                    calculate_num_of_extra_qubits(cg[4], qubit_index, eq_container, num_container, node_index)


def fliter_res(file, num=1000):
    operation_list = []
    eq_container = {}
    num_container = [0]
    result = 0
    for i, gate_list in enumerate(res[file]):
        if len(gate_list[-1]) in range(0, num):
            operation_list.append(gate_list)
            
            calculate_num_of_extra_qubits(gate_list[-1], gate_list[2].index, eq_container, num_container, i)
    
    return num_container[0]


def generate_new_operation_list(gate_list, node_index, gate_list_container):
    if node_index not in gate_list_container:
        gate_list_container[node_index] = []  # 为每个节点创建独立的映射
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
    # print(new_operation_list)
    # print('\n')

    # return new_operation_list
    eq = new_qc.qregs[-1]  # 获取辅助量子比特寄存器
    if node_index not in q_to_eq:
        q_to_eq[node_index] = {}  # 为每个节点创建独立的映射
    # if node_index not in gate_list_container:
    #     gate_list_container[node_index] = []  # 为每个节点创建独立的映射
    i = 1
    for gate in new_operation_list:
    # for gate in gate_list:
    #     if gate[:4] in gate_list_container[node_index]:
    #         continue
    #     gate_list_container[node_index].append(gate[:4])
                    
        ind = ind_container[0]
        # 确定目标量子比特的映射
        for qubit in gate[3]:
            if qubit.index != qubit_index:
                if qubit.index not in q_to_eq[node_index]:
                    q_to_eq[node_index][qubit.index] = ind
                    ind_container[0] += 1   # 更新 ind
        # print("Current q_to_eq mapping for node", node_index, ":", q_to_eq[node_index])  # 调试输出
        # 插入单比特门
        if len(gate) == 4 or (len(gate) == 5 and not gate[4]):
            # print('gate = ', gate)
            if len(gate[3]) == 2:
                # print('gate[3] = ', gate[3])
                # 确保控制和目标比特都使用当前节点内的辅助量子比特
                control_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                target_qubit = eq[q_to_eq[node_index][gate[3][1].index]] if gate[3][1].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                
                if not gate[2]:  # 无参数双比特门
                    getattr(new_qc, gate[1])(control_qubit, target_qubit)
                else:  # 有参数双比特门
                    if gate[1] == 'cu1':
                        gate[1] = 'cp'
                    para = gate[2][0]
                    # print('***************************')
                    # print(gate[1], para, control_qubit, target_qubit)
                    # print('***************************')
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
                        # print(gate[1], para, target_qubit)
                        getattr(new_qc, gate[1])(para, target_qubit)
        
        # 插入双比特门
        else:
            # 如果存在嵌套操作，递归处理，并更新 `ind_container` 防止重用
            # if len(gate) == 5 and gate[4]:
            #     generate_gate(gate[4], new_qc, qubit_index, q_to_eq, ind_container, node_index, gate_list_container)
            # print('+++++++++++++++++++++++')
            # print(gate)
            # print('+++++++++++++++++++++++')
            # if gate[0] in ['cp', 'cx']:
            if len(gate[3]) == 2:
                # print('gate[3] = ', gate[3])
                # 确保控制和目标比特都使用当前节点内的辅助量子比特
                control_qubit = eq[q_to_eq[node_index][gate[3][0].index]] if gate[3][0].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                target_qubit = eq[q_to_eq[node_index][gate[3][1].index]] if gate[3][1].index in q_to_eq[node_index] else new_qc.qubits[qubit_index]
                
                if not gate[2]:  # 无参数双比特门
                    getattr(new_qc, gate[1])(control_qubit, target_qubit)
                else:  # 有参数双比特门
                    if gate[1] == 'cu1':
                        gate[1] = 'cp'
                    para = gate[2][0]
                    # print('***************************')
                    # print(gate[1], para, control_qubit, target_qubit)
                    # print('***************************')
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
                        # print(gate[0], para, target_qubit)
                        getattr(new_qc, gate[1])(para, target_qubit)
        
        if i == len(new_operation_list):
            new_qc.data[-1][0].label = "test"
            # print("reset")
        
        i += 1



def generate_monitoring_circuit(file, res, num=1000):
    # file = '../quantum circuits/' + file
    operation_list = []
    eq_container = {}
    num_container = [0]
    result = 0
    for i, gate_list in enumerate(res[file]):
        if len(gate_list[-1]) in range(0, num):
            # print(gate_list)
            # print(';;;;;;;;;;;;;;;;;;;;;;')
            # print('\n')
            operation_list.append(gate_list)
            # if i[1] in ['cp', 'cx']:
            #     num_container[0] += 1
            
            calculate_num_of_extra_qubits(gate_list[-1], gate_list[2].index, eq_container, num_container, i)

    # print(file, num_container[0], '\n')
    
    qc = QuantumCircuit.from_qasm_file(file)
    transpiled_qc = transpile(qc, simulator)
    
    new_qc = QuantumCircuit()
    for reg in transpiled_qc.qregs:
        new_qc.add_register(QuantumRegister(reg.size, reg.name))
    for reg in transpiled_qc.cregs:
        new_qc.add_register(ClassicalRegister(reg.size, reg.name))
    if new_qc.num_clbits < new_qc.num_qubits:
        new_qc.add_register(ClassicalRegister(new_qc.num_qubits - new_qc.num_clbits, 'ec'))
    
    # 增加额外的辅助量子比特寄存器
    if num_container[0] != 0:
        additional_qubits = QuantumRegister(num_container[0], 'eq')
        new_qc.add_register(additional_qubits)
    
    # 遍历并插入所有门
    ind_container = [0]
    q_to_eq = {}
    gate_list_container = {}
    level = 0
    for i, (instr, qargs, cargs) in enumerate(transpiled_qc.data):
        new_qc.append(instr, qargs, cargs)  # 添加原始的门操作
        
        for op in operation_list:
            if i == op[0]:
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                # print(op, '\n')
                # print('VVVVVVVVVVVVVVVVVVVVVVVVVVVVV')
                for qubit in qargs:
                    if transpiled_qc.find_bit(qubit) == op[2]:
                        # 插入测量、复位和恢复门操作
                        new_qc.measure(new_qc.qubits[op[2].index], new_qc.clbits[op[2].index])
                        new_qc.reset(new_qc.qubits[op[2].index])
                        eq = new_qc.qregs[-1]
                        # print(op[-1], '\n')
                        generate_gate(op[-1], new_qc, op[2].index, q_to_eq, ind_container, i, gate_list_container)
    return new_qc



def find_probablity_new(qc, simulator):
    res = []
    partial_circuit = QuantumCircuit()
    for reg in qc.qregs:
        new_reg = QuantumRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    for reg in qc.cregs:
        new_reg = ClassicalRegister(reg.size, reg.name)
        partial_circuit.add_register(new_reg)
    # print(qc.num_qubits)
    # qc = transpile(qc, simulator)
    for i, gate in enumerate(qc.data):
        # 创建一个新的电路，将所有先前的门加入到新的电路中
        # partial_circuit = QuantumCircuit(qc.num_qubits)
        partial_circuit.data = qc.data[:i+1]  # 只包含到第 i 步的操作
        
    
        # 模拟部分电路并获取态向量
        job = simulator.run(partial_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # 将态向量转换为量子态对象
        quantum_state = Statevector(statevector)
        # quantum_state = Statevector.from_instruction(partial_circuit)
    
        # 提取与该门操作相关的量子比特的状态
        qubits_involved = [partial_circuit.find_bit(qubit) for qubit in gate[1]]  # 获取与该门相关的量子比特
        for qubit in qubits_involved:
            # if gate[0].name == 'u3':
            #     print('sjhfjg;kg;r')
            # 使用 partial_trace 仅提取该量子比特的状态
            qubit_state = partial_trace(quantum_state, [q for q in range(qc.num_qubits) if q != qubit.index])
            prob_zero = qubit_state.data[0, 0].real
            prob_one = qubit_state.data[1, 1].real
            res.append([i, gate[0].name, qubit, prob_zero, prob_one, gate[0].params, gate[0].label])
    return res



if __name__ == "__main__":   
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--shots', type=int, default=8192)
    # 解析命令行参数
    args = parser.parse_args()

    with open('../data/opeartion_list.pkl', 'rb') as pickle_file:
        res = pickle.load(pickle_file)

    unwanted_circuit_list = []
    with open('../quantum circuits/statevector_list.pkl', 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                # list = []
                qc = QuantumCircuit.from_qasm_file(item[0])
                for i, gate in enumerate(transpile(qc, simulator).data):
                    if gate[0].name == 'ccx':
                        unwanted_circuit_list.append(str(item[0]))
                        break
            except EOFError:
                break
    
    res_new = {}
    for key, value in res.items():
        # if fliter_res(key) <= 10 and key not in [str(i) for i in unwanted_circuit_list]:
        # if key not in unwanted_circuit_list:
        if fliter_res(key) <= args.num and key not in [str(i) for i in unwanted_circuit_list]:
            res_new[key] = value
    
    print(len(res_new))
    # exit(0)
    
    comparison_circuit = {}
    for key in res_new:
        try:
            comparison_circuit[key] = generate_monitoring_circuit(key, res)
        except Exception as e:
            print(e)
    
    """result = {}
    simulator = Aer.get_backend('statevector_simulator')
    pm = PassManager(passes.RemoveFinalMeasurements())
    for key, value in comparison_circuit.items():
        # print(key)
        new_qc_no_meas = pm.run(value)
        result[key] =  find_probablity_new(new_qc_no_meas, simulator)
    
    count = 0
    for key, value in result.items():
        flag = True
        print(key)
        for i in value:
            if i[-1] == 'test' and i[2][1][0][0].name != 'eq':
                if i[3] < 0.99 and i[4] < 0.99:
                    print(i)
                    flag = False
        print('***********************', '\n')
       
        if not flag:
            count += 1
    print('\n', count, '\n')
    
    shots = args.shots
    execution_list = []
    for key in res_new:
        qc = QuantumCircuit.from_qasm_file(key)
        new_qc = comparison_circuit[key]
        result_qc = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
        result_new_qc = execute(new_qc, backend=Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
        execution_list.append([{key:[result_qc, result_new_qc, calculate_tvd(result_qc, result_new_qc, shots)]}])
    with open('../data/execution_result.pkl', 'wb') as file:
        pickle.dump(execution_list, file)"""
    
    with open('../data/comparison_circuit.pkl', 'wb') as pickle_file:
        pickle.dump(comparison_circuit, pickle_file)
    

