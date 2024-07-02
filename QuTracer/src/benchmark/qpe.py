# author: Peiyi Li (pli11@ncsu.edu)

from qiskit import QuantumCircuit
from utils import get_ancestors_circuit
from math import pi

#generate original circuit
def generate_circuit(test_qubits, parameters, num_reps, angle = 1*pi/4):
    circuit=QuantumCircuit(test_qubits)
    for qubit in range(test_qubits-1):
        circuit.h(qubit)
    
    circuit.x(test_qubits-1)
    
    repetitions = 1
    for counting_qubit in range(test_qubits-1):
        for i in range(repetitions):
            circuit.cp(angle, counting_qubit, test_qubits-1); # controlled-T
        repetitions *= 2
    
    rotation_angle=-pi/2
    for i in range(test_qubits-1, 1, -1):
        qbit=i-1
        
        circuit.h(qbit)
        rotation_angle=-pi
        rotation_angle_cof=1/2
        for j in range(qbit, 0, -1):
            target_qbit = j-1
            rotation_angle = rotation_angle*rotation_angle_cof
            circuit.cp(rotation_angle,qbit,target_qbit)
        
    circuit.h(0)    
        
    return circuit

#generate equivalent original circuit under commute rule
def generate_equivalent_circuit(test_qubits, parameters, num_reps, qubit_index, angle = 1*pi/4):
    circuit=QuantumCircuit(test_qubits)
    for qubit in range(test_qubits-1):
        circuit.h(qubit)
    
    circuit.x(test_qubits-1)
    
    repetitions = 2**qubit_index
    for counting_qubit in range(qubit_index, test_qubits-1):
        for i in range(repetitions):
            circuit.cp(angle, counting_qubit, test_qubits-1); # controlled-T
        repetitions *= 2
    
    rotation_angle=-pi/2
    for i in range(test_qubits-1, 1, -1):
        qbit=i-1
        
        circuit.h(qbit)
        rotation_angle=-pi
        rotation_angle_cof=1/2
        for j in range(qbit, 0, -1):
            target_qbit = j-1
            rotation_angle = rotation_angle*rotation_angle_cof
            circuit.cp(rotation_angle,qbit,target_qbit)
        
    circuit.h(0)
    
    return circuit

def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps, qubit_index, node_index_list, layer_index=None):
    assert(len(node_index_list)==1)
    node_index =node_index_list[0]
    
    circuit=generate_equivalent_circuit(test_qubits,parameters,num_reps,qubit_index)
    qc=get_ancestors_circuit(circuit, qubit_index,node_index)
    
    qubit_index_list = []
    qubit_index_list.append(qubit_index)
    
    return qc,qubit_index_list
    
    


