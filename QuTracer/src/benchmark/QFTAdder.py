# author: Peiyi Li (pli11@ncsu.edu)

from qiskit import QuantumCircuit
from utils import get_ancestors_circuit
from math import pi

#generate original circuit
def generate_circuit(test_qubits, parameters, num_reps):
    #generate the 7qubit QFT adder
    #to do: make the function more general
    test_qubits=7
    qbit_index=3
    
    circuit=QuantumCircuit(test_qubits)
    circuit.x(0)
    circuit.x(1)
    circuit.x(2)
    circuit.x(3)
    circuit.x(4)
    circuit.x(5)

    ########################################
    circuit.h(3+qbit_index)
    circuit.cp(pi/2, 3+qbit_index, 2+qbit_index)
    circuit.h(2+qbit_index)
    circuit.cp(pi/4, 3+qbit_index, 1+qbit_index)
    circuit.cp(pi/2, 2+qbit_index, 1+qbit_index)
    circuit.h(1+qbit_index)
    circuit.cp(pi/8, 3+qbit_index, 0+qbit_index)
    circuit.cp(pi/4, 2+qbit_index, 0+qbit_index)
    circuit.cp(pi/2, 1+qbit_index, 0+qbit_index)
    circuit.h(0+qbit_index)
    ######################################
    circuit.cp(pi,0,3)
    circuit.cp(pi/2,0,4)
    circuit.cp(pi/4,0,5)

    circuit.cp(pi,1,4)
    circuit.cp(pi/2,1,5)

    circuit.cp(pi,2,5)
    circuit.cp(pi/2,2,6)
    circuit.cp(pi/4,1,6)
    circuit.cp(pi/8,0,6)

    ######################################
    circuit.h(0+qbit_index)
    circuit.cp(-pi/2, 1+qbit_index, 0+qbit_index)
    circuit.cp(-pi/4, 2+qbit_index, 0+qbit_index)
    circuit.cp(-pi/8, 3+qbit_index, 0+qbit_index)
    circuit.h(1+qbit_index)
    circuit.cp(-pi/2, 2+qbit_index, 1+qbit_index)
    circuit.cp(-pi/4, 3+qbit_index, 1+qbit_index)
    circuit.h(2+qbit_index)
    circuit.cp(-pi/2, 3+qbit_index, 2+qbit_index)
    circuit.h(3+qbit_index)
    ######################################

    return circuit

#generate equivalent original circuit under commute rule
def generate_equivalent_circuit(test_qubits, parameters, num_reps, qubit_index):
    #to do: make the function more general
    if qubit_index == 3:
        check_qubit_index=1
        circuit=QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.h(1)
        circuit.cp(pi,0,1)
        
    elif qubit_index == 4: 
        check_qubit_index=4
        circuit=QuantumCircuit(7)
        qbit_index=3
        circuit.x(0)
        circuit.x(1)
        circuit.x(2)
        circuit.x(3)
        circuit.x(4)
        circuit.x(5)
        circuit.h(3+qbit_index)
        circuit.cp(pi/2, 3+qbit_index, 2+qbit_index)
        circuit.h(2+qbit_index)
        circuit.cp(pi/4, 3+qbit_index, 1+qbit_index)
        circuit.cp(pi/2, 2+qbit_index, 1+qbit_index)
        circuit.h(1+qbit_index)
        circuit.cp(pi/8, 3+qbit_index, 0+qbit_index)
        circuit.cp(pi/4, 2+qbit_index, 0+qbit_index)
        circuit.cp(pi/2, 1+qbit_index, 0+qbit_index)
        circuit.h(0+qbit_index)
        circuit.cp(pi,0,3)
        circuit.cp(pi/2,0,4)
        circuit.cp(pi,1,4)
        circuit.h(0+qbit_index)
        circuit.cp(-pi/2, 1+qbit_index, 0+qbit_index)
    
    elif qubit_index == 5:
        check_qubit_index=5
        circuit=QuantumCircuit(7)
        qbit_index=3
        circuit.x(0)
        circuit.x(1)
        circuit.x(2)
        circuit.x(3)
        circuit.x(4)
        circuit.x(5)
        circuit.h(3+qbit_index)
        circuit.cp(pi/2, 3+qbit_index, 2+qbit_index)
        circuit.h(2+qbit_index)
        circuit.cp(pi/4, 3+qbit_index, 1+qbit_index)
        circuit.cp(pi/2, 2+qbit_index, 1+qbit_index)
        circuit.h(1+qbit_index)
        circuit.cp(pi/8, 3+qbit_index, 0+qbit_index)
        circuit.cp(pi/4, 2+qbit_index, 0+qbit_index)
        circuit.cp(pi/2, 1+qbit_index, 0+qbit_index)
        circuit.h(0+qbit_index)
        circuit.cp(pi,0,3)
        circuit.cp(pi/2,0,4)
        circuit.cp(pi/4,0,5)
        circuit.cp(pi,1,4)
        circuit.cp(pi/2,1,5)
        circuit.cp(pi,2,5)
        circuit.h(0+qbit_index)
        circuit.cp(-pi/2, 1+qbit_index, 0+qbit_index)
        circuit.cp(-pi/4, 2+qbit_index, 0+qbit_index)
        circuit.h(1+qbit_index)
        circuit.cp(-pi/2, 2+qbit_index, 1+qbit_index)

    elif qubit_index == 6:
        check_qubit_index=6
        circuit=generate_circuit(test_qubits, parameters, num_reps)
        
    return circuit, check_qubit_index

def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps, qubit_index, node_index_list, layer_index=None):
    #to do: make the function more general
    
    assert(len(node_index_list)==1)
    node_index =node_index_list[0]
    
    circuit,check_qubit_index = generate_equivalent_circuit(test_qubits,parameters,num_reps,qubit_index)
    qc=get_ancestors_circuit(circuit, check_qubit_index,node_index)
    
    qubit_index_list = []
    qubit_index_list.append(check_qubit_index)
    
    return qc,qubit_index_list
    #return circuit,qubit_index_list
    
    


