# author: Peiyi Li (pli11@ncsu.edu)

from qiskit import QuantumCircuit
from utils import get_ancestors_circuit
from math import pi

#generate original circuit
def generate_circuit(test_qubits, parameters, num_reps):
    #generate the 4qubit QFT multiplier
    #to do: make the function more general
    
    circuit=QuantumCircuit(4)
    circuit.x(0)
    circuit.x(1)

    ########################################
    circuit.h(3)
    circuit.cp(pi/2, 2, 3)
    circuit.h(2)
    ######################################

    circuit.mcp(pi,[0,1],2)
    circuit.mcp(pi/2,[0,1],3)

    ######################################
    circuit.h(2)
    circuit.cp(-pi/2, 2, 3)
    circuit.h(3)
    ######################################

    return circuit

#generate equivalent original circuit under commute rule
def generate_equivalent_circuit(test_qubits, parameters, num_reps, qubit_index):
    #to do: make the function more general
    if qubit_index == 2:
        circuit=QuantumCircuit(3)
        
        circuit.x(0)
        circuit.x(1)
        circuit.h(2)
        circuit.mcp(pi,[0,1],2)
        
    elif qubit_index == 3:
        circuit=QuantumCircuit(4)
        circuit.x(0)
        circuit.x(1)

        ########################################
        circuit.h(3)
        circuit.cp(pi/2, 2, 3)
        circuit.h(2)
        ######################################

        circuit.mcp(pi,[0,1],2)
        circuit.mcp(pi/2,[0,1],3)

        ######################################
        circuit.h(2)
        circuit.cp(-pi/2, 2, 3)
        
        ######################################
    
    return circuit

def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps, qubit_index, node_index_list, layer_index=None):
    assert(len(node_index_list)==1)
    node_index =node_index_list[0]
    
    circuit=generate_equivalent_circuit(test_qubits,parameters,num_reps,qubit_index)
    qc=get_ancestors_circuit(circuit, qubit_index,node_index)
    #print(circuit.num_qubits)
    qubit_index_list = []
    qubit_index_list.append(qubit_index)
    
    return qc,qubit_index_list
    
    


