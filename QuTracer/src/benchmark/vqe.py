# author: Peiyi Li (pli11@ncsu.edu)

from qiskit import QuantumCircuit
from utils import get_ancestors_circuit

#generate original circuit
def generate_circuit(test_qubits, parameters, num_reps):
    qc = QuantumCircuit(test_qubits)

    for i in range(test_qubits):
        qc.ry(parameters[i], i)
        
    for j in range(num_reps):
        for i in range(0,test_qubits-1):
            qc.cz(i,i+1)
        for i in range(test_qubits):
            qc.ry(parameters[(1+j)*test_qubits+i], i)
    return qc
#generate equivalent original circuit under commute rule
def generate_equivalent_circuit(test_qubits, parameters, num_reps, qubit_index):
    qc = QuantumCircuit(test_qubits)

    for i in range(test_qubits):
        qc.ry(parameters[i], i)
        
    for j in range(num_reps):
        if (qubit_index-1) < 0: 
            for i in range(0,test_qubits-1):
                qc.cz(i,i+1)
        else:
            for i in range(qubit_index-1,test_qubits-1):
                qc.cz(i,i+1)

            for i in range(qubit_index-1,0,-1):
                qc.cz(i,i-1)
            
        for i in range(test_qubits):
            qc.ry(parameters[(1+j)*test_qubits+i], i)
            
    return qc


def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps, qubit_index, node_index_list, layer_index=None):
    assert(len(node_index_list)==1)
    node_index =node_index_list[0]
    
    qc = generate_equivalent_circuit(test_qubits, parameters, num_reps, qubit_index)
        
    trimmed_qc=get_ancestors_circuit(qc, qubit_index, node_index)
    
    qubit_index_list = []
    qubit_index_list.append(qubit_index)
    return trimmed_qc,qubit_index_list
    
    


