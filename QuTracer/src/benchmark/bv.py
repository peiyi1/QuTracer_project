# author: Peiyi Li (pli11@ncsu.edu)

from qiskit import QuantumCircuit
from utils import get_ancestors_circuit

#generate original circuit
def generate_circuit(test_qubits, parameters, num_reps):
    circuit=QuantumCircuit(test_qubits)
    for qubit in range(test_qubits):
        circuit.h(qubit)
    
    circuit.z(test_qubits-1)

    for counting_qubit in range(test_qubits-1):
        circuit.cx(counting_qubit, test_qubits-1); # controlled-T
    
    for qubit in range(test_qubits-1):
        circuit.h(qubit)
    
    return circuit


def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps, qubit_index, node_index_list, layer_index=None):
    qubit_index_list = []
    qubit_index_list.append(0)
    
    qc=QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.z(1)
    qc.cx(0,1)
    
    return qc,qubit_index_list
    
    


