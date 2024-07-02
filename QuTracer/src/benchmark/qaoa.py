# author: Peiyi Li (pli11@ncsu.edu)

import networkx as nx
from qiskit import QuantumCircuit

#using QAOA to solve maxcut problems, the graph used is two-regular graph, the function create_edges generates edges for this graph. 
def create_edges(n):
    # initialize an empty list to store the tuples
    result = []
    # loop from 0 to n-1
    for i in range(n-1):
        # make a tuple of i and i+1
        t = (i, i+1)
        # append the tuple to the result list
        result.append(t)
    result.append((0,n-1))
    # return the result list
    return result


def generate_circuit(test_qubits, parameters, num_reps=None):
    # create two-regular graph G
    G = nx.Graph()
    G.add_nodes_from([i for i in range(test_qubits)])
    edges_list=create_edges(test_qubits)
    G.add_edges_from(edges_list)

    #generate the circuit based on the graph and the parameters
    nqubits = len(G.nodes())
    p = len(parameters)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    
    beta = parameters[:p]
    gamma = parameters[p:]
    
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    
    for irep in range(0, p):
        
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
            
        
    return qc
def generate_circuit_til_certain_layer(test_qubits, parameters, num_reps=None, qubit_index=None, node_index_list=None, layer_index=None):
    
    nqubits = 4 + layer_index*2
    qc = QuantumCircuit(nqubits)
    p = len(parameters)//2  # number of alternating unitaries
    beta = parameters[:p]
    gamma = parameters[p:]
    
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    
    p = len(parameters[:(layer_index+1)*2])//2
    edge_list=create_edges(nqubits)[:-1]
    node_list=[i for i in range(nqubits)]
    
    for irep in range(0, p):
        
        for pair in edge_list:
            qc.rzz(2 * gamma[irep], pair[0], pair[1])
        
        edge_list = edge_list[1:-1]
        node_list = node_list[1:-1]
        if irep < (p-1):
            for i in node_list:
                qc.rx(2 * beta[irep], i)
                
            
    return qc,node_list

    
    

