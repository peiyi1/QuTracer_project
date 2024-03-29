#author: Peiyi Li (pli11@ncsu.edu)
def obtain_initial_physical_qubits_index(qc, logical_qubits, changed_mapping):
    layout = qc._layout
    #initial_virtual_layout=layout.initial_virtual_layout(filter_ancillas=True)
    initial_index_layout=layout.initial_index_layout(filter_ancillas=True)
    #print(initial_virtual_layout)
    #print(initial_index_layout)
    
    record_qubits_index=[]
    for qbit in logical_qubits:
        index=changed_mapping[qbit]
        physical_qubit=initial_index_layout[index]
        record_qubits_index.append(physical_qubit)
    return record_qubits_index