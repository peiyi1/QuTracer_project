# author: Peiyi Li (pli11@ncsu.edu)

from .mlrecon_methods import identify_frag_targets,identify_stitches

def obtain_prep_dict(wire_path_map):
    stitches=identify_stitches(wire_path_map)
    
    prep_dict={}
    for meas_frag_qubit, prep_frag_qubit in stitches.items():
        #meas_frag, meas_qubit = meas_frag_qubit
        prep_frag, prep_qubit = prep_frag_qubit
        if prep_frag not in prep_dict.keys():
            prep_dict[prep_frag]=(prep_qubit,)
        else:
            prep_dict[prep_frag]+=(prep_qubit,)
            
    return prep_dict

def obtain_meas_dict(wire_path_map):
    stitches=identify_stitches(wire_path_map)
    
    meas_dict={}
    for meas_frag_qubit, prep_frag_qubit in stitches.items():
        meas_frag, meas_qubit = meas_frag_qubit
        #prep_frag, prep_qubit = prep_frag_qubit
        if meas_frag not in meas_dict.keys():
            meas_dict[meas_frag]=(meas_qubit,)
        else:
            meas_dict[meas_frag]+=(meas_qubit,)
            
    return meas_dict
        
def obtain_prep_info(wire_path_map):
    prep_dict=obtain_prep_dict(wire_path_map)
    
    prep_fragment_index_list=[]
    prep_qubits_list=[]
    for prep_fragment_index,prep_qubits in prep_dict.items():
        prep_fragment_index_list.append(prep_fragment_index)
        prep_qubits_list.append(prep_qubits)
    #print(prep_fragment_index_list)
    #print(prep_qubits_list)
    return (prep_fragment_index_list,prep_qubits_list)

def obtain_meas_info(wire_path_map):
    meas_dict=obtain_meas_dict(wire_path_map)
    
    meas_fragment_index_list=[]
    meas_qubits_list=[]
    for meas_fragment_index,meas_qubits in meas_dict.items():
        meas_fragment_index_list.append(meas_fragment_index)
        meas_qubits_list.append(meas_qubits)
    #print(meas_fragment_index_list)
    #print(meas_qubits_list)
    return (meas_fragment_index_list,meas_qubits_list)

#########################################################################
from qiskit import ClassicalRegister

def generate_circuit_adding_measurement(qc,measured_qubits,prep_qubits):
    new_qc=qc.copy()
    
    subcircuit=QuantumCircuit(1)
    subcircuit.barrier(0)
    
    for qbit in prep_qubits:
        new_qc=new_qc.compose(subcircuit,[qbit.index],front=True)
    
    c_reg = ClassicalRegister(len(measured_qubits), 'c')
    new_qc.add_register(c_reg)
    
    i=0
    for qbit in measured_qubits:
        new_qc.barrier(qbit.index)
        new_qc.measure(qbit.index,i)
        i+=1
    return new_qc    

#generate circuits after cutting
from .circuit_cutter import cut_circuit
def generate_circuit_after_cutting(circuit, check_qubits, check_position):
    #specify the info of cut
    cuts = list(zip((circuit.qubits[qbit] for qbit in check_qubits), check_position))
    #do circuit cutting
    fragments, wire_path_map = cut_circuit(circuit, cuts)
    
    #obtain the info of prep circuits
    prep_fragment_index_list,prep_qubits_list=obtain_prep_info(wire_path_map)
    assert(len(prep_fragment_index_list)==1)
    assert(len(prep_qubits_list)==1)
    prep_fragment_index=prep_fragment_index_list[0]
    prep_qubits=prep_qubits_list[0]
    
    #obtain the info of meas circuits
    meas_fragment_index_list,meas_qubits_list=obtain_meas_info(wire_path_map)
    meas_fragment_index=meas_fragment_index_list[0]
    meas_qubits=meas_qubits_list[0]
    
    #obtain the prep circuit and add measurement
    qc=fragments[prep_fragment_index]
    #obtain the meas circuit(to do: may have multiple framgment of meas circuits)
    meas_qc=fragments[meas_fragment_index]
    #layer n (n>0)
    if (len(meas_fragment_index_list)==1) and (prep_fragment_index==meas_fragment_index):   
        prep_qc=generate_circuit_adding_measurement(qc,meas_qubits+prep_qubits,prep_qubits)
    #layer 0     
    else:
        prep_qc=generate_circuit_adding_measurement(qc,prep_qubits,prep_qubits)
    return prep_qc,meas_qc,prep_qubits,meas_qubits
###########################################################################
def obtain_initial_physical_qubits_index(qc, logical_qubits):
    layout = qc._layout
    
    initial_virtual_layout=layout.initial_virtual_layout(filter_ancillas=False)
    initial_index_layout=layout.initial_index_layout(filter_ancillas=False)
    
    record_qubits_index=[]
    
    for logical_qubit in logical_qubits:
        for physical_qubit in initial_index_layout:
            if initial_virtual_layout[physical_qubit]==logical_qubit:
                record_qubits_index.append(physical_qubit)
                break
    return record_qubits_index

def obtain_final_physical_qubits_index_for_measurement(qc):
    record_qubits_index=[]
    for quantum_register in qc.data:
        if(quantum_register[0].name == 'measure'):
            #print(quantum_register[1][0])
            qubit_index = qc.find_bit(quantum_register[1][0]).index
            record_qubits_index.append(qubit_index)
    return record_qubits_index
############################################################################
def get_gate_information(circuit):
    gate_info = []
    for instruction in circuit.data:
        gate = instruction[0]
        qubits = instruction[1]
        params = gate.params
        gate_info.append({
            "name": gate.name,
            "qubits": [qubit.index for qubit in qubits],
            "params": params
        })
    return gate_info
###############################################################################
from qiskit import QuantumCircuit
def create_circuit_for_term1(qc,subcircuit,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,obs='I'):
    new_qc=qc.copy()
    if (subcircuit):
        new_qc=qc.compose(subcircuit,new_prep_qubit_index_transpiled,front=True)
    
    # Create a Classical Register with 1 bits.
    c_reg = ClassicalRegister(1, 'c')
    new_qc.add_register(c_reg)
    if(obs=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    elif(obs=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[0])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])  
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[0],0)
    
    return new_qc

def create_circuit_for_term4(qc,subcircuit,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,obs='I'):
    subcircuit0=QuantumCircuit(1)
    subcircuit0.z(0)
    new_qc0=qc.compose(subcircuit0,new_prep_qubit_index_transpiled,front=True)
    new_qc1=new_qc0.compose(subcircuit,new_prep_qubit_index_transpiled,front=True)
    
    c_reg = ClassicalRegister(1, 'c')
    new_qc1.add_register(c_reg)
    
    new_qc1.z(new_prep_qubit_index_transpiled_measurement[0])

    if(obs=='X'):
        new_qc1.h(new_prep_qubit_index_transpiled_measurement[0])
    elif(obs=='Y'):
        new_qc1.sdg(new_prep_qubit_index_transpiled_measurement[0])  
        new_qc1.h(new_prep_qubit_index_transpiled_measurement[0])  
    new_qc1.measure(new_prep_qubit_index_transpiled_measurement[0],0)
    
    return new_qc1

def create_circuits_with_certain_prep_meas_cutting_1q_layer_0(qc,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,prep_state,obs='I'):
    subcircuit=QuantumCircuit(1)
    #prep_state
    if(prep_state=='Z_n'):
        subcircuit.x(0)
    elif(prep_state=='X_p'):
        subcircuit.h(0)
    elif(prep_state=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
        
    new_qc=qc.compose(subcircuit,new_prep_qubit_index_transpiled,front=True)
    
    # Create a Classical Register with 1 bits.
    c_reg = ClassicalRegister(1, 'c')
    new_qc.add_register(c_reg)
    qubits_final_index=new_prep_qubit_index_transpiled_measurement[0]
    if(obs=='X'):
        new_qc.h(qubits_final_index)
    elif(obs=='Y'):
        new_qc.sdg(qubits_final_index)  
        new_qc.h(qubits_final_index)  
    new_qc.measure(qubits_final_index,0)
    
    return new_qc

def create_circuits_with_certain_prep_meas_cutting_1q_layer_n(qc,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,prep_state,obs_meas='I',obs_final='I'):
    subcircuit=QuantumCircuit(1)
    #prep_state
    if(prep_state=='Z_n'):
        subcircuit.x(0)
    elif(prep_state=='X_p'):
        subcircuit.h(0)
    elif(prep_state=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
        
    new_qc=qc.compose(subcircuit,new_prep_qubit_index_transpiled,front=True)
    
    # Create a Classical Register with 2 bits.
    c_reg = ClassicalRegister(2, 'c')
    new_qc.add_register(c_reg)
    if(obs_meas=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    elif(obs_meas=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[0])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    
    if(obs_final=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    elif(obs_final=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[1])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[0],0)
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[1],1)
    
    return new_qc

def create_circuits_with_certain_prep_meas_cutting_2q_layer_0(qc,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,prep_state,obs='II'):
    subcircuit=QuantumCircuit(1)
    #prep_state[0]
    if(prep_state[-1]=='Z_n'):
        subcircuit.x(0)
    elif(prep_state[-1]=='X_p'):
        subcircuit.h(0)
    elif(prep_state[-1]=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state[-1]=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state[-1]=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
    new_qc=qc.compose(subcircuit,[new_prep_qubit_index_transpiled[0]],front=True)
    
    subcircuit=QuantumCircuit(1)
    #prep_state[1]
    if(prep_state[-2]=='Z_n'):
        subcircuit.x(0)
    elif(prep_state[-2]=='X_p'):
        subcircuit.h(0)
    elif(prep_state[-2]=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state[-2]=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state[-2]=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
    new_qc=new_qc.compose(subcircuit,[new_prep_qubit_index_transpiled[1]],front=True)
           
    
    # Create a Classical Register with 1 bits.
    c_reg = ClassicalRegister(2, 'c')
    new_qc.add_register(c_reg)
    if(obs[-1]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    elif(obs[-1]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[0])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    if(obs[-2]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    elif(obs[-2]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[1])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[0],0)
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[1],1)
    
    return new_qc

def create_circuits_with_certain_prep_meas_cutting_2q_layer_n(qc,new_prep_qubit_index_transpiled_measurement,new_prep_qubit_index_transpiled,prep_state,obs_meas='II',obs_final='II'):
    subcircuit=QuantumCircuit(1)
    #prep_state[0]
    if(prep_state[-1]=='Z_n'):
        subcircuit.x(0)
    elif(prep_state[-1]=='X_p'):
        subcircuit.h(0)
    elif(prep_state[-1]=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state[-1]=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state[-1]=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
    new_qc=qc.compose(subcircuit,[new_prep_qubit_index_transpiled[0]],front=True)
    
    subcircuit=QuantumCircuit(1)
    #prep_state[1]
    if(prep_state[-2]=='Z_n'):
        subcircuit.x(0)
    elif(prep_state[-2]=='X_p'):
        subcircuit.h(0)
    elif(prep_state[-2]=='X_n'):
        subcircuit.x(0)
        subcircuit.h(0)
    elif(prep_state[-2]=='Y_p'):
        subcircuit.h(0)
        subcircuit.s(0)
    elif(prep_state[-2]=='Y_n'):
        subcircuit.h(0)
        subcircuit.sdg(0)
    new_qc=new_qc.compose(subcircuit,[new_prep_qubit_index_transpiled[1]],front=True)
           
    
    # Create a Classical Register with 1 bits.
    c_reg = ClassicalRegister(4, 'c')
    new_qc.add_register(c_reg)
    if(obs_meas[-1]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    elif(obs_meas[-1]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[0])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[0])
    if(obs_meas[-2]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    elif(obs_meas[-2]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[1])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[1])
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[0],0)
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[1],1)
    
    if(obs_final[-1]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[2])
    elif(obs_final[-1]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[2])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[2])
    if(obs_final[-2]=='X'):
        new_qc.h(new_prep_qubit_index_transpiled_measurement[3])
    elif(obs_final[-2]=='Y'):
        new_qc.sdg(new_prep_qubit_index_transpiled_measurement[3])  
        new_qc.h(new_prep_qubit_index_transpiled_measurement[3])
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[2],2)
    new_qc.measure(new_prep_qubit_index_transpiled_measurement[3],3)
    
    return new_qc
#####################################################################################
X_eigen_states = ['X_p', 'X_n']
Y_eigen_states = ['Y_p', 'Y_n']
Z_eigen_states = ['Z_p', 'Z_n']
I_eigen_states = ['Z_p', 'Z_n']

dict_for_create_circuit_term_1_4={'X':[X_eigen_states], 'Y':[Y_eigen_states]}
dict_for_create_circuit_term_2_3={'X':[Y_eigen_states], 'Y':[X_eigen_states]}
dict_for_create_circuit_term_common={'Z':[Z_eigen_states]}

dict_for_create_circuit={'XX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],
                         'XY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],
                         'YX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],
                         'YY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],
                         'XZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],
                         'YZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],
                         'ZX':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],
                         'ZY':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],
                         'ZZ':[(Z_eigen_states,Z_eigen_states)],
                        }
obs_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}

from qiskit import  transpile
def generate_execution_circuits_list(prep_qc, meas_qc, backend, qubits_initial_index_list, qubits_final_index_list, obs_list, prep_state_list):
    qc_list=[]
    
    #generate circuits for layer 0 and layer n (n>0)
    #circuits for layer 0 
    if len(prep_state_list) == 4:
        #cutting 1 qubit
        if len(qubits_initial_index_list)==1:
            #when obs_list just has one observable, decide if can do further optimizaiton
            if len(obs_list)==1:
                obs = obs_list[0]
                #can do further optimization (to do: need to fill in the case when can not do further optimization)
                if meas_qc.num_qubits == 1:
                    #circuit for term1
                    circ = create_circuit_for_term1(prep_qc,meas_qc,qubits_final_index_list,qubits_initial_index_list,obs=obs) 
                    qc_list.append(circ)
                    #circuit for term4
                    circ = create_circuit_for_term4(prep_qc,meas_qc,qubits_final_index_list,qubits_initial_index_list,obs=obs) 
                    qc_list.append(circ)
                    #circuit for term2 and term3
                    for prep_state in prep_state_list:
                        circ = create_circuits_with_certain_prep_meas_cutting_1q_layer_0(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_dict[obs])
                        qc_list.append(circ)
                    #circuit for term2 and term3 (denominator)
                    for prep_state in prep_state_list:
                        circ = create_circuits_with_certain_prep_meas_cutting_1q_layer_0(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_dict['I'])
                        qc_list.append(circ)
        #cutting more than 1 qubit
        else:
            for obs in obs_list:
                for prep_state_1 in prep_state_list:
                    for prep_state_0 in prep_state_list:
                        prep_state=[prep_state_1,prep_state_0]
                        circ = create_circuits_with_certain_prep_meas_cutting_2q_layer_0(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs)
                        qc_list.append(circ)
    
    #circuits for layer n
    else:
        if len(qubits_initial_index_list)==1:
            #create circuits for term 1 and 4
            for obs_final in obs_list:
                for obs_meas,prep_state_list in dict_for_create_circuit_term_1_4.items():
                    for prep_state in prep_state_list[0]:
                        circ = create_circuits_with_certain_prep_meas_cutting_1q_layer_n(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_meas,obs_final)
                        qc_list.append(circ)
            #create circuits for term 2 and 3
            #to do: need to process the case when all three obs are measured
            for obs_final in obs_list + ['I']:
                for obs_meas,prep_state_list in dict_for_create_circuit_term_2_3.items():
                    for prep_state in prep_state_list[0]:
                        circ = create_circuits_with_certain_prep_meas_cutting_1q_layer_n(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_meas,obs_dict[obs_final])
                        qc_list.append(circ)
            #create circuits for common part
            for obs_final in ['X','Y','Z']:
                for obs_meas,prep_state_list in dict_for_create_circuit_term_common.items():
                    for prep_state in prep_state_list[0]:
                        circ = create_circuits_with_certain_prep_meas_cutting_1q_layer_n(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_meas,obs_final)
                        qc_list.append(circ)
        else:
            for obs_final in obs_list:
                for obs_meas,prep_state_list in dict_for_create_circuit.items():
                    for (prep_state_1_list,prep_state_0_list) in prep_state_list:
                        for prep_state_1 in prep_state_1_list:
                            for prep_state_0 in prep_state_0_list:
                                prep_state=[prep_state_1,prep_state_0]
                                circ = create_circuits_with_certain_prep_meas_cutting_2q_layer_n(prep_qc,qubits_final_index_list,qubits_initial_index_list,prep_state,obs_meas,obs_final)
                                qc_list.append(circ)
                            
    new_qc_list = transpile(qc_list, backend, layout_method='trivial',optimization_level=3)
    #print(len(new_qc_list))
    return new_qc_list