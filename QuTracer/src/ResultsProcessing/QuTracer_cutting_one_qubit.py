#author: Peiyi Li (pli11@ncsu.edu)
###################################################################################
import numpy as np

#obtain density matrix of intial state
initial_rho_1qubit=np.array([[1,0,], [0,0,]])

# Define the Hadamard gate
Hgates = (1/np.sqrt(2)) * np.array([[1, 1], 
                               [1, -1]])
# pauli I matrix
observable_i = np.array([[1, 0 ],
                       [0, 1]])
# pauli Z matrix
observable_z = np.array([[1, 0, ],
                       [0, -1]])
# pauli X matrix
observable_x = np.array([[0, 1, ],
                       [1, 0]])
# pauli Y matrix
observable_y = np.array([[0, -1j, ],
                       [1j, 0]])
def obtain_h_gate():
    return Hgates

def obtain_sdg_gate():
    matrix=[[1,0], [0,-1j]]
    return matrix

def obtain_matrix_x():
    matrix= [[0,1], [1,0]]
    return matrix

def obtain_matrix_y():
    matrix= [[0, -1j],[1j, 0]]
    return matrix

def obtain_matrix_z():
    matrix= [[1,0],[0,-1]]
    return matrix

def obtain_matrix_i():
    matrix= [[1,0],[0,1]]
    return matrix

def matrix_multiply(A, B):
    # Get the number of rows and columns for matrices A and B
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    # Ensure the matrices can be multiplied
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Perform the matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return np.array(result)

matrix_list = [obtain_matrix_i(),obtain_matrix_x(),obtain_matrix_y(),obtain_matrix_z()]
pauli_list = ['I','X','Y','Z']

#calculate trace for different obs
def obtain_trace_for_4_obs(U, rho=initial_rho_1qubit):
    # Calculate the density matrix after applying the quantum gate
    if hasattr(U, 'any'):
        rho_final = U @ rho @ U.conj().T
    elif U is None:
        rho_final = rho

    tr_list=[]
    for obs in matrix_list:
        tr = np.trace(np.dot(obs, rho_final)).real 
        tr_list.append(tr)
    tr_dict = {'I':tr_list[0],'X':tr_list[1],'Y':tr_list[2],'Z':tr_list[3]}
  
    return tr_dict

#calculate density matrix based on trace
def obtain_density_matrix_from_trace(trace_dict):
    pauli_dict = {'I':obtain_matrix_i(),'X':obtain_matrix_x(),'Y':obtain_matrix_y(),'Z':obtain_matrix_z()}
    
    matrix = np.multiply(pauli_dict['I'],trace_dict['I']*0.5)
    
    for key in ['X','Y','Z']:
        inter_result=np.multiply(pauli_dict[key],trace_dict[key]*0.5)
        matrix=np.add(matrix,inter_result)
    return  matrix 

#calculate output distribution for different obs
def obtain_output_dist_for_4_obs( U,rho=initial_rho_1qubit):
    
    # Calculate the density matrix after applying the quantum gate
    if hasattr(U, 'any'):
        rho_final = U @ rho @ U.conj().T
    elif U is None:
        rho_final = rho
        
    output_dist={}
    for obs in pauli_list:
        if obs=='X':
            transfer_U = obtain_h_gate()
        elif obs=='Y':
            transfer_U = obtain_h_gate() @ obtain_sdg_gate() 
        else:
            transfer_U = obtain_matrix_i()
        transfer_U= np.array(transfer_U) 
        results = transfer_U @ rho_final @ transfer_U.conj().T
        
        result_dict={}
        result_dict['0']=results[0][0].real 
        result_dict['1']=results[1][1].real
        output_dist[obs]=result_dict
  
    return output_dist
##################################################################################### 
def obtain_ry_matrix(theta):
    """
    Returns the matrix representation of the RY gate for a given angle theta.
    
    Parameters:
    - theta (float): The rotation angle.
    
    Returns:
    - numpy.ndarray: The 2x2 matrix representation of the RY gate.
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])
def obtain_rx_matrix(theta):
    """
    Returns the matrix representation of the RY gate for a given angle theta.
    
    Parameters:
    - theta (float): The rotation angle.
    
    Returns:
    - numpy.ndarray: The 2x2 matrix representation of the RY gate.
    """
    return np.array([
        [np.cos(theta / 2), -1j*np.sin(theta / 2)],
        [-1j*np.sin(theta / 2), np.cos(theta / 2)]
    ])

def obtain_rotation_gate_matrix(gate_name, gate_params):
    if gate_name=='rx':
        gate_matrix=obtain_rx_matrix(gate_params)
    elif gate_name=='ry':
        gate_matrix=obtain_ry_matrix(gate_params)
    
    return gate_matrix

#obtain meas_trace_dict after applying local gates
def obtain_meas_trace_dict(gate_name, gate_params=None, initial_rho = initial_rho_1qubit):
    gate_matrix = observable_i
    
    if type(gate_name)==list:
        for gate in gate_name:
            #to do : include more type of gate
            if gate == 'h':
                single_gate_matrix = obtain_h_gate()
            elif (gate == 'x') :
                single_gate_matrix = obtain_matrix_x()
            elif (gate == 'rx') or (gate == 'ry') :
                single_gate_matrix = obtain_rotation_gate_matrix(gate, gate_params)
            gate_matrix = matrix_multiply(single_gate_matrix, gate_matrix)
            
    else:
        if gate_name == 'h':
            gate_matrix = obtain_h_gate()
        elif (gate_name == 'rx') or (gate_name == 'ry') :
            gate_matrix = obtain_rotation_gate_matrix(gate_name, gate_params)
        elif gate_name == None:
            gate_matrix = None
    
    meas_trace_dict = obtain_trace_for_4_obs(gate_matrix, rho = initial_rho)
    
    return meas_trace_dict

#obtain output distribution after applying local gates
def obtain_meas_output_dist(gate_name, gate_params=None, initial_rho = initial_rho_1qubit):
    gate_matrix = observable_i
    
    if type(gate_name)==list:
        for gate in gate_name:
            #to do : include more type of gate
            if gate == 'h':
                single_gate_matrix = obtain_h_gate()
            elif (gate == 'x') :
                single_gate_matrix = obtain_matrix_x()
            gate_matrix = matrix_multiply(single_gate_matrix, gate_matrix)
            
    else:
        if gate_name == 'h':
            gate_matrix = obtain_h_gate()
        elif (gate_name == 'rx') or (gate_name == 'ry') :
            gate_matrix = obtain_rotation_gate_matrix(gate_name, gate_params)
        elif gate_name == None:
            gate_matrix = None
            
    output_dist = obtain_output_dist_for_4_obs(gate_matrix, rho = initial_rho)
        
    return output_dist
###################################################################################
def replace_obs(obs):
    new_obs=''
    for pauli in obs:
        if pauli == 'I':
            new_obs = new_obs + 'Z'
        else:
            new_obs = new_obs + pauli
    return new_obs

import itertools
#operations for dictionary
def update_dict(d, length):
    # create a list of all possible binary keys of given length
    keys = [''.join(map(str, key)) for key in itertools.product([0, 1], repeat=length)]

    # loop through the keys
    for key in keys:
        # if the key is not in the dict, add it with value 0
        if key not in d:
            d[key] = 0

    # Calculate the sum of all values
    total = sum(d.values())

    # If total is not zero, normalize the values
    if total != 0:
        for key in d:
            d[key] /= total

    # return the updated dict
    return d

def calculate_trace(counts,obs):
    new_counts=update_dict(counts,1)
    if (obs=='I'):
        tr=new_counts['0']+new_counts['1']
    else:
        tr=new_counts['0']-new_counts['1'] 
    return tr
###################################################################################
obs_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}

def obtain_counts_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    counts_dict={}
    
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
            
    #when obs_list just has one observable, check if already done further optimizaiton
    if len(obs_list)==1:
        obs = obs_list[0]
        #to do: should check if done further optimization
        further_opt=True
        
        if further_opt:
            #for term1:
            key=check_layer_index + '-' + check_qubits + '-' + 'term1' + '-' + obs
            counts_dict[key]=counts_list[counts_index]
            counts_index+=1
            #for term4:
            key=check_layer_index + '-' + check_qubits + '-' + 'term4' + '-' + obs
            counts_dict[key]=counts_list[counts_index]
            counts_index+=1
            #for calculating remaining terms:
            for prep_state in prep_state_list:
                key=check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_dict[obs]
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1
            for prep_state in prep_state_list:
                key=check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_dict['I']
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1  
                
    return counts_dict, counts_index

def complete_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    trace_dict={}
    
    #obtain counts_dict
    counts_dict, counts_index = obtain_counts_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits)
    
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    #obtain trace dict
    for obs in obs_list + ['I']:
        #deal with key contains term1 and term4
        #for term1:
        key=check_layer_index + '-' + check_qubits + '-' + 'term1' + '-' + obs
        if key in counts_dict.keys():
                counts = counts_dict[key]
                trace = calculate_trace(counts,obs)
                update_key=tuple([check_layer_index, check_qubits, 'term1', obs])
                trace_dict.update({update_key:trace})
                
        #for term4:
        key=check_layer_index + '-' + check_qubits + '-' + 'term4' + '-' + obs
        if key in counts_dict.keys():
                counts = counts_dict[key]
                trace = calculate_trace(counts,obs)
                update_key=tuple([check_layer_index, check_qubits, 'term4', obs])
                trace_dict.update({update_key:trace})
                
        #deal with other keys
        for prep_state in prep_state_list:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_dict[obs]
                counts = None
                if 'I' in obs_dict[obs]:
                    new_key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + replace_obs(obs_dict[obs])
                    counts = counts_dict[new_key]
                else:
                    counts = counts_dict[key]
                trace = calculate_trace(counts,obs_dict[obs])
                update_key=tuple([check_layer_index, check_qubits, prep_state, obs_dict[obs]])
                trace_dict.update({update_key:trace})
    return trace_dict, counts_index

def obtain_prep_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    #to do: need to consider the situation when not using further optimizaiton for the first layer
    updated_trace_dict = {}
    
    #obtain trace_dict
    trace_dict, counts_index = complete_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits)
    #print(trace_dict)
    #print(counts_index)
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
        
    for obs in obs_list + ['I']:
        updated_trace_dict[(check_layer_index, check_qubits, 'I', obs_dict[obs])] = trace_dict[(check_layer_index, check_qubits, 'Z_p', obs_dict[obs])] + trace_dict[(check_layer_index, check_qubits, 'Z_n', obs_dict[obs])] 
        updated_trace_dict[(check_layer_index, check_qubits, 'X', obs_dict[obs])] = 2*trace_dict[(check_layer_index, check_qubits, 'X_p', obs_dict[obs])] - trace_dict[(check_layer_index, check_qubits, 'Z_p', obs_dict[obs])] - trace_dict[(check_layer_index, check_qubits, 'Z_n', obs_dict[obs])] 
        updated_trace_dict[(check_layer_index, check_qubits, 'Y', obs_dict[obs])] = 2*trace_dict[(check_layer_index, check_qubits, 'Y_p', obs_dict[obs])] - trace_dict[(check_layer_index, check_qubits, 'Z_p', obs_dict[obs])] -trace_dict[(check_layer_index, check_qubits, 'Z_n', obs_dict[obs])] 
        updated_trace_dict[(check_layer_index, check_qubits, 'Z', obs_dict[obs])] = trace_dict[(check_layer_index, check_qubits, 'Z_p', obs_dict[obs])] - trace_dict[(check_layer_index, check_qubits, 'Z_n', obs_dict[obs])] 
        
        key = tuple([check_layer_index, check_qubits, 'term1', obs])
        if key in trace_dict.keys():
            updated_trace_dict[key] = trace_dict[key]
        key = tuple([check_layer_index, check_qubits, 'term4', obs])
        if key in trace_dict.keys():
            updated_trace_dict[key] = trace_dict[key]    
            
    return updated_trace_dict, counts_index

###########################################################################
def calculate_term1(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    #to do: the situation when there is no opt
    key = tuple([check_layer_index, check_qubits, 'term1', obs])
    
    term = prep_trace_dict[key] 
    
    return term

def calculate_term4(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    #to do: the situation when there is no opt
    key = tuple([check_layer_index, check_qubits, 'term4', obs])
    
    term = prep_trace_dict[key] 
    
    return term

def calculate_term2(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
        
    trace = 0
    
    term_pauli_coef_dict = {'X':1j,'Y':-1j,'Z':1,'I':1}
    term_obs_coef_dict = {'X':-1j,'Y':1j,'Z':1,'I':1}
    term_pauli_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}
    for pauli in ['X','Y','Z','I']:
        key=tuple([check_layer_index, check_qubits, term_pauli_dict[pauli], term_pauli_dict[obs]])
        trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term_pauli_coef_dict[pauli]
    trace = trace/2*term_obs_coef_dict[obs]  
    
    return trace

def calculate_term3(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
        
    trace = 0
    
    term_pauli_coef_dict = {'X':-1j,'Y':1j,'Z':1,'I':1}
    term_obs_coef_dict = {'X':1j,'Y':-1j,'Z':1,'I':1}
    term_pauli_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}
    for pauli in ['X','Y','Z','I']:
        key=tuple([check_layer_index, check_qubits, term_pauli_dict[pauli], term_pauli_dict[obs]])
        trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term_pauli_coef_dict[pauli]
    trace = trace/2*term_obs_coef_dict[obs]  
    
    return trace
########################################################################################
def calculate_trace_pcs_for_certain_obs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs):#,value=None):
    trace = 0
    
    #if value != None:
    #    trace = value
    if obs == 'I':
        trace = 1
    elif obs == 'Z':
        trace = meas_trace_dict['Z']
    else:
        term1 = calculate_term1(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs)
        term2 = calculate_term2(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs)
        term3 = calculate_term3(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs)
        term4 = calculate_term4(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,obs)
        
        term2_denominator = calculate_term2(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'I')
        term3_denominator = calculate_term3(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'I')
        
        trace = (term1+term2+term3+term4)/(2+term2_denominator+term3_denominator)
        
    return trace
###########################################################################################    
def obtain_density_matrix_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,value_x=None,value_y=None,value_z=None):
    trace_dict = {'X':0,'Y':0,'Z':0,'I':0}
    
    trace_dict['I'] = calculate_trace_pcs_for_certain_obs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'I')
    
    if value_x != None:
        trace_dict['X'] = value_x
    else:
        trace_dict['X'] = calculate_trace_pcs_for_certain_obs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'X')
     
    if value_y != None:
        trace_dict['Y'] = value_y
    else:
        trace_dict['Y'] = calculate_trace_pcs_for_certain_obs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'Y')
        
    if value_z != None:
        trace_dict['Z'] = value_z
    else:
        trace_dict['Z'] = calculate_trace_pcs_for_certain_obs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,'Z')    
    
    
    #print(trace_dict)
    
    density_matrix = obtain_density_matrix_from_trace(trace_dict)
    
    return density_matrix
####################################################################################
#below functions are related to results processing for circuit layer n (n > 0)
####################################################################################
X_eigen_states = ['X_p', 'X_n']
Y_eigen_states = ['Y_p', 'Y_n']
Z_eigen_states = ['Z_p', 'Z_n']
I_eigen_states = ['Z_p', 'Z_n']

dict_for_create_circuit_term_1_4={'X':[X_eigen_states], 'Y':[Y_eigen_states]}
dict_for_create_circuit_term_2_3={'X':[Y_eigen_states], 'Y':[X_eigen_states]}
dict_for_create_circuit_term_common={'Z':[Z_eigen_states]}

def obtain_mitigated_info(k,obs_meas,mitigated_output_dist):
    mitigated_dist=mitigated_output_dist[obs_meas]
    p=mitigated_dist[k[1:2]]
    return p

def obtain_noisy_info(k,small_counts_dict):
    p=0
    sum=0
    for key,value in small_counts_dict.items():
        sum += value
        if key[1:2] == k[1:2]:
             p += value
    return p/sum

#obtain counts on current layer (without info from previous layer)
def obtain_counts_dict_layer_n(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
        
    counts_dict={}
    #counts_index=0
    #create circuits for term 1 and 4
    for obs_final in obs_list:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_1_4.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_final + obs_meas
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1
    #create circuits for term 2 and 3
    #to do: need to process the case when all three obs are measured
    for obs_final in obs_list + ['I']:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_2_3.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_dict[obs_final] + obs_meas
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1
    #create circuits for common part
    for obs_final in ['X','Y','Z']:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_common.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_final + obs_meas
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1
                        
    return counts_dict, counts_index
#obtain counts on current layer (based on the info from previous layer)
def obtain_counts_dict_based_on_previous_layer_info(mitigated_output_dist,counts_list,counts_index,prep_state_list,obs_list,check_layer_index,check_qubits):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    counts_dict, counts_index = obtain_counts_dict_layer_n(counts_list,counts_index,prep_state_list,obs_list,check_layer_index,check_qubits)

    new_counts_dict = {}                        
    
    #for term 1 and 4
    for obs_final in obs_list:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_1_4.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_final + obs_meas
                small_counts_dict=counts_dict[key]
                new_small_counts_dict={}
                for k,v in small_counts_dict.items():
                    new_v = v * obtain_mitigated_info(k,obs_meas,mitigated_output_dist) / obtain_noisy_info(k,small_counts_dict)
                    new_small_counts_dict[k] = new_v
                new_counts_dict[key] = new_small_counts_dict
    #for term 2 and 3
    #to do: need to process the case when all three obs are measured
    for obs_final in obs_list + ['I']:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_2_3.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_dict[obs_final] + obs_meas
                small_counts_dict=counts_dict[key]
                new_small_counts_dict={}
                for k,v in small_counts_dict.items():
                    new_v = v * obtain_mitigated_info(k,obs_meas,mitigated_output_dist) / obtain_noisy_info(k,small_counts_dict)
                    new_small_counts_dict[k] = new_v
                new_counts_dict[key] = new_small_counts_dict
    #for common part
    for obs_final in ['X','Y','Z']:
        for obs_meas,prep_state_list in dict_for_create_circuit_term_common.items():
            for prep_state in prep_state_list[0]:
                key = check_layer_index + '-' + check_qubits + '-' + prep_state + '-' + obs_final + obs_meas
                small_counts_dict=counts_dict[key]
                new_small_counts_dict={}
                for k,v in small_counts_dict.items():
                    new_v = v * obtain_mitigated_info(k,obs_meas,mitigated_output_dist) / obtain_noisy_info(k,small_counts_dict)
                    new_small_counts_dict[k] = new_v
                new_counts_dict[key] = new_small_counts_dict
     
    return new_counts_dict, counts_index
##########################################################################
def obtain_eigenvalue(bitstring, obs):
    result=1
    for i in range(len(bitstring)):
        if ( obs[i] != 'I' ) and ( bitstring[i] == '1' ):
            result = result*(-1)
            
    return result
    
def obtain_trace(counts,obs):
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        
        obj = obtain_eigenvalue(bitstring, obs)
        avg += obj * count
        sum_count += count
        
    return avg/sum_count

def split_string(input_string, delimiter='-'):
    # Split the input string using the specified delimiter
    return input_string.split(delimiter)

def obtain_obs_list(trace_dict):
    obs_list=[]
    for key in trace_dict.keys():
        obs = key[3]
        if obs not in obs_list:
            obs_list.append(obs)
    return obs_list

#replace z with i
def replace_obs_with_i(obs):
    new_obs=''
    for pauli in obs:
        if pauli == 'Z':
            new_obs = new_obs + 'I'
        else:
            new_obs = new_obs + pauli
    return new_obs

###################################################################
def complete_trace_dict_layer_n(counts_dict, prep_state_list,obs_list,check_layer_index,check_qubits):
    trace_dict={}
    
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
        
    #to do: complete the trace dict  
    #obtain trace dict
    for key in counts_dict:
        counts = counts_dict[key]
        split_key = split_string(key)
        trace = obtain_trace(counts,split_key[3])
        update_key=tuple(split_key)
        trace_dict.update({update_key:trace})
        
        #complete the trace dict
        if split_key[3]=='ZZ':
            #zi
            new_obs = 'ZI'
            trace = obtain_trace(counts,new_obs)
            update_key=tuple([split_key[0],split_key[1],split_key[2],new_obs])
            trace_dict.update({update_key:trace})
            #iz
            new_obs = 'IZ'
            trace = obtain_trace(counts,new_obs)
            update_key=tuple([split_key[0],split_key[1],split_key[2],new_obs])
            trace_dict.update({update_key:trace})
            #ii
            new_obs = 'II'
            trace = obtain_trace(counts,new_obs)
            update_key=tuple([split_key[0],split_key[1],split_key[2],new_obs])
            trace_dict.update({update_key:trace})
        elif 'Z' in split_key[3]:
            new_obs = replace_obs_with_i(split_key[3])
            trace = obtain_trace(counts,new_obs)
            update_key=tuple([split_key[0],split_key[1],split_key[2],new_obs])
            trace_dict.update({update_key:trace})
            
    return trace_dict
def obtain_meas_prep_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits):
    updated_trace_dict = {}
    
    #obtain trace_dict
    trace_dict = complete_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits)
    #print(trace_dict)
    
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    obs_list = obtain_obs_list(trace_dict)
    #print('obs_list:',obs_list)
    for obs in obs_list:
            key = tuple([check_layer_index, check_qubits, 'X_p', obs])
            if key in trace_dict.keys():
                updated_trace_dict[(check_layer_index, check_qubits, 'X', obs)] = trace_dict[(check_layer_index, check_qubits, 'X_p', obs)] - trace_dict[(check_layer_index, check_qubits, 'X_n', obs)]  
            key = tuple([check_layer_index, check_qubits, 'Y_p', obs])
            if key in trace_dict.keys():
                updated_trace_dict[(check_layer_index, check_qubits, 'Y', obs)] = trace_dict[(check_layer_index, check_qubits, 'Y_p', obs)] - trace_dict[(check_layer_index, check_qubits, 'Y_n', obs)] 
            key = tuple([check_layer_index, check_qubits, 'Z_p', obs])
            if key in trace_dict.keys():  
                updated_trace_dict[(check_layer_index, check_qubits, 'I', obs)] = trace_dict[(check_layer_index, check_qubits, 'Z_p', obs)] + trace_dict[(check_layer_index, check_qubits, 'Z_n', obs)] 
                updated_trace_dict[(check_layer_index, check_qubits, 'Z', obs)] = trace_dict[(check_layer_index, check_qubits, 'Z_p', obs)] - trace_dict[(check_layer_index, check_qubits, 'Z_n', obs)]
        
    return updated_trace_dict
########################################################################
def combine_meas_prep_trace_results_of_certain_obs_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    trace=0
    for pauli in ['I','X','Y','Z']:
        key=tuple([check_layer_index, check_qubits, pauli ,obs+pauli])
        trace+=trace_dict[key]
    trace = trace/2
    return trace

def calculate_term1_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    trace=0
    for pauli in ['I','X','Y','Z']:
        key=tuple([check_layer_index, check_qubits, pauli ,obs+pauli])
        trace+=trace_dict[key]
    trace = trace/2
    return trace

def calculate_term2_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    trace=0
    
    term_pauli_coef_dict = {'X':1j,'Y':-1j,'Z':1,'I':1}
    term_obs_coef_dict = {'X':-1j,'Y':1j,'Z':1,'I':1}
    term_pauli_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}
    
    for pauli in ['I','X','Y','Z']:
        key=tuple([check_layer_index, check_qubits, term_pauli_dict[pauli] ,term_pauli_dict[obs]+pauli])
        trace+=trace_dict[key]*term_pauli_coef_dict[pauli]
        
    trace = trace/2*term_obs_coef_dict[obs]
    
    return trace

def calculate_term3_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    trace=0
    
    term_pauli_coef_dict = {'X':-1j,'Y':1j,'Z':1,'I':1}
    term_obs_coef_dict = {'X':1j,'Y':-1j,'Z':1,'I':1}
    term_pauli_dict = {'X':'Y', 'Y':'X', 'Z':'I', 'I':'Z'}
    
    for pauli in ['I','X','Y','Z']:
        key=tuple([check_layer_index, check_qubits, term_pauli_dict[pauli] ,term_pauli_dict[obs]+pauli])
        trace+=trace_dict[key]*term_pauli_coef_dict[pauli]
        
    trace = trace/2*term_obs_coef_dict[obs]
    
    return trace

def calculate_term4_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    if type(check_layer_index) == int:
        check_layer_index = str(check_layer_index)
    if type(check_qubits) == int:
        check_qubits = str(check_qubits)
    
    trace=0
    
    term_pauli_coef_dict = {'X':-1,'Y':-1,'Z':1,'I':1}
    term_obs_coef_dict = {'X':-1,'Y':-1,'Z':1,'I':1}
    term_pauli_dict = {'X':'X', 'Y':'Y', 'Z':'Z', 'I':'I'}
    
    for pauli in ['I','X','Y','Z']:
        key=tuple([check_layer_index, check_qubits, term_pauli_dict[pauli] ,term_pauli_dict[obs]+pauli])
        trace+=trace_dict[key]*term_pauli_coef_dict[pauli]
        
    trace = trace/2*term_obs_coef_dict[obs]
    
    return trace

def calculate_trace_pcs_for_certain_obs_layer_n(check_layer_index,check_qubits,trace_dict,obs):
    trace = 0
    
    if obs == 'I':
        trace = 1

    else:
        term1 = calculate_term1_layer_n(check_layer_index,check_qubits,trace_dict,obs)
        term2 = calculate_term2_layer_n(check_layer_index,check_qubits,trace_dict,obs)
        term3 = calculate_term3_layer_n(check_layer_index,check_qubits,trace_dict,obs)
        term4 = calculate_term4_layer_n(check_layer_index,check_qubits,trace_dict,obs)
        
        term2_denominator = calculate_term2_layer_n(check_layer_index,check_qubits,trace_dict,'I')
        term3_denominator = calculate_term3_layer_n(check_layer_index,check_qubits,trace_dict,'I')
        trace = (term1+term2+term3+term4)/(2+term2_denominator+term3_denominator)
        
    return trace
#######################################################################
def obtain_density_matrix_pcs_layer_n(check_layer_index,check_qubits,trace_dict,value_x=None,value_y=None,value_z=None):
    new_trace_dict = {'X':0,'Y':0,'Z':0,'I':0}
    
    new_trace_dict['I'] = 1
    
    if value_x != None:
        new_trace_dict['X'] = value_x
    else:
        new_trace_dict['X'] = calculate_trace_pcs_for_certain_obs_layer_n(check_layer_index,check_qubits,trace_dict,'X')
     
    if value_y != None:
        new_trace_dict['Y'] = value_y
    else:
        new_trace_dict['Y'] = calculate_trace_pcs_for_certain_obs_layer_n(check_layer_index,check_qubits,trace_dict,'Y')
        
    if value_z != None:
        new_trace_dict['Z'] = value_z
    else:
        new_trace_dict['Z'] = calculate_trace_pcs_for_certain_obs_layer_n(check_layer_index,check_qubits,trace_dict,'Z')    
    
    density_matrix = obtain_density_matrix_from_trace(new_trace_dict)
    
    return density_matrix