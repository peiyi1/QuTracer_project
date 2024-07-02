#author: Peiyi Li (pli11@ncsu.edu)
###################################################################################
import numpy as np

#obtain density matrix of intial state
initial_rho_2qubit=np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])

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

def obtain_matrix_ii():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_i())
    return matrix

def obtain_matrix_ix():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_x())
    return matrix

def obtain_matrix_iy():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_y())
    return matrix

def obtain_matrix_iz():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_z())
    return matrix

def obtain_matrix_xi():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_i())
    return matrix
def obtain_matrix_xx():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_x())
    return matrix
def obtain_matrix_xy():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_y())
    return matrix
def obtain_matrix_xz():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_z())
    return matrix

def obtain_matrix_yi():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_i())
    return matrix
def obtain_matrix_yx():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_x())
    return matrix
def obtain_matrix_yy():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_y())
    return matrix
def obtain_matrix_yz():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_z())
    return matrix

def obtain_matrix_zi():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_i())
    return matrix
def obtain_matrix_zx():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_x())
    return matrix
def obtain_matrix_zy():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_y())
    return matrix
def obtain_matrix_zz():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_z())
    return matrix

matrix_list = [obtain_matrix_ii(),obtain_matrix_ix(),obtain_matrix_iy(),obtain_matrix_iz(),obtain_matrix_xi(),obtain_matrix_xx(),obtain_matrix_xy(),obtain_matrix_xz(),obtain_matrix_yi(),obtain_matrix_yx(),obtain_matrix_yy(),obtain_matrix_yz(),obtain_matrix_zi(),obtain_matrix_zx(),obtain_matrix_zy(),obtain_matrix_zz()]
pauli_list = ['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']

#calculate trace for different obs
def obtain_trace_for_16_obs(U, rho=initial_rho_2qubit):
    
    # Calculate the density matrix after applying the quantum gate
    rho_final = U @ rho @ U.conj().T
    
    tr_list=[]
    for obs in matrix_list:
        tr = np.trace(np.dot(obs, rho_final)).real 
        tr_list.append(tr)
    tr_dict = {'II':tr_list[0],'IX':tr_list[1],'IY':tr_list[2],'IZ':tr_list[3],'XI':tr_list[4],'XX':tr_list[5],'XY':tr_list[6],'XZ':tr_list[7],'YI':tr_list[8],'YX':tr_list[9],'YY':tr_list[10],'YZ':tr_list[11],'ZI':tr_list[12],'ZX':tr_list[13],'ZY':tr_list[14],'ZZ':tr_list[15]}
  
    return tr_dict

#calculate density matrix based on trace
def obtain_density_matrix_from_trace(trace_dict):
    pauli_dict={'II':obtain_matrix_ii(),'IX':obtain_matrix_ix(),'IY':obtain_matrix_iy(),'IZ':obtain_matrix_iz(),'XI':obtain_matrix_xi(),'XX':obtain_matrix_xx(),'XY':obtain_matrix_xy(),'XZ':obtain_matrix_xz(),'YI':obtain_matrix_yi(),'YX':obtain_matrix_yx(),'YY':obtain_matrix_yy(),'YZ':obtain_matrix_yz(),'ZI':obtain_matrix_zi(),'ZX':obtain_matrix_zx(),'ZY':obtain_matrix_zy(),'ZZ':obtain_matrix_zz()}
    
    matrix=np.multiply(pauli_dict['II'],trace_dict['II']*0.25)
    
    for key in ['IX','IY','IZ','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
        inter_result=np.multiply(pauli_dict[key],trace_dict[key]*0.25)
        matrix=np.add(matrix,inter_result)
    return  matrix 

#calculate output distribution for different obs
def obtain_output_dist_for_16_obs( U,rho=initial_rho_2qubit):
    
    # Calculate the density matrix after applying the quantum gate
    rho_final = U @ rho @ U.conj().T
    
    output_dist={}
    for obs in pauli_list:
        if obs[0]=='X':
            transfer_U_0 = obtain_h_gate()
        elif obs[0]=='Y':
            transfer_U_0 = obtain_h_gate() @ obtain_sdg_gate() 
        else:
            transfer_U_0 = obtain_matrix_i()
            
        if obs[1]=='X':
            transfer_U_1 = obtain_h_gate()
        elif obs[1]=='Y':
            transfer_U_1 = obtain_h_gate() @ obtain_sdg_gate() 
        else:
            transfer_U_1 = obtain_matrix_i()
            
        transfer_U = np.kron(transfer_U_0,transfer_U_1)
        results = transfer_U @ rho_final @ transfer_U.conj().T
        
        result_dict={}
        result_dict['00']=results[0][0].real 
        result_dict['01']=results[1][1].real
        result_dict['10']=results[2][2].real
        result_dict['11']=results[3][3].real
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
def obtain_meas_trace_dict(gate_name, gate_params=None, initial_rho = initial_rho_2qubit):
    meas_trace_dict = None
    if gate_name == 'h':
        h_gate = obtain_h_gate()
        h_gate_tensor = np.kron(h_gate,h_gate)
        meas_trace_dict = obtain_trace_for_16_obs(h_gate_tensor, rho = initial_rho)
    elif gate_name == 'rx':
        gate_matrix = obtain_rotation_gate_matrix(gate_name, gate_params)
        gate_matrix_tensor = np.kron(gate_matrix,gate_matrix)
        meas_trace_dict = obtain_trace_for_16_obs(gate_matrix_tensor, rho = initial_rho)
    return meas_trace_dict

#obtain output distribution after applying local gates
def obtain_meas_output_dist(gate_name, gate_params=None, initial_rho = initial_rho_2qubit):
    output_dist = None
    if gate_name == 'h':
        h_gate = obtain_h_gate()
        h_gate_tensor = np.kron(h_gate,h_gate)
        output_dist = obtain_output_dist_for_16_obs(h_gate_tensor, rho = initial_rho)
    elif gate_name == 'rx':
        gate_matrix = obtain_rotation_gate_matrix(gate_name, gate_params)
        gate_matrix_tensor = np.kron(gate_matrix,gate_matrix)
        output_dist = obtain_output_dist_for_16_obs(gate_matrix_tensor, rho = initial_rho)
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
    new_counts=update_dict(counts,2)
    if (obs=='II'):
        tr=new_counts['00']+new_counts['01']+new_counts['10']+new_counts['11'] 
    elif (obs[-1]=='I'):
        tr=new_counts['00']+new_counts['01']-new_counts['10']-new_counts['11'] 
    elif (obs[-2]=='I'):
        tr=new_counts['00']-new_counts['01']+new_counts['10']-new_counts['11'] 
    else:
        tr=new_counts['00']-new_counts['01']-new_counts['10']+new_counts['11'] 
    return tr
###################################################################################
def obtain_counts_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    counts_dict={}
    #counts_index=0
    for obs in obs_list:
        for prep_state_1 in prep_state_list:
            for prep_state_0 in prep_state_list:
                key=str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + obs
                counts_dict[key]=counts_list[counts_index]
                counts_index+=1
    return counts_dict, counts_index

def complete_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    trace_dict={}
    
    #obtain counts_dict
    counts_dict, counts_index = obtain_counts_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits)
    
    #obtain trace dict
    for obs in obs_list + ['II','XI','YI','ZI','IX','IY','IZ',]:
        for prep_state_1 in prep_state_list:
            for prep_state_0 in prep_state_list:
                key = str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + obs
                counts = None
                if 'I' in obs:
                    new_key = str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + replace_obs(obs)
                    counts = counts_dict[new_key]
                else:
                    counts = counts_dict[key]
                trace = calculate_trace(counts,obs)
                update_key=tuple([str(check_layer_index), check_qubits, prep_state_1, prep_state_0, obs])
                trace_dict.update({update_key:trace})
    return trace_dict, counts_index

def obtain_prep_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    updated_trace_dict = {}
    
    #obtain trace_dict
    trace_dict, counts_index = complete_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits)
    
    for obs in obs_list + ['II','XI','YI','ZI','IX','IY','IZ',]:
        updated_trace_dict[(str(check_layer_index), check_qubits, 'II', obs)] = trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)] 
        updated_trace_dict[(str(check_layer_index), check_qubits, 'ZI', obs)] = trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)] 
        updated_trace_dict[(str(check_layer_index), check_qubits, 'IZ', obs)] = trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)] 
        updated_trace_dict[(str(check_layer_index), check_qubits, 'ZZ', obs)] = trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        
        updated_trace_dict[(str(check_layer_index), check_qubits, 'XI', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_p', obs)] + 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'IX', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'X_p', obs)] + 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'X_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'YI', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_p', obs)] + 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'IY', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Y_p', obs)] + 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Y_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        
        updated_trace_dict[(str(check_layer_index), check_qubits, 'XX', obs)] = 4*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'X_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_n', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'X_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'X_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'YY', obs)] = 4*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Y_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_n', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Y_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Y_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'YX', obs)] = 4*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'X_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_n', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'X_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'X_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'XY', obs)] = 4*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Y_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_n', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Y_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Y_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        
        updated_trace_dict[(str(check_layer_index), check_qubits, 'ZX', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'X_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'X_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'XZ', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'X_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'ZY', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Y_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Y_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        updated_trace_dict[(str(check_layer_index), check_qubits, 'YZ', obs)] = 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_p', obs)] - 2*trace_dict[(str(check_layer_index), check_qubits, 'Y_p', 'Z_n', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_p', obs)] - trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_p', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_p', 'Z_n', obs)] + trace_dict[(str(check_layer_index), check_qubits, 'Z_n', 'Z_n', obs)]
        
    return updated_trace_dict, counts_index

###########################################################################
def combine_meas_prep_trace_results(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]
        inter_results[obs].append(trace/4)
    return inter_results
def combine_meas_prep_trace_results_for_pcs_part1(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]
        inter_results[obs].append(trace/4)
        
    term_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':-1,'XX':1,'XY':1,'XZ':-1,'YI':-1,'YX':1,'YY':1,'YZ':-1,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term_coef_dict[pauli]
        inter_results[obs].append(trace/4*term_coef_dict[obs])
    
    term_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':-1,'XX':-1,'XY':-1,'XZ':-1,'YI':-1,'YX':-1,'YY':-1,'YZ':-1,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term_coef_dict[pauli]
        inter_results[obs].append(trace/4*term_coef_dict[obs])
    
    term_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':1,'XX':-1,'XY':-1,'XZ':1,'YI':1,'YX':-1,'YY':-1,'YZ':1,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term_coef_dict[pauli]
        inter_results[obs].append(trace/4*term_coef_dict[obs])
    
    return inter_results

def combine_meas_prep_trace_results_for_pcs_part2(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':1j,'XX':-1,'XY':1,'XZ':1j,'YI':-1j,'YX':1,'YY':-1,'YZ':-1j,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':-1j,'XX':-1,'XY':1,'XZ':-1j,'YI':1j,'YX':1,'YY':-1,'YZ':1j,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    term_2_3_pauli_dict={'II':'ZZ','IX':'ZY','IY':'ZX','IZ':'ZI','XI':'YZ','XX':'YY','XY':'YX','XZ':'YI','YI':'XZ','YX':'XY','YY':'XX','YZ':'XI','ZI':'IZ','ZX':'IY','ZY':'IX','ZZ':'II'}
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':-1j,'XX':1,'XY':-1,'XZ':-1j,'YI':1j,'YX':-1,'YY':1,'YZ':1j,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':1j,'XX':1,'XY':-1,'XZ':1j,'YI':-1j,'YX':-1,'YY':1,'YZ':-1j,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    return inter_results

def combine_meas_prep_trace_results_for_pcs_part3(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':1,'XX':1j,'XY':-1j,'XZ':1,'YI':1,'YX':1j,'YY':-1j,'YZ':1,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':1,'XX':-1j,'XY':1j,'XZ':1,'YI':1,'YX':-1j,'YY':1j,'YZ':1,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    term_2_3_pauli_dict={'II':'IZ','IX':'IY','IY':'IX','IZ':'II','XI':'XZ','XX':'XY','XY':'XX','XZ':'XI','YI':'YZ','YX':'YY','YY':'YX','YZ':'YI','ZI':'ZZ','ZX':'ZY','ZY':'ZX','ZZ':'ZI'}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':-1,'XX':-1j,'XY':1j,'XZ':-1,'YI':-1,'YX':-1j,'YY':1j,'YZ':-1,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':-1,'XX':1j,'XY':-1j,'XZ':-1,'YI':-1,'YX':1j,'YY':-1j,'YZ':-1,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    return inter_results

def combine_meas_prep_trace_results_for_pcs_part4(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    
    term2_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':1j,'XX':1j,'XY':1j,'XZ':1j,'YI':-1j,'YX':-1j,'YY':-1j,'YZ':-1j,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':-1j,'XX':-1j,'XY':-1j,'XZ':-1j,'YI':1j,'YX':1j,'YY':1j,'YZ':1j,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    term_2_3_pauli_dict={'II':'ZI','IX':'ZX','IY':'ZY','IZ':'ZZ','XI':'YI','XX':'YX','XY':'YY','XZ':'YZ','YI':'XI','YX':'XX','YY':'XY','YZ':'XZ','ZI':'II','ZX':'IX','ZY':'IY','ZZ':'IZ'}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    term2_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':1j,'XX':-1j,'XY':-1j,'XZ':1j,'YI':-1j,'YX':1j,'YY':1j,'YZ':-1j,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':-1j,'XX':1j,'XY':1j,'XZ':-1j,'YI':1j,'YX':-1j,'YY':-1j,'YZ':1j,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]])
            trace+=meas_trace_dict[pauli]*prep_trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    
    return inter_results

def combine_meas_prep_trace_results_for_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    new_dict={}
    dict1=combine_meas_prep_trace_results_for_pcs_part1(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    dict2=combine_meas_prep_trace_results_for_pcs_part2(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    dict3=combine_meas_prep_trace_results_for_pcs_part3(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    dict4=combine_meas_prep_trace_results_for_pcs_part4(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    for key in dict1.keys():
        new_list=dict1[key]+dict2[key]+dict3[key]+dict4[key]
        new_dict[key]=new_list
        
    return new_dict

def obtain_trace_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict):
    #obtain combined results from prep_traces and meas_trace
    inter_results = combine_meas_prep_trace_results_for_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    
    #create trace dict
    trace ={}
    denominator=sum(i for i in inter_results['II'])
    
    for i,v_list in inter_results.items():
        trace[i]=sum(i for i in v_list)/denominator
        
    return trace    

def obtain_density_matrix_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict,value_x=None,value_y=None,value_z=None):
    #obtain combined results from prep_traces and meas_trace
    trace_dict = obtain_trace_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
    density_matrix = obtain_density_matrix_from_trace(trace_dict)
    
    return density_matrix
            
#################################################################################### 
#below functions are related to results processing for circuit layer n (n > 0)
#################################################################################### 

X_eigen_states = ['X_p', 'X_n']
Y_eigen_states = ['Y_p', 'Y_n']
Z_eigen_states = ['Z_p', 'Z_n']
I_eigen_states = ['Z_p', 'Z_n']

dict_for_create_circuit={'XX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'XY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'YX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'YY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'XZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'YZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'ZX':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'ZY':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'ZZ':[(Z_eigen_states,Z_eigen_states)],
                        }
  
def obtain_mitigated_info(k,obs_meas,mitigated_output_dist):
    mitigated_dist=mitigated_output_dist[obs_meas]
    p=mitigated_dist[k[2:4]]
    return p

def obtain_noisy_info(k,small_counts_dict):
    p=0
    sum=0
    for key,value in small_counts_dict.items():
        sum += value
        if key[2:4] == k[2:4]:
             p += value
    return p/sum

#obtain counts on current layer (without info from previous layer)
def obtain_counts_dict_layer_n(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits):
    counts_dict={}
    #counts_index=0
    for obs_final in obs_list:
            for obs_meas,prep_state_list in dict_for_create_circuit.items():
                for (prep_state_1_list,prep_state_0_list) in prep_state_list:
                    for prep_state_1 in prep_state_1_list:
                        for prep_state_0 in prep_state_0_list:
                            key=str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + obs_final + obs_meas
                            counts_dict[key]=counts_list[counts_index]
                            counts_index+=1
    return counts_dict, counts_index

#obtain counts on current layer (based on the info from previous layer)
def obtain_counts_dict_based_on_previous_layer_info(mitigated_output_dist,counts_list,counts_index,prep_state_list,obs_list,check_layer_index,check_qubits):
    counts_dict, counts_index = obtain_counts_dict_layer_n(counts_list,counts_index,prep_state_list,obs_list,check_layer_index,check_qubits)

    new_counts_dict={}
    for obs_final in obs_list:
            for obs_meas,prep_state_list in dict_for_create_circuit.items():
                for (prep_state_1_list,prep_state_0_list) in prep_state_list:
                    for prep_state_1 in prep_state_1_list:
                        for prep_state_0 in prep_state_0_list:
                            key=str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + obs_final + obs_meas
                            small_counts_dict=counts_dict[key]
                            new_small_counts_dict={}
                            for k,v in small_counts_dict.items():
                                new_v = v * obtain_mitigated_info(k,obs_meas,mitigated_output_dist) / obtain_noisy_info(k,small_counts_dict)
                                new_small_counts_dict[k] = new_v
                        
                            new_counts_dict[key] = new_small_counts_dict
                            
                        
    return new_counts_dict, counts_index
##################################################################################
complete_dict_for_create_circuit={'XX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'XY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'YX':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'YY':[(X_eigen_states,X_eigen_states),(Y_eigen_states,Y_eigen_states),(X_eigen_states,Y_eigen_states),(Y_eigen_states,X_eigen_states)],'XZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'YZ':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'ZX':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'ZY':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'ZZ':[(Z_eigen_states,Z_eigen_states)],'XI':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'YI':[(X_eigen_states,Z_eigen_states),(Y_eigen_states,Z_eigen_states)],'IX':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'IY':[(Z_eigen_states,X_eigen_states),(Z_eigen_states,Y_eigen_states)],'II':[(Z_eigen_states,Z_eigen_states)],'IZ':[(Z_eigen_states,Z_eigen_states)],'ZI':[(Z_eigen_states,Z_eigen_states)],}

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

def complete_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits):
    trace_dict={}
    
    #obtain trace dict
    for obs_final in obs_list + ['II','XI','YI','ZI','IX','IY','IZ',]:
            for obs_meas,prep_state_list in complete_dict_for_create_circuit.items():
                for (prep_state_1_list,prep_state_0_list) in prep_state_list:
                    for prep_state_1 in prep_state_1_list:
                        for prep_state_0 in prep_state_0_list:
                            key=str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + obs_final + obs_meas
                            
                            counts = None
                            if 'I' in obs_final + obs_meas:
                                new_key = str(check_layer_index) + '-' + check_qubits + '-' + prep_state_1 + '-' + prep_state_0 + '-' + replace_obs(obs_final + obs_meas)
                                counts = counts_dict[new_key]
                            else:
                                counts = counts_dict[key]
                            trace = obtain_trace(counts,obs_final + obs_meas)
                            update_key=tuple([str(check_layer_index), check_qubits, prep_state_1, prep_state_0, obs_final + obs_meas])
                            trace_dict.update({update_key:trace})
                
    return trace_dict

obs_transfer_dict={'II':['II','ZI','IZ','ZZ'],'IX':['IX','ZX','IY','ZY'],'IY':['IY','ZY','IX','ZX'],'IZ':['IZ','ZZ','II','ZI'],
                   'XI':['XI','YI','XZ','YZ'],'XX':['XX','YX','XY','YY'],'XY':['XY','YY','XX','YX'],'XZ':['XZ','YZ','XI','YI'],
                   'YI':['YI','XI','YZ','XZ'],'YX':['YX','XX','YY','XY'],'YY':['YY','XY','YX','XX'],'YZ':['YZ','XZ','YI','XI'],
                   'ZI':['ZI','II','ZZ','IZ'],'ZX':['ZX','IX','ZY','IY'],'ZY':['ZY','IY','ZX','IX'],'ZZ':['ZZ','IZ','ZI','II']}

def obtain_meas_prep_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits):
    value_dict = complete_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits)
    
    new_trace_dict={}
    for obs_final in obs_list+['II','XI','YI','ZI','IX','IY','IZ',]:
            for obs_meas,pauli_prep_list in obs_transfer_dict.items():
                for pauli_prep in pauli_prep_list:
                    obs = obs_final + obs_meas
                    if pauli_prep == 'II':
                        new_trace_dict[(str(check_layer_index), check_qubits,'II',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p','Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_p','Z_n',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_n',obs)]
                        
                    elif pauli_prep == 'XI':
                        new_trace_dict[(str(check_layer_index), check_qubits,'XI',obs)] = value_dict[(str(check_layer_index), check_qubits,'X_p', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'X_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_n', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_n', 'Z_n',obs)]
                        
                    elif pauli_prep == 'YI':
                        new_trace_dict[(str(check_layer_index), check_qubits,'YI',obs)] = value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Z_n',obs)]
                        
                    elif pauli_prep == 'ZI':
                        new_trace_dict[(str(check_layer_index), check_qubits,'ZI',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_n',obs)]
                        
                    elif pauli_prep == 'IX':
                        new_trace_dict[(str(check_layer_index), check_qubits,'IX',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'X_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'X_n',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'X_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'X_n',obs)]
                    elif pauli_prep == 'IY':
                        new_trace_dict[(str(check_layer_index), check_qubits,'IY',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Y_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Y_n',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Y_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Y_n',obs)]
                    elif pauli_prep == 'IZ':
                        new_trace_dict[(str(check_layer_index), check_qubits,'IZ',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_n',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_n',obs)]
                        
                    elif pauli_prep == 'XX':
                        new_trace_dict[(str(check_layer_index), check_qubits,'XX',obs)] = value_dict[(str(check_layer_index), check_qubits,'X_p', 'X_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_p', 'X_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_n', 'X_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'X_n', 'X_n',obs)]
                    elif pauli_prep == 'YY':
                        new_trace_dict[(str(check_layer_index), check_qubits,'YY',obs)] = value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Y_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Y_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Y_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Y_n',obs)]
                    elif pauli_prep == 'ZZ':
                        new_trace_dict[(str(check_layer_index), check_qubits,'ZZ',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Z_n',obs)]
                     
                    elif pauli_prep == 'XY':
                        new_trace_dict[(str(check_layer_index), check_qubits,'XY',obs)] = value_dict[(str(check_layer_index), check_qubits,'X_p', 'Y_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_p', 'Y_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_n', 'Y_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'X_n', 'Y_n',obs)]
                    elif pauli_prep == 'YX':
                        new_trace_dict[(str(check_layer_index), check_qubits,'YX',obs)] = value_dict[(str(check_layer_index), check_qubits,'Y_p', 'X_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_p', 'X_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_n', 'X_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Y_n', 'X_n',obs)]
                    elif pauli_prep == 'XZ':
                        new_trace_dict[(str(check_layer_index), check_qubits,'XZ',obs)] = value_dict[(str(check_layer_index), check_qubits,'X_p', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'X_n', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'X_n', 'Z_n',obs)]
                    elif pauli_prep == 'ZX':
                        new_trace_dict[(str(check_layer_index), check_qubits,'ZX',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'X_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'X_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'X_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'X_n',obs)]
                    elif pauli_prep == 'YZ':
                        new_trace_dict[(str(check_layer_index), check_qubits,'YZ',obs)] = value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Z_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_p', 'Z_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Z_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Y_n', 'Z_n',obs)]
                    elif pauli_prep == 'ZY':
                        new_trace_dict[(str(check_layer_index), check_qubits,'ZY',obs)] = value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Y_p',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_p', 'Y_n',obs)] - value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Y_p',obs)] + value_dict[(str(check_layer_index), check_qubits,'Z_n', 'Y_n',obs)]
                
    return new_trace_dict
##################################################################################
def combine_meas_prep_trace_results_layer_n(check_layer_index,check_qubits,trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs+pauli])
            trace+=trace_dict[key]
        inter_results[obs].append(trace/4)
    return inter_results

def combine_trace_results_for_pcs_part1(check_layer_index,check_qubits,trace_dict):
    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs+pauli])
            trace+=trace_dict[key]
        inter_results[obs].append(trace/4)
        
    term4_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':-1,'XX':1,'XY':1,'XZ':-1,'YI':-1,'YX':1,'YY':1,'YZ':-1,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs+pauli])
            trace+=trace_dict[key]*term4_coef_dict[pauli]
        inter_results[obs].append(trace/4*term4_coef_dict[obs])
        
    term4_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':-1,'XX':-1,'XY':-1,'XZ':-1,'YI':-1,'YX':-1,'YY':-1,'YZ':-1,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs+pauli])
            trace+=trace_dict[key]*term4_coef_dict[pauli]
        inter_results[obs].append(trace/4*term4_coef_dict[obs])
     
    term4_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':1,'XX':-1,'XY':-1,'XZ':1,'YI':1,'YX':-1,'YY':-1,'YZ':1,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, pauli ,obs+pauli])
            trace+=trace_dict[key]*term4_coef_dict[pauli]
        inter_results[obs].append(trace/4*term4_coef_dict[obs])
    return inter_results

def combine_trace_results_for_pcs_part2(check_layer_index,check_qubits,trace_dict):
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':1j,'XX':-1,'XY':1,'XZ':1j,'YI':-1j,'YX':1,'YY':-1,'YZ':-1j,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':-1j,'XX':-1,'XY':1,'XZ':-1j,'YI':1j,'YX':1,'YY':-1,'YZ':1j,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    term_2_3_pauli_dict={'II':'ZZ','IX':'ZY','IY':'ZX','IZ':'ZI','XI':'YZ','XX':'YY','XY':'YX','XZ':'YI','YI':'XZ','YX':'XY','YY':'XX','YZ':'XI','ZI':'IZ','ZX':'IY','ZY':'IX','ZZ':'II'}

    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
        
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':-1j,'XX':1,'XY':-1,'XZ':-1j,'YI':1j,'YX':-1,'YY':1,'YZ':1j,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':1j,'XX':1,'XY':-1,'XZ':1j,'YI':-1j,'YX':-1,'YY':1,'YZ':-1j,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
     
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    return inter_results
def combine_trace_results_for_pcs_part3(check_layer_index,check_qubits,trace_dict):
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':1,'XX':1j,'XY':-1j,'XZ':1,'YI':1,'YX':1j,'YY':-1j,'YZ':1,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':1,'XX':-1j,'XY':1j,'XZ':1,'YI':1,'YX':-1j,'YY':1j,'YZ':1,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}
    term_2_3_pauli_dict={'II':'IZ','IX':'IY','IY':'IX','IZ':'II','XI':'XZ','XX':'XY','XY':'XX','XZ':'XI','YI':'YZ','YX':'YY','YY':'YX','YZ':'YI','ZI':'ZZ','ZX':'ZY','ZY':'ZX','ZZ':'ZI'}

    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
        
    term2_coef_dict= {'II':1,'IX':1j,'IY':-1j,'IZ':1,'XI':-1,'XX':-1j,'XY':1j,'XZ':-1,'YI':-1,'YX':-1j,'YY':1j,'YZ':-1,'ZI':1,'ZX':1j,'ZY':-1j,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1j,'IY':1j,'IZ':1,'XI':-1,'XX':1j,'XY':-1j,'XZ':-1,'YI':-1,'YX':1j,'YY':-1j,'YZ':-1,'ZI':1,'ZX':-1j,'ZY':1j,'ZZ':1}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
     
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    return inter_results
def combine_trace_results_for_pcs_part4(check_layer_index,check_qubits,trace_dict):
    term2_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':1j,'XX':1j,'XY':1j,'XZ':1j,'YI':-1j,'YX':-1j,'YY':-1j,'YZ':-1j,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':1,'IY':1,'IZ':1,'XI':-1j,'XX':-1j,'XY':-1j,'XZ':-1j,'YI':1j,'YX':1j,'YY':1j,'YZ':1j,'ZI':1,'ZX':1,'ZY':1,'ZZ':1}
    term_2_3_pauli_dict={'II':'ZI','IX':'ZX','IY':'ZY','IZ':'ZZ','XI':'YI','XX':'YX','XY':'YY','XZ':'YZ','YI':'XI','YX':'XX','YY':'XY','YZ':'XZ','ZI':'II','ZX':'IX','ZY':'IY','ZZ':'IZ'}

    inter_results={'II':[],'IX':[],'IY':[],'IZ':[],'XI':[],'XX':[],'XY':[],'XZ':[],'YI':[],'YX':[],'YY':[],'YZ':[],'ZI':[],'ZX':[],'ZY':[],'ZZ':[]}
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
        
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
        
    term2_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':1j,'XX':-1j,'XY':-1j,'XZ':1j,'YI':-1j,'YX':1j,'YY':1j,'YZ':-1j,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}
    term3_coef_dict= {'II':1,'IX':-1,'IY':-1,'IZ':1,'XI':-1j,'XX':1j,'XY':1j,'XZ':-1j,'YI':1j,'YX':-1j,'YY':-1j,'YZ':1j,'ZI':1,'ZX':-1,'ZY':-1,'ZZ':1}

    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term2_coef_dict[pauli]
        inter_results[obs].append(trace/4*term3_coef_dict[obs])
     
    for obs in inter_results.keys():
        trace=0
        for pauli in ['IX','IY','IZ','II','XX','XY','XZ','XI','YX','YY','YZ','YI','ZX','ZY','ZZ','ZI']:
            key=tuple([str(check_layer_index), check_qubits, term_2_3_pauli_dict[pauli] ,term_2_3_pauli_dict[obs]+pauli])
            trace+=trace_dict[key]*term3_coef_dict[pauli]
        inter_results[obs].append(trace/4*term2_coef_dict[obs])
    return inter_results
def combine_trace_results_for_pcs(check_layer_index,check_qubits,trace_dict):
    new_dict={}
    dict1=combine_trace_results_for_pcs_part1(check_layer_index,check_qubits,trace_dict)
    dict2=combine_trace_results_for_pcs_part2(check_layer_index,check_qubits,trace_dict)
    dict3=combine_trace_results_for_pcs_part3(check_layer_index,check_qubits,trace_dict)
    dict4=combine_trace_results_for_pcs_part4(check_layer_index,check_qubits,trace_dict)
    for key in dict1.keys():
        new_list=dict1[key]+dict2[key]+dict3[key]+dict4[key]
        new_dict[key]=new_list
        
    return new_dict

def obtain_trace_pcs_layer_n(check_layer_index,check_qubits,trace_dict):
    #obtain combined results from prep_traces and meas_trace
    inter_results = combine_trace_results_for_pcs(check_layer_index,check_qubits,trace_dict)
    
    #create trace dict
    trace ={}
    denominator=sum(i for i in inter_results['II'])
    
    for i,v_list in inter_results.items():
        trace[i]=sum(i for i in v_list)/denominator
        
    return trace         

def obtain_density_matrix_pcs_layer_n(check_layer_index,check_qubits,trace_dict,value_x=None,value_y=None,value_z=None):
    #obtain combined results from prep_traces and meas_trace
    trace_dict = obtain_trace_pcs_layer_n(check_layer_index,check_qubits,trace_dict)
    density_matrix = obtain_density_matrix_from_trace(trace_dict)
    
    return density_matrix
            
   