# author: Peiyi Li (pli11@ncsu.edu)
############################################################################
import math

def ceiling_division(dividend, divisor):
    """
    Perform division and return the smallest integer greater than or equal to the division result.
    
    Parameters:
        dividend (int): The number to be divided.
        divisor (int): The number by which the dividend is divided.
        
    Returns:
        int: The smallest integer greater than or equal to the division result.
    """
    return math.ceil(dividend / divisor)


#extract qubits from config
def extract_numbers(string):
    # Split the string by the dash
    numbers = string.split('-')
    # Convert the split strings to integers
    num1, num2 = int(numbers[0]), int(numbers[1])
    return num1, num2

############################################################################
#select quantum backend
from qiskit_ibm_provider import IBMProvider
def select_backend(hardware):
    provider = IBMProvider()
    backend = provider.get_backend(hardware)
    return backend

#obtain info of the 2qubit basis gate for a backend
def obtain_info_2q_basis_gate(backend):
    #backend=select_backend(hardware)
    backend_properties = backend.properties()
    
    two_qubit_gates=set()
    for gate in backend_properties.gates:
        if len(gate.qubits) == 2:
            two_qubit_gates.add(gate.gate)
            
    assert(len(two_qubit_gates)==1)
    basis_gate_2q = list(two_qubit_gates)[0]        
    return basis_gate_2q

#obtain average number of 2qubit basis gates in a list of circuits
def obtain_average_num_2q_basis_gate(circuits_list,backend):
    basis_gate_2q = obtain_info_2q_basis_gate(backend)
    
    total_num_basis_gate_2q=0
    total_num_circuits = 0
    
    for circ in circuits_list:
        total_num_basis_gate_2q += circ.count_ops()[basis_gate_2q]
        total_num_circuits +=1
        
    average_num_basis_gate_2q = ceiling_division(total_num_basis_gate_2q,total_num_circuits)
    return average_num_basis_gate_2q
###########################################################################
from qiskit import  transpile
import numpy as np

def obtain_best_trans_qc(qc,transpile_times,backend,optimization_level=3):
    basis_gate_2q = obtain_info_2q_basis_gate(backend)
    
    trans_qc_list = transpile([qc]*transpile_times, backend, optimization_level=optimization_level)
    gate_count = [circ.count_ops()[basis_gate_2q] for circ in trans_qc_list]

    best_idx = np.argmin(gate_count)
    best_qc = trans_qc_list[best_idx]

    return best_qc
###########################################################################
#run a list of circuits on a simulator
def execute_circuits_on_simulator(circuits_list, simulator, shots):
    counts_list=[]
    for i in range(0,len(circuits_list),300):
        start=i
        end=i
        if(i+300)<len(circuits_list):
            end=i+300
        else:
            end=len(circuits_list)
        
        result=simulator.run(circuits_list[start:end], shots=shots).result()    
        counts=result.get_counts()
        if type(counts)==list:
            counts_list+=counts
        else:
            counts_list.append(counts)
    return counts_list

def execute_circuits_on_real_machine(circuits_list, backend, shots):
    #job_list=[]
    job_id_list=[]
    for i in range(0,len(circuits_list),300):
        start=i
        end=i
        if(i+300)<len(circuits_list):
            end=i+300
        else:
            end=len(circuits_list)
        
        job=backend.run(circuits_list[start:end], shots=shots)
        #print(f"Job ID: {job.job_id()}")
        #job_list.append(job) 
        job_id_list.append(job.job_id())
    return job_id_list

def retrive_job_from_real_machine(provider, job_id_list):
    counts_list=[]
    for job_id in job_id_list:
        # Retrieve the job by its id
        job = provider.backend.retrieve_job(job_id)
        result = job.result()
        counts = result.get_counts()
        if type(counts)==list:
            counts_list+=counts
        else:
            counts_list.append(counts)
    
    return counts_list

###########################################################################
#function related to QAOA
def obtain_info_1q_gate_qaoa(layer_index,parameters):
    p = len(parameters)//2
    beta = parameters[:p]
    
    rep = (len(parameters[:(layer_index+1)*2])//2) - 1
    return 2 * beta[rep]
    