import argparse
import yaml

import warnings
warnings.filterwarnings('ignore')
#########################################################################
#parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('yamlfile', metavar='file.yaml', nargs=1, help='YAML configuration file')
args = parser.parse_args()
yamlfile = args.yamlfile[0]

#load the configuration file
with open(yamlfile) as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)
    
##########################################################################
#load hardware info
hardware=configuration['hardware']
from utils import select_backend, obtain_info_2q_basis_gate
backend = select_backend(hardware)
basis_gate_2q = obtain_info_2q_basis_gate(backend)

#choose simulator
from qiskit.providers.aer import AerSimulator
simulator = AerSimulator()
from qiskit.providers.aer.noise import NoiseModel
noise_model = NoiseModel.from_backend(backend=backend)
simulator_noisy = AerSimulator(noise_model=noise_model)

##########################################################################
from qiskit import qpy
running_circuits=[]
#load original circuit
save_path = configuration['results_save_path'] + '/generate_circuits_original' 
save_name= '/original_circuit.qpy'
with open(save_path + save_name, "rb") as f:
    circuits = qpy.load(f)
    running_circuits=running_circuits+circuits
    
#calculate average counts of 2qubit basis gates for the original circuit
from utils import obtain_average_num_2q_basis_gate
average_num_2q_basis_gate_original_circuit = obtain_average_num_2q_basis_gate(running_circuits, backend)

############################################################################
enable_QuTracer=configuration['enable_QuTracer']
if enable_QuTracer:
    #recored the start point of the running circuits
    running_circuits_QuTracer_start = len(running_circuits)
    
    #obtain the info about layer checked
    check_layer_index_list = configuration['QuTracer_check_layer_index']
    #obtain the info about qubits checked and position checked
    check_qubits_dict=configuration['QuTracer_check_qubits']
    
    #check each of the layers
    for check_layer_index in check_layer_index_list:
        #check qubits and position
        for check_qubits, check_position in check_qubits_dict.items():
            if(check_position[check_layer_index]!=None):
                #load circuit
                save_path = configuration['results_save_path'] + '/generate_circuits_QuTracer/circuit_layer_' + str(check_layer_index)
                if type(check_qubits) == int:
                    check_qubits = str(check_qubits)
                save_name= '/qc_list_checking_qubit_' + check_qubits + '.qpy'
                with open(save_path + save_name, "rb") as f:
                    circuits = qpy.load(f)
                    running_circuits=running_circuits+circuits
                
    #recored the end point of the running circuits
    running_circuits_QuTracer_end = len(running_circuits)
    
    #calculate average counts of 2qubit basis gates
    average_num_2q_basis_gate_QuTracer = obtain_average_num_2q_basis_gate(running_circuits[running_circuits_QuTracer_start:running_circuits_QuTracer_end], backend)

#############################################################################
enable_jigsaw=configuration['enable_jigsaw']
if enable_jigsaw:
    #recored the start point of the running circuits
    running_circuits_jigsaw_start = len(running_circuits)
    
    #obtain the info about qubits checked 
    jigsaw_check_qubits=configuration['jigsaw_check_qubits']
    
    #check each of the layers
    for check_qubits in jigsaw_check_qubits:
            #load circuit
            save_path = configuration['results_save_path'] + '/generate_circuits_jigsaw' 
            save_name= '/jigsaw_circuit_' + str(check_qubits[0]) + '-' + str(check_qubits[1]) + '.qpy'
            with open(save_path + save_name, "rb") as f:
                circuits = qpy.load(f)
                running_circuits=running_circuits+circuits
                
    #recored the end point of the running circuits
    running_circuits_jigsaw_end = len(running_circuits)
    #calculate average counts of 2qubit basis gates
    average_num_2q_basis_gate_jigsaw = obtain_average_num_2q_basis_gate(running_circuits[running_circuits_jigsaw_start:running_circuits_jigsaw_end], backend)

##################################################################################
import json
import os
from utils import ceiling_division

#create folder to save results
save_path = configuration['results_save_path'] + '/execute_circuits'
if not os.path.exists(save_path):
    os.makedirs(save_path)    
################################################################################
#execute all the circuits
from utils import execute_circuits_on_simulator, execute_circuits_on_real_machine

#specify shots number
shots_original_circuits =  configuration['shots']
shots_remaining_circuits =  configuration['shots']//len(configuration['measured_qubits'])

#execute circuits (ideal simulation)
ideal_counts_original_circuits = execute_circuits_on_simulator(running_circuits[0:1], simulator, shots_original_circuits)
ideal_counts_remaining_circuits = execute_circuits_on_simulator(running_circuits[1:], simulator, shots_remaining_circuits)

#decide to do simulation or running on real machine based on the value of enable_running_on_real_machine
enable_running_on_real_machine = configuration['enable_running_on_real_machine']
#execute circuits (noisy simulation)
if not enable_running_on_real_machine:
    noisy_counts_original_circuits = execute_circuits_on_simulator(running_circuits[0:1], simulator_noisy, shots_original_circuits)
    noisy_counts_remaining_circuits = execute_circuits_on_simulator(running_circuits[1:], simulator_noisy, shots_remaining_circuits)
else:
    job_id_list_original_circuit = execute_circuits_on_real_machine(running_circuits[0:1], backend, shots_original_circuits)
    job_id_list_remaining_circuit = execute_circuits_on_real_machine(running_circuits[1:], backend, shots_remaining_circuits)
    #save job_id_list
    save_name= '/job_id_list_original_circuit.json'
    with open(save_path + save_name, "w") as f:
        json.dump(job_id_list_original_circuit, f)
    save_name= '/job_id_list_remaining_circuit.json'
    with open(save_path + save_name, "w") as f:
        json.dump(job_id_list_remaining_circuit, f)

##################################################################################  
#save results
#for the info of original circuit    
#save ideal_counts_original_circuit
save_name= '/ideal_counts_original_circuit.json'
with open(save_path + save_name, "w") as f:
    json.dump(ideal_counts_original_circuits[0:1], f)
    
#save noisy_counts_original_circuit    
if not enable_running_on_real_machine:    
    save_name= '/noisy_counts_original_circuit.json'
    with open(save_path + save_name, "w") as f:
        json.dump(noisy_counts_original_circuits[0:1], f)

#save average_num_2q_basis_gate_original_circuit
save_name= '/average_num_2q_basis_gate_original_circuit.json'
with open(save_path + save_name, "w") as f:
    json.dump(average_num_2q_basis_gate_original_circuit, f)
    
#save normalized_shots_original_circuit
normalized_shots_original_circuit = ceiling_division(shots_original_circuits, shots_original_circuits)
save_name= '/normalized_shots_original_circuit.json'
with open(save_path + save_name, "w") as f:
    json.dump(normalized_shots_original_circuit, f)

if enable_QuTracer:
    #save ideal_counts_QuTracer
    save_name= '/ideal_counts_QuTracer.json'
    with open(save_path + save_name, "w") as f:
        json.dump(ideal_counts_remaining_circuits[running_circuits_QuTracer_start-1:running_circuits_QuTracer_end-1], f)
    #save noisy_counts_QuTracer
    if not enable_running_on_real_machine:
        save_name= '/noisy_counts_QuTracer.json'
        with open(save_path + save_name, "w") as f:
            json.dump(noisy_counts_remaining_circuits[running_circuits_QuTracer_start-1:running_circuits_QuTracer_end-1], f)
    #save average_num_2q_basis_gate_QuTracer
    save_name= '/average_num_2q_basis_gate_QuTracer.json'
    with open(save_path + save_name, "w") as f:
        json.dump(average_num_2q_basis_gate_QuTracer, f)
    #save normalized_shots_QuTracer   
    save_name = '/normalized_shots_QuTracer.json'
    normalized_shots_QuTracer = ceiling_division(shots_original_circuits+shots_remaining_circuits*(running_circuits_QuTracer_end-running_circuits_QuTracer_start), shots_original_circuits)
    with open(save_path + save_name, "w") as f:
        json.dump(normalized_shots_QuTracer,f)
        
if enable_jigsaw:
    #save ideal_counts_QuTracer
    save_name= '/ideal_counts_jigsaw.json'
    with open(save_path + save_name, "w") as f:
        json.dump(ideal_counts_remaining_circuits[running_circuits_jigsaw_start-1:running_circuits_jigsaw_end-1], f)
    #save noisy_counts_QuTracer
    if not enable_running_on_real_machine:
        save_name= '/noisy_counts_jigsaw.json'
        with open(save_path + save_name, "w") as f:
            json.dump(noisy_counts_remaining_circuits[running_circuits_jigsaw_start-1:running_circuits_jigsaw_end-1], f)
    #save average_num_2q_basis_gate_QuTracer
    save_name= '/average_num_2q_basis_gate_jigsaw.json'
    with open(save_path + save_name, "w") as f:
        json.dump(average_num_2q_basis_gate_jigsaw, f)
    #save normalized_shots_QuTracer   
    save_name = '/normalized_shots_jigsaw.json'
    normalized_shots_jigsaw = ceiling_division(shots_remaining_circuits*(running_circuits_jigsaw_end-running_circuits_jigsaw_start), shots_original_circuits)
    with open(save_path + save_name, "w") as f:
        json.dump(normalized_shots_jigsaw,f)
    