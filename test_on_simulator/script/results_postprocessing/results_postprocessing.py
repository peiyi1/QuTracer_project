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
import json
import os
from qiskit.quantum_info import hellinger_fidelity
from ResultsProcessing import norm_dict, two_bit_weight, bayesian_reconstruct

#load the global output (ideal simulation)
save_path = configuration['results_save_path'] + '/execute_circuits'
save_name= '/ideal_counts_original_circuit.json'
with open(save_path + save_name, "r") as f:
    counts_list = json.load(f)
global_output_ideal = counts_list[0]
#for i in range(configuration['test_qubits']):
#    print(two_bit_weight(norm_dict(global_output_ideal),i))
    
#load the global output (noisy)
save_path = configuration['results_save_path'] + '/execute_circuits'
save_name= '/noisy_counts_original_circuit.json'
with open(save_path + save_name, "r") as f:
    counts_list = json.load(f)
global_output_noisy = counts_list[0]
#for i in range(configuration['test_qubits']):
#    print(two_bit_weight(norm_dict(global_output_noisy),i))

#calculate hellinger_fidelity
fidelity = hellinger_fidelity(global_output_ideal, global_output_noisy)
#print('hellinger_fidelity:',fidelity)

#create folder to save results
save_path = configuration['results_save_path'] + '/results'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#save results
save_name= '/hellinger_fidelity_original_circuit.json'
with open(save_path + save_name, "w") as f:
    json.dump(fidelity, f)

##########################################################################
enable_jigsaw=configuration['enable_jigsaw']
#update the global output distribution
if enable_jigsaw: 
    #load configs that are needed
    #obtain the info about qubits needed to update
    jigsaw_check_qubits=configuration['jigsaw_check_qubits']
    
    #load the noise mitigated local output distribution
    save_path = configuration['results_save_path'] + '/execute_circuits'
    save_name= '/noisy_counts_jigsaw.json'
    with open(save_path + save_name, "r") as f:
        local_counts_jigsaw = json.load(f)
            
    #obtain local output distribution
    local_output_dist=[]
    for i in range(len(jigsaw_check_qubits)):
            local_output_dist.append([norm_dict(local_counts_jigsaw[i]), i, 0])
    #print(local_output_dist)
    
    #update the global output
    updated_global_output = bayesian_reconstruct(norm_dict(global_output_noisy),local_output_dist)
    #calculate hellinger_fidelity
    fidelity_jigsaw = hellinger_fidelity(global_output_ideal, updated_global_output)
    #print('hellinger_fidelity:',fidelity_jigsaw)
    #calculate hellinger_fidelity improvement compared with original circuit
    fidelity_improvement_jigsaw = (fidelity_jigsaw - fidelity)/fidelity
    #print('hellinger_fidelity_improvement:',fidelity_improvement_jigsaw)
    
    #save results
    save_path = configuration['results_save_path'] + '/results'
    save_name= '/hellinger_fidelity_jigsaw.json'
    with open(save_path + save_name, "w") as f:
        json.dump(fidelity_jigsaw, f)
    save_name= '/hellinger_fidelity_improvement_jigsaw.json'
    with open(save_path + save_name, "w") as f:
        json.dump(fidelity_improvement_jigsaw, f)
        
##########################################################################
from ResultsProcessing import obtain_prep_trace_dict,obtain_meas_trace_dict
from ResultsProcessing import obtain_density_matrix_pcs
from ResultsProcessing import obtain_meas_output_dist
from ResultsProcessing import obtain_counts_dict_based_on_previous_layer_info,obtain_meas_prep_trace_dict_layer_n,obtain_density_matrix_pcs_layer_n

from utils import obtain_info_1q_gate_qaoa
#if QuTracer is enabled, postprocessing the data of QuTracer
enable_QuTracer=configuration['enable_QuTracer']
if enable_QuTracer:
    #load counts_list:
    save_path = configuration['results_save_path'] + '/execute_circuits'
    save_name= '/noisy_counts_QuTracer.json'
    with open(save_path + save_name, "r") as f:
        counts_list = json.load(f)
 
    #load other configs that are needed
    #obtain the info about layer checked
    check_layer_index_list = configuration['QuTracer_check_layer_index']
    #obtain the info about qubits checked and position checked
    check_qubits_dict=configuration['QuTracer_check_qubits']
    
    #init counts index:
    counts_index=0
    #check each of the layers
    for check_layer_index in check_layer_index_list:
        #check qubits and position
        for check_qubits in check_qubits_dict:
            #load prep_state_list and obs_list
            prep_state_list=configuration['prep_state_list_layer_'+str(check_layer_index)]
            obs_list=configuration['obs_list_layer_'+str(check_layer_index)]
            #check layer 0
            if check_layer_index == 0:
                #calculate prep_trace
                prep_trace_dict, counts_index = obtain_prep_trace_dict(counts_list,counts_index, prep_state_list,obs_list,check_layer_index,check_qubits)
                localized_initial_gates = configuration['localized_initial_gates']
                #calculate meas_trace
                #print(localized_initial_gates)
                meas_trace_dict=obtain_meas_trace_dict(localized_initial_gates)
                #print(meas_trace_dict)
            
                #combine prep_trace and meas_trace
                inter_matrix = obtain_density_matrix_pcs(check_layer_index,check_qubits,meas_trace_dict,prep_trace_dict)
                
            #check layer n (n>0)
            else:
                #load mitigated output info from previous layer:
                with open(save_path + save_name, "r") as f:
                    mitigated_output_dist = json.load(f)
                counts_dict, counts_index = obtain_counts_dict_based_on_previous_layer_info(mitigated_output_dist,counts_list,counts_index,prep_state_list,obs_list,check_layer_index,check_qubits)
                #print(len(counts_dict))
                
                #calculate meas_prep_trace
                meas_prep_trace_dict = obtain_meas_prep_trace_dict_layer_n(counts_dict,prep_state_list,obs_list,check_layer_index,check_qubits)
                
                #inter_results = combine_meas_prep_trace_results_layer_n(check_layer_index,check_qubits,meas_prep_trace_dict)
                #print(inter_results)
                inter_matrix = obtain_density_matrix_pcs_layer_n(check_layer_index,check_qubits,meas_prep_trace_dict)
                
            #obtain info for single gate
            gate_name = configuration['localized_gates']
            gate_params = obtain_info_1q_gate_qaoa(check_layer_index,configuration['parameters'])
            #print(gate_params)
            
            #obtain final results
            results_final_trace=obtain_meas_trace_dict(gate_name, gate_params,initial_rho=inter_matrix)
            #print(results_final_trace)
            results_final_output_dist=obtain_meas_output_dist(gate_name, gate_params,initial_rho=inter_matrix)
            #print(results_final_output_dist)
            
            #create folder to save results
            save_path = configuration['results_save_path'] + '/results_postprocessing_QuTracer/circuit_layer_' + str(check_layer_index)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #save results
            save_name= '/results_final_trace_' + check_qubits + '.json'
            with open(save_path + save_name, "w") as f:
                json.dump(results_final_trace, f)
            save_name= '/results_final_output_dist_' + check_qubits + '.json'
            with open(save_path + save_name, "w") as f:
                json.dump(results_final_output_dist, f)
            
##########################################################################
from utils import extract_numbers
#update the global output distribution
if enable_QuTracer: 
    #load configs that are needed
    #obtain the info about layer checked
    check_layer_index_list = configuration['QuTracer_check_layer_index']
    last_layer_index = check_layer_index_list[-1]
    #obtain the info about qubits needed to update
    update_qubits_dict=configuration['QuTracer_update_qubits']
    
    #obtain local output distribution
    local_output_dist=[]
    for updated_qubits, checked_qubits in update_qubits_dict.items():
            #load the noise mitigated local output distribution
            save_path = configuration['results_save_path'] + '/results_postprocessing_QuTracer/circuit_layer_' + str(last_layer_index)
            save_name= '/results_final_output_dist_' + checked_qubits + '.json'
            with open(save_path + save_name, "r") as f:
                results_final_output_dist = json.load(f)
                #print(results_final_output_dist['ZZ'])
                local_output_dist.append([norm_dict(results_final_output_dist['ZZ']), extract_numbers(updated_qubits)[0], 0])
    #print(local_output_dist)
    
    #update the global output
    updated_global_output=bayesian_reconstruct(norm_dict(global_output_noisy),local_output_dist)
    #calculate hellinger_fidelity
    fidelity_QuTracer = hellinger_fidelity(global_output_ideal, updated_global_output)
    #print('hellinger_fidelity:',fidelity_QuTracer)
    #calculate hellinger_fidelity improvement compared with original circuit
    fidelity_improvement_QuTracer = (fidelity_QuTracer - fidelity)/fidelity
    #print('hellinger_fidelity_improvement:',fidelity_improvement_QuTracer)
    
    #save results
    save_path = configuration['results_save_path'] + '/results'
    save_name= '/hellinger_fidelity_QuTracer.json'
    with open(save_path + save_name, "w") as f:
        json.dump(fidelity_QuTracer, f)
    save_name= '/hellinger_fidelity_improvement_QuTracer.json'
    with open(save_path + save_name, "w") as f:
        json.dump(fidelity_improvement_QuTracer, f)
            