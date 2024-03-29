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
#load info of benchmark
from importlib import import_module
benchmark = import_module(configuration['benchmark'])
parameters = configuration['parameters']
test_qubits = configuration['test_qubits']

#load hardware info
hardware=configuration['hardware']
from utils import select_backend
backend = select_backend(hardware)

#load other config info
enable_noise_aware_mapping=configuration['enable_noise_aware_mapping']
transpile_times=configuration['transpile_times']

##########################################################################
#generate original circuit
circuit=benchmark.qaoa_circuits(test_qubits,parameters)
#adding measurment for the original circuits
from qiskit import ClassicalRegister
c_reg = ClassicalRegister(test_qubits, 'c')
circuit.add_register(c_reg)
for i in range(test_qubits):
    circuit.measure(i,i)
    
#generate the transpiled circuits for the original circuit
from utils import obtain_best_trans_qc
best_qc = obtain_best_trans_qc(circuit,transpile_times,backend)
#do noise aware mapping based on the value of enable_noise_aware_mapping, 
import mapomatic as mm
from qiskit import  transpile
if enable_noise_aware_mapping:
    best_small_qc,changed_mapping = mm.deflate_circuit(best_qc)
    layouts = mm.matching_layouts(best_small_qc, backend)
    scores = mm.evaluate_layouts(best_small_qc, layouts, backend)
    best_qc = transpile(best_small_qc, backend, initial_layout=scores[0][0])
    
#save the original circuits
import os
from qiskit import qpy
#create folder to save results
save_path = configuration['results_save_path'] + '/generate_circuits_original'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#save circuit
save_name= '/original_circuit.qpy'
with open(save_path + save_name, "wb") as f:
    qpy.dump(best_qc, f)
        
#############################################################################
#generate jigsaw circuits based on the value of enable_jigsaw
from CircuitCutter import obtain_final_physical_qubits_index_for_measurement
enable_jigsaw=configuration['enable_jigsaw']
if enable_jigsaw:
    #obtain the qubits index for final measurment
    final_qubits_index_for_measurement = obtain_final_physical_qubits_index_for_measurement(best_qc)
    
    #load jigsaw_check_qubits:
    jigsaw_check_qubits = configuration['jigsaw_check_qubits'] 
    
    #generate circuits for checked qubits:
    for check_qubits in jigsaw_check_qubits:
        #record the qubits index for checked qubits
        qubits_index_for_measurement=[]
        qubits_index_for_measurement.append(final_qubits_index_for_measurement[check_qubits[0]])
        qubits_index_for_measurement.append(final_qubits_index_for_measurement[check_qubits[1]])
        
        #generate the circuits 
        best_qc.remove_final_measurements(inplace=True)
        c_reg = ClassicalRegister(len(qubits_index_for_measurement), 'c')
        best_qc.add_register(c_reg)
        i=0
        for qbit in qubits_index_for_measurement:
            best_qc.measure(qbit,i)
            i+=1
            
        #save circuits
        #create folder to save results
        save_path = configuration['results_save_path'] + '/generate_circuits_jigsaw'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #save circuit
        save_name= '/jigsaw_circuit_' + str(check_qubits[0]) + '-' + str(check_qubits[1]) + '.qpy'
        with open(save_path + save_name, "wb") as f:
            qpy.dump(best_qc, f)
            
#################################################################################### 
from utils import extract_numbers
from CircuitCutter import generate_circuit_after_cutting, generate_execution_circuits_list
from CircuitCutter import obtain_initial_physical_qubits_index, obtain_final_physical_qubits_index_for_measurement

#generate QuTracer circuits based on the value of enable_QuTracer
enable_QuTracer=configuration['enable_QuTracer']
if enable_QuTracer:
    #obtain the info about layer checked
    check_layer_index_list = configuration['QuTracer_check_layer_index']
    #obtain the info about qubits checked and position checked
    check_qubits_dict=configuration['QuTracer_check_qubits']

    #check each layers
    for check_layer_index in check_layer_index_list:
        #check qubits and position
        for check_qubits, check_position in check_qubits_dict.items():
            #obtain the checking qubits
            check_qubits_extracted = extract_numbers(check_qubits)
            #obtain the checking position
            midpoint = len(check_position[check_layer_index]) // 2 
            check_position_left=check_position[check_layer_index][:midpoint]
            check_position_right=check_position[check_layer_index][midpoint:]
            
            #generate circuits
            circuit, check_qubits_extracted = benchmark.generate_circuits_til_certain_layer(check_layer_index,parameters)
            qc, prep_qubits, meas_qubits = generate_circuit_after_cutting(circuit,check_qubits_extracted,check_position_left)
            best_qc = obtain_best_trans_qc(qc,transpile_times,backend)
            qubits_initial_index_list = obtain_initial_physical_qubits_index(best_qc,prep_qubits)
            qubits_final_index_list = obtain_final_physical_qubits_index_for_measurement(best_qc)
            if enable_noise_aware_mapping:
                best_small_qc,changed_mapping = mm.deflate_circuit(best_qc)
                layouts = mm.matching_layouts(best_small_qc, backend)
                scores = mm.evaluate_layouts(best_small_qc, layouts, backend)
                best_qc = transpile(best_small_qc, backend, initial_layout=scores[0][0])
    
                qubits_initial_index_list = mm.obtain_initial_physical_qubits_index(best_qc, prep_qubits, changed_mapping)
                qubits_final_index_list = obtain_final_physical_qubits_index_for_measurement(best_qc)
            
            best_qc.remove_final_measurements(inplace=True)
            
            #generate transpiled circuits list and save the list 
            prep_state_list = configuration['prep_state_list_layer_'+str(check_layer_index)]
            obs_list = configuration['obs_list_layer_'+str(check_layer_index)]
            new_qc_list = generate_execution_circuits_list(best_qc, backend, qubits_initial_index_list, qubits_final_index_list, obs_list, prep_state_list)
            
            #create folder to save results
            save_path = configuration['results_save_path'] + '/generate_circuits_QuTracer/circuit_layer_' + str(check_layer_index)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #save circuit
            save_name= '/qc_list_checking_qubit_' + check_qubits + '.qpy'
            with open(save_path + save_name, "wb") as f:
                qpy.dump(new_qc_list, f)
        
        
