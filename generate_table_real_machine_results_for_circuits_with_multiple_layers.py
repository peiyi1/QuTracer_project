import csv
import json


labels = ['vqe_reps2_n12','vqe_reps3_n12','vqe_reps2_n15','vqe_reps3_n15','qaoa_reps2_n10','qaoa_reps3_n10']    
    

with open('real_machine_results_for_circuits_with_multiple_layers.csv', 'w') as csvfile:
    fieldnames = ['workload', 'normalized_shots_number_Original', 'normalized_shots_number_JigSaw',
                  'normalized_shots_number_QuTracer', 'average_counts_for_2q_basis_gates_Original', 'average_counts_for_2q_basis_gates_JigSaw',
                  'average_counts_for_2q_basis_gates_QuTracer', 'fidelity_Original', 'fidelity_JigSaw',
                  'fidelity_QuTracer', #'fidelity_improvement_QuTracer',
                 ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for name in labels:
        ################################################
        #load results from /results
        save_path ='./test_on_real_machine/saved_data/' + name
        save_name= '/results/hellinger_fidelity_original_circuit.json'
        with open(save_path + save_name, "r") as f:
            fidelity_Original = json.load(f)

        save_name= '/results/hellinger_fidelity_jigsaw.json'
        with open(save_path + save_name, "r") as f:
            fidelity_JigSaw = json.load(f)
            
        save_name= '/results/hellinger_fidelity_QuTracer.json'
        with open(save_path + save_name, "r") as f:
            fidelity_QuTracer = json.load(f)
        
        save_name= '/results/hellinger_fidelity_improvement_QuTracer.json'
        with open(save_path + save_name, "r") as f:
            fidelity_improvement_QuTracer = json.load(f)
            
        ################################################
        #load results from /results
        save_name= '/execute_circuits/average_num_2q_basis_gate_original_circuit.json'
        with open(save_path + save_name, "r") as f:
            average_counts_for_2q_basis_gates_Original = json.load(f)
        
        save_name= '/execute_circuits/average_num_2q_basis_gate_QuTracer.json'
        with open(save_path + save_name, "r") as f:
            average_counts_for_2q_basis_gates_QuTracer = json.load(f)
        
        save_name= '/execute_circuits/average_num_2q_basis_gate_jigsaw.json'
        with open(save_path + save_name, "r") as f:
            average_counts_for_2q_basis_gates_JigSaw = json.load(f)
    
        save_name= '/execute_circuits/normalized_shots_original_circuit.json'
        with open(save_path + save_name, "r") as f:
            normalized_shots_number_Original = json.load(f)
            
        save_name= '/execute_circuits/normalized_shots_jigsaw.json'
        with open(save_path + save_name, "r") as f:
            normalized_shots_number_JigSaw = json.load(f)
            
        save_name= '/execute_circuits/normalized_shots_QuTracer.json'
        with open(save_path + save_name, "r") as f:
            normalized_shots_number_QuTracer = json.load(f)
    
        writer.writerow({'workload':name, 'fidelity_Original':float("{:.2f}".format(fidelity_Original)),'fidelity_JigSaw':float("{:.2f}".format(fidelity_JigSaw)),'fidelity_QuTracer':float("{:.2f}".format(fidelity_QuTracer)),#'fidelity_improvement_QuTracer':"{:.2%}".format(fidelity_improvement_QuTracer),
                        'average_counts_for_2q_basis_gates_Original':average_counts_for_2q_basis_gates_Original,
                         'average_counts_for_2q_basis_gates_QuTracer':average_counts_for_2q_basis_gates_QuTracer,
                         'average_counts_for_2q_basis_gates_JigSaw': average_counts_for_2q_basis_gates_JigSaw,
                         'normalized_shots_number_Original':normalized_shots_number_Original,
                         'normalized_shots_number_JigSaw':normalized_shots_number_JigSaw,
                         'normalized_shots_number_QuTracer':normalized_shots_number_QuTracer,
                        })
        