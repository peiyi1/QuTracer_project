#!/bin/bash

cd script/results_postprocessing

python results_postprocessing.py ../../yaml_file/qaoa_reps1_n10.yaml 
python results_postprocessing.py ../../yaml_file/qaoa_reps2_n10.yaml 
python results_postprocessing.py ../../yaml_file/qaoa_reps3_n10.yaml 
python results_postprocessing.py ../../yaml_file/qaoa_reps4_n10.yaml 
python results_postprocessing.py ../../yaml_file/qaoa_reps5_n10.yaml 

