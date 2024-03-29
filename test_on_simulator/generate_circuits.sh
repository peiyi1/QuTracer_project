#!/bin/bash

cd script/generate_circuits

python generate_circuits.py ../../yaml_file/qaoa_reps1_n10.yaml 
python generate_circuits.py ../../yaml_file/qaoa_reps2_n10.yaml 
python generate_circuits.py ../../yaml_file/qaoa_reps3_n10.yaml 
python generate_circuits.py ../../yaml_file/qaoa_reps4_n10.yaml 
python generate_circuits.py ../../yaml_file/qaoa_reps5_n10.yaml 