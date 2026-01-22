#!/usr/bin/env python
import sys
import os
from team_code_cau import load_challenge_model, run_challenge_model
sys.path.append('..')
from helper_code import *

if __name__ == '__main__':
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]
    
    os.makedirs(output_folder, exist_ok=True)
    
    model = load_challenge_model(model_folder, verbose=1)
    patient_files = find_patient_files(data_folder)
    
    for patient_file in patient_files:
        patient_data = load_patient_data(patient_file)
        recordings = load_recordings(data_folder, patient_data)
        
        classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose=0)
        
        patient_id = patient_data.split('\n')[0].split()[0]
        output_file = os.path.join(output_folder, patient_id + '.csv')
        save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)
