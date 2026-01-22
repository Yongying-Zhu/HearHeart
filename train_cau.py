#!/usr/bin/env python
import sys
from team_code_cau import train_challenge_model

if __name__ == '__main__':
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]
    verbose = 1
    
    train_challenge_model(data_folder, model_folder, verbose)
