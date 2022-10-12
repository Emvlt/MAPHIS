import torch
import argparse
from train_loops import train_loop

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyper_parameters = {
        'n_input_channels': 3,
        'learning_rate': 1e-4,
        'batch_size':8,
        'epochs':10,
        'threshold_value':0.9, 
        'n_samples':1000,
        'architecture_parameters':{
            "nGaborFilters": 64, 
            "ngf": 4, 
            "supportSizes": [5, 7, 9, 11]
            }
    }

    train_loop(args.process ,device, model=None, hyper_parameters=hyper_parameters, training_type=args.training_type)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=False, type=str, default= 'segmentation')
    parser.add_argument('--training_type', required=False, type=str, default= 'real')
    args = parser.parse_args()
    main(args)

