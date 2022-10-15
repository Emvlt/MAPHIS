import sys
sys.path.append('..')
from utils import constants
import json
from models_folder import models
import torch

def load_model(operation:str, ncIn:int, parameters = None, training_type=str):
    parameter_path = constants.MODELSPATH.joinpath(f'parameters/{operation}.json')
    print(f"------ Attempting to load saved Parameters at {parameter_path} ------")
    if parameter_path.is_file():
        print(f"------ Loading Parameters from {parameter_path} ------")
        parameters = json.load(open(str(parameter_path)))
    elif not parameter_path.is_file() and parameters is None:
        print(f'------ No saved parameters, using default ------')
        parameters = json.load(open(str(constants.MODELSPATH.joinpath('default_model_parameters.json'))))[operation]

    print(f"------ Instanciating {operation.capitalize()} Model ------")
    if operation == 'segmentation':
        model = models.SegmentationModel(parameters, ncIn, 1+len(constants.HIGHLEVELFEATURES))
    else:
        print(f'{operation.capitalize()} not implemented')
        return None

    saved_model_path = constants.MODELSPATH.joinpath(f'saves/{operation}_{training_type}_state_dict.pth')
    print(f"------ Attempting to load saved Model at {saved_model_path} ------")
    if saved_model_path.is_file():
        print(f"Loading {operation.capitalize()} Saved Model")
        model.load_state_dict(torch.load(str(saved_model_path)))
    else:
        print('No saved model, passing')
    return model
