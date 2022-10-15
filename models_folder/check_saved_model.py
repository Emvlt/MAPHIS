import torch
from models import SegmentationModel
import sys
import json
sys.path.append('..')
from utils import constants

parameters = json.load(open(str(constants.MODELSPATH.joinpath('default_model_parameters.json'))))['segmentation']


model = SegmentationModel(parameters, 3, 1+len(constants.HIGHLEVELFEATURES))

saved_model_path = constants.MODELSPATH.joinpath(f'saves/segmentation_real_state_dict.pth')

saved_model = torch.load(str(saved_model_path))

for i in [5,7,9,11]:
    for p in ['weight', 'bias']:
        saved_model[f'gabor_filters.{i}.{p}'] = saved_model[f'gaborFilters.{i}.{p}']
        del saved_model[f'gaborFilters.{i}.{p}']

for k in saved_model:
    print(k)

torch.save(saved_model, saved_model_path)
saved_model = torch.load(str(saved_model_path))
for k in saved_model:
    print(k)
