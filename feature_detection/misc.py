import matplotlib.pyplot as plt
import numpy as np
from dataset_utils import get_testfile_paths, Thumbnails
import sys
sys.path.append('..')
import utils.constants as constants

folder_path = get_testfile_paths('Luton')

for j, train_path in enumerate(folder_path):     
    thumbnails_dataset = Thumbnails(cityName='Luton', tileName=train_path.stem, n_input_channels=3)
    tgt = np.transpose(thumbnails_dataset.paddedTarget,(1,2,0))
    for channel_index, high_level_feature_name in enumerate(constants.HIGHLEVELFEATURES):
        print(f'Displaying {high_level_feature_name} ...')

