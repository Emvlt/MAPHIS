import sys
sys.path.append('..')
import utils.constants as constants
import utils.morphological_tools as morph_tools
import json
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np

def get_testfile_paths(city_name:str):
    path_to_test_files = constants.CITIESFOLDERPATH.glob(f'{city_name}/*{constants.FILEEXTENSION}')
    print('Test files paths loaded...')
    return path_to_test_files

class RealFeatures(Dataset):
    def __init__(self, n_input_channels:int) -> None:
        self.path_to_data = constants.TRAININGPATH
        self.n_input_channels = n_input_channels

    def __len__(self):
        return int(len((list(self.path_to_data.glob('*'))))/2)

    def __getitem__(self, index):
        image = np.load(self.path_to_data.joinpath(f'input_{index}.npy'))
        target  = np.load(self.path_to_data.joinpath(f'target_{index}.npy'))
        if self.n_input_channels ==3:
            image = np.concatenate(( np.expand_dims(morph_tools.erosion(image),2), np.expand_dims(image,2), np.expand_dims(morph_tools.dilation(image),2)), axis=2)

        image, target = torch.from_numpy(np.transpose(image,(2,0,1))), torch.from_numpy(np.transpose(target,(2,0,1)))
        k = np.random.randint(0,4)
        image = torch.rot90(image, k, dims=(1,2))
        target = torch.rot90(target, k, dims=(1,2))
        return image, target

class Thumbnails(Dataset):
    def __init__(self, city_name:str,  tileName:str, n_input_channels:int) -> None:
        print(f'Opening tile {tileName} for the city of {city_name}')
        self.tile_shape  = json.load(open(constants.TILESDATAPATH.joinpath('cities_tile_shapes.json')))[city_name]
        self.tiling_dict = json.load(open(constants.TILESDATAPATH.joinpath('tiles_dict.json')))[self.tile_shape]
        self.coordinates = self.tiling_dict['coordinates']
        w_pad = self.tiling_dict['width_padding']
        h_pad = self.tiling_dict['height_padding']

        self.padding_operation_background = nn.ConstantPad2d((h_pad, h_pad, w_pad, w_pad),1)
        self.padding_operation_target     = nn.ConstantPad2d((h_pad, h_pad, w_pad, w_pad),0)

        background = morph_tools.just_open(constants.CITIESFOLDERPATH.joinpath( f'{city_name}/{tileName}/{constants.FILEEXTENSION}'), mode = 'grayscale')

        if n_input_channels ==1:
            self.paddedBackground:torch.Tensor = self.padding_operation_background(ToTensor()(np.transpose(background)))

        elif n_input_channels == 3:
            background = np.concatenate(( np.expand_dims(morph_tools.erosion(background),2), np.expand_dims(background,2), np.expand_dims(morph_tools.dilation(background),2)), axis=2)
            self.paddedBackground:torch.Tensor = self.padding_operation_background(ToTensor()(np.transpose(background,(1,0,2))))

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        coord_dict = self.coordinates[f'{index}']
        sample = {'coord_dict': coord_dict}
        sample['background'] = self.paddedBackground[:,coord_dict['w_low']:coord_dict['w_high'], coord_dict['h_low']:coord_dict['h_high']]
        return sample

