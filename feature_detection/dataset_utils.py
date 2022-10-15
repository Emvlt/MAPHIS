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
from torch.utils.data import DataLoader

def get_testfile_paths(city_name:str):
    path_to_test_files = constants.CITYPATH[city_name].glob(f'*{constants.FILEEXTENSION}')
    print('Test files paths loaded...')
    return path_to_test_files

class Windows(Dataset):
    def __init__(self, n_input_channels:int) -> None:
        self.path_to_data = constants.TRAININGFOLDERPATH
        self.n_input_channels = n_input_channels

    def __len__(self):
        return int(len((list(self.path_to_data.glob('*'))))/2)

    def __getitem__(self, index):
        image = np.load(self.path_to_data.joinpath(f'input_{index}.npy'))
        target  = np.load(self.path_to_data.joinpath(f'target_{index}.npy'))
        if self.n_input_channels ==3:
            image = np.concatenate(( np.expand_dims(morph_tools.erosion(image),2), np.expand_dims(image,2), np.expand_dims(morph_tools.dilation(image),2)), axis=2)

        # Put to pytorch format

        image, target = torch.from_numpy(np.transpose(image,(2,0,1))), torch.from_numpy(np.transpose(target,(2,0,1)))

        k = np.random.randint(0,4)
        image = torch.rot90(image, k, dims=(1,2))
        target = torch.rot90(target, k, dims=(1,2))
        return image, target

class Thumbnails(Dataset):
    def __init__(self, cityName:str,  tileName:str, n_input_channels:int) -> None:
        print(f'Opening tile {tileName} for the city of {cityName}')
        self.tilingParameters = json.load(open(constants.DATASETFOLDERPATH.joinpath('tiling_parameters.json')))
        self.tilesCoordinates = self.tilingParameters['coordinates']

        self.paddingOpBKG = nn.ConstantPad2d((constants.HEIGHTPADDING, constants.HEIGHTPADDING, constants.WIDTHPADDING, constants.WIDTHPADDING),1)
        self.paddingOpTGT = nn.ConstantPad2d((constants.HEIGHTPADDING, constants.HEIGHTPADDING, constants.WIDTHPADDING, constants.WIDTHPADDING),0)
        background = morph_tools.just_open((constants.CITYPATH[cityName] / f'{tileName}{constants.FILEEXTENSION}'), mode = 'grayscale')

        if n_input_channels ==1:
            self.paddedBackground:torch.Tensor = self.paddingOpBKG(ToTensor()(np.transpose(background)))

        elif n_input_channels == 3:
            background = np.concatenate(( np.expand_dims(morph_tools.erosion(background),2), np.expand_dims(background,2), np.expand_dims(morph_tools.dilation(background),2)), axis=2)
            self.paddedBackground:torch.Tensor = self.paddingOpBKG(ToTensor()(np.transpose(background,(1,0,2))))


    def __len__(self):
        return len(self.tilesCoordinates)

    def __getitem__(self, index):
        coordDict = self.tilesCoordinates[f'{index}']
        sample = {'coordDict': coordDict}
        sample['background'] = self.paddedBackground[:,coordDict['xLow']:coordDict['xHigh'], coordDict['yLow']:coordDict['yHigh']]
        #sample['target'] = self.paddedTarget[:,coordDict['xLow']:coordDict['xHigh'], coordDict['yLow']:coordDict['yHigh']]
        return sample

    def get(self, keyword):
        if keyword =='target':
            return self.paddedTarget.numpy()[0]
        elif keyword =='background':
            return self.paddedBackground.numpy()
