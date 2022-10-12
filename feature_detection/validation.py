## We want to write scripts to assess the performance of a feature extraction procedure
## The reference data are:
##      - a json file with the properties of all reference shapes
##      - a jpg file with the reference image
## We want to achieve:
##      - a ratio between detected area and reference area close to 1.
##      - minimise the number of false positives.
##      - make sure the detected and reference objects at position (x,y) have the same properties.
## We then want the output of the NN to be a binary map
##      --> sigmoid as activation function + nn.Threshold?
import pathlib
import sys
import json
from typing import Dict, List
import torch
from torch import  nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.append('..')
import utils.constants as constants
import models.models as models
from dataset_utils import get_testfile_paths, Thumbnails, re_tile
from loading_utils import load_model
from metrics_utils import write_test_metrics, report_to_tune

import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import utils.morphological_tools as morph_tools

import numpy as np
import cv2

import os
from draw_fake_shapes import make_batch

def test_loop(test_folders_path:List[pathlib.Path], feature_name:str, operation:str, device:torch.cuda.Device, model, hyper_parameters:Dict, histo_dict:Dict, n_channels:int) -> Dict:

    test_loss = {'BCE':0, 'area':0, 'shape_distributions':{'A_distribution':[], 'H_distribution':[], 'W_distribution':[]}}
    print('------ Beginning Test Loop ------')
    for test_path in test_folders_path:
        city_name = test_path.parent.stem

        print(f'Testing file {test_path.stem}')
        segmented, masked, target = re_tile(city_name, test_path, feature_name, n_channels, model, device)
        #segmented = morph_tools.unpad(morph_tools.tile_scale(segmented, str(constants.DATASETFOLDERPATH / 'scaling_mat.npy')))
        #thresholded = morph_tools.threshold(segmented, hyper_parameters['threshold_value'])
        ## Qualitative inspection
        segmented = morph_tools.threshold(segmented, 0.9)
        masked = morph_tools.threshold(masked, 0.9)
        images_path = constants.MAPHISFOLDERPATH.joinpath(f'feature_detection/images/{city_name}/{test_path.stem}')
        images_path.mkdir(exist_ok=True, parents=True)
        masked = np.transpose(masked)
        segmented = np.transpose(segmented)
        target = np.transpose(morph_tools.normalise(target))
        cv2.imwrite(str(images_path / f'{feature_name}_{operation}_target.jpg'), np.uint8(target*255))
        ## Quantitative assessment
        #   - Loss function : BCE
        #test_loss['BCE']  += BCE
        #   - Area covered  : np.sum(cv2.bitwise_and(target, segmented))/np.sum(target)
        target = np.uint8(target)
        segmented = np.uint8(segmented)
        masked = np.uint8(masked)
        A = cv2.bitwise_and(target, segmented)
        AC = (np.sum(A)/np.sum(target) )
        print(f'----- Total area coverage for {feature_name} : {AC} -----')
        for kwd in constants.HIGHLEVELFEATURES[feature_name]:
            target = morph_tools.open_and_binarise(str(constants.RAWPATH.joinpath(f'{city_name}/{test_path.stem}/{kwd}{constants.FILEEXTENSION}')), kwd, 1)
            A = cv2.bitwise_and(np.uint8(target), np.uint8(segmented))
            AC = (np.sum(A)/np.sum(target) )
            M = cv2.bitwise_and(np.uint8(target), np.uint8(masked))
            cv2.imwrite(str(images_path / f'{feature_name}_{operation}_{kwd}.jpg'),  M*255)
            print(f'----- Area coverage for sub-feature {kwd} : {AC} -----')

        segmented = np.uint8(segmented*255)
        cv2.imwrite(str(images_path / f'{feature_name}_{operation}.jpg'), np.uint8(masked*255))
        cv2.imwrite(str(images_path / f'{feature_name}_{operation}_segmented.jpg'), segmented)


        #test_loss['area'] += AC / len(test_folders_path)
        #   - Shape_distribution :
        contours = morph_tools.extract_contours(segmented)
        test_histo_dict = morph_tools.parse_contours_to_histogram(contours, histo_dict)
        for distribution_kwd in constants.SHAPEDISTRIBUTION:
            test_loss['shape_distributions'][distribution_kwd] = test_histo_dict[distribution_kwd]

    return test_loss

def train_loop_real(feature_name:str, city_name:str, operation:str, device:torch.cuda.Device, model, hyper_parameters:Dict, training_type='tile_only'):
    train_folders_path, test_folders_path = split_folders(city_name, 0.95)
    model = load_model(feature_name, operation, training_type, device)

    optimizer = optim.Adam(model.unet.parameters(), lr=hyper_parameters['learning_rate'])

    criterion = nn.BCELoss()
    kl_loss   = nn.KLDivLoss()

    writer = SummaryWriter(log_dir = f'runs/{feature_name}')
    statistics_file_path = constants.STATISTICSPATH.joinpath(f'{feature_name}.json')

    if not statistics_file_path.is_file():
        print(f'Computing {feature_name} shape statistics')
        histo_dict = morph_tools.compute_shape_histogram(test_folders_path, feature_name, to_save=True)
        with open(statistics_file_path, 'w') as file:
            json.dump(histo_dict, file, indent=4)

    histo_dict:Dict = json.load(open(statistics_file_path))
    ## Write start statistics
    for key in histo_dict.keys():
        histo_dict[key]['histogram'] = torch.FloatTensor(histo_dict[key]['histogram'])
        writer.add_histogram(f'{key} target', histo_dict[key]['histogram'], global_step=0)

    #epoch_loss_test:Dict = test_loop(test_folders_path, feature_name, operation, device, model, hyper_parameters, histo_dict, hyper_parameters['n_channels'])
    #write_test_metrics(writer, epoch_loss_test, histo_dict, kl_loss, step=0)

    print('------ Beginning Train Loop ------')
    for epoch in range(hyper_parameters['epochs']):
        epoch_loss_train = 0
        for j, train_path in enumerate(train_folders_path):
            print(f'Training on file {train_path.stem}')
            tile_loss = 0
            thumbnails_dataset = Thumbnails(cityName=city_name, tileName=train_path.stem, high_level_feature_name=feature_name)
            tileDataloader = DataLoader(thumbnails_dataset, batch_size=4, shuffle=True, num_workers=2)
            for i, data in enumerate(tileDataloader):
                optimizer.zero_grad()
                tile, target = data['background'].float().cuda(device), data['target'].float().cuda(device)
                out  = model(tile)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                tile_loss += loss.item()

            epoch_loss_train += tile_loss

        g_step =int(epoch*len(train_folders_path)+j)
        writer.add_scalar(tag='epoch_loss_train', scalar_value=epoch_loss_train/len(train_folders_path), global_step=g_step)

        #epoch_loss_test:Dict = test_loop(test_folders_path, feature_name, operation, device, model, hyper_parameters, histo_dict,  hyper_parameters['n_channels'])
        #write_test_metrics(writer, epoch_loss_test, histo_dict, kl_loss, step=g_step)
        torch.save(model.state_dict(), constants.MODELSPATH / f'saves/{feature_name}_{operation}_{training_type}_state_dict.pth')


def infer_tiles(test_folders_path, feature_name, model, device, n_channels):
    for test_path in test_folders_path:
        infer_tile(test_path, feature_name, model, device, n_channels)

def infer_tile(test_path:pathlib.Path, feature_name, model, device, n_channels):
    city_name = test_path.parent.stem
    print(f'Testing file {test_path.stem}')
    segmented, masked, target = re_tile(city_name, test_path, feature_name, n_channels, model, device)
    #segmented = morph_tools.unpad(morph_tools.tile_scale(segmented, str(constants.DATASETFOLDERPATH / 'scaling_mat.npy')))
    #thresholded = morph_tools.threshold(segmented, hyper_parameters['threshold_value'])
    ## Qualitative inspection
    segmented = morph_tools.threshold(segmented, 0.9)
    segmented = np.uint8(segmented*255)
    images_path = constants.MAPHISFOLDERPATH.joinpath(f'feature_detection/images/{city_name}/{test_path.stem}')
    images_path.mkdir(exist_ok=True, parents=True)
    segmented = np.transpose(segmented)
    cv2.imwrite(str(images_path / f'{feature_name}.jpg'), segmented)

def inference_loop(feature_name:str, city_name:str, operation:str, device:torch.cuda.Device, model, hyper_parameters:Dict, training_type='tile_only' ):
    test_folders_path = split_folders(city_name, 0, 'evaluation')
    model = load_model(feature_name, operation, training_type)
    device = "cuda:0"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    infer_tiles(test_folders_path, feature_name, model, device, hyper_parameters['n_channels'])


def evaluation_loop(feature_name:str, city_name:str, operation:str, device:torch.cuda.Device, model, hyper_parameters:Dict, training_type='tile_only' ):
    test_folders_path = split_folders(city_name, 0, 'evaluation')
    model = load_model(feature_name, operation, training_type)
    device = "cuda:0"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    kl_loss   = nn.KLDivLoss()
    writer = SummaryWriter(log_dir = f'runs/{feature_name}')
    statistics_file_path = constants.STATISTICSPATH.joinpath(f'{feature_name}.json')

    if not statistics_file_path.is_file():
        print(f'Computing {feature_name} shape statistics')
        histo_dict = morph_tools.compute_shape_histogram(test_folders_path, feature_name, to_save=True)
        with open(statistics_file_path, 'w') as file:
            json.dump(histo_dict, file, indent=4)

    histo_dict:Dict = json.load(open(statistics_file_path))
    ## Write start statistics
    for key in histo_dict.keys():
        histo_dict[key]['histogram'] = torch.FloatTensor(histo_dict[key]['histogram'])
        writer.add_histogram(f'{key} target', histo_dict[key]['histogram'], global_step=0)

    epoch_loss_test:Dict = test_loop(test_folders_path, feature_name, operation, device, model, hyper_parameters, histo_dict, hyper_parameters['n_channels'])   #
    write_test_metrics(writer, epoch_loss_test, histo_dict, kl_loss, step=0)


def main(args):
    city_name = constants.CITYKEY[args.cityKey]['Town']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyper_parameters = {
        'learning_rate': 1e-4,
        'batch_size':8,
        'epochs':40,
        'threshold_value':0.9,
        'n_samples':1000,
        'n_channels':3
    }
    config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([2, 4, 8, 16])
    }
    #train_loop_synthetic( args.featureName, city_name,  args.process ,device, model=None, hyper_parameters=hyper_parameters)
    #evaluation_loop( args.featureName, city_name,  args.process ,device, model=None, hyper_parameters=hyper_parameters)
    inference_loop( args.featureName, city_name,  args.process ,device, model=None, hyper_parameters=hyper_parameters)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--featureName', required=False, type=str, default= 'buildings')
    parser.add_argument('--process', required=False, type=str, default= 'segmentation')
    args = parser.parse_args()
    main(args)
