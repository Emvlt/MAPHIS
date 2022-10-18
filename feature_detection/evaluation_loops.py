import sys
import pathlib
from typing import Dict
sys.path.append('..')
import utils.constants as constants
import utils.morphological_tools as morph_tools
import argparse
import loading_utils
import dataset_utils as du
import torch
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw_fake_shapes as df

def infer_tile(tile_dataloader, model, device, n_output_channels):
    segmented = torch.zeros((n_output_channels, constants.TILEWIDTH+2*constants.WIDTHPADDING, constants.TILEHEIGHT+2*constants.HEIGHTPADDING))

    for data in tqdm(tile_dataloader):
        coords = data['coordDict']
        output = model(255*data['background'].float().cuda(device))[0].detach().cpu()
        segmented[:,coords['xLow']:coords['xHigh'], coords['yLow']:coords['yHigh']] += output

    return morph_tools.unpad_tile((segmented.numpy()))

def process_tile(city_name, tile_path:pathlib.Path, save_folder_path:pathlib.Path, hyper_parameters:Dict, model, device:torch.device, n_output_channels:int):
    save_file_path = save_folder_path.joinpath(tile_path.stem)
    if save_file_path.is_dir():
        print(f'Tile {tile_path.stem} at location {save_file_path} has already been processed. \n Would you like to overwrite the saved files?')
        answer = ''
        while answer not in ['Y','N']:
            answer = input("Would you like to proceed anyway? (Y/N):")
            if answer == 'N':
                print(f'Aborting feature extraction for tile {tile_path.stem}')
                return

    save_file_path.mkdir(parents=True, exist_ok=True)
    tile_dataset = du.Thumbnails(cityName=city_name, tileName=tile_path.stem, n_input_channels=hyper_parameters['n_input_channels'])
    tile_dataloader = du.DataLoader(tile_dataset, batch_size=1, shuffle=True, num_workers=0)
    thresholded_tile = np.where(0.95<infer_tile(tile_dataloader, model, device, n_output_channels),1,0)
    print(f'Tile {tile_path.stem} inferred and thresholded.')
    background = thresholded_tile[0]
    if args.save:
        print('Saving the background...')
        cv2.imwrite(f'{save_file_path}/background.jpg', np.uint8(background*255))
        print('Saving the colored image...')
        cv2.imwrite(f'{save_file_path}/colored.jpg', np.transpose(np.uint8(thresholded_tile[1:]*255),[1,2,0]))
        for feature_index, feature_name in enumerate(constants.HIGHLEVELFEATURES):
            print(f'Saving the feature {feature_name}...')
            cv2.imwrite(f'{save_file_path}/{feature_name}.jpg', np.uint8(thresholded_tile[1+feature_index]*255))

def main(args):
    n_output_channels = 1 + len(constants.HIGHLEVELFEATURES)

    hyper_parameters = {
        'n_input_channels': 3
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the city name

    city_name = constants.CITYKEY[args.city_key]['Town']
    print(f'Processing city : {city_name}')
    if args.save:
        save_folder_path = constants.PROCESSEDPATH.joinpath(city_name)
        print(f'Saving files at {save_folder_path}')
    # Get the file_paths
    tile_paths = du.get_testfile_paths(city_name)
    # Load the model
    model = loading_utils.load_model(args.process, ncIn=3, training_type=args.training_type)
    model.to(device)
    # Iterate over tiles and infer the image
    for tile_path in tile_paths:
        if args.save:
            process_tile(city_name, tile_path, save_folder_path, hyper_parameters, model, device, n_output_channels)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default= '0')
    parser.add_argument('--process', required=False, type=str, default= 'segmentation')
    parser.add_argument('--training_type', required=False, type=str, default= 'real')
    parser.add_argument('--save', required=False, type=bool, default= True)
    args = parser.parse_args()
    main(args)
