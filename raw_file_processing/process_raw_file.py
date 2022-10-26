"""Scripts to process raw feature extraction"""

from typing import Dict, List
import pathlib
import argparse
import sys

import json
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from utils import constants
from utils import morphological_tools as morph_tools

def get_key(max_dim:int, max_size_dict):
    for key, size in max_size_dict.items():
        if max_dim<=size:
            return key
    return 'xl'

def trim_and_save(mat:np.ndarray, mask:np.ndarray, key:str, indices_dict:Dict, save_path : pathlib.Path):
    if np.any(mat):
        contours = morph_tools.extract_contours(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY))
        (pos_x, pos_y, width, height) = cv2.boundingRect(contours[0])
        key = get_key(max(width,height), constants.DRAWING['max_size_dict'])
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}{constants.FILEEXTENSION}')), mat[pos_y:pos_y+height, pos_x:pos_x+width])
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}_mask{constants.FILEEXTENSION}')), mask[pos_y:pos_y+height, pos_x:pos_x+width])
        indices_dict[key] +=1
    return indices_dict

def tile_and_save(mat, height, width, key, save_path, indices_dict, zer):
    if key != 'xl':
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}{constants.FILEEXTENSION}')), mat)
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}_mask{constants.FILEEXTENSION}')), zer)
        indices_dict[key] +=1
    else:
        q_w, r_w = width//(362), width%362
        q_h, r_h = height//(362), height%362
        if q_h==0 or q_w==0:
            if q_h == 0:
                for i in range(q_w+1):
                    if i ==0:
                        to_save     = mat[:,:362]
                        to_save_zer = zer[:,:362]
                    else:
                        to_save     = mat[:,(i-1)*362+r_w:i*362+r_w]
                        to_save_zer = zer[:,(i-1)*362+r_w:i*362+r_w]

                    indices_dict = trim_and_save(to_save, to_save_zer, key, indices_dict, save_path)

            else:
                for j in range(q_h+1):
                    if j==0:
                        to_save     = mat[:362]
                        to_save_zer = zer[:362]
                    else:
                        to_save     = mat[(j-1)*362+r_h:j*362+r_h]
                        to_save_zer = zer[(j-1)*362+r_h:j*362+r_h]

                    indices_dict = trim_and_save(to_save, to_save_zer, key, indices_dict, save_path)

        else:
            for i in range(q_w+1):
                for j in range(q_h+1):
                    if j==0:
                        to_save     = mat[:362,:362]
                        to_save_zer = zer[:362,:362]
                    else:
                        to_save     = mat[(j-1)*362+r_h:j*362+r_h,(i-1)*362+r_w:i*362+r_w]
                        to_save_zer = zer[(j-1)*362+r_h:j*362+r_h,(i-1)*362+r_w:i*362+r_w]

                    indices_dict = trim_and_save(to_save, to_save_zer, key, indices_dict, save_path)

    return indices_dict

def process_contour(contour:List, background:np.ndarray, save_path:pathlib.Path, indices_dict:Dict)-> Dict:
    (pos_x, pos_y, width, height) = cv2.boundingRect(contour)
    zero_mat = np.zeros((height, width,3), np.uint8)
    cv2.drawContours(zero_mat, [contour], 0, (255,255,255), -1, offset=(-pos_x, -pos_y))
    image = cv2.bitwise_and(background[pos_y:pos_y+height, pos_x:pos_x+width], zero_mat)
    key = get_key(max(width,height), constants.DRAWING['max_size_dict'])
    return tile_and_save(image, height, width, key, save_path, indices_dict, zero_mat)

def extract_features_tile(feature_name:str, city_name:str, tile_path:pathlib.Path, progress_dict:Dict):
    if feature_name not in progress_dict.keys():
        progress_dict[city_name][feature_name] = {tile_path.stem:''}

    if  progress_dict[city_name][feature_name][tile_path.stem] == 'True':
        print(f"The feature {feature_name} of the tile {tile_path.stem} has already been extracted, ! extracting it again will duplicate extracted elements ! ")
        answer = ''
        while answer not in ['Y','N']:
            answer = input("Would you like to proceed anyway? (Y/N):")
            if answer == 'N':
                print(f'Aborting feature extraction for tile {tile_path.stem}')
                return

    print(f'Processing feature {feature_name} on tile {tile_path.stem}')
    save_path = constants.IMAGESFOLDERPATH.joinpath(f'{feature_name}')
    indices_dict = {}
    for tshirt_size in constants.DRAWING['max_size_dict']:
        save_path.joinpath(tshirt_size).mkdir(exist_ok=True, parents=True)
        indices_dict[tshirt_size] = int(len(list(save_path.joinpath(tshirt_size).glob(f'*{constants.FILEEXTENSION}')))/2)
    file_path = tile_path.joinpath(f'{feature_name}{constants.FILEEXTENSION}')
    feature_layer = np.uint8(morph_tools.open_and_binarise(str(file_path), 'grayscale', feature_name))
    background:np.ndarray = np.uint8(morph_tools.just_open(str(constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_path.stem}{constants.FILEEXTENSION}')), 'color'))
    contours = morph_tools.extract_contours(feature_layer)
    for contour in tqdm(contours):
        _, false_positive = morph_tools.is_false_positive(contour)
        if not false_positive:
            indices_dict = process_contour(contour, background, save_path, indices_dict)
    progress_dict[city_name][feature_name][tile_path.stem] = 'True'

def extract_raw_features_city(feature_name:str, city_name:str):
    city_path = constants.RAWPATH.joinpath(f'{city_name}').glob('*')
    with open(constants.IMAGESFOLDERPATH.joinpath('progress.json'), encoding="utf-8") as progress_dict_path:
        progress_dict = json.load(progress_dict_path)
    for tile_path in city_path:
        if tile_path.is_dir():
            if tile_path.joinpath(f'{feature_name}.jpg').is_file():
                print('Beginning feature extraction')
                extract_features_tile(feature_name, city_name, tile_path, progress_dict)
            else:
                print(f'No file found for {tile_path.stem} and feature name {feature_name}, passing')
                progress_dict[city_name][feature_name][tile_path.stem] = 'NA'

def extract_all_raw_features(city_name:str):
    for feature_name in constants.FEATURENAMES:
        extract_raw_features_city(feature_name, city_name)

def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    if args.process == 'extract_features_city':
        extract_raw_features_city(args.feature_name, city_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '36', choices = constants.CITYKEY.keys())
    parser.add_argument('--tile_name', required=False, type=str, default= '0105033010251')
    parser.add_argument('--feature_name', required=False, type=str, default= 'embankments', choices = constants.FEATURENAMES)
    parser.add_argument('--process', required=False, type=str, default= 'extract_features_city')
    args = parser.parse_args()
    main(args)
