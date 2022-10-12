"""Scripts to process raw feature extraction"""

from typing import Dict
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

def tile_and_save(mat,H, W, key, save_path, indices_dict, zer):
    if key != 'xl':
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}{constants.FILEEXTENSION}')), mat)
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}_mask{constants.FILEEXTENSION}')), zer)
        indices_dict[key] +=1
    else:
        q_w, r_w = W//(362), W%362
        q_h, r_h = H//(362), H%362
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

def extract_features_tile(feature_name:str, city_name:str, tile_path:pathlib.Path):
    print(f'Processing feature {feature_name} on tile {tile_path.stem}')
    save_path = constants.IMAGESFOLDERPATH.joinpath(f'{feature_name}')
    indices_dict = {}
    for tshirt_size in constants.DRAWING['max_size_dict']:
        save_path.joinpath(tshirt_size).mkdir(exist_ok=True, parents=True)
        indices_dict[tshirt_size] = int(len(list(save_path.joinpath(tshirt_size).glob(f'*{constants.FILEEXTENSION}')))/2)
    print(indices_dict)
    file_path = tile_path.joinpath(f'{feature_name}{constants.FILEEXTENSION}')
    print(str(constants.CITYPATH[city_name].joinpath(f'{tile_path.stem}{constants.FILEEXTENSION}')))
    feature_layer = np.uint8(morph_tools.open_and_binarise(str(file_path), 'grayscale', feature_name))
    background:np.ndarray = np.uint8(morph_tools.just_open(str(constants.CITYPATH[city_name].joinpath(f'{tile_path.stem}{constants.FILEEXTENSION}')), 'color'))

    contours = morph_tools.extract_contours(feature_layer)

    for contour in tqdm(contours):
        _, false_positive = morph_tools.is_false_positive(contour)
        if not false_positive:
            (pos_x, pos_y, width, height) = cv2.boundingRect(contour)
            zer = np.zeros((height, width,3), np.uint8)
            cv2.drawContours(zer, [contour], 0, (255,255,255), -1, offset=(-pos_x, -pos_y))
            image = cv2.bitwise_and(background[pos_y:pos_y+height, pos_x:pos_x+width], zer)
            key = get_key(max(width,height), constants.DRAWING['max_size_dict'])
            indices_dict = tile_and_save(image, height, width, key, save_path, indices_dict, zer)

def extract_raw_features_city_wide(feature_name:str, city_name:str):
    city_path = constants.RAWPATH.joinpath(f'{city_name}').glob('*')
    for tile_path in city_path:
        if tile_path.is_dir():
            if tile_path.joinpath(f'{feature_name}.jpg').is_file():
                extract_features_tile(feature_name, city_name, tile_path)

def extract_all_raw_features(city_name:str):
    for feature_name in constants.FEATURENAMES:
        extract_raw_features_city_wide(feature_name, city_name)

'''
def write_json_progress():
    progress_dict = {'Luton':{}}
    path = constants.IMAGESFOLDERPATH
    for feature_name in ['buildings', 'labels', 'rail', 'rivers', 'stamps_large_font', 'stamps_small_font', 'trees']:
        progress_dict['Luton'][feature_name] = {}
        for tile_path in constants.RAWPATH.joinpath('Luton').glob('*'):
            if tile_path.joinpath(f'{feature_name}.jpg').is_file():
                progress_dict['Luton'][feature_name][tile_path.stem] = 'True'
            else:
                progress_dict['Luton'][feature_name][tile_path.stem] = 'NA'
    with open(path.joinpath(f'progress.json'), 'w') as out_file:
            out_file.write(json.dumps(progress_dict, indent=4))
'''

def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    if args.feature_name not in constants.FEATURENAMES:
        raise ValueError ('Wrong feature_name argument')
    if args.process == 'extract_features_city_wide':
        extract_raw_features_city_wide(args.feature_name, city_name)
    else:
        raise ValueError ('Wrong process arg value')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '36')
    parser.add_argument('--tile_name', required=False, type=str, default= '0105033010251')
    parser.add_argument('--feature_name', required=False, type=str, default= 'embankments')
    parser.add_argument('--process', required=False, type=str, default= 'extract_features_city_wide')
    args = parser.parse_args()
    #main(args)
