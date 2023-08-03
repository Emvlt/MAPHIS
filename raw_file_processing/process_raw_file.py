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
import utils.constants as constants
import utils.morphological_tools as morph_tools

WINDOWSIZE = int(constants.THUMBNAILSIZE/2)

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
    if key != 'xl' and 530<height*width:
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

def add_feature_to_tile(tile:np.ndarray, city_name:str, tile_name:str, feature_name:str):
    p = constants.RAWPATH / f'{city_name}/{tile_name}/{feature_name}{constants.FILEEXTENSION}'
    if p.is_file():
        print(f'{feature_name} file found, adding...')
        tile = cv2.bitwise_or(tile, morph_tools.closing(np.uint8(morph_tools.open_and_binarise(p, 'grayscale', feature_name))))
    return tile

def load_feature_layer(feature_name:str, city_name:str, tile_name:str):
    target_tile = morph_tools.empty_tile(city_name)
    if feature_name == 'text':
        for sub_feature_name in ['labels', 'stamps_small_font', 'stamps_large_font']:
            target_tile = add_feature_to_tile(target_tile, city_name, tile_name, sub_feature_name)

    else:
        target_tile = add_feature_to_tile(target_tile, city_name, tile_name, feature_name)

    return target_tile

def load_feature_condition(feature_name:str, city_name:str, tile_name:str):
    if feature_name=='text':
        for sub_feature_name in ['labels', 'stamps_small_font', 'stamps_large_font']:
            p = constants.RAWPATH / f'{city_name}/{tile_name}/{sub_feature_name}{constants.FILEEXTENSION}'
            if p.is_file():
                return True
        return False
    else:
        feature_layer_path = constants.RAWPATH.joinpath(f'{city_name}/{tile_name}/{feature_name}{constants.FILEEXTENSION}')
        if feature_layer_path.is_file():
            return True
        return False

def extract_features_with_context_tile(feature_name:str, city_name:str, tile_name:str, real_tile_path:pathlib.Path, save_folder_path:pathlib.Path, n_features:int):
    if load_feature_condition(feature_name, city_name, tile_name):
        print(f'Processing feature {feature_name} for tile {tile_name}')
        feature_layer = load_feature_layer(feature_name, city_name, tile_name)
        # Load the real_tile
        original_tile = np.uint8(morph_tools.just_open(real_tile_path, 'grayscale'))
        # Load the target array (made up of masks of all modalities)
        [tile_width, tile_height] = json.load(open(constants.TILESDATAPATH.joinpath('cities_tile_sizes.json')))[city_name]

        background = np.zeros((tile_height, tile_width), np.uint8)
        target = np.zeros( (1+len(constants.FEATURENAMES), tile_height, tile_width), np.uint8)

        for channel_index, other_feature_name in enumerate(constants.FEATURENAMES):
            print(f'Processing {other_feature_name} ...')
            target_tile = load_feature_layer(other_feature_name, city_name, tile_name)

            target[1+channel_index] = target_tile
            background = cv2.bitwise_or(background,  target_tile)

        background = 1 - background

        # Transpose Arrays
        original_tile = np.transpose(original_tile)
        feature_layer = np.transpose(feature_layer)
        target        = np.transpose(target,(2,1,0))
        background    = np.transpose(background)

        # Pad once for good for bordering features
        original_tile = np.pad(original_tile, ((WINDOWSIZE,WINDOWSIZE),(WINDOWSIZE,WINDOWSIZE)), constant_values=255)
        target = np.pad(target, ((WINDOWSIZE,WINDOWSIZE),(WINDOWSIZE,WINDOWSIZE),(0,0)), constant_values=0)
        background = np.pad(background, ((WINDOWSIZE,WINDOWSIZE),(WINDOWSIZE,WINDOWSIZE)), constant_values=1)

        target[:,:,0] = background

        assert np.shape(original_tile)[:1] == np.shape(target)[:1]

        contours = morph_tools.extract_contours(feature_layer)
        for contour in tqdm(contours):
            area, f_p = morph_tools.is_false_positive(contour)
            if not f_p:
                window_save_path = save_folder_path.joinpath(f'input_{n_features}.npy')
                tgt_save_path    = save_folder_path.joinpath(f'target_{n_features}.npy')

                M = cv2.moments(contour)
                c_x, c_y = int(M['m01'] / (M['m00'] + 1e-5)), int(M['m10'] / (M['m00'] + 1e-5))
                # Get window
                window = original_tile[WINDOWSIZE + c_x-WINDOWSIZE:WINDOWSIZE + c_x+WINDOWSIZE, WINDOWSIZE + c_y-WINDOWSIZE:WINDOWSIZE + c_y+WINDOWSIZE]

                tgt    = target[WINDOWSIZE + c_x-WINDOWSIZE:WINDOWSIZE + c_x+WINDOWSIZE, WINDOWSIZE + c_y-WINDOWSIZE:WINDOWSIZE + c_y+WINDOWSIZE]

                assert np.shape(window) == (512,512)
                assert np.shape(tgt)    == (512,512, 1+len(constants.FEATURENAMES))

                # save matrices
                np.save(window_save_path, window)
                np.save(tgt_save_path, tgt)

                n_features +=1

def check_feature_processed(progress_dict:Dict, city_name:str, feature_name, tile_name:str) -> bool:
    if tile_name in progress_dict[city_name][feature_name]:
        print(f"The feature {feature_name} of the tile {tile_name} has already been extracted, ! extracting it again will duplicate extracted elements ! ")
        answer = ''
        while answer not in ['Y','N']:
            answer = input("Would you like to proceed anyway? (Y/N):")
            if answer == 'N':
                print(f'Aborting feature extraction for tile {tile_name}')
                return False
            return True
    return True

def extract_features_with_context(feature_name:str, city_name:str):
    all_tiles_path = constants.CITIESFOLDERPATH.joinpath(city_name).glob(f'*{constants.FILEEXTENSION}')
    progress_dict_path = constants.TRAININGPATH.joinpath('progress.json')
    if progress_dict_path.is_file():
        progress_dict = json.load(open(progress_dict_path))
        if city_name not in  progress_dict:
            progress_dict = {city_name:{feature_name:[]}}
        elif feature_name not in progress_dict[city_name]:
            progress_dict[city_name][feature_name] = []
    else:
        progress_dict = {city_name:{feature_name:[]}}
        with open(progress_dict_path, 'w') as out_file:
            json.dump(progress_dict, out_file)

    save_folder_path = constants.TRAININGPATH
    save_folder_path.mkdir(exist_ok=True, parents=True)
    n_features = int(len(list(save_folder_path.glob('*.npy')))/2)

    #for real_tile_path in all_tiles_path:
    for real_tile_path in [pathlib.Path(r'D:\MAPHIS\datasets\cities\Luton\0105033010241.jpg')]:
        tile_name = real_tile_path.stem
        if check_feature_processed(progress_dict, city_name, feature_name, tile_name):
            extract_features_with_context_tile(feature_name, city_name, tile_name, real_tile_path, save_folder_path, n_features)
            progress_dict[city_name][feature_name].append(tile_name)

    with open(progress_dict_path, 'w') as out_file:
        json.dump(progress_dict, out_file)

def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    if args.process == 'extract_features_city':
        extract_raw_features_city(args.feature_name, city_name)
    elif args.process == 'extract_features_with_context':
        for ft in constants.FEATURENAMES:
            extract_features_with_context(ft, city_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '36', choices = constants.CITYKEY.keys())
    parser.add_argument('--tile_name', required=False, type=str, default= '0105033010251')
    parser.add_argument('--feature_name', required=False, type=str, default= 'embankments', choices = constants.FEATURENAMES)
    parser.add_argument('--process', required=False, type=str, default= 'extract_features_with_context')
    args = parser.parse_args()
    main(args)
