import pathlib
from PIL import Image
import constants
import numpy as np
import cv2
from tqdm import tqdm

def convert_tiff(tile_name:pathlib.Path, tiff_folder_path:pathlib.Path):
    print(f'Converting tile {tile_name} at path {tiff_folder_path} to jpg...')
    in_path  = tiff_folder_path.joinpath(f'{tile_name}.tif')
    out_path = tiff_folder_path.joinpath(f'{tile_name}.jpg')
    if out_path.is_file():
        print(f'File {out_path.stem}.jpg already exists, passing.')
        return
    cv2.imwrite(str(out_path), np.uint8(np.asarray(Image.open(in_path))*255))

def process_folder(tiff_folder_path:pathlib.Path):
    for tile_path in tqdm(tiff_folder_path.glob('*.tif')):
        convert_tiff(tile_path.stem, tiff_folder_path)

def process_all_folders():
    for city_dict in constants.CITYKEY.values():
        city_name = city_dict['Town']
        city_path = constants.CITIESFOLDERPATH.joinpath(city_name)
        if city_path.is_dir():
            print(f'Processing city {city_name}')
            process_folder(city_path)
        else:
            print(f'{city_path} is not a valid path...')
            input('Would you like to continue?')

def test_processing(city_name:str, tile_name:str):
    in_path  = constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_name}.tif')
    out_path = constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_name}.jpg')
    cv2.imwrite(out_path, np.uint8(np.asarray(Image.open(in_path))*255))

if __name__=='__main__':
    process_all_folders()
