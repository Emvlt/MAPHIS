import time
from typing import Dict
import pathlib
import requests
import argparse

from sympy import re
import utils.constants as constants

def json_online_save(request_url:str, tile_dict:Dict):
    print('Uploading json...')
    t0 = time.time()
    res = requests.post(url=request_url, json = tile_dict)
    print(f'The post request {request_url} has returned the status {res.status_code}')
    print(f'Elapsed Time to Upload Data: {time.time()-t0}')

def thumbnail_online_save(thumbnail_path:pathlib.Path, zoom_url:str, x_value:str, y_value:str):
    t0 = time.time()
    request = f'{zoom_url}/{x_value}/{y_value}'
    print(request)
    #res = requests.post(url=request, files={'file':open(thumbnail_path, 'rb')})
    #print(f'The post request {request} has returned the status {res.status_code} in {time.time()-t0:.2f} seconds')
    return 0

def process_zoom_level(city_path:pathlib.Path, zoom_level:int, city_url:str):
    zoomed_folder = city_path.joinpath(f'{zoom_level}')
    zoom_url = f'{city_url}/{zoom_level}'
    for x_folder_path in zoomed_folder.glob('*'):
        if x_folder_path.is_dir():
            x_folder_name = x_folder_path.stem
            for thumbnail_path in x_folder_path.glob('*'):
                file_name = thumbnail_path.stem
                thumbnail_online_save(thumbnail_path, zoom_url, x_folder_name, file_name)
        else:
            print(f'{x_folder_path} is not a folder')

def process_city(city_name:str):
    city_path = constants.TILESDATAPATH.joinpath(city_name)
    city_url = f'http://13.40.112.22/v1alpha1/features/upload/images/{city_name}'
    if city_path.is_dir():
        for zoom_level in range(4,5):
            process_zoom_level(city_path, zoom_level, city_url)

    else:
        print(f'{city_path} is not a folder path, check that you have processed city')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '70', choices = constants.CITYKEY.keys())
    args = parser.parse_args()
    city_name = constants.CITYKEY[args.city_key]['Town']
    process_city(city_name)
