import pathlib
from PIL import Image
import constants

def convert_tiff(tile_name:pathlib.Path, tiff_folder_path:pathlib.Path):
    print(f'Converting tile {tile_name} at path {tiff_folder_path} to jpg...')
    in_path  = tiff_folder_path.joinpath(f'{tile_name}.tif')
    out_path = tiff_folder_path.joinpath(f'{tile_name}.jpg')
    if out_path.is_file():
        print(f'File {out_path.stem}.jpg already exists, passing.')
        return
    #Image.open(in_path).save(out_path, "JPEG", quality=100)

def process_folder(tiff_folder_path:pathlib.Path):
    for tile_path in tiff_folder_path.glob('*.tif'):
        convert_tiff(tile_path.stem, tiff_folder_path)

def process_all_folders():
    for city_name, city_path in constants.CITYPATH.items():
        print(f'Processing city {city_name}')
        process_folder(city_path)

def test_processing(city_name:str, tile_name:str):
    in_path  = constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_name}.tif')
    out_path = constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_name}.jpg')
    Image.open(in_path).save(out_path, "JPEG", quality=100)

if __name__=='__main__':
    process_all_folders()
    #test_processing('Demo', '1301071160071')
