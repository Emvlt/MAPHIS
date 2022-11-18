from typing import List
import shutil
import pathlib
import sys
sys.path.append('..')
import utils.constants as constants

def copy_sample(old_feature:str, new_feature:str, shirt_size:str, old_index:int, new_index:int):
    old_img_path = constants.IMAGESFOLDERPATH.joinpath(f'{old_feature}/{shirt_size}/{old_index}.jpg')
    new_img_path = constants.IMAGESFOLDERPATH.joinpath(f'{new_feature}/{shirt_size}/{new_index}.jpg')
    shutil.copyfile(old_img_path, new_img_path)
    old_mask_path = constants.IMAGESFOLDERPATH.joinpath(f'{old_feature}/{shirt_size}/{old_index}_mask.jpg')
    new_mask_path = constants.IMAGESFOLDERPATH.joinpath(f'{new_feature}/{shirt_size}/{new_index}_mask.jpg')
    shutil.copyfile(old_mask_path, new_mask_path)

def compare_dir_sizes(shirt_size:str, old_features:List, new_feature:str):
    counter = 0
    merged_samples = len(list(constants.IMAGESFOLDERPATH.joinpath(f'{new_feature}/{shirt_size}').glob('*')))
    for old_feature in old_features:
        path_to_folder = constants.IMAGESFOLDERPATH.joinpath(f'{old_feature}/{shirt_size}')
        counter += len(list(path_to_folder.glob('*')))
    return merged_samples == counter

def perform_merge(shirt_size:str, old_features:List, new_feature:str, new_feature_dir:pathlib.Path):
    new_shirt_dir = new_feature_dir.joinpath(shirt_size)
    new_shirt_dir.mkdir(exist_ok=True,parents=True)
    new_index = 0
    for old_feature in old_features:
        print(f'Processing {shirt_size} for {old_feature}')
        path_to_folder = constants.IMAGESFOLDERPATH.joinpath(f'{old_feature}/{shirt_size}')
        n_samples = int(len(list(path_to_folder.glob('*')))/2)
        for old_index in range(n_samples):
            copy_sample(old_feature, new_feature, shirt_size, old_index, new_index)
            new_index += 1

def check_dir_size(shirt_size:str, old_features:List, new_feature:str):
    if compare_dir_sizes(shirt_size, old_features, new_feature):
        print(f'{shirt_size} Features may have been merged already: there are the same number of samples in {old_features} and {new_feature}')
        answer = ''
        while answer not in ['Y','N']:
            answer = input("Would you like to proceed anyway? (Y/N):")
            if answer == 'N':
                print(f'Aborting feature merge from {old_features} to {new_feature}')
                return False
            return True
    return True

def merge(old_features:List, new_feature:str):
    new_feature_dir = constants.IMAGESFOLDERPATH.joinpath(new_feature)
    new_feature_dir.mkdir(exist_ok=True,parents=True)
    for shirt_size in ['xs', 's', 'm', 'l']:
        if check_dir_size(shirt_size, old_features, new_feature ):
            perform_merge(shirt_size, old_features, new_feature, new_feature_dir)

if __name__=='__main__':
    old_features = ['labels', 'stamps_small_font', 'stamps_large_font']
    new_feature  = 'text'
    merge(old_features, new_feature)