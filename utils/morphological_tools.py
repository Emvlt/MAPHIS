"""Definition of the morphological tools in one place"""

from typing import List, Tuple
import pathlib

import cv2
import numpy as np
from tqdm import tqdm
from imutils import grab_contours

import utils.constants as constants

def normalise(array:np.ndarray):
    return (array-array.min())/(array.max()-array.min())

def dilation(src:np.float32, dilate_size=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1),
                                    (dilate_size, dilate_size))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilate_size=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_size + 1, 2 * dilate_size + 1), (dilate_size, dilate_size))
    return cv2.erode(src.astype('uint8'), element)

def just_open(path, mode:str) -> np.ndarray:
    path = str(path)
    if mode not in ['grayscale', 'color']:
        raise ValueError(f'Wrong mode {mode} value, must be color or grayscale')

    if mode == 'grayscale':
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return cv2.imread(path, cv2.IMREAD_COLOR)

def open_and_normalise(path, mode:str) -> np.ndarray:
    return normalise(just_open(path, mode))

def open_and_binarise(path, mode:str, feature_name:str) -> np.ndarray:
    return np.where(just_open(path, mode)<=constants.COLORTHRESHOLD[feature_name],1,0)

def blend(image_a, image_b, weight)->np.ndarray:
    background = cv2.addWeighted(image_a, weight, image_b, 1-weight, 0.0)
    return background

def unpad_tile(mat:np.ndarray)->np.ndarray:
    return mat[:,constants.WIDTHPADDING:constants.WIDTHPADDING+constants.TILEWIDTH, constants.HEIGHTPADDING:constants.HEIGHTPADDING+constants.TILEHEIGHT]

def tile_scale(mat:np.ndarray, scaling_matrix_path:str) ->np.ndarray:
    return mat/np.load(scaling_matrix_path)

def threshold(mat:np.ndarray, threshold_value:float)-> np.ndarray:
    return np.where(mat < threshold_value, 0, 1)

def extract_contours(mat:np.ndarray) -> List:
    contours = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    return contours

def empty_tile(val=None):
    return np.zeros( (constants.TILEHEIGHT, constants.TILEWIDTH), np.uint8) if val is None else np.ones( (constants.TILEHEIGHT, constants.TILEWIDTH), np.uint8)*val

def load_background(city_name, tile_name):
    target = empty_tile()
    for background_element in constants.BACKGROUNDKWDS:
        path = constants.RAWPATH / f'{city_name}/{tile_name}/{background_element}{constants.FILEEXTENSION}'
        if path.is_file():
            target = cv2.bitwise_or(target, np.uint8(open_and_binarise(str(path), 'grayscale', background_element)))
    return target

def load_high_level_feature(city_name, tile_name, high_level_feature_name):
    target = empty_tile()
    for feature_name in constants.HIGHLEVELFEATURES[high_level_feature_name]:
        path = constants.RAWPATH / f'{city_name}/{tile_name}/{feature_name}{constants.FILEEXTENSION}'
        if path.is_file():
            target = cv2.bitwise_or(target, np.uint8(open_and_binarise(str(path), 'grayscale', feature_name)))
    return target

def is_false_positive(contour:List) -> Tuple[float, bool]:
    area = cv2.contourArea(contour)
    if area < 530:
        return area, True
    return area, False

def histo(array, bins=None)->np.ndarray:
    if bins is None:
        return np.histogram(array, bins=100)
    return np.histogram(array, bins=bins)

def compute_tile_histogram(file_path:pathlib.Path, feature_name:str, areas:list, heights:list, widths:list):
    print(f'Processing feature {feature_name} on tile {file_path.parent}/{file_path.stem}')
    if file_path.is_file():
        mat = np.uint8(open_and_binarise(str(file_path), 'grayscale', feature_name))
        contours = extract_contours(mat)
        for contour in tqdm(contours):
            area, false_positive = is_false_positive(contour)
            if not false_positive:
                (_,_, width, height) = cv2.boundingRect(contour)
                areas.append(area)
                heights.append(height)
                widths.append(width)
    return areas, heights, widths

def compute_shape_histogram(test_folders_paths:List[pathlib.Path], feature_name:str, to_save:bool):
    # feature_name is now text, imprints, trees
    # for each tile, we have to compile the shape distributions of each sub-feature
    # we finally return a dictionnary of histogram and bins.
    histogram_dict = {shape_dist_name:{'histogram':[], 'bins':[]} for shape_dist_name in constants.SHAPEDISTRIBUTION}
    widths_list = []
    heights_list = []
    areas_list = []
    for test_folder_path in test_folders_paths:
        print(f'Processing folder {test_folder_path.stem}')
        if feature_name in constants.HIGHLEVELFEATURES:
            for sub_feature_name in constants.HIGHLEVELFEATURES[feature_name]:
                file_path = constants.RAWPATH.joinpath(f'{test_folder_path.parent.stem}/{test_folder_path.stem}/{sub_feature_name}{constants.FILEEXTENSION}')
                if file_path.exists():
                    areas_list, heights_list, widths_list = compute_tile_histogram(file_path, sub_feature_name, areas_list, heights_list, widths_list)
                else:
                    print(f'No {sub_feature_name} for tile {file_path.stem}, passing...')

        else:
            file_path = constants.RAWPATH.joinpath(f'{test_folder_path.parent.stem}/{test_folder_path.stem}/{feature_name}{constants.FILEEXTENSION}')
            if file_path.exists():
                areas_list, heights_list, widths_list = compute_tile_histogram(file_path, feature_name, areas_list, heights_list, widths_list)
            else:
                print(f'No {feature_name} for tile {file_path.stem}, passing...')

    if to_save:
        area_histo, area_bins = histo(areas_list)
        histogram_dict['A_distribution']['histogram'] = area_histo.tolist()
        histogram_dict['A_distribution']['bins'] = area_bins.tolist()
        height_histo, height_bins = histo(heights_list)
        histogram_dict['H_distribution']['histogram'] = height_histo.tolist()
        histogram_dict['H_distribution']['bins'] = height_bins.tolist()
        width_histo, width_bins = histo(widths_list)
        histogram_dict['W_distribution']['histogram'] = width_histo.tolist()
        histogram_dict['W_distribution']['bins'] = width_bins.tolist()
    else:
        area_histo, area_bins = histo(areas_list)
        histogram_dict['A_distribution']['histogram'] = area_histo
        histogram_dict['A_distribution']['bins'] = area_bins
        height_histo, height_bins = histo(heights_list)
        histogram_dict['H_distribution']['histogram'] = height_histo
        histogram_dict['H_distribution']['bins'] = height_bins
        width_histo, width_bins = histo(widths_list)
        histogram_dict['W_distribution']['histogram'] = width_histo
        histogram_dict['W_distribution']['bins'] = width_bins
    return histogram_dict
