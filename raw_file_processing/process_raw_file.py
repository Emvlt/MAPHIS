from typing import Dict
import cv2
import pathlib
from cv2 import bitwise_and
from imutils import grab_contours
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import utils.constants as constants
import utils.morphological_tools as morph_tools   

def getBoundaries(tfwDataPath:pathlib.Path) -> dict:
    tfwRaw = open(tfwDataPath, 'r').read()
    xDiff = float(tfwRaw.split("\n")[0])
    yDiff = float(tfwRaw.split("\n")[3])
    westBoundary = float(tfwRaw.split("\n")[4])
    northBoundary = float(tfwRaw.split("\n")[5])
    eastBoundary = westBoundary + (constants.TILEWIDTH - 1) * xDiff
    southBoundary = northBoundary + (constants.TILEHEIGHT - 1) * yDiff
    return {'westBoundary':westBoundary, 'northBoundary':northBoundary,
            'eastBoundary':eastBoundary, 'southBoundary':southBoundary, 
            'xDiff':xDiff, 'yDiff':yDiff, 'lattitude_length': (constants.TILEWIDTH - 1) * xDiff, 'longitude_length':(constants.TILEHEIGHT - 1) * yDiff}



def get_key(max_dim:int, max_size_dict):
    for key, size in max_size_dict.items():
        if max_dim<=size: 
            return key 
    return 'xl'

def trim_and_save(mat:np.ndarray, mask:np.ndarray, key:str, indices_dict:Dict, save_path : pathlib.Path):
    if np.any(mat):
        contours = morph_tools.extract_contours(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY))
        (x, y, w, h) = cv2.boundingRect(contours[0])
        key = get_key(max(w,h), constants.DRAWING['max_size_dict'])
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}{constants.FILEEXTENSION}')), mat[y:y+h, x:x+w])
        cv2.imwrite(str(save_path.joinpath(f'{key}/{indices_dict[key]}_mask{constants.FILEEXTENSION}')), mask[y:y+h, x:x+w])
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

def extract_features(feature_name, city_name):
    
    city_path = constants.RAWPATH.joinpath(f'{city_name}').glob('*')
    
    for tile_path in city_path:
        if tile_path.is_dir():
            for sub_feature_name in constants.HIGHLEVELFEATURESDRAW[feature_name]:
                save_path = constants.B_IMAGESFOLDERPATH.joinpath(f'{feature_name}/{sub_feature_name}')
                save_path.joinpath('xs').mkdir(exist_ok=True, parents=True)
                save_path.joinpath('s').mkdir(exist_ok=True, parents=True)
                save_path.joinpath('m').mkdir(exist_ok=True, parents=True)
                save_path.joinpath('l').mkdir(exist_ok=True, parents=True)
                xs_n = int(len(list(save_path.joinpath('xs').glob(f'*{constants.FILEEXTENSION}')))/2)
                s_n  = int(len(list(save_path.joinpath('s').glob(f'*{constants.FILEEXTENSION}')))/2)
                m_n  = int(len(list(save_path.joinpath('m').glob(f'*{constants.FILEEXTENSION}')))/2)
                l_n  = int(len(list(save_path.joinpath('l').glob(f'*{constants.FILEEXTENSION}')))/2)
                indices_dict = {'xs':xs_n, 's':s_n, 'm':m_n, 'l':l_n}

                filePath = tile_path.joinpath(f'{sub_feature_name}{constants.FILEEXTENSION}')
                print(f'Processing feature {sub_feature_name} on tile {tile_path.stem}')
                if filePath.is_file():
                    featureLayer = np.uint8(morph_tools.open_and_binarise(str(filePath), sub_feature_name, ratio=1))
                    bkg = np.uint8(morph_tools.open_and_colorise(str(constants.CITYPATH[city_name].joinpath(f'{tile_path.stem}{constants.FILEEXTENSION}')), ratio=1))

                    contours = morph_tools.extract_contours(featureLayer)

                    for contour in tqdm(contours):
                        area, false_positive = morph_tools.is_false_positive(contour)
                        if not false_positive:                
                            (x, y, W, H) = cv2.boundingRect(contour)
                            zer = np.zeros((H, W,3), np.uint8)
                            cv2.drawContours(zer, [contour], 0, (255,255,255), -1, offset=(-x, -y))                            
                            a = bitwise_and(bkg[y:y+H, x:x+W], zer)
                            key = get_key(max(W,H), constants.DRAWING['max_size_dict'])
                            indices_dict = tile_and_save(a, H, W, key, save_path, indices_dict, zer)                            
                            
        
def process_file(cityName, tileName, featureName):
    print(f'City: {cityName} ; \n Tile: {tileName}; \n Feature: {featureName}')
    pathToShapeFolder = constants.MAPHISFOLDERPATH.joinpath(f'/datasets/extractedShapes/{cityName}/{tileName}/{featureName}')
    pathToShapeFolder.mkdir(parents=True, exist_ok=True)

    filePath = constants.MAPHISFOLDERPATH.joinpath(f'datasets/raw/{cityName}/{tileName}/{featureName}.jpg')
    if not filePath.is_file():
        print(f'{featureName}.jpg not in {filePath}')
        return None    

    jsonSavePath = constants.MAPHISFOLDERPATH.joinpath(f'datasets/layers/{featureName}/{cityName}')
    jsonSavePath.mkdir(parents=True, exist_ok=True)

    featureLayer = np.uint8(morph_tools.open_and_binarise(str(filePath), featureName, ratio=1))
    
    contours = cv2.findContours(featureLayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)

    shapeDict = {"mapName":f'{tileName}', f'{featureName}':{}}

    index = 0
    for contour in tqdm(contours):
        if not morph_tools.is_false_positive(contour, featureName):
            
            area = cv2.contourArea(contour)
            shapeDict[f'{featureName}'][f'{index}'] ={}
            shapeDict[f'{featureName}'][f'{index}']['area'] = area
            M = cv2.moments(contour)
            perimeter = cv2.arcLength(contour,True)
            shapeDict[f'{featureName}'][f'{index}']['perimeter'] = perimeter
            _, radiusEnclosingCircle = cv2.minEnclosingCircle(contour)
            areaCircle = 3.1416 * radiusEnclosingCircle * radiusEnclosingCircle
            circleness = area/areaCircle
            shapeDict[f'{featureName}'][f'{index}']['circleness'] = circleness
            (x, y, W, H) = cv2.boundingRect(contour)
            shapeDict[f'{featureName}'][f'{index}']['rectangleness'] = area / (W*H)
            shapeDict[f'{featureName}'][f'{index}']['H'] = H 
            shapeDict[f'{featureName}'][f'{index}']['W'] = W
            shapeDict[f'{featureName}'][f'{index}']['xTile'] = x
            shapeDict[f'{featureName}'][f'{index}']['yTile'] = y
            shapeDict[f'{featureName}'][f'{index}']['savePath'] = str(pathToShapeFolder / f'{index}.npy')
            np.save(pathToShapeFolder / f'{index}.npy', contour)
            index+=1

    

    with open(jsonSavePath /f'{tileName}.json', 'w') as outfile:
        json.dump(shapeDict, outfile, indent=4)

def main(args):
    cityName = constants.CITYKEY[args.cityKey]['Town']
    if args.process == 'display':
        display_file(cityName, tileName=args.tileName, featureName=args.featureName)
    elif args.process == 'process':
        display_file(cityName, tileName=args.tileName, featureName=args.featureName)
        process_file(cityName, tileName=args.tileName, featureName=args.featureName)
    elif args.process == 'contourPpties':
        get_contour_ppties(cityName, tileName=args.tileName, featureName=args.featureName)
    elif args.process == 'extract_features':
        extract_features(args.featureName, cityName)

    else:
        raise ValueError ('Wrong process arg value')

       
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--tileName', required=False, type=str, default= '0105033010251')
    parser.add_argument('--featureName', required=False, type=str, default= 'neighbourhoods')
    parser.add_argument('--process', required=False, type=str, default= 'contourPpties')
    args = parser.parse_args()
    main(args)


