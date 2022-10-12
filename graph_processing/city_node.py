
import statistics
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pathlib
import cv2
import sys
sys.path.append('..')
import utils.morphological_tools as morph_tools
import utils.constants as constants 
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Tuple, Dict
from edge import Edge, Node as feature_node
from pathfinding import explore, explore_street_names
from sklearn.linear_model import LinearRegression

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

@dataclass
class TileNode():
    def __init__(self, cityName:str, tileName:str, featureName:str, ratio:float) -> None:
        self.nodes: Dict[feature_node] = {} 
        self.cityName = cityName 
        self.tileName = tileName
        self.featureName:str = featureName
        self.ratio:float = ratio
        self.boundaries_geolocalised = getBoundaries(constants.CITYPATH[cityName].joinpath(f'{tileName}.tfw'))
        self.neighbours = {
            'north':None,
            'north_east':None,
            'east':None, 
            'south_east':None,
            'south':None,
            'south_west':None,
            'west':None, 
            'north_west':None
            }
        self.center_lattitude = (self.boundaries_geolocalised['westBoundary'] + self.boundaries_geolocalised['eastBoundary'])/2
        self.center_longitude = (self.boundaries_geolocalised['northBoundary'] + self.boundaries_geolocalised['southBoundary'])/2

        self.ref_image_path = constants.CITYPATH[cityName].joinpath(f'{tileName}.jpg')
        self.classifiedLayerPath = constants.CLASSIFIEDPATH.joinpath(f'{self.featureName}/{self.cityName}/{self.tileName}.json')
        self.jpegLayersPath = constants.RAWPATH.joinpath(f'{self.cityName}/{self.tileName}')

        self.is_classified:bool = False 
       
        self.classes:Dict = json.load(open(constants.CLASSESFILES[featureName]))
        self.inv_map:Dict = {v: k for k, v in self.classes.items()}

        self.road_nodes:Dict[feature_node] = {key:node for key,node in self.nodes.items() if self.inv_map[node.label] =='number'}
        self.initialise_graph()

    def initialise_graph(self):
        load_path = constants.GRAPHPATH.joinpath(f'tile_graphs/{self.cityName}/{self.tileName}/{self.featureName}.json')
        
        print(f'Attempting to load from saved graph at {load_path.name} from {load_path.parent}')
        if load_path.is_file():
            print(f"Success, loading")
            self.load_from_json(load_path)    
            self.is_classified = True
        else:
            print(f'No saved graph file found, fetching classified folder at {self.classifiedLayerPath}')
            if self.classifiedLayerPath.is_file():
                print('Constructing New Graph')
                self.construct_graph(load_path)
                self.is_classified = True
            else:
                print(f'displaying base image')

    def construct_graph(self, savePath=None): 
        dataframe = pd.read_json(self.classifiedLayerPath).transpose()
        for index, row in dataframe.iterrows():
            if self.inv_map[row['class']] !='false positive':
                new_node = feature_node(int(row['xTile']*self.ratio), int(row['yTile']*self.ratio), row['class'], key = f'{self.tileName}_{index}')
                self.nodes[f'{self.tileName}_{index}'] = new_node
                if self.inv_map[new_node.label] =='number':
                    self.road_nodes[new_node.key] = new_node

        if savePath is not None:
            savePath.parent.mkdir(parents=True, exist_ok=True)
            self.save_to_json(savePath)

    def save_to_json(self, savePath):
        graph_dict = {
            'cityName':self.cityName,
            'tileName':self.tileName,
            'featureName':self.featureName,
            'nodes': [node.scale_coordinates(1/self.ratio).serialise_node for node in self.nodes.values()]
        }
        with open(f'{savePath}', 'w') as out_f:
            json.dump(graph_dict, out_f, indent = 4)

    def load_from_json(self, loadPath):
        load_dict : Dict = json.load(open(loadPath))
        self.road_nodes : Dict[feature_node] = {}
        for node in load_dict['nodes']:
            new_node = feature_node(int(node['x']*self.ratio), int(node['y']*self.ratio), node['label'], node['key'])
            self.nodes[node['key']] = new_node
            if self.inv_map[node['label']] =='number':
                self.road_nodes[node['key']] = new_node

    def is_equal_to(self, other_node: object):
        for key, value in self.boundaries_geolocalised.items():
            if other_node[key] != value:
                return False
        return True

    @property
    def to_string(self) -> str:
        returnString = f'cityName : {self.cityName}, tileName : {self.tileName} \n'
        returnString += f'center_longitude : {self.center_longitude} \ncenter_lattitude : {self.center_lattitude} \n'
        returnString += 'Coordinates : \n'
        for key, value in self.boundaries_geolocalised.items():
            returnString += f'\t {key} : {value} \n'
        returnString += '\n'
        returnString += 'Neighbours : \n'
        for key, value in self.neighbours.items():
            returnString += f'\t {key} : {value} \n'
        return returnString
        
    @property
    def serialise_coords(self):
        node_dict = {
            'cityName':self.cityName,
            'tileName':self.tileName,
            'featureName':self.featureName,
            'boundaries_geolocalised':self.boundaries_geolocalised,
            'nodes': [node.scale_coordinates(1/self.ratio).serialise_node for node in self.nodes.values()],
            'neighbours': [n.serialise_coords for n in self.neighbours.values()]
        }
        return node_dict

    @property
    def serialise_node(self) -> dict:
        return_dict = self.serialise_coords
        return_dict['neighbours'] = [n.serialise_coords for n in self.neighbours]
        return return_dict

    def get_all_background(self, val_range:int):
        background = morph_tools.open_and_binarise(str(self.jpegLayersPath.joinpath(f'{constants.BACKGROUNDKWDS[0]}.jpg')), constants.BACKGROUNDKWDS[0], self.ratio)
        for background_kwd in constants.BACKGROUNDKWDS[1:]:                
            if self.jpegLayersPath.joinpath(f'{background_kwd}.jpg').is_file():
                bkg = morph_tools.open_and_binarise(str(self.jpegLayersPath.joinpath(f'{background_kwd}.jpg')), background_kwd,  self.ratio)
                cv2.bitwise_or(background, bkg)
        return np.uint8(background*val_range)

    def display_lights(self, val_range = 255,light_profile = False):
        max_rad = 50
        min_rad = 250
        rad_step = 50

        bkg_msk = morph_tools.open_and_binarise(str(self.jpegLayersPath.joinpath(f'{constants.BACKGROUNDKWDS[0]}.jpg')), constants.BACKGROUNDKWDS[0], self.ratio)
        bkg_msk = cv2.cvtColor(np.uint8(bkg_msk), cv2.COLOR_GRAY2BGR)
        mask = np.zeros((int(constants.TILEHEIGHT*self.ratio), int(constants.TILEWIDTH*self.ratio),3), np.uint8)
        for node in self.nodes.values():
            if self.inv_map[node.label] in ['L', 'L.P', 'Post', 'S.P']:
                cv2.circle(mask, (node.x,  node.y), max_rad, (1,1,1), -1)

        if light_profile:
            return mask*(1-bkg_msk)*val_range
        else:
            background = morph_tools.open_and_colorise(str(constants.CITYPATH[self.cityName].joinpath(f'{self.tileName}.jpg')), self.ratio)
            return cv2.bitwise_and(background, mask*(1-bkg_msk))*val_range

    def display_roads(self, val_range=1):
        background = self.get_all_background(val_range=val_range)
        for node in self.road_nodes.values():
            print(node.key)
        for road_node in self.road_nodes.values():
            cv2.circle(background,(road_node.x,  road_node.y), 4, (0,0,0), -1)
        return cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)*255

    def display_all(self, val_range=1):
        background = self.get_all_background(val_range=val_range)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)*255
        for node in self.nodes.values():
            if self.inv_map[int(node.label)] != 'false positive':
                cv2.putText(background, self.inv_map[int(node.label)],(node.x, node.y), 1, 1,(0,0,255)) 
        return background

    def display_element(self, savePath=None, element='roads'):
        if element ==  'roads':
            display_mat = self.display_roads()
        elif element ==  'lights':
            display_mat = self.display_lights()
        elif element ==  'all':
            display_mat = self.display_all()
        
        if savePath is not None:
            savePath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(savePath), display_mat)
        else:
            return display_mat

    def load_tile_part(self,x_low, x_high, y_low, y_high):
        mat = morph_tools.open_and_colorise(str(self.ref_image_path),self.ratio)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        return np.transpose(mat)[x_low: x_high, y_low: y_high]

    def return_display_element(self, el='', val_range=255):
        if self.is_classified:
            if el == '':
                mat = self.get_all_background(val_range=val_range)
            elif el =='original':
                mat = morph_tools.just_open(str(self.ref_image_path), mode='color')
                #mat = morph_tools.open_and_colorise(str(self.ref_image_path),self.ratio)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
            else:

                mat = self.display_element(savePath = None, element=el)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        else:
            mat = morph_tools.just_open(str(self.ref_image_path), mode='color')
            #mat = morph_tools.open_and_colorise(str(self.ref_image_path),self.ratio)
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        
        return np.transpose(mat) 

    def display_neighbours(self, ratio=0.1):
        w = int(constants.TILEWIDTH*ratio)
        h = int(constants.TILEHEIGHT*ratio)
        mat = np.zeros((int(w*3),int(h*3)), np.uint8)
        mat[w:w*2, h:h*2] = self.return_display_element()
        for neighbour_cardinal, neighbour_node in self.neighbours.items():
            if neighbour_node is not None:
                neighbour_coordinates = constants.CARDINALTOCARTESIANPLOT[neighbour_cardinal]
                print(neighbour_cardinal, neighbour_coordinates)
                mat[w*(1+neighbour_coordinates[0]):w*(2+neighbour_coordinates[0]), h*(1+neighbour_coordinates[1]):h*(2+neighbour_coordinates[1]) ] =  neighbour_node.return_display_element()
        plt.matshow(np.transpose(mat,(1,0)))
        plt.show()
        plt.clf()

    def extract_feature(self, feature_name, feature_dict, global_index, w_offset, h_offset, epsilon=0.0010):
        #def extract_feature(self, feature_name, feature_dict, global_index, w_offset, h_offset, epsilon=0.0010):
        print(f'Processing {feature_name} for tile {self.tileName}')
        mat = morph_tools.just_open( constants.PROCESSEDPATH.joinpath(f'{self.cityName}/{self.tileName}/{feature_name}{constants.FILEEXTENSION}'), 'grayscale')
        file_to_extract = np.uint8(np.transpose(mat)/255)
        contours = morph_tools.extract_contours(file_to_extract)
        tile_dict = {
                "type": "FeatureCollection",
                "features": []
                }
        local_index = 0
        for contour in tqdm(contours):
            _, false_positive = morph_tools.is_false_positive(contour)
            if not false_positive:  
                contour = cv2.approxPolyDP(contour, epsilon, True)
                contour = cv2.convexHull(contour)
                tile_dict["features"].append({
                    "type": "Feature",
                    "properties": {
                        "class": f"{feature_name}_{local_index}"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [contour_to_polygon_geojson(contour, w_offset, h_offset)]
                    }
                    })
                feature_dict['features'].append({
                    "type": "Feature",
                    "properties": {
                        "class": f"{feature_name}_{global_index}"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [contour_to_polygon_geojson(contour, w_offset, h_offset)]
                    }
                    })        
                global_index += 1
                local_index  += 1
        
        return feature_dict, global_index, tile_dict

    def get_street_names(self):
        if self.featureName != 'stamps_large_font':
            return None
        else:
            # Get background            
            background = np.transpose(self.get_all_background(1))
            # Sort nodes from left to right (reading direction) = ascending x
            sorted_nodes:List[feature_node] = sorted(self.nodes.values(), key=lambda item: item.x)
            sorted_nodes_dict = {value.key:value for value in sorted_nodes}
            visited_nodes = []
            words = []
            for sorted_node in sorted_nodes:
                if not sorted_node.is_in(visited_nodes):
                    visited, distances= explore_street_names(background, sorted_node, sorted_nodes_dict, visited=[], distances=[], inv_map=self.inv_map)
                    chars = ''
                    for n in visited:
                        visited_nodes.append(n)
                        chars += f'{self.inv_map[n.label]}'
                    i = 1
                    low_b = 0
                    temp = ''
                    while i < len(chars):
                        if 1.5*distances[i-1] < distances[i]:
                            temp = visited[low_b: i]
                            temp.sort(key=lambda node:node.x)
                            words.append(''.join([f'{self.inv_map[n.label]}' for n in temp]))
                            low_b = i
                        i+=1
                    temp = visited[low_b: i]
                    temp.sort(key=lambda node:node.x)
                    words.append(''.join([f'{self.inv_map[n.label]}' for n in temp]))
                    #words.append(chars[low_b: i])
            print(words)
                    
            
def contour_to_projected(contour, offset_x, offset_y):
    x = contour[0][0]
    y = contour[0][1]
    return ((int(offset_x+x), int(offset_y+y)))

def contour_to_polygon_geojson(contours, offset_x, offset_y):
    lines = []
    lines.append(contour_to_projected(contours[0], offset_x, offset_y))
    for c in reversed(contours):
        lines.append(contour_to_projected(c, offset_x, offset_y))
    return lines
