

from dataclasses import dataclass
import sys
from typing import List, Dict
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
import pandas as pd
import json
from tqdm import tqdm

import utils.morphological_tools as morph_tools
import utils.constants as constants
from feature_node import Node as feature_node
from pathfinding import explore_street_names


def get_boundaries(tfw_data_path:pathlib.Path) -> dict:
    tfw_raw = open(tfw_data_path, 'r').read()
    x_diff = float(tfw_raw.split("\n")[0])
    y_diff = float(tfw_raw.split("\n")[3])
    west_boundary = float(tfw_raw.split("\n")[4])
    north_boundary = float(tfw_raw.split("\n")[5])
    east_boundary = west_boundary + (constants.TILEWIDTH - 1) * x_diff
    south_boundary = north_boundary + (constants.TILEHEIGHT - 1) * y_diff
    return {'west_boundary':west_boundary, 'north_boundary':north_boundary,
            'east_boundary':east_boundary, 'south_boundary':south_boundary,
            'x_diff':x_diff, 'y_diff':y_diff, 'lattitude_length': (constants.TILEWIDTH - 1) * x_diff, 'longitude_length':(constants.TILEHEIGHT - 1) * y_diff}

@dataclass
class TileNode():
    def __init__(self, city_name:str, tile_name:str, feature_name:str, ratio:float) -> None:
        self.nodes: Dict[feature_node] = {}
        self.city_name = city_name
        self.tile_name = tile_name
        self.feature_name:str = feature_name
        self.ratio:float = ratio
        self.boundaries_geolocalised = get_boundaries(constants.CITYPATH[city_name].joinpath(f'{tile_name}.tfw'))
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
        self.center_lattitude = (self.boundaries_geolocalised['west_boundary'] + self.boundaries_geolocalised['east_boundary'])/2
        self.center_longitude = (self.boundaries_geolocalised['north_boundary'] + self.boundaries_geolocalised['south_boundary'])/2

        self.ref_image_path = constants.CITYPATH[city_name].joinpath(f'{tile_name}.jpg')
        self.classified_layer_path = constants.CLASSIFIEDPATH.joinpath(f'{self.feature_name}/{self.city_name}/{self.tile_name}.json')
        self.jpeg_layers_path = constants.RAWPATH.joinpath(f'{self.city_name}/{self.tile_name}')

        self.is_classified:bool = False

        self.classes:Dict = json.load(open(constants.CLASSESFILES[feature_name]))
        self.inv_map:Dict = {v: k for k, v in self.classes.items()}

        self.road_nodes:Dict[feature_node] = {key:node for key,node in self.nodes.items() if self.inv_map[node.label] =='number'}
        self.initialise_graph()

    def initialise_graph(self):
        load_path = constants.GRAPHPATH.joinpath(f'tile_graphs/{self.city_name}/{self.tile_name}/{self.feature_name}.json')

        print(f'Attempting to load from saved graph at {load_path.name} from {load_path.parent}')
        if load_path.is_file():
            print(f"Success, loading")
            self.load_from_json(load_path)
            self.is_classified = True
        else:
            print(f'No saved graph file found, fetching classified folder at {self.classified_layer_path}')
            if self.classified_layer_path.is_file():
                print('Constructing New Graph')
                self.construct_graph(load_path)
                self.is_classified = True
            else:
                print(f'displaying base image')

    def construct_graph(self, savePath=None):
        dataframe = pd.read_json(self.classified_layer_path).transpose()
        for index, row in dataframe.iterrows():
            if self.inv_map[row['class']] !='false positive':
                new_node = feature_node(int(row['xTile']*self.ratio), int(row['yTile']*self.ratio), row['class'], key = f'{self.tile_name}_{index}')
                self.nodes[f'{self.tile_name}_{index}'] = new_node
                if self.inv_map[new_node.label] =='number':
                    self.road_nodes[new_node.key] = new_node

        if savePath is not None:
            savePath.parent.mkdir(parents=True, exist_ok=True)
            self.save_to_json(savePath)

    def save_to_json(self, savePath):
        graph_dict = {
            'city_name':self.city_name,
            'tile_name':self.tile_name,
            'feature_name':self.feature_name,
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
        return_String = f'city_name : {self.city_name}, tile_name : {self.tile_name} \n'
        return_String += f'center_longitude : {self.center_longitude} \ncenter_lattitude : {self.center_lattitude} \n'
        return_String += 'Coordinates : \n'
        for key, value in self.boundaries_geolocalised.items():
            return_String += f'\t {key} : {value} \n'
        return_String += '\n'
        return_String += 'Neighbours : \n'
        for key, value in self.neighbours.items():
            return_String += f'\t {key} : {value} \n'
        return return_String

    @property
    def serialise_coords(self):
        node_dict = {
            'city_name':self.city_name,
            'tile_name':self.tile_name,
            'feature_name':self.feature_name,
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
        background = morph_tools.open_and_binarise(str(self.jpeg_layers_path.joinpath(f'{constants.BACKGROUNDKWDS[0]}.jpg')), constants.BACKGROUNDKWDS[0], self.ratio)
        for background_kwd in constants.BACKGROUNDKWDS[1:]:
            if self.jpeg_layers_path.joinpath(f'{background_kwd}.jpg').is_file():
                bkg = morph_tools.open_and_binarise(str(self.jpeg_layers_path.joinpath(f'{background_kwd}.jpg')), background_kwd,  self.ratio)
                cv2.bitwise_or(background, bkg)
        return np.uint8(background*val_range)

    def display_lights(self, val_range = 255,light_profile = False):
        max_rad = 50

        bkg_msk = morph_tools.open_and_binarise(str(self.jpeg_layers_path.joinpath(f'{constants.BACKGROUNDKWDS[0]}.jpg')), constants.BACKGROUNDKWDS[0], self.ratio)
        bkg_msk = cv2.cvtColor(np.uint8(bkg_msk), cv2.COLOR_GRAY2BGR)
        mask = np.zeros((int(constants.TILEHEIGHT*self.ratio), int(constants.TILEWIDTH*self.ratio),3), np.uint8)
        for node in self.nodes.values():
            if self.inv_map[node.label] in ['L', 'L.P', 'Post', 'S.P']:
                cv2.circle(mask, (node.pos_x,  node.pos_y), max_rad, (1,1,1), -1)

        if light_profile:
            return mask*(1-bkg_msk)*val_range
        else:
            background = morph_tools.open_and_colorise(str(constants.CITYPATH[self.city_name].joinpath(f'{self.tile_name}.jpg')), self.ratio)
            return cv2.bitwise_and(background, mask*(1-bkg_msk))*val_range

    def display_roads(self, val_range=1):
        background = self.get_all_background(val_range=val_range)
        for node in self.road_nodes.values():
            print(node.key)
        for road_node in self.road_nodes.values():
            cv2.circle(background,(road_node.pos_x,  road_node.pos_y), 4, (0,0,0), -1)
        return cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)*255

    def display_all(self, val_range=1):
        background = self.get_all_background(val_range=val_range)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)*255
        for node in self.nodes.values():
            if self.inv_map[int(node.label)] != 'false positive':
                cv2.putText(background, self.inv_map[int(node.label)],(node.pos_x, node.pos_y), 1, 1,(0,0,255))
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
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
            else:

                mat = self.display_element(savePath = None, element=el)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        else:
            mat = morph_tools.just_open(str(self.ref_image_path), mode='color')
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

        return np.transpose(mat)

    def display_neighbours(self, ratio=0.1):
        width = int(constants.TILEWIDTH*ratio)
        height = int(constants.TILEHEIGHT*ratio)
        mat = np.zeros((int(width*3),int(height*3)), np.uint8)
        mat[width:width*2, height:height*2] = self.return_display_element()
        for neighbour_cardinal, neighbour_node in self.neighbours.items():
            if neighbour_node is not None:
                neighbour_coordinates = constants.CARDINALTOCARTESIANPLOT[neighbour_cardinal]
                print(neighbour_cardinal, neighbour_coordinates)
                mat[width*(1+neighbour_coordinates[0]):width*(2+neighbour_coordinates[0]), height*(1+neighbour_coordinates[1]):height*(2+neighbour_coordinates[1]) ] =  neighbour_node.return_display_element()
        plt.matshow(np.transpose(mat,(1,0)))
        plt.show()
        plt.clf()

    def extract_feature(self, feature_name, feature_dict, global_index, w_offset, h_offset, epsilon=0.0010):
        #def extract_feature(self, feature_name, feature_dict, global_index, w_offset, h_offset, epsilon=0.0010):
        print(f'Processing {feature_name} for tile {self.tile_name}')
        mat = morph_tools.just_open( constants.PROCESSEDPATH.joinpath(f'{self.city_name}/{self.tile_name}/{feature_name}{constants.FILEEXTENSION}'), 'grayscale')
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
        if self.feature_name != 'stamps_large_font':
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
                    index = 1
                    low_b = 0
                    temp = ''
                    while index < len(chars):
                        if 1.5*distances[index-1] < distances[index]:
                            temp = visited[low_b: index]
                            temp.sort(key=lambda node:node.x)
                            words.append(''.join([f'{self.inv_map[n.label]}' for n in temp]))
                            low_b = index
                        index+=1
                    temp = visited[low_b: index]
                    temp.sort(key=lambda node:node.x)
                    words.append(''.join([f'{self.inv_map[n.label]}' for n in temp]))
            print(words)


def contour_to_projected(contour, offset_x, offset_y):
    return ((int(offset_x+contour[0][0]), int(offset_y+contour[0][1])))

def contour_to_polygon_geojson(contours, offset_x, offset_y):
    lines = []
    lines.append(contour_to_projected(contours[0], offset_x, offset_y))
    for contour in reversed(contours):
        lines.append(contour_to_projected(contour, offset_x, offset_y))
    return lines
