from typing import Dict
from collections import Counter
import math
from operator import add

import json
import numpy as np
import requests

import cv2
import sys

sys.path.append('..')
import utils.constants as constants
import city_node, pathfinding


citiesPath =  constants.CITIESFOLDERPATH
backgroundsPath = constants.RAWPATH

class Graph():
    def __init__(self, city_name:str, ratio:float, feature_name='stamps_small_font') -> None:
        self.city_name = city_name
        self.ratio = ratio
        self.feature_name = feature_name
        self.cityPath = constants.CITYPATH[city_name]
        self.nodes:Dict[city_node.TileNode] = {}
        self.road_nodes:Dict[city_node.TileNode] = {}
        self.sub_trees = []
        self.ordered_nodes = {}
        self.n_tile_lattitude = 0
        self.n_tile_longitude = 0
        self.tile_h = constants.TILEHEIGHT
        self.tile_w = constants.TILEWIDTH
        self.ordered = False

    def populate_graph(self):
        tfwFilesPath = [x for x in self.cityPath.iterdir() if x.suffix == '.tfw']
        self.n_tiles = len(tfwFilesPath)
        for tfwFilePath in tfwFilesPath:
            tile_name = tfwFilePath.stem
            new_node = city_node.TileNode(self.city_name, tile_name=tile_name, feature_name = self.feature_name, ratio=self.ratio)
            self.nodes[tfwFilePath] = new_node

    def contains(self, w_b:float, n_b:float) -> bool:
        for node in self.nodes.values():
            if (math.floor(node.boundaries_geolocalised['north_boundary'])== math.floor(n_b) and math.floor(node.boundaries_geolocalised['west_boundary']) == math.floor(w_b)) or (math.ceil(node.boundaries_geolocalised['north_boundary'] )== math.ceil(n_b) and math.ceil(node.boundaries_geolocalised['west_boundary']) == math.ceil(w_b)):
                return node
        return None

    def get_lattitude_length(self):
        return next(iter(self.nodes.values())).boundaries_geolocalised['lattitude_length']

    def get_longitude_length(self):
        return next(iter(self.nodes.values())).boundaries_geolocalised['longitude_length']

    def __str__(self)-> str:
        for node in self.nodes:
            print('----------- Printing new node -----------')
            print(node.to_string)
        return ''

    def save_to_json(self, savePath):
        return_list = [node.serialise_node for node in self.nodes]
        with open(f'{savePath}', 'w') as out_f:
            json.dump(return_list, out_f, indent = 4)

    def load_from_json(self, loadPath):
        load_list = json.load(open(loadPath))
        for node in load_list:
            new_node = city_node.Node(node['city_name'], node['cityPath'], node['tile_name'])
            for neighbour in node['neighbours']:
                new_node_n = city_node.Node(neighbour['city_name'], neighbour['cityPath'], neighbour['tile_name'])
                new_node.add_neighbour(new_node_n)
            self.nodes.append(new_node)

    def order_graph(self):
        print('------------ Ordering Graph ------------')
        west_boundary  = min(self.nodes.values(), key=lambda x: x.boundaries_geolocalised['west_boundary']).boundaries_geolocalised['west_boundary']
        north_boundary = max(self.nodes.values(), key=lambda x: x.boundaries_geolocalised['north_boundary']).boundaries_geolocalised['north_boundary']
        n_tile_lattitude = self.get_n_tile_lattitude()
        n_tile_longitude = self.get_n_tile_longitude()
        self.n_tile_lattitude = n_tile_lattitude
        self.n_tile_longitude = n_tile_longitude
        lat_lenght = self.get_lattitude_length()
        lon_length = self.get_longitude_length()
        for n_b_index in range(n_tile_longitude):
            n_b = north_boundary + (lon_length * n_b_index)
            for w_b_index in range(n_tile_lattitude):
                w_b = west_boundary + (lat_lenght * w_b_index)
                node = self.contains(w_b, n_b)
                if node is not None:
                    self.ordered_nodes[(w_b_index,n_b_index)] = node
        self.ordered = True


    def make_neighbours(self):
        if not self.ordered:
            self.order_graph()
        for key, node in self.ordered_nodes.items():
            for cartesian_tuple, cardinal in constants.CARTESIANTOCARDINAL.items():
                neighbourd_coords = tuple(map(add, key, cartesian_tuple))
                if neighbourd_coords in self.ordered_nodes:
                    node.neighbours[cardinal] = self.ordered_nodes[neighbourd_coords]
            node.display_neighbours(0.1)

    def get_n_tile_lattitude(self):
        return len(Counter([node.boundaries_geolocalised['west_boundary'] for node in self.nodes.values()]).keys())

    def get_n_tile_longitude(self):
        return len(Counter([node.boundaries_geolocalised['north_boundary'] for node in self.nodes.values()]).keys())

    def compute_path(self, tile_0_coords=(0,1), node_0=2, tile_1_coords=(2,0), node_1=72, savePath='facking_roads'):
        if not self.ordered:
            self.order_graph()
        background, road_nodes, display_background = self.display_city_subset(tile_0_coords, tile_1_coords)
        background = np.uint8(background)
        start_key = f'{self.ordered_nodes[tile_0_coords].tile_name}_{node_0}'
        end_key   = f'{self.ordered_nodes[tile_1_coords].tile_name}_{node_1}'
        path_nodes = pathfinding.compute_path(start_key=start_key, end_key=end_key, dict_of_nodes=road_nodes, background=background)
        display_background = np.uint8(display_background)
        display_background = cv2.cvtColor(display_background, cv2.COLOR_GRAY2BGR)

        for i in range(len(path_nodes)-1):
            x0, y0 = path_nodes[i].x  , path_nodes[i].y
            x1, y1 = path_nodes[i+1].x, path_nodes[i+1].y
            cv2.circle(display_background,(y0, x0), int(100*self.ratio), (0,0,255), -1)
            cv2.line(display_background,(y0,x0),(y1,x1),(0,0,255),int(50*self.ratio))

        x0, y0 = path_nodes[-1].x  , path_nodes[-1].y
        cv2.circle(display_background,(y0, x0), int(100*self.ratio), (0,0,255), -1)


        if savePath is not None:
            cv2.imwrite(f'{savePath}_path.jpg', np.transpose(display_background,(1,0,2)))
        else:
            return display_background

    def display_city_subset(self, tile_0_coords, tile_1_coords, val_range=255, element=''):
        road_nodes = {}
        x0, y0 = min([tile_0_coords[0], tile_1_coords[0]]), min([tile_0_coords[1], tile_1_coords[1]])
        x1, y1 = max([tile_0_coords[0], tile_1_coords[0]]), max([tile_0_coords[1], tile_1_coords[1]])
        offset_x = int(self.tile_w*self.ratio)
        offset_y = int(self.tile_h*self.ratio)
        subset = np.ones(((1+x1-x0)*offset_x, (1+y1-y0)*offset_y))
        display_subset = np.ones(((1+x1-x0)*offset_x, (1+y1-y0)*offset_y))*255
        print(np.shape(subset))

        for i in range(x0,x1+1):
            for j in range(y0,y1+1):
                if (i,j) in self.ordered_nodes:
                    current_tile_node:city_node.TileNode = self.ordered_nodes[(i,j)]
                    print(i,j, current_tile_node.tile_name)
                    for road_node_key, road_node_value in current_tile_node.road_nodes.items():
                        road_nodes[road_node_key] = city_node.feature_node(road_node_value.x+((i-x0)*offset_x), road_node_value.y+((j-y0)*offset_y), label='number', key=road_node_key)

                    subset[(i-x0)*offset_x:(i-x0+1)*offset_x, (j-y0)*offset_y:(j-y0+1)*offset_y] = current_tile_node.return_display_element(el=element, val_range=val_range)
                    display_subset[(i-x0)*offset_x:(i-x0+1)*offset_x, (j-y0)*offset_y:(j-y0+1)*offset_y] = current_tile_node.return_display_element(el='original', val_range=val_range)
                else:
                    subset[(i-x0)*offset_x:(i-x0+1)*offset_x, (j-y0)*offset_y:(j-y0+1)*offset_y] = np.ones((offset_x, offset_y))*255
        return subset, road_nodes, display_subset

    def display_element(self, savePath=None, element = ''):
        if not self.ordered:
            self.order_graph()
        cityDisplay = np.zeros((int(self.n_tile_lattitude*self.tile_w*self.ratio), int(self.n_tile_longitude*self.tile_h*self.ratio)))
        for i in range(self.n_tile_lattitude):
            for j in range(self.n_tile_longitude):
                if (i,j) in self.ordered_nodes:
                    print(f'----- Processing node ({i}, {j})-----')
                    cityDisplay[i*int(self.tile_w*self.ratio):(i+1)*int(self.tile_w*self.ratio), j*int(self.tile_h*self.ratio):(j+1)*int(self.tile_h*self.ratio)] = self.ordered_nodes[(i,j)].return_display_element(element)
        if savePath is not None:
            savePath.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(savePath),np.transpose(cityDisplay))
        else:
            return np.transpose(cityDisplay)

    def extract_features_city_wide(self, feature_list=constants.HIGHLEVELFEATURES):
        if not self.ordered:
            self.order_graph()

        save_folder_path = constants.TILESDATAPATH.joinpath(f'overlays/{self.city_name}')
        save_folder_path.mkdir(parents=True, exist_ok=True)
        for feature_name in feature_list:
            global_index = 0
            feature_dict = {
                "type": "FeatureCollection",
                "features": []
                }
            for tile_h_index in range(self.n_tile_longitude):
                h_offset = self.tile_h * tile_h_index
                for tile_w_index in range(self.n_tile_lattitude):
                    w_offset = self.tile_w * tile_w_index
                    if (tile_w_index,tile_h_index) in self.ordered_nodes:
                        feature_dict, global_index, tile_dict = self.ordered_nodes[(tile_w_index,tile_h_index)].extract_feature(feature_name, feature_dict, global_index, w_offset, h_offset)
                        #request = f'http://13.40.112.22/v1alpha1/features/{self.city_name}/{self.ordered_nodes[(tile_w_index,tile_h_index)].tile_name}_{feature_name}/insert/replace'
                        #res = requests.post(url=request, json = tile_dict)
            #request = f'http://13.40.112.22/v1alpha1/features/{self.city_name}/{feature_name}/insert/replace'
            #res = requests.post(url=request, json = feature_dict)
            #print(f'The post request {request} has returned the status {res}')
            with open(save_folder_path.joinpath(f'{feature_name}.js'), 'w') as out_file:
                out_file.write(f'var tile_data_{feature_name} = {json.dumps(feature_dict)};' )

    def make_dataset(self, high_level_feature_dict=constants.HIGHTOLOWDISPLAY):
        if not self.ordered:
            self.order_graph()
        save_folder_path = constants.GRAPHPATH.joinpath(f'datasets/{self.city_name}')
        save_folder_path.mkdir(parents=True, exist_ok=True)
        for feature_name, sub_features_list in high_level_feature_dict:
            global_index = 0
            for sub_feature_name in sub_features_list:
                for tile_h_index in range(self.n_tile_longitude):
                    h_offset = self.tile_h * tile_h_index
                    for tile_w_index in range(self.n_tile_lattitude):
                        w_offset = self.tile_w * tile_w_index
                        if (tile_w_index,tile_h_index) in self.ordered_nodes:
                            '''feature_dict = {
                                'identifier':global_index,
                                'class':sub_feature_name,
                                'tile_name':self.ordered_nodes[(tile_w_index,tile_h_index)].tile_name,
                                'center_x_relative':,
                                'center_y_relative':,
                                'center_x_absolute':,
                                'center_y_absolute':
                                }'''


    def make_tiles(self):
        save_path = constants.DATASETFOLDERPATH.joinpath(f'tiles_data/{self.city_name}')
        max_zoom_depth = 5 # We compute up to 5 zoom level, but could be more
        # First, we determine the maximum zoom level
        # For that zoom level, one pixel on the map is one pixel on the display
        if not self.ordered:
            self.order_graph()
        pixel_width  = self.n_tile_lattitude*self.tile_w
        pixel_height = self.n_tile_longitude*self.tile_h
        max_dim = max(pixel_width, pixel_height)
        i = 13
        while 2**i <= max_dim:
            i += 1
        self.max_zoom_level = i
        # We then iterate over zoom levels
        zoom_depth = 0
        while zoom_depth < max_zoom_depth:
            print(f'Processing zoom level {self.max_zoom_level-zoom_depth-8}')
            save_path_zoom = save_path.joinpath(f'{self.max_zoom_level-zoom_depth-8}')
            input_tile_size = 2**(8+zoom_depth)
            print(f'Thumbnail size : {input_tile_size}')
            # For each zoom depth, we compute
                # the number of thumbnails (horizontal and vertical)
                # the number of thumbnails that fit in our big_tiles
            thumbnails_per_tile_width  = self.tile_w // input_tile_size + 1
            thumbnails_per_tile_height = self.tile_h // input_tile_size + 1
            mesh_width  = thumbnails_per_tile_width*input_tile_size
            mesh_height = thumbnails_per_tile_height*input_tile_size
            diff_width  = mesh_width  - self.tile_w
            diff_height = mesh_height - self.tile_h
            # We then iterate over the TILE COORDINATES
            for tile_h_index in range(self.n_tile_longitude):
                mesh_h_index_offset = tile_h_index * thumbnails_per_tile_height
                # We now specify the part of the tile that will be queried
                start_h = diff_height*tile_h_index
                end_h   = self.tile_h
                extent_h = end_h - start_h
                for tile_w_index in range(self.n_tile_lattitude):
                    mesh_w_index_offset = tile_w_index * thumbnails_per_tile_width
                    # We begin by creating a (thumbnails_per_tile_width, thumbnails_per_tile_height) empty matrix
                    current_mesh = np.ones((mesh_width, mesh_height), np.uint8)*255

                    # We now specify the part of the tile that will be queried
                    start_w = diff_width*tile_w_index
                    end_w   = self.tile_w
                    extent_w = end_w - start_w
                    if (tile_w_index,tile_h_index) in self.ordered_nodes:
                        tile:np.ndarray = self.ordered_nodes[(tile_w_index,tile_h_index)].return_display_element('original')[start_w:end_w, start_h: end_h]
                    else:
                        tile:np.ndarray = np.ones((extent_w, extent_h), np.uint8)*255
                    # We can now position the tile on the mesh
                    current_mesh[:extent_w, :extent_h] = tile
                    # We can remove the tile from the memory
                    del tile

                    # Now is time to load the bits from the neighbours
                    #   Load bit below
                    if tile_h_index != self.n_tile_longitude - 1:
                        if (tile_w_index,tile_h_index+1) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index,tile_h_index+1)].return_display_element('original')[start_w:end_w, : diff_height*(tile_h_index+1)]
                        else:
                            tile = np.ones((extent_w, diff_height*(tile_h_index+1)), np.uint8)*255

                        current_mesh[:mesh_width-diff_width*(1+tile_w_index), mesh_height-diff_height*(1+tile_h_index):] = tile
                        del tile

                    if tile_w_index != self.n_tile_lattitude - 1:
                        if (tile_w_index+1,tile_h_index) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index+1,tile_h_index)].return_display_element('original')[:diff_width*(tile_w_index+1), start_h:end_h]
                        else:
                            tile = np.ones((diff_width*(tile_w_index+1), extent_h), np.uint8)*255
                        current_mesh[mesh_width-diff_width*(1+tile_w_index):, :mesh_height-diff_height*(1+tile_h_index)] = tile
                        del tile

                    if tile_h_index != self.n_tile_longitude - 1 and tile_w_index != self.n_tile_lattitude - 1:
                        if (tile_w_index+1,tile_h_index+1) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index+1,tile_h_index+1)].return_display_element('original')[:diff_width*(tile_w_index+1), :diff_height*(tile_h_index+1)]
                        else:
                            tile = np.ones((diff_width*(tile_w_index+1), diff_height*(tile_h_index+1)), np.uint8)*255
                        current_mesh[mesh_width-diff_width*(1+tile_w_index):, mesh_height-diff_height*(1+tile_h_index):] = tile
                        del tile

                    # current_mesh now holds the appropriate information. We can then move on to splitting the mesh into appropriate thumbnails
                    print(f'Processing mesh coordinate: ({tile_h_index, tile_w_index})')
                    print(f'Current_mesh_shape : {current_mesh.shape}')
                    for th_index_w in range(thumbnails_per_tile_width):
                        save_path_zoom_width = save_path_zoom.joinpath(f'{th_index_w + mesh_w_index_offset}')
                        save_path_zoom_width.mkdir(parents=True, exist_ok=True)
                        start_mesh_w = th_index_w*input_tile_size
                        end_mesh_w = (th_index_w+1)*input_tile_size
                        for th_index_h in range(thumbnails_per_tile_height):
                            save_name = save_path_zoom_width.joinpath(f'{th_index_h + mesh_h_index_offset}.jpg')
                            start_mesh_h = th_index_h*input_tile_size
                            end_mesh_h = (th_index_h+1)*input_tile_size
                            thumbnail = current_mesh[start_mesh_w:end_mesh_w, start_mesh_h:end_mesh_h]
                            thumbnail = np.transpose(cv2.resize(thumbnail, dsize=(256,256), interpolation=cv2.INTER_CUBIC))
                            cv2.imwrite(str(save_name), thumbnail)
                            request = f'http://13.40.112.22/v1alpha1/upload/images/{self.city_name}/{self.max_zoom_level-zoom_depth-8}/{th_index_w + mesh_w_index_offset}/{th_index_h+ mesh_h_index_offset}'
                            res = requests.post(url=request, data = thumbnail)

            zoom_depth += 1
