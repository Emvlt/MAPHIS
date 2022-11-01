from typing import Dict,List
from collections import Counter
import math
from operator import add
import time
import pathlib

import json
import matplotlib.pyplot as plt
import numpy as np

import cv2
import sys

from sympy import ordered

sys.path.append('..')
import utils.constants as constants
import utils.upload_utils as upload_utils
import city_node, pathfinding


cities_path =  constants.CITIESFOLDERPATH
backgrounds_path = constants.RAWPATH

class Graph():
    def __init__(self, city_name:str, ratio:float, feature_name='stamps_small_font') -> None:
        self.city_name = city_name
        self.ratio = ratio
        self.feature_name = feature_name
        self.city_path = constants.CITIESFOLDERPATH.joinpath(city_name)
        self.nodes:Dict[str, city_node.TileNode] = {}
        self.road_nodes:Dict[city_node.TileNode] = {}
        self.sub_trees = []
        self.ordered_nodes = {}
        self.n_tile_lattitude = 0
        self.n_tile_longitude = 0
        self.tile_w, self.tile_h = 0,0
        self.ordered = False

    def populate_graph(self):
        if not constants.TILESDATAPATH.joinpath('cities_tilesizes.json').is_file():
            self.get_city_tilesizes()
        with open(constants.TILESDATAPATH.joinpath('cities_tilesizes.json'), 'r') as file_path:
            city_size = json.load(file_path)[self.city_name]
        self.tile_w, self.tile_h = city_size[0], city_size[1]
        tfwFilesPath = [x for x in self.city_path.iterdir() if x.suffix == '.tfw']
        self.n_tiles = len(tfwFilesPath)
        for tfwFilePath in tfwFilesPath:
            tile_name = tfwFilePath.stem
            new_node = city_node.TileNode(self.city_name, tile_name=tile_name, feature_name = self.feature_name, ratio=self.ratio, width=self.tile_w, height=self.tile_h)
            self.nodes[tile_name] = new_node

    def get_lattitude_length(self):
        return next(iter(self.nodes.values())).boundaries_geolocalised['lattitude_length']

    def get_longitude_length(self):
        return next(iter(self.nodes.values())).boundaries_geolocalised['longitude_length']

    def __str__(self)-> str:
        for node in self.nodes.values():
            print('----------- Printing new node -----------')
            print(node.to_string)
        return ''

    def save_to_json(self, savePath):
        return_list = [node.serialise_node for node in self.nodes.values()]
        with open(f'{savePath}', 'w') as out_f:
            json.dump(return_list, out_f)

    def get_n_tile_lattitude(self):
        return len(Counter([node.boundaries_geolocalised['west_boundary'] for node in self.nodes.values()]).keys())

    def get_n_tile_longitude(self):
        return len(Counter([node.boundaries_geolocalised['north_boundary'] for node in self.nodes.values()]).keys())

    ### CORE ###
    def order_graph(self):
        ### Why it is bad: there is no empty node pattern to load should there be no tile at a given coordinate. To address that:
        ### - Change city_node as by being default an empty node that returns a blank tile
        ### - Change condition if node is not None
        print(f'------------ Ordering {self.city_name} Graph ------------')
        west_boundary  = min(self.nodes.values(), key=lambda x: x.boundaries_geolocalised['west_boundary']).boundaries_geolocalised['west_boundary']
        north_boundary = max(self.nodes.values(), key=lambda x: x.boundaries_geolocalised['north_boundary']).boundaries_geolocalised['north_boundary']
        self.n_tile_lattitude = self.get_n_tile_lattitude()
        self.n_tile_longitude = self.get_n_tile_longitude()
        lat_lenght = self.get_lattitude_length()
        lon_length = self.get_longitude_length()
        for node in self.nodes.values():
            c_x = round((node.boundaries_geolocalised['west_boundary']-west_boundary)/lat_lenght)
            c_y = round((node.boundaries_geolocalised['north_boundary']-north_boundary)/lon_length)
            for n in self.ordered_nodes:
                if c_x == n[0] and c_y == n[1]:
                    print(self.ordered_nodes[n].tile_name, node.tile_name)
                    print(n, c_x, c_y )
                    print(((node.boundaries_geolocalised['west_boundary']-west_boundary)/lat_lenght))
                    print(((node.boundaries_geolocalised['north_boundary']-north_boundary)/lon_length))
            self.ordered_nodes[(c_x,c_y)] = node

        if len(self.ordered_nodes) != self.n_tiles:
            print(f'Ordered Nodes Count : {len(self.ordered_nodes)}')
            print(f'Tiles Count         : {self.n_tiles}')

        assert len(self.ordered_nodes) == self.n_tiles
        self.ordered = True

    def get_city_tilesizes(self):
        n:city_node.TileNode = next(iter(self.nodes.values()))
        tilesize = list(n.return_display_element('original').shape)
        update_city_dict(self.city_name, tilesize, 'cities_tilesizes')

    def extract_features_city_wide(self, feature_list=constants.HIGHLEVELFEATURES):
        if not self.ordered:
            self.order_graph()

        save_folder_path = constants.TILESDATAPATH.joinpath(f'overlays/{self.city_name}')
        save_folder_path.mkdir(parents=True, exist_ok=True)
        for feature_name in feature_list:
            global_index = 0
            city_dict = {
                "type": "FeatureCollection",
                "features": []
                }
            for tile_h_index in range(self.n_tile_longitude):
                h_offset = self.tile_h * tile_h_index
                for tile_w_index in range(self.n_tile_lattitude):
                    w_offset = self.tile_w * tile_w_index
                    if (tile_w_index,tile_h_index) in self.ordered_nodes:
                        city_dict, global_index, tile_dict = self.ordered_nodes[(tile_w_index,tile_h_index)].extract_feature(feature_name, city_dict, global_index, w_offset, h_offset)
                        ### Local save tile dict
                        json_local_save(self.city_name, feature_name, tile_dict, True, self.ordered_nodes[(tile_w_index,tile_h_index)].tile_name)
                        ### Online save tile dict
                        request = f'http://13.40.112.22/v1alpha1/features/{self.city_name}/{self.ordered_nodes[(tile_w_index,tile_h_index)].tile_name}_{feature_name}/insert/replace'
                        upload_utils.json_online_save(request, tile_dict)
            ### Online save city dict
            request = f'http://13.40.112.22/v1alpha1/features/{self.city_name}/{feature_name}/insert/replace'
            upload_utils.json_online_save(request, city_dict)
            json_local_save(self.city_name, feature_name, city_dict, False)

    def get_city_size(self):
        if not self.ordered:
            self.order_graph()
        pixel_width  = self.n_tile_lattitude*self.tile_w
        pixel_height = self.n_tile_longitude*self.tile_h
        update_city_dict(self.city_name, [pixel_width, pixel_height], 'cities_sizes')

    '''def make_tiles(self):
        ### SUSPECTED BRAK POINT: VERY LARGE CITIES
        ## Check if file exists locally
        if not self.ordered:
            self.order_graph()
        pixel_width  = self.n_tile_lattitude*self.tile_w
        pixel_height = self.n_tile_longitude*self.tile_h
        print([self.n_tile_lattitude, self.n_tile_longitude], [pixel_width, pixel_height])
        if constants.TILESDATAPATH.joinpath(f'{self.city_name}').is_dir():
            print(f'Folder {self.city_name} at location {constants.TILESDATAPATH} has already been processed. \n Would you like to overwrite the saved files?')
            answer = ''
            while answer not in ['Y','N']:
                answer = input("Would you like to proceed anyway? (Y/N):")
                if answer == 'N':
                    print(f'Aborting feature extraction for tile {self.city_name}')
                    return

        save_path = constants.TILESDATAPATH.joinpath(f'{self.city_name}')
        max_zoom_depth = 5 # We compute up to 5 zoom level, but could be more
        # First, we determine the maximum zoom level
        # For that zoom level, one pixel on the map is one pixel on the display
        if not self.ordered:
            self.order_graph()
        pixel_width  = self.n_tile_lattitude*self.tile_w
        pixel_height = self.n_tile_longitude*self.tile_h
        print([pixel_width, pixel_height])
        update_city_dict(self.city_name, [pixel_width, pixel_height], 'cities_sizes')
        max_dim = max(pixel_width, pixel_height)
        i = 13
        while 2**i <= max_dim:
            i += 1
        self.max_zoom_level = i
        # We then iterate over zoom levels
        #zoom_depth = 0
        zoom_depth = 4
        while zoom_depth < max_zoom_depth:
            current_zoom_level = self.max_zoom_level-zoom_depth-8
            print(f'Processing zoom level {current_zoom_level}')
            save_path_zoom = save_path.joinpath(f'{current_zoom_level}')
            input_tile_size = 2**(8+zoom_depth)
            print(f'Thumbnail size : {input_tile_size}')
            # For each zoom depth, we compute
                # the number of thumbnails (horizontal and vertical)
                # the number of thumbnails that fit in our big_tiles
            thumbnails_per_tile_width  = self.tile_w // input_tile_size + 1
            thumbnails_per_tile_height = self.tile_h // input_tile_size + 1
            mesh_w  = thumbnails_per_tile_width*input_tile_size
            mesh_h = thumbnails_per_tile_height*input_tile_size
            diff_width  = mesh_w  - self.tile_w
            diff_height = mesh_h - self.tile_h
            # We then iterate over the TILE COORDINATES
            tile_h_index = 0
            #for tile_h_index in range(self.n_tile_longitude):
            ## Get the offset first allocation out of the while and increment it, rather than get it from a multiply
            while tile_h_index < self.n_tile_longitude:
                mesh_h_index_offset = tile_h_index * thumbnails_per_tile_height
                # We now specify the part of the tile that will be queried
                start_h = diff_height*tile_h_index ### ADD MODULO SELF.TILE_h ?
                end_h   = self.tile_h
                extent_h = end_h - start_h
                dh = diff_height*(tile_h_index+1)

                print(f'H {tile_h_index} \n Start h: {start_h} \n End h: {end_h} \n Extent h: {extent_h} \n dh: {dh}')

                if self.tile_h < dh:


                    tile_h_index +=1


                try:
                    assert (dh + extent_h) // mesh_h == (dh + extent_h) / mesh_h
                except AssertionError:
                    print(extent_h, dh, mesh_h)
                    return
                assert dh<self.tile_h
                for tile_w_index in range(self.n_tile_lattitude):
                    ### ADD TILE_W_INDEX OFFSET = SELF.TILE_W // (TILE_W_INDEX*DIFF_WIDTH)
                    mesh_w_index_offset = tile_w_index * thumbnails_per_tile_width
                    # We begin by creating a (thumbnails_per_tile_width, thumbnails_per_tile_height) empty matrix
                    t0= time.time()
                    current_mesh = np.ones((mesh_w, mesh_h), np.uint8)*255

                    # We now specify the part of the tile that will be queried
                    start_w = diff_width*tile_w_index  ### ADD MODULO SELF.TILE_W ?
                    end_w   = self.tile_w
                    extent_w = end_w - start_w
                    dw = diff_width*(tile_w_index+1)

                    print(f'W {tile_w_index} \n Start w: {start_w} \n End w: {end_w} \n Extent w: {extent_w} \n dw: {dw}')
                    try:
                        assert (dw + extent_w) // mesh_w == (dw + extent_w) / mesh_w
                    except AssertionError:
                        print((dw + extent_w) // mesh_w, (dw + extent_w) / mesh_w)
                        print(self.tile_w, dw, mesh_w)
                        return
                    assert dw < self.tile_w

                    if (tile_w_index,tile_h_index) in self.ordered_nodes:
                        tile:np.ndarray = self.ordered_nodes[(tile_w_index,tile_h_index)].return_display_element('original')[start_w:end_w, start_h: end_h]
                        print(f'Found tile at position {(tile_w_index,tile_h_index)}, processing...')
                    else:
                        tile:np.ndarray = np.ones((extent_w, extent_h), np.uint8)*255
                        print(f'No tile found at position {(tile_w_index,tile_h_index)}, allocating an empty tile')
                    # We can now position the tile on the mesh
                    current_mesh[:extent_w, :extent_h] = tile
                    # We can remove the tile from the memory

                    # Now is time to load the bits from the neighbours
                    #   Load bit below
                    if tile_h_index != self.n_tile_longitude - 1:
                        if (tile_w_index,tile_h_index+1) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index,tile_h_index+1)].return_display_element('original')[start_w:end_w, : dh]
                        else:
                            tile = np.ones((extent_w, dh), np.uint8)*255

                        current_mesh[:mesh_w-diff_width*(tile_w_index+1), mesh_h-diff_height*(1+tile_h_index):] = tile


                    if tile_w_index != self.n_tile_lattitude - 1:
                        if (tile_w_index+1,tile_h_index) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index+1,tile_h_index)].return_display_element('original')[:dw, start_h:end_h]
                        else:
                            tile = np.ones((dw, extent_h), np.uint8)*255
                        current_mesh[mesh_w-diff_width*(1+tile_w_index):, :mesh_h-diff_height*(1+tile_h_index)] = tile

                    if tile_h_index != self.n_tile_longitude - 1 and tile_w_index != self.n_tile_lattitude - 1:
                        if (tile_w_index+1,tile_h_index+1) in self.ordered_nodes:
                            tile = self.ordered_nodes[(tile_w_index+1,tile_h_index+1)].return_display_element('original')[:dw, :dh]
                        else:
                            tile = np.ones((dw, dh), np.uint8)*255
                        current_mesh[mesh_w-diff_width*(1+tile_w_index):, mesh_h-diff_height*(1+tile_h_index):] = tile


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
                            thumbnail_local_save(save_name, thumbnail)
                            #request = f'http://13.40.112.22/v1alpha1/upload/images/{self.city_name}/{current_zoom_level}/{th_index_w + mesh_w_index_offset}/{th_index_h+ mesh_h_index_offset}'
                            #res = requests.post(url=request, files=cv2.imread(str(save_name)))
                            #print(f'The post request {request} has returned the status {res.status_code}')
                    print(f'Time from allocating mesh memory to upload all thumbnails:{time.time()-t0}')

                tile_h_index += 1
            zoom_depth += 1
    '''
    ### UGLY AF: REFACTOR TILE ALLOCATION
    def make_tiles(self):
        ### SUSPECTED BRAK POINT: VERY LARGE CITIES
        ## Check if file exists locally
        if constants.TILESDATAPATH.joinpath(f'{self.city_name}').is_dir():
            print(f'Folder {self.city_name} at location {constants.TILESDATAPATH} has already been processed. \n Would you like to overwrite the saved files?')
            answer = ''
            while answer not in ['Y','N']:
                answer = input("Would you like to proceed anyway? (Y/N):")
                if answer == 'N':
                    print(f'Aborting feature extraction for tile {self.city_name}')
                    return
        if not self.ordered:
            self.order_graph()
        pixel_width  = self.n_tile_lattitude*self.tile_w
        pixel_height = self.n_tile_longitude*self.tile_h
        print([self.n_tile_lattitude, self.n_tile_longitude], [pixel_width, pixel_height])

        save_path = constants.TILESDATAPATH.joinpath(f'{self.city_name}')
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
            current_zoom_level = self.max_zoom_level-zoom_depth-8
            print(f'Processing zoom level {current_zoom_level}')
            save_path_zoom = save_path.joinpath(f'{current_zoom_level}')
            input_tile_size = 2**(8+zoom_depth)
            thumbnails_per_tile_height = self.tile_h // input_tile_size + 1
            thumbnails_per_tile_width  = self.tile_w // input_tile_size + 1
            mesh_h = thumbnails_per_tile_height*input_tile_size
            mesh_w = thumbnails_per_tile_width*input_tile_size
            dh = mesh_h-self.tile_h
            dw = mesh_w-self.tile_w

            tile_h_index = 0
            start_h = 0
            while tile_h_index < self.n_tile_longitude:
                mesh_h_index_offset = tile_h_index * thumbnails_per_tile_height
                h_offset = self.tile_h-start_h
                h_condition = mesh_h-h_offset < self.tile_h
                if h_condition:
                    ### ALLOCATING END INDICES FOR TILES
                    end_h_j = mesh_h-h_offset
                    ### ASSERTING THAT ALL THE MESH IS ALLOCATED
                    assert h_offset + end_h_j == mesh_h
                    ### w index iteration
                    tile_w_index = 0
                    start_w = 0
                    while tile_w_index < self.n_tile_lattitude:
                        mesh_w_index_offset = tile_w_index * thumbnails_per_tile_width
                        t0 = time.time()
                        w_offset = self.tile_w-start_w
                        w_condition = mesh_w-w_offset <  self.tile_w
                        if w_condition:
                            end_w_j = mesh_w-w_offset
                            assert w_offset + end_w_j == mesh_w
                            mesh = np.ones((mesh_w, mesh_h), np.uint8)*255
                            ##  +-----------------+---------+
                            ##  |                 |         |
                            ##  |      i,j        |  i+1,j  |
                            ##  |                 |         |
                            ##  +-----------------+---------+
                            ##  |      i, j+1     | i+1,j+1 |
                            ##  +-----------------+---------+
                            # (i,j)
                            mesh = self.allocate_tile(mesh,   tile_w_index,   tile_h_index, start_w, self.tile_w, start_h, self.tile_h,        0, w_offset, 0, h_offset)
                            # (i+1,j)
                            mesh = self.allocate_tile(mesh, tile_w_index+1,   tile_h_index,       0,     end_w_j, start_h, self.tile_h, w_offset,   mesh_w, 0, h_offset)
                            # (i,j+1)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+1, start_w, self.tile_w,       0,     end_h_j,        0, w_offset, h_offset, mesh_h)
                            # (i+1,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+1,       0,     end_w_j,       0,     end_h_j, w_offset,   mesh_w, h_offset, mesh_h)
                            start_w  += dw
                        else:
                            end_w_j = self.tile_w
                            end_w_k = mesh_w-w_offset-self.tile_w
                            assert w_offset + end_w_j + end_w_k == mesh_w
                            ##  +-------+---------+---------+
                            ##  |       |         |         |
                            ##  | i,j   |  i+1,j  | i+2,j   |
                            ##  |       |         |         |
                            ##  +-------+---------+---------+
                            ##  | i,j+1 | i+1,j+1 | i+2,j+1 |
                            ##  +-------+---------+---------+
                            # (i,j)
                            mesh = self.allocate_tile(mesh,   tile_w_index,   tile_h_index, start_w, self.tile_w, start_h, self.tile_h,                0,         w_offset, 0, h_offset)
                            # (i+1,j)
                            mesh = self.allocate_tile(mesh, tile_w_index+1,   tile_h_index,       0,     end_w_j, start_h, self.tile_h, w_offset        , w_offset+end_w_j, 0, h_offset)
                            # (i+2,j)
                            mesh = self.allocate_tile(mesh, tile_w_index+2,   tile_h_index,       0,     end_w_k, start_h, self.tile_h, w_offset+end_w_j,           mesh_w, 0, h_offset)
                            # (i,j+1)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+1, start_w, self.tile_w,       0,     end_h_j,                0,         w_offset, h_offset, mesh_h)
                            # (i+1,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+1,       0,     end_w_j,       0,     end_h_j, w_offset        , w_offset+end_w_j, h_offset, mesh_h)
                            # (i+2,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+2, tile_h_index+1,       0,     end_w_k,       0,     end_h_j, w_offset+end_w_j,           mesh_w, h_offset, mesh_h)
                            ### Index allocation
                            tile_w_index += 1
                            start_w = dw-w_offset
                        self.save_mesh(save_path_zoom, mesh, input_tile_size, tile_h_index, tile_w_index, thumbnails_per_tile_width, thumbnails_per_tile_height, mesh_h_index_offset, mesh_w_index_offset)
                        print(f'Time from allocating mesh memory to upload all thumbnails:{time.time()-t0}')
                        tile_w_index +=1
                    ### INCREMENTING INDICES
                    start_h  += dh

                else:
                    ### ALLOCATING END INDICES FOR TILES
                    end_h_j = self.tile_h
                    end_h_k = mesh_h-h_offset-self.tile_h
                    ### ASSERTING THAT ALL THE MESH IS ALLOCATED
                    assert h_offset + end_h_j + end_h_k == mesh_h
                    ### w index iteration
                    tile_w_index = 0
                    start_w = 0
                    while tile_w_index < self.n_tile_lattitude:
                        mesh_w_index_offset = tile_w_index * thumbnails_per_tile_width
                        t0 = time.time()
                        w_offset = self.tile_w-start_w
                        w_condition = mesh_w-w_offset <  self.tile_w
                        if w_condition:
                            end_w_j = mesh_w-w_offset
                            assert w_offset + end_w_j == mesh_w
                            mesh = np.ones((mesh_w, mesh_h), np.uint8)*255
                            ##  +-----------------+---------+
                            ##  |   i,j           |  i+1,j  |
                            ##  +-----------------+---------+
                            ##  |                 |         |
                            ##  |   i,j+1         | i+1,j+1 |
                            ##  |                 |         |
                            ##  +-----------------+---------+
                            ##  |   i,j+2         | i+1,j+2 |
                            ##  +-----------------+---------+
                            # (i,j)
                            mesh = self.allocate_tile(mesh,   tile_w_index,   tile_h_index, start_w, self.tile_w, start_h, self.tile_h,        0, w_offset,        0, h_offset)
                            # (i+1,j)
                            mesh = self.allocate_tile(mesh, tile_w_index+1,   tile_h_index,       0,     end_w_j, start_h, self.tile_h, w_offset,   mesh_w,        0, h_offset)
                            # (i,j+1)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+1, start_w, self.tile_w,       0,     end_h_j,        0, w_offset, h_offset, h_offset+end_h_j)
                            # (i+1,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+1,       0,     end_w_j,       0,     end_h_j, w_offset,   mesh_w, h_offset, h_offset+end_h_j)
                            # (i,j+2)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+2, start_w, self.tile_w,       0,     end_h_k,        0, w_offset, h_offset+end_h_j, mesh_h)
                            # (i+1,j+2)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+2,       0,     end_w_j,       0,     end_h_k, w_offset,   mesh_w, h_offset+end_h_j, mesh_h)
                            start_w  += dw
                        else:
                            end_w_j = self.tile_w
                            end_w_k = mesh_w-w_offset-self.tile_w
                            assert w_offset + end_w_j + end_w_k == mesh_w
                            ##  +-------+---------+---------+
                            ##  | i,j   | i+1,j   |  i+2,j  |
                            ##  +-------+---------+---------+
                            ##  |       |         |         |
                            ##  | i,j+1 | i+1,j+1 | i+2,j+1 |
                            ##  |       |         |         |
                            ##  +-------+---------+---------+
                            ##  | i,j+2 | i+1,j+2 | i+2,j+2 |
                            ##  +-------+---------+---------+
                            # (i,    j)
                            mesh = self.allocate_tile(mesh,   tile_w_index,   tile_h_index, start_w, self.tile_w, start_h, self.tile_h,                0,         w_offset, 0, h_offset)
                            # (i+1,  j)
                            mesh = self.allocate_tile(mesh, tile_w_index+1,   tile_h_index,       0,     end_w_j, start_h, self.tile_h, w_offset        , w_offset+end_w_j, 0, h_offset)
                            # (i+2,  j)
                            mesh = self.allocate_tile(mesh, tile_w_index+2,   tile_h_index,       0,     end_w_k, start_h, self.tile_h, w_offset+end_w_j,           mesh_w, 0, h_offset)
                            # (i,  j+1)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+1, start_w, self.tile_w,       0,     end_h_j,                0,         w_offset, h_offset, h_offset+end_h_j)
                            # (i+1,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+1,       0,     end_w_j,       0,     end_h_j, w_offset        , w_offset+end_w_j, h_offset, h_offset+end_h_j)
                            # (i+2,j+1)
                            mesh = self.allocate_tile(mesh, tile_w_index+2, tile_h_index+1,       0,     end_w_k,       0,     end_h_j, w_offset+end_w_j,           mesh_w, h_offset, h_offset+end_h_j)
                            # (i,  j+2)
                            mesh = self.allocate_tile(mesh,   tile_w_index, tile_h_index+2, start_w, self.tile_w,       0,     end_h_k,                0,         w_offset, h_offset+end_h_j, mesh_h)
                            # (i+1,j+2)
                            mesh = self.allocate_tile(mesh, tile_w_index+1, tile_h_index+2,       0,     end_w_j,       0,     end_h_k, w_offset        , w_offset+end_w_j, h_offset+end_h_j, mesh_h)
                            # (i+2,j+2)
                            mesh = self.allocate_tile(mesh, tile_w_index+2, tile_h_index+2,       0,     end_w_k,       0,     end_h_k, w_offset+end_w_j,           mesh_w, h_offset+end_h_j, mesh_h)
                            ### Index allocation
                            tile_w_index += 1
                            start_w = dw-w_offset
                        self.save_mesh(save_path_zoom, mesh, input_tile_size, tile_h_index, tile_w_index, thumbnails_per_tile_width, thumbnails_per_tile_height, mesh_h_index_offset, mesh_w_index_offset)
                        print(f'Time from allocating mesh memory to upload all thumbnails:{time.time()-t0}')
                        tile_w_index +=1
                    ### INCREMENTING INDICES
                    tile_h_index += 1
                    start_h = dh-h_offset



                tile_h_index += 1

            zoom_depth += 1

    def allocate_tile(self, mesh, tile_w_index, tile_h_index, start_w_tile, end_w_tile, start_h_tile, end_h_tile, start_w_mesh, end_w_mesh, start_h_mesh, end_h_mesh):
        if (tile_w_index,tile_h_index) in self.ordered_nodes:
            tile:np.ndarray = self.ordered_nodes[(tile_w_index,tile_h_index)].return_display_element('original')[start_w_tile:end_w_tile, start_h_tile:end_h_tile]
            print(f'Found tile at position {(tile_w_index,tile_h_index)}, processing...')
        else:
            tile = np.ones((end_w_tile-start_w_tile, end_h_tile-start_h_tile), np.uint8)*255
            print(f'No tile found at position {(tile_w_index,tile_h_index)}, allocating an empty tile')
        mesh[start_w_mesh:end_w_mesh, start_h_mesh:end_h_mesh] = tile
        return mesh

    def save_mesh(self, save_path_zoom:pathlib.Path, mesh:np.ndarray, input_tile_size, tile_h_index, tile_w_index, thumbnails_per_tile_width, thumbnails_per_tile_height, mesh_h_index_offset, mesh_w_index_offset):
        print(f'Processing mesh coordinate: ({tile_h_index, tile_w_index})')
        print(f'Current_mesh_shape : {mesh.shape}')
        for th_index_w in range(thumbnails_per_tile_width):
            save_path_zoom_width = save_path_zoom.joinpath(f'{th_index_w + mesh_w_index_offset}')
            save_path_zoom_width.mkdir(parents=True, exist_ok=True)
            start_mesh_w = th_index_w*input_tile_size
            end_mesh_w = (th_index_w+1)*input_tile_size
            for th_index_h in range(thumbnails_per_tile_height):
                save_name = save_path_zoom_width.joinpath(f'{th_index_h + mesh_h_index_offset}.jpg')
                start_mesh_h = th_index_h*input_tile_size
                end_mesh_h = (th_index_h+1)*input_tile_size
                thumbnail = mesh[start_mesh_w:end_mesh_w, start_mesh_h:end_mesh_h]
                thumbnail = np.transpose(cv2.resize(thumbnail, dsize=(256,256), interpolation=cv2.INTER_CUBIC))
                thumbnail_local_save(save_name, thumbnail)

    ### EXPERIMENTAL ###

    def make_neighbours(self):
        if not self.ordered:
            self.order_graph()
        for key, node in self.ordered_nodes.items():
            for cartesian_tuple, cardinal in constants.CARTESIANTOCARDINAL.items():
                neighbourd_coords = tuple(map(add, key, cartesian_tuple))
                if neighbourd_coords in self.ordered_nodes:
                    node.neighbours[cardinal] = self.ordered_nodes[neighbourd_coords]
            node.display_neighbours(0.1)

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

def thumbnail_local_save(save_path:pathlib.Path, thumbnail:np.ndarray):
    cv2.imwrite(str(save_path), thumbnail)

def json_local_save(city_name:str, feature_name:str, data_dict:Dict, tile:bool, tile_name=None):
    print('Saving Json Locally')
    t0 = time.time()
    if tile_name is not None:
        save_kwd = f'{city_name}/{tile_name}/{feature_name}.json'
    else:
        save_kwd = f'{city_name}/{feature_name}.json'

    with open(constants.SHAPESDATAPATH.joinpath(save_kwd), 'w') as f:
        json.dump(data_dict, f)

    if not tile:
        ### Local save city dict as js file
        with open(constants.TILESDATAPATH.joinpath(f'overlays/{city_name}/{feature_name}.js'), 'w') as out_file:
            out_file.write(f'var tile_data_{feature_name} = {json.dumps(data_dict)};' )

    print(f'Elapsed Time Saving City Data Locally: {time.time()-t0}')

def check_if_json(file_path:pathlib.Path):
    if not file_path.is_file():
        empty_dict = {}
        with open(file_path, 'w') as fp:
            json.dump(empty_dict, fp, indent=4)


def update_city_dict(city_name: str, city_size:List, dict_name:str):
    dict_path = constants.TILESDATAPATH.joinpath(f'{dict_name}.json')
    check_if_json(dict_path)
    with open(dict_path, 'r') as file_path:
        city_sizes_dict:Dict = json.load(file_path)
    if city_name not in  city_sizes_dict:
        city_sizes_dict[city_name] = city_size
        with open(dict_path, 'w') as file_path:
            json.dump(city_sizes_dict, file_path, indent=4)
