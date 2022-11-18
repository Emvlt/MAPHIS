import sys
from typing import Dict
sys.path.append('..')

import json

import utils.constants as constants

def make_tile_shape_dict():
    path_to_tile_size = constants.TILESDATAPATH.joinpath('cities_tile_sizes.json')

    tile_size_dict:Dict = json.load(open(path_to_tile_size))

    tile_shape_dict = {}

    for city_name, tile_size in tile_size_dict.items():
        if tile_size == [10800, 7200]:
            tile_shape_dict[city_name] = 'small'
        elif tile_size == [11400, 7590]:
            tile_shape_dict[city_name] = 'large'
        else:
            print(tile_size)
            input()

    with open(constants.TILESDATAPATH.joinpath('cities_tile_shapes.json'), 'w') as out_file:
        json.dump(tile_shape_dict, out_file)


def make_tiling_dict():
    tile_dict = constants.TILEDICT
    thumnail_size = 512
    for shape, shape_dict in tile_dict.items():
        print(f'Processing {shape} shapes')
        shape_dict['coordinates'] = {}
        nTiles = 0
        for row_index in range(tile_dict[shape]['n_rows']+1):
                h_low  = (thumnail_size - tile_dict[shape]['height_stride'])*row_index
                h_high = thumnail_size*(row_index+1) - tile_dict[shape]['height_stride']*row_index
                for col_index in range(tile_dict[shape]['n_cols']+1):
                    w_low  = (thumnail_size - tile_dict[shape]['width_stride'])*col_index
                    w_high = thumnail_size*(col_index+1) - tile_dict[shape]['width_stride']*col_index
                    shape_dict['coordinates'][nTiles] = {'w_low':w_low, 'w_high':w_high, 'h_low':h_low, 'h_high':h_high}
                    nTiles+=1

    with open(constants.TILESDATAPATH.joinpath(f'tiles_dict.json'), 'w') as outfile:
        json.dump(tile_dict, outfile)

def main():
    make_tiling_dict()

if __name__ =='__main__':
    main()