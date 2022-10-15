import argparse
import sys
import pathlib

sys.path.append('..')
from utils import constants

import city_node


def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    graph = city_node.TileNode(city_name, args.tile_name, args.feature_name, ratio=0.1)
    graph.initialise_graph()
    element = 'roads'
    graph.display_element(pathlib.Path(f'images/{args.tile_name}/{element}.jpg'), element)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '36')
    parser.add_argument('--tile_name', required=False, type=str, default= '0105033010241')
    parser.add_argument('--feature_name', required=False, type=str, default= 'stamps_small_font')
    args = parser.parse_args()
    main(args)
