import argparse
import sys
import pathlib
sys.path.append('..')
from utils import constants
import city_node
import matplotlib.pyplot as plt

def main(args):
    cityName = constants.CITYKEY[args.cityKey]['Town']
    graph = city_node.TileNode(cityName, args.tileName, args.featureName, ratio=0.1)
    graph.initialise_graph()
    el='roads'
    graph.display_element(savePath=pathlib.Path(f'images/{args.tileName}/{el}.jpg'), element=el)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--tileName', required=False, type=str, default= '0105033010241')
    parser.add_argument('--featureName', required=False, type=str, default= 'stamps_small_font')
    args = parser.parse_args()
    main(args)