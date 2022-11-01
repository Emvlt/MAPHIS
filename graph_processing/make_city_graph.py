import argparse
import sys
sys.path.append('..')
import utils.constants as constants
import city_graph


def construct_graph(city_name:str):
    graph = city_graph.Graph(city_name, ratio=1)
    graph.populate_graph()
    graph.make_tiles()

def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    construct_graph(city_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '1')
    args = parser.parse_args()

    construct_graph('Barrow-in-Furness')
    construct_graph('Sheerness')
