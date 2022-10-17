import argparse
import sys
sys.path.append('..')
import utils.constants as constants
import city_graph


def construct_graph(city_name:str, save_path=None):
    graph = city_graph.Graph(city_name, ratio=1)
    graph.populate_graph()
    graph.extract_features_city_wide()
    #graph.make_tiles()
    #graph.display_element(pathlib.Path(f'images/{city_name}.jpg'))
    #graph.populate_graph()
    '''
    el = 'lights'
    graph.display_element(save_path=pathlib.Path(f'images/{city_name}/{el}.jpg'), ratio = 0.1, element=el)
    '''
    #graph.compute_path()
    #graph.make_neighbours()
    '''for el in ['roads', 'lights']:
        graph.display_element(save_path=pathlib.Path(f'images/{city_name}/{el}.jpg'), element=el)'''

def main(args):
    city_name = constants.CITYKEY[args.city_key]['Town']
    construct_graph(city_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default = '36')
    args = parser.parse_args()
    main(args)
