import argparse
import sys
sys.path.append('..')
import utils.constants as constants
import city_graph
import pathlib

feature_dict = {
    'labels':'class',
    'buildings':'area',
    'trees':'area',
    'stamps_small_font':'class',
    'stamps_large_font':'class'
    }


def construct_graph(cityName:str, savePath=None):    
    graph = city_graph.Graph(cityName, ratio=1)
    graph.populate_graph()
    #graph.compute_path((1,2), 3, (1,2), 277, 'test_path_0105033050091')
    #graph.extract_features_city_wide(feature_list=['text', 'vegetation', 'imprint'])
    #graph.make_dataset()
    graph.make_tiles()
    #graph.display_element(pathlib.Path(f'images/{cityName}.jpg'))
    #graph.populate_graph()
    '''
    el = 'lights'
    graph.display_element(savePath=pathlib.Path(f'images/{cityName}/{el}.jpg'), ratio = 0.1, element=el)
    '''
    #graph.compute_path()
    #graph.make_neighbours()
    '''for el in ['roads', 'lights']:
        graph.display_element(savePath=pathlib.Path(f'images/{cityName}/{el}.jpg'), element=el)'''

def main(args):
    cityName = constants.CITYKEY[args.cityKey]['Town']
    construct_graph(cityName)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityKey', required=False, type=str, default = '71')
    args = parser.parse_args()
    main(args)