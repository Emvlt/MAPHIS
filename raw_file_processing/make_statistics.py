import sys
sys.path.append('..')
import json
from utils import constants

def count_features():
    shirt_sizes = constants.DRAWING['max_size_dict'].keys()
    dict_to_save = {
        'shirt_to_probability':{shirt:0 for shirt in shirt_sizes},
        'shirt_to_feature_probability':{shirt:{'features':[], 'probabilities':[]} for shirt in shirt_sizes},

        }
    save_path = constants.IMAGESFOLDERPATH.joinpath(f'statistics.json')

    for feature_name in constants.FEATURENAMES:
        dict_to_save[feature_name] = {}
        for indice in shirt_sizes:
            n_items = int(len(list(constants.IMAGESFOLDERPATH.joinpath(f'{feature_name}/{indice}').glob(f'*{constants.FILEEXTENSION}')))/2)
            dict_to_save[feature_name][indice] =  n_items
            dict_to_save['shirt_to_probability'][indice] += n_items

    sigma = sum(dict_to_save['shirt_to_probability'].values())
    for key in dict_to_save['shirt_to_probability']:
        dict_to_save['shirt_to_probability'][key] /= sigma

    for indice in shirt_sizes:
        for feature_name in constants.FEATURENAMES:
            dict_to_save['shirt_to_feature_probability'][indice]['features'].append(feature_name)
            dict_to_save['shirt_to_feature_probability'][indice]['probabilities'].append(int(len(list(constants.IMAGESFOLDERPATH.joinpath(f'{feature_name}/{indice}').glob(f'*{constants.FILEEXTENSION}')))/2))
        s = sum(dict_to_save['shirt_to_feature_probability'][indice]['probabilities'])
        dict_to_save['shirt_to_feature_probability'][indice]['probabilities'] = [x/s for x in dict_to_save['shirt_to_feature_probability'][indice]['probabilities']]


    with open(str(save_path), 'w') as out_file:
        json.dump(dict_to_save, out_file, indent=4)

if __name__=='__main__':
    count_features()
