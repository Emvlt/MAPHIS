import sys
sys.path.append('..')
import utils.constants as constants
import utils.morphological_tools as morph_tools
import argparse
import loading_utils
import dataset_utils as du
import torch
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import map_drawer.draw_fake_shapes as df

def infer_tile(tile_dataloader, model, device, n_output_channels):
    segmented = torch.zeros((n_output_channels, constants.TILEWIDTH+2*constants.WIDTHPADDING, constants.TILEHEIGHT+2*constants.HEIGHTPADDING))

    for data in tqdm(tile_dataloader):
        coords = data['coordDict']
        output = model(255*data['background'].float().cuda(device))[0].detach().cpu()
        segmented[:,coords['xLow']:coords['xHigh'], coords['yLow']:coords['yHigh']] += output

    return morph_tools.unpad_tile((segmented.numpy()))


def main(args):
    n_output_channels = 1 + len(constants.HIGHLEVELFEATURES)

    hyper_parameters = {
        'n_input_channels': 3
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the city name
    city_name = constants.CITYKEY[args.city_key]['Town']

    if args.save:
        save_folder_path = constants.PROCESSEDPATH.joinpath(city_name)
        
    # Get the file_paths
    test_paths = du.get_testfile_paths(city_name)
    # Load the model
    model = loading_utils.load_model(args.process, ncIn=3, training_type=args.training_type)
    model.to(device)
    # Iterate over tiles and infer the image
    for test_path in test_paths:
        if args.save:
            save_file_path = save_folder_path.joinpath(test_path.stem)
            save_file_path.mkdir(parents=True, exist_ok=True)
        tile_dataset = du.Thumbnails(cityName=city_name, tileName=test_path.stem, n_input_channels=hyper_parameters['n_input_channels'])
        tile_dataloader = du.DataLoader(tile_dataset, batch_size=1, shuffle=True, num_workers=0)
        thresholded_tile = np.where(0.95<infer_tile(tile_dataloader, model, device, n_output_channels),1,0)
        background = thresholded_tile[0]
        if args.save:
            cv2.imwrite(f'{save_file_path}/{args.training_type}_background.jpg', np.uint8(background*255))
            cv2.imwrite(f'{save_file_path}/{args.training_type}_colored.jpg', np.transpose(np.uint8(thresholded_tile[1:]*255),[1,2,0]))
        for feature_index, feature_name in enumerate(constants.HIGHLEVELFEATURES):
            if args.save:
                cv2.imwrite(f'{save_file_path}/{args.training_type}_{feature_name}.jpg', np.uint8(thresholded_tile[1+feature_index]*255))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_key', required=False, type=str, default= '36')
    parser.add_argument('--process', required=False, type=str, default= 'segmentation')
    parser.add_argument('--training_type', required=False, type=str, default= 'real')
    parser.add_argument('--save', required=False, type=bool, default= True)
    args = parser.parse_args()
    main(args)