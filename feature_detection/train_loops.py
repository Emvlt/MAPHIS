from random import paretovariate

from loading_utils import load_model
from dataset_utils import Windows, DataLoader
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('..')
import utils.constants as constants
from torch.utils.tensorboard import SummaryWriter
from draw_fake_shapes import make_batch
import numpy as np
import cv2
import pathlib

from monai import metrics, networks, transforms, losses
import matplotlib.pyplot as plt

def train_loop(operation:str, device:torch.cuda.Device, model, hyper_parameters:Dict, training_type:str):

    if training_type not in ['synthetic', 'real', 'hybrid']:
        raise ValueError (f'Wrong argument training type {training_type}, can only be synthetic, real or hybrid')

    image_save_path = pathlib.Path(f'images/{training_type}')
    image_save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = constants.MODELSPATH.joinpath('saves')
    model_save_path.mkdir(parents=True, exist_ok=True)
    '''
    # MONAI Fancy attention Unet
    model = networks.nets.UNETR(in_channels=hyper_parameters['n_input_channels'], out_channels=4, img_size=512, pos_embed='conv', norm_name='instance', spatial_dims=2)
    '''
    # Good Ol' Unet
    model = load_model(operation, ncIn=hyper_parameters['n_input_channels'], parameters = hyper_parameters['architecture_parameters'], training_type=training_type)

    device = "cuda:0"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters['learning_rate'])

    cross_entropy = nn.BCELoss()
    dice_loss = losses.DiceLoss()

    dice_metric = metrics.DiceMetric(reduction='mean_batch')
    surface_dice = metrics.SurfaceDiceMetric(class_thresholds=[0.9,0.9,0.9], reduction='mean_batch')

    writer = SummaryWriter(log_dir = f'runs/{operation}_{training_type}')

    minimum_loss = 1

    threshold = nn.Threshold(0.9 ,1)

    print(f'------ Beginning {training_type.capitalize()} Train Loop ------')
    for epoch in range(hyper_parameters['epochs']):
        epoch_loss_train = 0
        if training_type =='synthetic':
            for i in range(hyper_parameters['n_samples']):
                input_image, input_target = make_batch(hyper_parameters['batch_size'], hyper_parameters['n_input_channels'])
                input_image     = torch.from_numpy(input_image).float().to(device)
                input_target = torch.from_numpy(input_target).float().to(device)

                optimizer.zero_grad()

                out  = model(input_image)

                loss = cross_entropy(out, input_target) + dice_loss(out, input_target)

                loss.backward()
                optimizer.step()
                epoch_loss_train += loss.item()

                if i %100 == 0:
                    cv2.imwrite(image_save_path.joinpath( f'{operation}_target.jpg'), np.uint8((input_image[0,0]*(1-input_target[0,0])).detach().cpu()*255))
                    for feature_index, f_name in enumerate(constants.HIGHLEVELFEATURES):
                        cv2.imwrite(image_save_path.joinpath(f'{operation}_{f_name}.jpg'), np.uint8(out[0,1+feature_index].detach().cpu()*255))

                    d_m = dice_loss(out, input_target).item()
                    c_e = cross_entropy(out, input_target).item()
                    print(f'Epoch {epoch} / {hyper_parameters["epochs"]}; {i} / {hyper_parameters["n_samples"]}; \n Running loss : {loss.item()} \n Dice Metric : {d_m} \n BCE_loss: {c_e} ')

                    writer.add_scalar(tag='running_loss', scalar_value=loss.item(), global_step=epoch*hyper_parameters['n_samples']+i)
                    writer.add_scalar(tag='dice_loss', scalar_value=d_m, global_step=epoch*hyper_parameters['n_samples']+i)
                    writer.add_scalar(tag='bce_loss', scalar_value=c_e, global_step=epoch*hyper_parameters['n_samples']+i)

        elif training_type == 'real':
            real_dataset = Windows(hyper_parameters['n_input_channels'])
            data_loader  = DataLoader(real_dataset, batch_size=hyper_parameters['batch_size'], shuffle=True, num_workers=0)
            for i, data in enumerate(data_loader):

                input_image, input_target = data[0], data[1]
                input_image  = input_image.float().to(device)
                input_target = input_target.float().to(device)

                optimizer.zero_grad()

                out  = model(input_image)

                loss = cross_entropy(out, input_target) + dice_loss(out, input_target)

                loss.backward()
                optimizer.step()
                epoch_loss_train += loss.item()

                if i %100 == 0:
                    cv2.imwrite(str(image_save_path.joinpath( f'{operation}_target.jpg')), np.uint8((input_image[0,0]*(1-input_target[0,0])).detach().cpu()*255))
                    for feature_index, f_name in enumerate(constants.HIGHLEVELFEATURES):
                        cv2.imwrite(str(image_save_path.joinpath(f'{operation}_{f_name}.jpg')), np.uint8(out[0,1+feature_index].detach().cpu()*255))

                    d_m = dice_loss(out, input_target).item()
                    c_e = cross_entropy(out, input_target).item()
                    print(f'Epoch {epoch} / {hyper_parameters["epochs"]}; {i} / {hyper_parameters["n_samples"]}; \n Running loss : {loss.item()} \n Dice Metric : {d_m} \n BCE_loss: {c_e} ')

                    writer.add_scalar(tag='running_loss', scalar_value=loss.item(), global_step=epoch*hyper_parameters['n_samples']+i)
                    writer.add_scalar(tag='dice_loss', scalar_value=d_m, global_step=epoch*hyper_parameters['n_samples']+i)
                    writer.add_scalar(tag='bce_loss', scalar_value=c_e, global_step=epoch*hyper_parameters['n_samples']+i)




        g_step = int(epoch*hyper_parameters['n_samples'])
        scale_epoch_loss = epoch_loss_train/hyper_parameters['n_samples']
        writer.add_scalar(tag='epoch_loss_train', scalar_value=scale_epoch_loss, global_step=g_step)
        if scale_epoch_loss<minimum_loss:
            minimum_loss = scale_epoch_loss
            torch.save(model.state_dict(), constants.MODELSPATH / f'saves/{operation}_{training_type}_state_dict.pth')
