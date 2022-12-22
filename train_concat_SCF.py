import argparse
import os
import numpy as np

from test_local import PearsonCoeff as PearsonCoeff_l
from test_local import evaluation as evaluation_l
from test_global import PearsonCoeff as PearsonCoeff_g
from test_global import evaluation as evaluation_g
from utils.dataset_concatenated import Dataset
from utils.parse_config import parse_config

import torch
import torchsort
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def spearmanr_g(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

def spearmanr_l(pred, target, mask, **kw):
    N = set(mask.detach().cpu().numpy())
    coeff = 0
    mask = mask.unsqueeze(0)
    for i in N:
        coeff += spearmanr_g(pred[mask==i].unsqueeze(0), target[mask==i].unsqueeze(0), **kw)
    return coeff/len(N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM_global_concat_1.cfg", help="path to data config file")
    # parser.add_argument("--config", type=str, default="config/train_DISTS_global_concat.cfg", help="path to data config file")

    opt = parser.parse_args()
    print(opt)
    config = parse_config(opt.config)

    print(config)
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_folder']):
        os.mkdir(config['weights_folder'])

    mode = config['mode']
    # read training data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    loss_type = config['loss']

    if config['model'] == 'STSIM':
        from metrics.STSIM import *
        model = STSIM_M(dim=[82,10]).to(device).double()

        optimizer = optim.Adam([model.linear.weight, model.linear.bias], lr=lr)
        X1_train = torch.load('features/X1_train.pt')
        X2_train = torch.load('features/X2_train.pt')
        Y_train = torch.load('features/Y_train.pt')
        mask_train = torch.load('features/mask_train.pt')

        X1_valid = torch.load('features/X1_valid.pt')
        X2_valid = torch.load('features/X2_valid.pt')
        Y_valid = torch.load('features/Y_valid.pt')
        mask_valid = torch.load('features/mask_valid.pt')

        X1_test = torch.load('features/X1_test.pt')
        X2_test = torch.load('features/X2_test.pt')
        Y_test = torch.load('features/Y_test.pt')
        mask_test = torch.load('features/mask_test.pt')
        # train
        for i in range(epochs):
            running_loss = []

            pred = model(X1_train, X2_train)
            if loss_type == 'SRCC':
                if mode == 'local':
                    loss = -spearmanr_l(pred.T, Y_train.unsqueeze(1).T, mask_train)  # min neg ==> max
                if mode == 'global':
                    loss = -spearmanr_g(pred.T, Y_train.unsqueeze(1).T) # min neg ==> max
            elif loss_type == 'PLCC':
                if mode=='local':
                    loss = -PearsonCoeff_l(pred, Y_train, mask_train)  # min neg ==> max
                if mode=='global':
                    loss = -PearsonCoeff_g(pred, Y_train)  # min neg ==> max
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            writer.add_scalar('Loss/train', np.mean(running_loss), i)
            if i % 10 == 0:
                print('training iter ' + str(i) + ' :', np.mean(running_loss))

            with torch.no_grad():
                if i % evaluation_interval == 0:  # validation
                    running_loss = []
                    pred = model(X1_valid, X2_valid)
                    if loss_type == 'SRCC':
                        if mode == 'local':
                            loss = -spearmanr_l(pred.T, Y_valid.unsqueeze(1).T, mask_valid)  # min neg ==> max
                        if mode == 'global':
                            loss = -spearmanr_g(pred.T, Y_valid.unsqueeze(1).T)  # min neg ==> max
                    elif loss_type == 'PLCC':
                        if mode == 'local':
                            loss = -PearsonCoeff_l(pred, Y_valid, mask_valid)  # min neg ==> max
                        if mode=='global':
                            loss = -PearsonCoeff_g(pred, Y_valid)  # min neg ==> max
                    running_loss.append(loss.item())
                    writer.add_scalar('Loss/valid', np.mean(running_loss), i)
                    print('validation iter ' + str(i) + ' :', np.mean(running_loss))

                    pred = model(X1_test, X2_test)
                    if mode == 'local':
                        print(evaluation_l(pred, Y_test, mask_test))
                    if mode == 'global':
                        print(evaluation_g(pred, Y_test))

            if i % checkpoint_interval == 0:
                torch.save(model.state_dict(),
                           os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))

    # save config
    import json

    output_path = os.path.join(config['weights_folder'], 'config.json')
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file)