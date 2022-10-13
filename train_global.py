import argparse
import os
import numpy as np

from test_global import evaluation
from test_global import PearsonCoeff
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM_global.cfg", help="path to data config file")
    # parser.add_argument("--config", type=str, default="config/train_DISTS_global.cfg", help="path to data config file")

    opt = parser.parse_args()
    print(opt)
    config = parse_config(opt.config)

    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_folder']):
        os.mkdir(config['weights_folder'])

    # read training data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']
    dist = config['train']
    shuffle = bool(int(config['shuffle']))
    train_batch_size = int(config['train_batch_size'])
    trainset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle)

    # read validation data
    dist = config['valid']
    valid_batch_size = int(config['valid_batch_size'])
    validset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=shuffle)

    # # read test data
    # dist = config['test']
    # test_batch_size = int(config['test_batch_size'])
    # testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=shuffle)

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    loss_type = config['loss']

    if config['model'] == 'STSIM':
        from metrics.STSIM_VGG import *
        model = STSIM_VGG(config['dim']).to(device)
        # learnable parameters
        # import pdb;pdb.set_trace()
        optimizer = optim.Adam([model.linear.weight, model.linear.bias], lr=lr)
    elif config['model'] == 'DISTS':
        # model
        from metrics.DISTS_pt import *
        model = DISTS().to(device)
        optimizer = optim.Adam([model.alpha, model.beta], lr=lr)

    writer = SummaryWriter()
        # train
    for i in range(epochs):
        running_loss = []
        for X1_train, X2_train, Y_train, _, _ in train_loader:
            X1_train = F.interpolate(X1_train, size=256).float().to(device)
            X2_train = F.interpolate(X2_train, size=256).float().to(device)
            Y_train = Y_train.to(device)

            pred = model(X1_train, X2_train)
            if loss_type == 'MSE':
                loss = torch.mean((pred - Y_train) ** 2)
            elif loss_type == 'Coeff':
                loss = -PearsonCoeff(pred, Y_train)  # min neg ==> max
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        writer.add_scalar('Loss/train', np.mean(running_loss), i)
        if i % 10 == 0:
            print('training iter ' + str(i) + ' :', np.mean(running_loss))

        if i % evaluation_interval == 0:  # validation
            running_loss = []
            for X1_valid, X2_valid, Y_valid, _, _ in valid_loader:
                X1_valid = F.interpolate(X1_valid, size=256).float().to(device)
                X2_valid = F.interpolate(X2_valid, size=256).float().to(device)
                Y_valid = Y_valid.to(device)
                pred = model(X1_valid, X2_valid)
                if loss_type == 'MSE':
                    loss = torch.mean((pred - Y_valid) ** 2)
                elif loss_type == 'Coeff':
                    loss = -PearsonCoeff(pred, Y_valid)  # min neg ==> max
                running_loss.append(loss.item())
            writer.add_scalar('Loss/valid', np.mean(running_loss), i)
            print('validation iter ' + str(i) + ' :', np.mean(running_loss))
        if i % checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))

    # save config
    import json

    output_path = os.path.join(config['weights_folder'], 'config.json')
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file)