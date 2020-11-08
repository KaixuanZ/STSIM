import argparse
import os
import numpy as np

from test import evaluation
from test import PearsonCoeff
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM.cfg", help="path to data config file")

    opt = parser.parse_args()
    print(opt)
    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_path']):
        os.mkdir(config['weights_path'])

    # read training data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']
    dist_img_folder = config['train_img_folder']
    train_batch_size = int(config['train_batch_size'])
    trainset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    # read validation data
    dataset_dir = config['dataset_dir']
    dist_img_folder = config['valid_img_folder']
    valid_batch_size = int(config['valid_batch_size'])
    validset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size)

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    # model, STSIM or DISTS
    if config['model'] == 'STSIM':
        # prepare data
        X1_train, X2_train, Y_train, mask_train = next(iter(train_loader))
        X1_valid, X2_valid, Y_valid, mask_valid = next(iter(valid_loader))

        from steerable.sp3Filters import sp3Filters
        from metrics.STSIM import *
        m = Metric(sp3Filters, device)
        # STSIM-M features
        X1_train = m.STSIM_M(X1_train.double().to(device))
        X2_train = m.STSIM_M(X2_train.double().to(device))
        X1_valid = m.STSIM_M(X1_valid.double().to(device))
        X2_valid = m.STSIM_M(X2_valid.double().to(device))
        Y_train = Y_train.to(device)
        Y_valid = Y_valid.to(device)
        mask_train = mask_train.to(device)
        mask_valid = mask_valid.to(device)

        # learnable parameters
        model = STSIM_M([X1_train.shape[1], 10], device).double().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for i in range(epochs):
            pred = model(X1_train, X2_train)
            loss = -PearsonCoeff(pred, Y_train, mask_train)  # min neg ==> max
            #loss = torch.mean((pred - Y_train) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                print('training iter ' + str(i) + ' :', loss.item())
            if i % evaluation_interval == 0:    # validation
                pred = model(X1_valid, X2_valid)
                val = evaluation(pred, Y_valid, mask_valid)
                print('validation iter ' + str(i) + ' :', val)
            if i % checkpoint_interval == 0:    # save weights
                torch.save(model.state_dict(), os.path.join(config['weights_path'], 'epoch_' + str(i).zfill(4) + '.pt'))

    elif config['model'] == 'DISTS':
        # model
        from metrics.DISTS_pt import *
        model = DISTS().to(device)
        optimizer = optim.Adam([model.alpha, model.beta], lr=0.001)

        writer = SummaryWriter()
        # train
        for i in range(epochs):
            running_loss = []
            for X1_train, X2_train, Y_train, _ in train_loader:
                X1_train = F.interpolate(X1_train, size=256).float().to(device)
                X2_train = F.interpolate(X2_train, size=256).float().to(device)
                Y_train = Y_train.to(device)

                pred = model(X1_train, X2_train)
                loss = torch.mean((pred - Y_train) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
            writer.add_scalar('Loss/train', np.mean(running_loss), i)
            print('training iter ' + str(i) + ' :', np.mean(running_loss))

            if i % evaluation_interval == 0:    # validation
                running_loss = []
                for X1_valid, X2_valid, Y_valid, _ in valid_loader:
                    X1_valid = F.interpolate(X1_valid, size=256).float().to(device)
                    X2_valid = F.interpolate(X2_valid, size=256).float().to(device)
                    Y_valid = Y_valid.to(device)
                    pred = model(X1_valid, X2_valid)
                    loss = torch.mean((pred - Y_valid) ** 2)
                    running_loss.append(loss.item())
                writer.add_scalar('Loss/valid', np.mean(running_loss), i)
                print('validation iter ' + str(i) + ' :', np.mean(running_loss))
            if i % checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(config['weights_path'], 'epoch_' + str(i).zfill(4) + '.pt'))
