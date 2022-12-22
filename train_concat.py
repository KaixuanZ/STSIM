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

def extract_feats(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    for X1, X2, Y, mask in tqdm(data_loader):
        X1 = F.interpolate(X1, size=256).float().to(device)
        X2 = F.interpolate(X2, size=256).float().to(device)
        Y = Y.float().to(device)
        mask = mask.float().to(device)
        res1.append(model.forward_once(X1))
        res2.append(model.forward_once(X2))
        res3.append(Y)
        res4.append(mask)
    return torch.cat(res1), torch.cat(res2), torch.cat(res3), torch.cat(res4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM_global_concat.cfg", help="path to data config file")
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
    dist = config['test']
    test_batch_size = int(config['test_batch_size'])
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=shuffle)

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    loss_type = config['loss']

    if config['model'] == 'STSIM':
        from metrics.STSIM_VGG import *
        model = STSIM_VGG(config['dim']).to(device)

        optimizer = optim.Adam([model.linear.weight, model.linear.bias], lr=lr)
        X1_train, X2_train, Y_train, mask_train = extract_feats(model, train_loader)
        X1_valid, X2_valid, Y_valid, mask_valid = extract_feats(model, valid_loader)
        X1_test, X2_test, Y_test, mask_test = extract_feats(model, test_loader)
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
    elif config['model'] == 'DISTS':
        # model
        from metrics.DISTS_pt import *
        model = DISTS().to(device)
        optimizer = optim.Adam([model.alpha, model.beta], lr=lr)
        for i in range(epochs):
            running_loss = []
            for X1_train, X2_train, Y_train, mask_train in train_loader:
                X1_train = F.interpolate(X1_train, size=256).float().to(device)
                X2_train = F.interpolate(X2_train, size=256).float().to(device)
                Y_train = Y_train.float().to(device)
                mask_train = mask_train.float().to(device)
                pred = model(X1_train, X2_train)
                if loss_type == 'SRCC':
                    if mode == 'local':
                        loss = -spearmanr_l(pred.unsqueeze(1).T, Y_train.unsqueeze(1).T, mask_train)  # min neg ==> max
                    if mode == 'global':
                        loss = -spearmanr_g(pred.unsqueeze(1).T, Y_train.unsqueeze(1).T) # min neg ==> max
                elif loss_type == 'PLCC':
                    if mode=='local':
                        loss = -PearsonCoeff_l(pred.unsqueeze(1), Y_train, mask_train)  # min neg ==> max
                    if mode=='global':
                        loss = -PearsonCoeff_g(pred.unsqueeze(1), Y_train)  # min neg ==> max
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
                    for X1_valid, X2_valid, Y_valid, mask_valid in valid_loader:
                        X1_valid = F.interpolate(X1_valid, size=256).float().to(device)
                        X2_valid = F.interpolate(X2_valid, size=256).float().to(device)
                        Y_valid = Y_valid.float().to(device)
                        mask_valid = mask_valid.float().to(device)
                        pred = model(X1_valid, X2_valid)
                        if loss_type == 'SRCC':
                            if mode == 'local':
                                loss = -spearmanr_l(pred.unsqueeze(1).T, Y_valid.unsqueeze(1).T, mask_valid)  # min neg ==> max
                            if mode == 'global':
                                loss = -spearmanr_g(pred.unsqueeze(1).T, Y_valid.unsqueeze(1).T)  # min neg ==> max
                        elif loss_type == 'PLCC':
                            if mode == 'local':
                                loss = -PearsonCoeff_l(pred.unsqueeze(1), Y_valid, mask_valid)  # min neg ==> max
                            if mode=='global':
                                loss = -PearsonCoeff_g(pred.unsqueeze(1), Y_valid)  # min neg ==> max
                        running_loss.append(loss.item())
                        writer.add_scalar('Loss/valid', np.mean(running_loss), i)
                    print('validation iter ' + str(i) + ' :', np.mean(running_loss))
                    pred, Y, mask = [], [], []
                    for X1_test, X2_test, Y_test, mask_test in test_loader:
                        X1_test = F.interpolate(X1_test, size=256).float().to(device)
                        X2_test = F.interpolate(X2_test, size=256).float().to(device)
                        Y_test = Y_test.float().to(device)
                        mask_test = mask_test.float().to(device)
                        pred_test = model(X1_test, X2_test)
                        pred.append(pred_test)
                        Y.append(Y_test)
                        mask.append(mask_test)
                    pred = torch.cat(pred, dim=0)
                    Y = torch.cat(Y, dim=0)
                    mask_test = torch.cat(mask, dim=0)
                    if mode == 'local':
                        print(evaluation_l(pred.unsqueeze(1), Y, mask))
                    if mode == 'global':
                        print(evaluation_g(pred.unsqueeze(1), Y))

            if i % checkpoint_interval == 0:
                torch.save(model.state_dict(),
                           os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))

    # save config
    import json

    output_path = os.path.join(config['weights_folder'], 'config.json')
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file)