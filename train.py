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
    #parser.add_argument("--config", type=str, default="config/train_DISTS.cfg", help="path to data config file")

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

    # read test data
    dist = config['test']
    test_batch_size = int(config['test_batch_size'])
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=shuffle)

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    loss_type = config['loss']

    # model, STSIM or DISTS
    if config['model'] == 'STSIM':
        # prepare data
        X1_train, X2_train, Y_train, mask_train, pt_train = next(iter(train_loader))
        X1_valid, X2_valid, Y_valid, mask_valid, pt_valid = next(iter(valid_loader))
        X1_test, X2_test, Y_test, mask_test, pt_test = next(iter(test_loader))

        from metrics.STSIM import *
        m = Metric(config['filter'], device)
        # STSIM-M features
        X1_train = m.STSIM(X1_train.double().to(device))
        X2_train = m.STSIM(X2_train.double().to(device))
        X1_valid = m.STSIM(X1_valid.double().to(device))
        X2_valid = m.STSIM(X2_valid.double().to(device))
        Y_train = Y_train.to(device)
        Y_valid = Y_valid.to(device)
        mask_train = mask_train.to(device)
        mask_valid = mask_valid.to(device)
        pt_train = pt_train.to(device)
        pt_valid = pt_valid.to(device)

        # collect all data and estimate STSIM-M and STSIM-I
        X1 = torch.cat((X1_train, X1_valid))
        mask = torch.cat((mask_train, mask_valid))
        X_train = [X1[mask==i][0:1] for i in set(mask.detach().cpu().numpy())]
        mask_I = [mask[mask==i][0:1]  for i in set(mask.detach().cpu().numpy())]
        X_train.append(X2_train)
        X_train.append(X2_valid)
        mask_I.append(mask_train)
        mask_I.append(mask_valid)
        X_train = torch.cat(X_train)
        mask_I = torch.cat(mask_I)

        # STSIM-M
        weight_M = m.STSIM_M(X_train)
        torch.save(weight_M, os.path.join(config['weights_folder'],'STSIM-M.pt'))
        # STSIM-I
        weight_I = m.STSIM_I(X_train, mask = mask_I)
        torch.save(weight_I, os.path.join(config['weights_folder'],'STSIM-I.pt'))

        mode = int(config['mode'])
        # learnable parameters
        model = STSIM_M(config['dim'], mode, device).double().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        valid_perform = []
        valid_res = []
        for i in range(epochs):
            pred = model(X1_train, X2_train)
            if loss_type == 'MSE':
                loss = torch.mean((pred - Y_train) ** 2)
            elif loss_type == 'Coeff':
                # import pdb;pdb.set_trace()
                loss = -PearsonCoeff(pred, Y_train, mask_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % evaluation_interval == 0:    # validation
                print('training iter ' + str(i) + ' :', loss.item())
                pred = model(X1_valid, X2_valid)
                val = evaluation(pred, Y_valid, mask_valid)
                print('validation iter ' + str(i) + ' :', val)
                valid_perform.append(sum(val.values()))
                valid_res.append(val)
            if i % checkpoint_interval == 0:    # save weights
                torch.save(model.state_dict(), os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))
        idx = valid_perform.index(max(valid_perform))
        print('best model')
        print('epoch:', idx*evaluation_interval)
        print('performance on validation set:', valid_res[idx])
        config['weights_path'] = os.path.join(config['weights_folder'], 'epoch_' + str(idx*evaluation_interval).zfill(4) + '.pt')

        model.load_state_dict(torch.load(config['weights_path']))
        X1_test = m.STSIM(X1_test.double().to(device))
        X2_test = m.STSIM(X2_test.double().to(device))
        Y_test = Y_test.to(device)
        mask_test = mask_test.to(device)
        pred = model(X1_test, X2_test)
        test = evaluation(pred, Y_test, mask_test)
        print('performance on test set:', test)

        #import pdb;pdb.set_trace()

    elif config['model'] == 'DISTS':
        # model
        from metrics.DISTS_pt import *
        model = DISTS().to(device)
        optimizer = optim.Adam([model.alpha, model.beta], lr=lr)

        writer = SummaryWriter()
        # train
        for i in range(epochs):
            running_loss = []
            for X1_train, X2_train, Y_train, mask_train in train_loader:
                X1_train = F.interpolate(X1_train, size=256).float().to(device)
                X2_train = F.interpolate(X2_train, size=256).float().to(device)
                Y_train = Y_train.to(device)

                pred = model(X1_train, X2_train)
                if loss_type == 'MSE':
                    loss = torch.mean((pred - Y_train) ** 2)
                elif loss_type == 'Coeff':
                    loss = -PearsonCoeff(pred, Y_train, mask_train)  # min neg ==> max
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
            writer.add_scalar('Loss/train', np.mean(running_loss), i)
            print('training iter ' + str(i) + ' :', np.mean(running_loss))

            if i % evaluation_interval == 0:    # validation
                running_loss = []
                for X1_valid, X2_valid, Y_valid, mask_valid in valid_loader:
                    X1_valid = F.interpolate(X1_valid, size=256).float().to(device)
                    X2_valid = F.interpolate(X2_valid, size=256).float().to(device)
                    Y_valid = Y_valid.to(device)
                    pred = model(X1_valid, X2_valid)
                    if loss_type == 'MSE':
                        loss = torch.mean((pred - Y_valid) ** 2)
                    elif loss_type == 'Coeff':
                        loss = -PearsonCoeff(pred, Y_valid, mask_valid)  # min neg ==> max
                    running_loss.append(loss.item())
                writer.add_scalar('Loss/valid', np.mean(running_loss), i)
                print('validation iter ' + str(i) + ' :', np.mean(running_loss))
            if i % checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))
    
    # save config
    import json
    output_path = os.path.join(config['weights_folder'], 'config.json')
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file)