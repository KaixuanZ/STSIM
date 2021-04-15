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
    dist_img_folder = config['train_img_folder']
    shuffle = bool(int(config['shuffle']))
    train_batch_size = int(config['train_batch_size'])
    trainset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle)

    # read validation data
    dataset_dir = config['dataset_dir']
    dist_img_folder = config['valid_img_folder']
    valid_batch_size = int(config['valid_batch_size'])
    validset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size)

    epochs = int(config['epochs'])
    evaluation_interval = int(config['evaluation_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    loss_type = config['loss']

    # model, STSIM or DISTS
    if config['model'] == 'STSIM':
        # prepare data
        X1_train, X2_train, Y_train, mask_train = next(iter(train_loader))
        X1_valid, X2_valid, Y_valid, mask_valid = next(iter(valid_loader))

        k1=5
        k2=7

        X1_train_tmp = torch.cat((X1_train[mask_train <= k1], X1_valid[mask_valid <= k1]))
        X1_valid_tmp = torch.cat((X1_train[ (mask_train>k1)*(mask_train<=k2)], X1_valid[ (mask_valid>k1)*(mask_valid<=k2)]))
        X1_test_tmp = torch.cat((X1_train[ k2 < mask_train], X1_valid[ k2 < mask_valid]))
        X2_train_tmp = torch.cat((X2_train[mask_train <= k1], X2_valid[mask_valid <= k1]))
        X2_valid_tmp = torch.cat((X2_train[ (mask_train>k1)*(mask_train<=k2)], X2_valid[ (mask_valid>k1)*(mask_valid<=k2)]))
        X2_test_tmp = torch.cat((X2_train[ k2 < mask_train], X2_valid[ k2 < mask_valid]))
        Y_train_tmp = torch.cat((Y_train[mask_train <= k1], Y_valid[mask_valid <= k1]))
        Y_valid_tmp = torch.cat((Y_train[ (mask_train>k1)*(mask_train<=k2)], Y_valid[ (mask_valid>k1)*(mask_valid<=k2)]))
        Y_test_tmp = torch.cat((Y_train[ k2 < mask_train], Y_valid[ k2 < mask_valid]))
        mask_train_tmp = torch.cat((mask_train[mask_train <= k1], mask_valid[mask_valid <= k1]))
        mask_valid_tmp = torch.cat((mask_train[ (mask_train>k1)*(mask_train<=k2)], mask_valid[ (mask_valid>k1)*(mask_valid<=k2)]))
        mask_test_tmp = torch.cat((mask_train[ k2 < mask_train], mask_valid[ k2 < mask_valid]))

        X1_train = X1_train_tmp
        X1_valid = X1_valid_tmp
        X1_test = X1_test_tmp
        X2_train = X2_train_tmp
        X2_valid = X2_valid_tmp
        X2_test = X2_test_tmp
        Y_train = Y_train_tmp
        Y_valid = Y_valid_tmp
        Y_test = Y_test_tmp
        mask_train = mask_train_tmp
        mask_valid = mask_valid_tmp
        mask_test = mask_test_tmp
        import pdb;
        pdb.set_trace()
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

        # collect all data and estimate STSIM-M and STSIM-I
        X_train = [X1_train[mask_train==i][0:1] for i in set(mask_train.detach().cpu().numpy())]
        mask_I = [mask_train[mask_train==i][0:1]  for i in set(mask_train.detach().cpu().numpy())]
        X_train.append(X2_train)
        mask_I.append(mask_train)
        X_train = torch.cat(X_train)
        mask_I = torch.cat(mask_I)

        weight_M = m.STSIM_M(X_train)   #STSIM-M
        torch.save(weight_M, os.path.join(config['weights_folder'],'STSIM-M.pt'))
        weight_I = m.STSIM_I(X_train, mask = mask_I)    #STSIM-I
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
                loss = -PearsonCoeff(pred, Y_train, mask_train)  # min neg ==> max
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
        print('performance:', valid_res[idx])
        config['weights_path'] = os.path.join(config['weights_folder'], 'epoch_' + str(idx*evaluation_interval).zfill(4) + '.pt')

        import pdb;
        pdb.set_trace()

        model.load_state_dict(torch.load(config['weights_path']))
        X1_test = m.STSIM(X1_test.double().to(device))
        X2_test = m.STSIM(X2_test.double().to(device))
        Y_test = Y_test.to(device)
        mask_test = mask_test.to(device)
        pred = model(X1_test, X2_test)
        val = evaluation(pred, Y_test, mask_test)

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