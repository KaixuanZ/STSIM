import argparse
import os
import numpy as np

from test import evaluation
# from test_global import evaluation
from test import PearsonCoeff
# from test_global import PearsonCoeff
from utils.dataset_concatenated import Dataset
from utils.parse_config import parse_config

import torch
# import torchsort
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# def spearmanr(pred, target, **kw):
#     pred = torchsort.soft_rank(pred, **kw)
#     target = torchsort.soft_rank(target, **kw)
#     pred = pred - pred.mean()
#     pred = pred / pred.norm()
#     target = target - target.mean()
#     target = target / target.norm()
#     return (pred * target).sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM_global_concat.cfg", help="path to data config file")
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

    # read test data
    if config['model'] == 'STSIM':
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
        from metrics.STSIM import *
        m = Metric(config['filter'], device)
        # prepare data
        if config['filter'] == 'VGG':
            def load_data(dataloader):
                X1, X2, Y = [], [], []
                for x1, x2, y, _, in tqdm(dataloader):
                    x1 = F.interpolate(x1, size=256).double().to(device)
                    x2 = F.interpolate(x2, size=256).double().to(device)
                    y = y.to(device)
                    X1.append(m.STSIM(x1))
                    X2.append(m.STSIM(x2))
                    Y.append(y)
                    # import pdb;pdb.set_trace()
                X1 = torch.cat(X1, dim=0)
                X2 = torch.cat(X2, dim=0)
                Y = torch.cat(Y, dim=0)
                return X1, X2, Y
            X1_train, X2_train, Y_train = load_data(train_loader)
            X1_valid, X2_valid, Y_valid = load_data(valid_loader)
            X1_test, X2_test, Y_test = load_data(test_loader)
        else:
            def load_data(data_loader, metric, device):
                res1, res2, res3, res4 = [], [], [], []
                for X1, X2, Y, mask in tqdm(data_loader):
                    res1.append(metric.STSIM(X1.double().to(device)))
                    res2.append(metric.STSIM(X2.double().to(device)))
                    res3.append(Y.to(device))
                    res4.append(mask.to(device))
                return torch.cat(res1), torch.cat(res2), torch.cat(res3), torch.cat(res4)

            print('generating training set')
            X1_train, X2_train, Y_train, mask_train = load_data(train_loader, m, device)
            print('generating validation set')
            X1_valid, X2_valid, Y_valid, mask_valid = load_data(valid_loader, m, device)
            print('generating test set')
            X1_test, X2_test, Y_test, mask_test = load_data(test_loader, m, device)

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
                # loss = -PearsonCoeff(pred, Y_train)  - 0.05*F.relu(- pred[pt_train==1].mean() + pred[pt_train==0].mean()) # min neg ==> max
                loss = -PearsonCoeff(pred, Y_train, mask_train) # min neg ==> max
                # loss = -PearsonCoeff(pred, Y_train) # min neg ==> max
                # loss = -spearmanr(pred.T, Y_train.unsqueeze(1).T) # min neg ==> max
                # loss = -PearsonCoeff(pred, Y_train) - spearmanr(pred.T, Y_train.unsqueeze(1).T) # min neg ==> max
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % evaluation_interval == 0:  # validation
                print('training iter ' + str(i) + ' :', loss.item())
                pred = model(X1_valid, X2_valid)
                val = evaluation(pred, Y_valid, mask_valid)
                val = evaluation(pred, Y_valid, mask_valid)
                print('validation iter ' + str(i) + ' :', val)
                valid_perform.append(sum(val.values()))
                valid_res.append(val)
            if i % checkpoint_interval == 0:  # save weights
                torch.save(model.state_dict(),
                           os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))
        # import pdb;pdb.set_trace()
        idx = valid_perform.index(max(valid_perform))
        print('best model')
        print('epoch:', idx * evaluation_interval)
        print('performance on validation set:', valid_res[idx])
        config['weights_path'] = os.path.join(config['weights_folder'],
                                              'epoch_' + str(idx * evaluation_interval).zfill(4) + '.pt')

        model.load_state_dict(torch.load(config['weights_path']))
        pred = model(X1_test, X2_test)
        test = evaluation(pred, Y_test, mask_test)
        print('performance on test set:', test)

    elif config['model'] == 'DISTS':
        # model
        from metrics.DISTS_pt import *

        model = DISTS().to(device)
        optimizer = optim.Adam([model.alpha, model.beta], lr=lr)

        writer = SummaryWriter()
        # traintorchsort
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