import argparse
import os
import numpy as np

from utils.dataset_concatenated import Dataset
from utils.parse_config import parse_config

import torch


from tqdm import tqdm


def extract_feats(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    for X1, X2, Y, mask in tqdm(data_loader):
        X1 = X1.double().to(device)
        X2 = X2.double().to(device)
        Y = Y.double().to(device)
        mask = mask.double().to(device)
        res1.append(model.STSIM(X1))
        res2.append(model.STSIM(X2))
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


    from metrics.STSIM import *
    model = Metric(filter='SF', device='cuda')

    X1_train, X2_train, Y_train, mask_train = extract_feats(model, train_loader)
    X1_valid, X2_valid, Y_valid, mask_valid = extract_feats(model, valid_loader)
    X1_test, X2_test, Y_test, mask_test = extract_feats(model, test_loader)
    torch.save(X1_train, 'features/X1_train.pt')
    torch.save(X2_train, 'features/X2_train.pt')
    torch.save(Y_train, 'features/Y_train.pt')
    torch.save(mask_train, 'features/mask_train.pt')

    torch.save(X1_valid, 'features/X1_valid.pt')
    torch.save(X2_valid, 'features/X2_valid.pt')
    torch.save(Y_valid, 'features/Y_valid.pt')
    torch.save(mask_valid, 'features/mask_valid.pt')

    torch.save(X1_test, 'features/X1_test.pt')
    torch.save(X2_test, 'features/X2_test.pt')
    torch.save(Y_test, 'features/Y_test.pt')
    torch.save(mask_test, 'features/mask_test.pt')

