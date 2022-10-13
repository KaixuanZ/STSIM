import argparse
import numpy as np
from utils.dataset import Dataset
from utils.parse_config import parse_config
from metrics.STSIM_VGG import *
from metrics.DISTS_pt import *
import torch
import torch.nn.functional as F
import scipy.stats
import os
import time
from tqdm import tqdm

def SpearmanCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return np.abs(scipy.stats.spearmanr(X, Y).correlation)

def KendallCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return np.abs(scipy.stats.kendalltau(X, Y).correlation)

def PearsonCoeff(X, Y):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    X = X.squeeze(-1)

    X1 = X.double()
    X1 = X1 - X1.mean()
    X2 = Y.double()
    X2 = X2 - X2.mean()

    nom = torch.dot(X1, X2)
    denom = torch.sqrt(torch.sum(X1 ** 2) * torch.sum(X2 ** 2))

    coeff = torch.abs(nom / (denom + 1e-10))
    return coeff

def evaluation(pred, Y):
    res = {}

    PCoeff = PearsonCoeff(pred, Y).item()
    res['PLCC'] = float("{:.3f}".format(PCoeff))

    SCoeff = SpearmanCoeff(pred, Y)
    res['SRCC'] = float("{:.3f}".format(SCoeff))

    KCoeff = KendallCoeff(pred, Y)
    res['KRCC'] = float("{:.3f}".format(KCoeff))
    return res

def test_model(config, batch_size):
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']
    dist = config['dist']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    if config['model'] == 'STSIM':  # STSIM-VGG randomized weight
        model = STSIM_VGG(train_config['dim']).to(device)
    elif config['model'] == 'DISTS':
        # model = DISTS(weights_path='weights/weights_DISTS_original.pt').to(device)
        model = DISTS(weights_path=config['weights_path']).to(device)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=45)
    pred, Y, mask = [], [], []
    inference_time = []
    for X1_test, X2_test, Y_test, _, _ in test_loader:
        t1 = time.time()
        X1_test = F.interpolate(X1_test, size=256).float().to(device)
        X2_test = F.interpolate(X2_test, size=256).float().to(device)
        Y_test = Y_test.to(device)
        pred.append(model(X1_test, X2_test))
        Y.append(Y_test)
        t2 = time.time()
        if config['model'] == 'STSIM':
            print("STSIM-VGG test time:", t2 - t1)
        if config['model'] == 'DISTS':
            print("DISTS test time:", t2 - t1)
        inference_time.append(t2-t1)
    pred = torch.cat(pred, dim=0).detach()
    Y = torch.cat(Y, dim=0).detach()
    print("inference time per batch:", np.mean(inference_time[1:]))
    if config['model'] == 'STSIM':
        print("STSIM test:", evaluation(pred, Y))
    if config['model'] == 'DISTS':
        print("DISTS test:", evaluation(pred,Y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="config/test_DISTS_global.cfg", help="path to data config file")
    parser.add_argument("--config", type=str, default="config/test_STSIM_global.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=45, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)
    config = parse_config(opt.config)

    # now only for testing speed
    with torch.no_grad():
        test_model(config, opt.batch_size)