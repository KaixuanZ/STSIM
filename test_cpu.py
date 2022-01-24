import argparse
import numpy as np
from utils.dataset import Dataset
from utils.dataset_concatenated import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn.functional as F
import scipy.stats
from tqdm import tqdm

from joblib import Parallel, delayed

def SpearmanCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask)
    for i in N:
        X1 = X[mask == i]
        X2 = Y[mask == i]

        coeff += np.abs(scipy.stats.spearmanr(X1, X2).correlation)
    return coeff / len(N)


def KendallCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask)
    for i in N:
        X1 = X[mask == i]
        X2 = Y[mask == i]

        coeff += np.abs(scipy.stats.kendalltau(X1, X2).correlation)
    return coeff / len(N)

def PearsonCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask)
    for i in N:
        X1 = X[mask == i]
        X1 = X1 - X1.mean()
        X2 = Y[mask == i]
        X2 = X2 - X2.mean()

        nom = np.dot(X1, X2)
        denom = np.sqrt(np.sum(X1 ** 2) * np.sum(X2 ** 2))

        coeff += np.abs(nom / (denom + 1e-10))
    return coeff / len(N)

def evaluation(pred, Y, mask):
    res = {}

    PCoeff = PearsonCoeff(pred, Y, mask)
    res['PLCC'] = float("{:.3f}".format(PCoeff))

    SCoeff = SpearmanCoeff(pred, Y, mask)
    res['SRCC'] = float("{:.3f}".format(SCoeff))

    KCoeff = KendallCoeff(pred, Y, mask)
    res['KRCC'] = float("{:.3f}".format(KCoeff))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_STSIM_global.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=4080, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = config['dataset_dir'] #'concatenated'
    label_file = config['label_file']
    dist = config['dist']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    X1, X2, Y, mask = next(iter(test_loader))


    from metrics_cpu.metric import Metric

    X1 = X1.squeeze(1).numpy()
    X2 = X2.squeeze(1).numpy()
    Y = Y.numpy()
    mask = mask.numpy()

    m = Metric()

    stsim1 = []
    stsim2 = []

    def STSIM1(X1,X2):
        return m.STSIM(X1[0], X2[0])

    def STSIM2(X1,X2):
        return m.STSIM2(X1[0],X2[0])

    def STSIM_VGG(X1,X2):
        return m.STSIM_VGG(X1[0],X2[0])

    X1 = np.array_split(X1,X1.shape[0])
    X2 = np.array_split(X2,X2.shape[0])

    # stsim_vgg = Parallel(n_jobs=6)(delayed(STSIM_VGG)(X1[i],X2[i]) for i in tqdm(range(len(X1))))
    # print("STSIM-VGG test:", evaluation(np.array(stsim_vgg), Y, mask))
    # import pdb;pdb.set_trace()

    stsim1 = Parallel(n_jobs=-1)(delayed(STSIM1)(X1[i],X2[i]) for i in tqdm(range(len(X1))))
    stsim2 = Parallel(n_jobs=-1)(delayed(STSIM2)(X1[i],X2[i]) for i in tqdm(range(len(X1))))
    
    print("STSIM-1 test:", evaluation(np.array(stsim1), Y, mask))

    print("STSIM-2 test:", evaluation(np.array(stsim2), Y, mask))
    import pdb;
    pdb.set_trace()

