import argparse
import numpy as np
from utils.dataset import Dataset
from utils.parse_config import parse_config
from torchvision.utils import save_image, make_grid

import torch
import torch.nn.functional as F
import scipy.stats
import os

from PIL import Image, ImageOps
import torchvision.transforms as transforms

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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from metrics.STSIM import *
    m_g = Metric(filter, device=device)

    model = STSIM_M([82,10], mode=0, filter = 'SCF', device = device)
    model.load_state_dict(torch.load('weights/STSIM_01242022_SCF_global_mode0/epoch_0199.pt'))
    model.to(device).double()

    def read_img(imgpath):
        img = Image.open(imgpath)
        img = ImageOps.grayscale(img)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        return img
    path1 = '/home/kaixuan/Desktop/new_data/original'
    path2 = '/home/kaixuan/Desktop/new_data/test'

    res = []
    for file1 in sorted(os.listdir(path1)):
        X1 = read_img(os.path.join(path1, file1))
        res.append(X1)
        preds = []
        imgs = []
        for i in range(9):
            X2 = read_img(os.path.join(path2, file1[:2]+'_'+str(i)+'.png'))
            imgs.append(X2)
            pred = model(X1, X2)
            # print(pred)
            preds.append(pred.item())
        preds = torch.tensor(preds)
        _, idx = torch.sort(preds)
        tmp1 = torch.cat(imgs)
        res.append(tmp1[idx])

    # tmp = torch.tensor(res)
    # print(tmp)
    # tmp = tmp.reshape(10,9)
    # print(tmp)

    # import pdb;pdb.set_trace()

    grid = make_grid(torch.cat(res), 10)
    save_image(grid, os.path.join('tmp.png'))