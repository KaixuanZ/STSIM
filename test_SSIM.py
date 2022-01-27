import argparse
import numpy as np

from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn.functional as F
from torchvision import transforms
import scipy.stats
from tqdm import tqdm

def SpearmanCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch, or [N] some other metric's output
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
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
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
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
    N = set(mask.detach().cpu().numpy())
    X = X.squeeze(-1)
    for i in N:
        X1 = X[mask == i].double()
        X1 = X1 - X1.mean()
        X2 = Y[mask == i].double()
        X2 = X2 - X2.mean()

        nom = torch.dot(X1, X2)
        denom = torch.sqrt(torch.sum(X1 ** 2) * torch.sum(X2 ** 2))

        coeff += torch.abs(nom / (denom + 1e-10))
    return coeff / len(N)

def evaluation(pred, Y, mask):
    res = {}

    PCoeff = PearsonCoeff(pred, Y, mask).item()
    res['PLCC'] = float("{:.3f}".format(PCoeff))

    SCoeff = SpearmanCoeff(pred, Y, mask)
    res['SRCC'] = float("{:.3f}".format(SCoeff))

    KCoeff = KendallCoeff(pred, Y, mask)
    res['KRCC'] = float("{:.3f}".format(KCoeff))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=4080, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = config['dataset_dir']
    label_file = config['label_file']
    dist_img_folder = config['dist']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist_img_folder)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    X1, X2, Y, mask, _ = next(iter(test_loader))

    X1 = X1.to(device).double()
    X2 = X2.to(device).double()
    Y = Y.to(device).double()
    mask = mask.to(device).double()
    mask = mask*0   #global testing

    from metrics.SSIM import ssim
    pred = ssim(X1,X2)
    print("SSIM test:", evaluation(pred, Y, mask))

    from ssim import SSIM
    trans = transforms.ToPILImage()
    X1 = (X1*255).int().squeeze(1).detach().cpu()
    X2 = (X2*255).int().squeeze(1).detach().cpu()
    pred = []
    for i in tqdm(range(X1.shape[0])):
        pred.append(SSIM(trans(X1[i])).cw_ssim_value(trans(X2[i])))
    print("CW-SSIM test:", evaluation(torch.tensor(pred).to(device), Y, mask))
