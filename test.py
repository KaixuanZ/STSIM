import argparse
import numpy as np

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn.functional as F
import scipy.stats

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
    dist_img_folder = config['dist_img_folder']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    X1, X2, Y, mask = next(iter(test_loader))

    # test with different model
    if config['model'] == 'PSNR':
        tmp = torch.tensor([torch.max(X1[i]) for i in range(X1.shape[0])])
        pred = 10 * torch.log10(tmp * tmp / torch.mean((X1 - X2) ** 2, dim=[1, 2, 3]))
        print("PSNR test:", evaluation(pred, Y, mask))
    elif config['model'] == 'STSIM':
        from metrics.STSIM import *
        X1 = X1.to(device).double()
        X2 = X2.to(device).double()
        Y = Y.to(device).double()
        mask = mask.to(device).double()
        m_g = Metric(sp3Filters, device=device)

        pred = m_g.STSIM(X1, X2)
        print("STSIM-1 test:", evaluation(pred, Y, mask)) # {'PLCC': 0.834, 'SRCC': 0.82, 'KRCC': 0.708}

        pred = m_g.STSIM2(X1, X2)
        print("STSIM-2 test:", evaluation(pred, Y, mask))  #  {'PLCC': 0.899, 'SRCC': 0.881, 'KRCC': 0.775}

        model = STSIM_M(train_config['dim'], mode=int(train_config['mode']), device = device)
        model.load_state_dict(torch.load(config['weights_path']))
        model.to(device).double()
        pred = model(X1, X2)
        print("STSIM-M test:", evaluation(pred, Y, mask)) # for complex: {'PLCC': 0.983, 'SRCC': 0.979, 'KRCC': 0.944}
        import pdb;

        pdb.set_trace()

    elif config['model'] == 'DISTS':
        from metrics.DISTS_pt import *
        X1 = F.interpolate(X1.to(device), size=256).float()
        X2 = F.interpolate(X2.to(device), size=256).float()
        Y = Y.to(device)
        mask = mask.to(device)
        model = DISTS(weights_path=config['weights_path']).to(device)
        pred = []
        for i in range(2):  # 2*45 = 90, lack of GPU memory
            pred.append(model(X1[i * 45:(i + 1) * 45], X2[i * 45:(i + 1) * 45]))
        pred = torch.cat(pred, dim=0).detach()

        print("DISTS test:", evaluation(pred, Y, mask))  # {'PLCC': 0.9574348579861184, 'SRCC': 0.9213941434033467, 'KRCC': 0.8539799877032255}