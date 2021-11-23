import argparse
import numpy as np
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

def save_as_np(pred, Y, mask, pt):
    path = 'STSIM-Mf-local-v1'
    # os.mkdir(path)
    np.save(os.path.join(path,'pred_Mf.npy'), pred.detach().cpu().numpy())
    np.save(os.path.join(path,'label.npy'), Y.detach().cpu().numpy())    # label
    np.save(os.path.join(path,'mask.npy'), mask.detach().cpu().numpy())  # class
    np.save(os.path.join(path,'pt.npy'), pt.detach().cpu().numpy())  # perceptual threshold

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
    dist = config['dist']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist=dist)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    X1, X2, Y, mask, pt = next(iter(test_loader))



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

        filter = train_config['filter']
        m_g = Metric(filter, device=device)

        path = train_config['weights_path'].split('/')
        path[-1] = 'STSIM-M.pt'
        weight_M = torch.load('/'.join(path))
        pred = m_g.STSIM_M(X1, X2, weight=weight_M)
        print("STSIM-M test:", evaluation(pred, Y, mask))  #  {'PLCC': 0.874, 'SRCC': 0.834, 'KRCC': 0.73}

        path = train_config['weights_path'].split('/')
        path[-1] = 'STSIM-I.pt'
        weight_I = torch.load('/'.join(path))
        pred = m_g.STSIM_I(X1, X2, weight=weight_I)
        print("STSIM-I test:", evaluation(pred, Y, mask))  #  {'PLCC': 0.894, 'SRCC': 0.852, 'KRCC': 0.736}

        model = STSIM_M(train_config['dim'], mode=int(train_config['mode']), filter = filter, device = device)
        model.load_state_dict(torch.load(train_config['weights_path']))
        model.to(device).double()
        pred = model(X1, X2)
        print("STSIM-M (trained) test:", evaluation(pred, Y, mask)) # for complex: {'PLCC': 0.983, 'SRCC': 0.979, 'KRCC': 0.944}
        save_as_np(pred, Y, mask, pt)
        import pdb;pdb.set_trace()
    elif config['model'] == 'DISTS':
        test_loader = torch.utils.data.DataLoader(testset, batch_size=60)
        from metrics.DISTS_pt import *

        model = DISTS(weights_path=config['weights_path']).to(device)
        pred, Y, mask = [], [], []
        for X1_test, X2_test, Y_test, mask_test in test_loader:
            X1_test = F.interpolate(X1_test, size=256).float().to(device)
            X2_test = F.interpolate(X2_test, size=256).float().to(device)
            Y_test = Y_test.to(device)
            mask_test = mask_test.to(device)
            pred.append(model(X1_test, X2_test))
            Y.append(Y_test)
            mask.append(mask_test)
            #import pdb;pdb.set_trace()
        pred = torch.cat(pred, dim=0).detach()
        Y = torch.cat(Y, dim=0).detach()
        mask = torch.cat(mask, dim=0).detach()

        print("DISTS test:", evaluation(pred, Y, mask))  # {'PLCC': 0.9574348579861184, 'SRCC': 0.9213941434033467, 'KRCC': 0.8539799877032255}