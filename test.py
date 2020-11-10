import argparse

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torch.nn.functional as F

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
    return PearsonCoeff(pred, Y, mask).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=1000, help="size of each image batch")
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

    X1, X2, Y, mask = next(iter(test_loader))

    # test with different model
    if config['model'] == 'PSNR':
        pred = torch.log(torch.mean((X1 - X2)**2, dim = [1,2,3]))
        print("PSNR test:", evaluation(pred, Y, mask)) # 0.7319034371628678
    elif config['model'] == 'STSIM':
        from metrics.STSIM import *
        X1 = X1.to(device).double()
        X2 = X2.to(device).double()
        Y = Y.to(device).double()
        mask = mask.to(device).double()
        m_g = Metric(sp3Filters, device=device)
        pred = m_g.STSIM(X1, X2)
        print("STSIM-1 test:", evaluation(pred, Y, mask)) # 0.8158

        pred = m_g.STSIM2(X1, X2)
        print("STSIM-2 test:", evaluation(pred, Y, mask))  # 0.8517


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

        print("DISTS test:", evaluation(pred, Y, mask))  #0.9356