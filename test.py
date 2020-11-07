import argparse

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from utils.loss import PearsonCoeff
from utils.parse_config import parse_data_config

import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/test.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=1000, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    print(data_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    dataset_dir = data_config['dataset_dir']
    label_file = data_config['label_file']
    dist_img_folder = data_config['dist_img_folder']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    X1, X2, Y, mask = next(iter(test_loader))

    # test with different model
    if data_config['model'] == 'PSNR':
        pred = torch.mean((X1 - X2)**2, dim = [1,2,3])
        print("Pearson Coeff with Borda's rule:", PearsonCoeff(pred, Y, mask)) # 0.68791932
    elif data_config['model'] == 'STSIM':
        from metrics.STSIM import *
        X1 = X1.to(device).double()
        X2 = X2.to(device).double()
        Y = Y.to(device).double()
        mask = mask.to(device).double()
        m_g = Metric(sp3Filters, device=device)
        pred = m_g.STSIM(X1, X2)
        print("STSIM-1 Pearson Coeff with Borda's rule:", PearsonCoeff(pred, Y, mask)) # 0.8158

        pred = m_g.STSIM2(X1, X2)
        print("STSIM-2 Pearson Coeff with Borda's rule:", PearsonCoeff(pred, Y, mask))  # 0.8517
    elif data_config['model'] == 'DISTS':
        from metrics.DISTS_pt import *
        X1 = F.interpolate(X1.to(device), size=256).float()
        X2 = F.interpolate(X2.to(device), size=256).float()
        Y = Y.to(device)
        mask = mask.to(device)
        model = DISTS(weights_path=data_config['weights_path']).to(device)
        pred = []
        for i in range(2):  # 2*45 = 90, lack of GPU memory
            pred.append(model(X1[i * 45:(i + 1) * 45], X2[i * 45:(i + 1) * 45]))
        pred = torch.cat(pred, dim=0).detach()

        print("Pearson Coeff with Borda's rule:", PearsonCoeff(pred, Y, mask))  #0.9356