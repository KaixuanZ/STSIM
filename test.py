import argparse

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from utils.loss import PearsonCoeff
from utils.parse_config import parse_data_config

import torch
import torch.nn.functional as F

def PSNR(X1, X2):
    return torch.mean((X1 - X2)**2, dim = [1,2,3])

def test_stsim():
    # test STSIM-1 and STSIM-2
    image_dir = '/dataset/jana2012/'
    label_file = 'label.xlsx'
    dist_img_folder = 'test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = Dataset(data_dir=image_dir, label_file=label_file, dist_folder=dist_img_folder, grayscale=True)

    batch_size = 1000  # the actually batchsize <= total images in dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    X1, X2, Y, mask = next(iter(test_loader))
    X1 = X1.to(device).double()
    X2 = X2.to(device).double()
    Y = Y.to(device).double()
    mask = mask.to(device).double()

    m_g = Metric(sp3Filters, device=device)
    pred = m_g.STSIM(X1, X2)

    print("STSIM-1 (Borda's rule):",PearsonCoeff(pred, Y, mask)) # [0.81489032 0.81520277 0.81575464]    input is [0, 255]

    pred = m_g.STSIM2(X1, X2)
    print("STSIM-2 (Borda's rule):", PearsonCoeff(pred, Y, mask)) # [0.85011804 0.85057577 0.85170018]   input is [0, 255]

    import pdb;
    pdb.set_trace()

    model = STSIM_M([82, 10], device).double().to(device)
    X1 = m_g.STSIM_M(X1)
    X2 = m_g.STSIM_M(X2)

    path = os.path.join('weights', 'weights_STSIM_M_01000.pt')
    model.load_state_dict(torch.load(path))
    pred = model(X1, X2)
    print("STSIM-M (Borda's rule):", PearsonCoeff(pred, Y, mask))

    import pdb;
    pdb.set_trace()

def test_DISTS(model = None):
    image_dir = '/dataset/jana2012/'
    label_file = 'label.xlsx'
    dist_img_folder = 'test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = Dataset(data_dir=image_dir, label_file=label_file, dist_folder=dist_img_folder, grayscale=True)

    batch_size = 1000  # the actually batchsize <= total images in dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    X1, X2, Y, mask = next(iter(test_loader))
    X1 = X1.to(device)
    X2 = X2.to(device)
    Y = Y.to(device)
    mask = mask.to(device)
    X1 = F.interpolate(X1, size=256).float()
    X2 = F.interpolate(X2, size=256).float()
    if model is None:
        model = DISTS(weights_path = 'weights/weights_DISTS_finetuned.pt').to(device)
    pred = []
    for i in range(2):  # 2*45 = 90
        pred.append( model(X1[i*45:(i+1)*45],X2[i*45:(i+1)*45]) )
    pred = torch.cat(pred, dim=0).detach()
    print("DISTS (Borda's rule):", PearsonCoeff(pred, Y, mask))  # 0.9356 finetuned

    import pdb;
    pdb.set_trace()

def test(opt):
    pass
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
    grayscale = data_config['grayscale']
    testset = Dataset(data_dir=dataset_dir, label_file=label_file, dist_folder=dist_img_folder, grayscale=grayscale)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    X1, X2, Y, mask = next(iter(test_loader))

    # test with different model
    if data_config['model'] == 'PSNR':
        pred = PSNR(X1, X2)
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