from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from utils.loss import PearsonCoeff
from metrics.STSIM import *
from metrics.DISTS_pt import *

import torch
import torch.nn.functional as F

def test_PSNR():
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y, mask = dataset.getdata(batchsize)

    MSE = torch.mean((X1 - X2)**2, dim = [1,2,3])

    print("PSNR (Borda's rule):", PearsonCoeff(MSE, Y, mask))    # 0.68791932

    import pdb;
    pdb.set_trace()

def test_stsim():
    # test STSIM-1 and STSIM-2
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y, mask = dataset.getdata(batchsize)

    m_g = Metric(sp3Filters, device=device)
    pred = m_g.STSIM(X1, X2)
    print("STSIM-1 (Borda's rule):",PearsonCoeff(pred, Y, mask)) # [0.81489032 0.81520277 0.81575464]    input is [0, 255]

    pred = m_g.STSIM2(X1, X2)
    print("STSIM-2 (Borda's rule):", PearsonCoeff(pred, Y, mask)) # [0.85011804 0.85057577 0.85170018]   input is [0, 255]

    model = STSIM_M().double().to(device)
    X1 = m_g.STSIM_M(X1)
    X2 = m_g.STSIM_M(X2)

    data = torch.cat([X2, X1[::9, :]], dim=0)
    model.init_weight(data)
    coeff, pred = model(X1, X2, Y, mask)
    print("STSIM-M (Borda's rule):", PearsonCoeff(pred, Y, mask))

    import pdb;
    pdb.set_trace()

def test_DISTS(model = None):
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y, mask = dataset.getdata(batchsize)
    X1 = F.interpolate(X1, size=256).float()
    X2 = F.interpolate(X2, size=256).float()
    if model is None:
        model = DISTS(weights_path = 'weights/weights_DISTS_trained.pt').to(device)
    pred = []
    for i in range(2):  # 2*45 = 90
        pred.append( model(X1[i*45:(i+1)*45],X2[i*45:(i+1)*45]) )
    pred = torch.cat(pred, dim=0).detach()
    print("DISTS (Borda's rule):", PearsonCoeff(pred, Y, mask))  # [0.79882915 0.80043482 0.8006695 ] pre-trained, [0.81766776 0.81650565 0.81800537] 34 epoches

    import pdb;
    pdb.set_trace()

if __name__ == '__main__':
    test_DISTS()

    test_PSNR()

    test_stsim()