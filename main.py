import numpy as np

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from metrics.STSIM import Metric

import torch


def stsim_features():
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda:0')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y = dataset.getdata(batchsize)

    m_g = Metric(sp3Filters, device=device)
    sub_sample = False # no subsampling
    features1 = m_g.STSIM_M(X1, sub_sample)
    features2 = m_g.STSIM_M(X2, sub_sample)

    import pdb;
    pdb.set_trace()

def test_stsim():
    # test STSIM-1 and STSIM-2
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda:0')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y = dataset.getdata(batchsize)

    m_g = Metric(sp3Filters, device=device)
    pred = m_g.STSIM(X1, X2)
    corr = np.corrcoef(pred.cpu().numpy(), Y.cpu().numpy())[0, 1]
    print("STSIM-1:",np.abs(corr))  # 0.924

    pred = m_g.STSIM2(X1, X2)
    corr = np.corrcoef(pred.cpu().numpy(), Y.cpu().numpy())[0, 1]
    print("STSIM-2:",np.abs(corr))  # 0.926


test_stsim()

stsim_features()