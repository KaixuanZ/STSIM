import numpy as np

from steerable.sp3Filters import sp3Filters
from utils.dataset import Dataset
from metrics.STSIM import Metric

import torch

def Borda_rule(pred, label, N):
    '''
    expectation of Pearson's corr over all textures
    :param pred: values predicted by the metric
    :param label: ground truth label
    :param N: number of distortions per texture
    :return: Pearson's corr with Borda's rule
    '''
    coeff = 0
    pred = pred.reshape([N,-1]).cpu().numpy()
    label = label.reshape([N,-1]).cpu().numpy()
    for i in range(pred.shape[1]):
        corr = np.corrcoef(pred[:,i], label[:,i])[0,1]
        coeff += np.abs(corr)
    return coeff/pred.shape[1]


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
    print("STSIM-1 (Borda's rule):",Borda_rule(pred, Y, 9)) # 0.896

    pred = m_g.STSIM2(X1, X2)
    print("STSIM-2 (Borda's rule):", Borda_rule(pred, Y, 9)) # 0.886

    import pdb;
    pdb.set_trace()

test_stsim()

stsim_features()
