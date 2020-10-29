from tqdm import tqdm

from utils.dataset import Dataset
from metrics.DISTS_pt import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_DISTS():
    # test STSIM-1 and STSIM-2
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda:0')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000

    X1, _, Y = dataset.getdata(batchsize, augment=False)    # 90
    ref = F.interpolate(X1, size=256).float()

    ref = torch.unbind(ref, dim = 0)
    ref = [i.repeat(10,1,1,1) for i in ref]
    ref = torch.cat(ref, 0)                                 # 900

    image_dir = 'data/training_images'
    dataset = Dataset(image_dir, label_file, device)
    X1, Y = dataset.getdata(batchsize, augment=True)        # 900
    dist = F.interpolate(X1, size=256).float()
    Y = Y.float()

    model = DISTS(weights_path = 'weights/weights_DISTS.pt').to(device)

    epoch = 10
    batchsize = 20
    criterion = nn.MSELoss()
    optimizer = optim.SGD([model.alpha, model.beta], lr=0.001, momentum=0.9)

    for i in tqdm(range(epoch)):
        running_loss = 0
        for j in tqdm(range(ref.shape[0]//batchsize)):    # 900//20 = 45
            pred = model(ref[j*batchsize:(j+1)*batchsize], dist[j*batchsize:(j+1)*batchsize])
            loss = criterion(pred, Y[j*batchsize:(j+1)*batchsize,2])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(running_loss)
    import pdb;
    pdb.set_trace()

if __name__ == '__main__':
    train_DISTS()
