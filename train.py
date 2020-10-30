from tqdm import tqdm
import numpy as np

from utils.dataset import Dataset
from metrics.DISTS_pt import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

def prepare_data():
    '''
    :param dataset:
    :return: 80% for training, 20% for validation
    '''
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda:0')
    dataset = Dataset(image_dir, label_file, device)
    size = 900

    X, _, Y = dataset.getdata(size, augment=False)  # 90
    ref = F.interpolate(X, size=256).float()

    image_dir = 'data/training_images'
    dataset = Dataset(image_dir, label_file, device)
    X, Y = dataset.getdata(size, augment=True)  # 720
    X = F.interpolate(X, size=256).float()
    Y = Y.float()

    X = [X[i::10] for i in range(10)]
    Y = [Y[i::10] for i in range(10)]

    dist_train = torch.cat(X[:8], 0)       # 720
    dist_valid = torch.cat(X[8:], 0)       # 180
    Y_train = torch.cat(Y[:8], 0)       # 720
    Y_valid = torch.cat(Y[8:], 0)       # 180

    ref = torch.unbind(ref, dim=0)
    ref_train = [i.repeat(8, 1, 1, 1) for i in ref]
    ref_train = torch.cat(ref_train, 0)  # 720
    ref_valid = [i.repeat(2, 1, 1, 1) for i in ref]
    ref_valid = torch.cat(ref_valid, 0)  # 180

    return dist_train, ref_train, Y_train, dist_valid, ref_valid, Y_valid

def train_DISTS():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weights')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    dist_train, ref_train, Y_train, dist_valid, ref_valid, Y_valid = prepare_data()

    model = DISTS(weights_path = 'weights/weights_DISTS.pt').to(device)

    epoch = 100
    batchsize = 20
    optimizer = optim.SGD([model.alpha, model.beta], lr=0.002, momentum=0.9)

    writer = SummaryWriter()

    for i in range(epoch):
        # train
        running_loss = []
        for j in range(ref_train.shape[0]//batchsize):    # 720//20 = 36
            pred = model(ref_train[j*batchsize:(j+1)*batchsize], dist_train[j*batchsize:(j+1)*batchsize])
            # loss function should have two terms, the one below is E1, need to implement E2
            loss = torch.mean((pred - Y_train[j*batchsize:(j+1)*batchsize,2])**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        print("Iter " + str(i) + ", training loss:", np.mean(running_loss))
        writer.add_scalar('Loss/train', np.mean(running_loss), i)

        if (i+1)%5 ==0:
            # valid
            running_loss = []
            for j in range(ref_valid.shape[0]//batchsize):    # 180//20 = 9
                pred = model(ref_valid[j*batchsize:(j+1)*batchsize], dist_valid[j*batchsize:(j+1)*batchsize])
                loss = torch.mean((pred - Y_valid[j*batchsize:(j+1)*batchsize, 2])**2)
                running_loss.append(loss.item())
            print("Iter " + str(i) + ", validation loss:", np.mean(running_loss))
            writer.add_scalar('Loss/valid', np.mean(running_loss), i)
            # save model
            torch.save(model.state_dict(), os.path.join(args.weight_path, 'epoch_' + str(i).zfill(4) + '.pt'))


    import pdb;
    pdb.set_trace()

if __name__ == '__main__':
    train_DISTS()
