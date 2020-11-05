from utils.dataset import Dataset
from utils.loss import PearsonCoeff
from metrics.DISTS_pt import *
from metrics.STSIM import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


def prepare_data(interpolate = False):
    '''
    :param dataset:
    :return: 80% for training, 20% for validation
    '''
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)
    size = 900

    ref, _, _, _ = dataset.getdata(size, augment=False)  # 90 original image
    if interpolate:
        ref = F.interpolate(ref, size=256).double()
    ref = ref.double()

    image_dir = 'data/training_images'
    dataset = Dataset(image_dir, label_file, device)
    X, Y, mask = dataset.getdata(size, augment=True)  # 900
    if interpolate:
        X = F.interpolate(X, size=256).double()
    X = X.double()
    Y = Y.double()

    ref = torch.unbind(ref, dim=0)
    ref_train = [i.repeat(10, 1, 1, 1) for i in ref]
    ref_train = torch.cat(ref_train, 0)  # 720

    return X, ref_train, Y, mask
'''
def prepare_data(interpolate = False):
    :param dataset:
    :return: 80% for training, 20% for validation
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)
    size = 900

    ref, _, _, _ = dataset.getdata(size, augment=False)  # 90 original image
    if interpolate:
        ref = F.interpolate(ref, size=256).double()
    ref = ref.double()

    image_dir = 'data/training_images'
    dataset = Dataset(image_dir, label_file, device)
    X, Y, mask = dataset.getdata(size, augment=True)  # 900
    if interpolate:
        X = F.interpolate(X, size=256).double()
    X = X.double()
    Y = Y.double()

    X = [X[i::10] for i in range(10)]
    Y = [Y[i::10] for i in range(10)]
    mask = [mask[i::10] for i in range(10)]

    N = 10
    dist_train = torch.cat(X[:N], 0)       # 720
    dist_valid = torch.cat(X[8:], 0)       # 180
    Y_train = torch.cat(Y[:N], 0)       # 720
    Y_valid = torch.cat(Y[8:], 0)       # 180
    mask_train = torch.cat(mask[:N], 0)
    mask_valid = torch.cat(mask[8:], 0)

    ref = torch.unbind(ref, dim=0)
    ref_train = [i.repeat(N, 1, 1, 1) for i in ref]
    ref_train = torch.cat(ref_train, 0)  # 720
    ref_valid = [i.repeat(2, 1, 1, 1) for i in ref]
    ref_valid = torch.cat(ref_valid, 0)  # 180

    return dist_train, ref_train, Y_train, mask_train, dist_valid, ref_valid, Y_valid, mask_valid
'''

def train_DISTS():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dist_train, ref_train, Y_train, mask_train, dist_valid, ref_valid, Y_valid, mask_valid = prepare_data(interpolate=True)

    #model = DISTS(weights_path = 'weights/weights_DISTS_inital.pt').to(device)
    model = DISTS().to(device)

    epoch = 100
    batchsize = 60
    #optimizer = optim.SGD([model.alpha, model.beta], lr=0.01, momentum=0.9)
    optimizer = optim.Adam([model.alpha, model.beta], lr=0.001)

    writer = SummaryWriter()
    for i in range(epoch):
        # train
        running_loss = []
        for j in range(ref_train.shape[0]//batchsize):    # 720//60 = 12
            pred = model(ref_train[j*batchsize:(j+1)*batchsize], dist_train[j*batchsize:(j+1)*batchsize])
            # loss function should have two terms, the one below is E1, need to implement E2
            # loss = torch.mean((pred - Y_train[j*batchsize:(j+1)*batchsize])**2)
            loss = - PearsonCoeff(pred, Y_train[j*batchsize:(j+1)*batchsize], mask_train[j*batchsize:(j+1)*batchsize])
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
                # loss = torch.mean((pred - Y_valid[j*batchsize:(j+1)*batchsize])**2)
                loss = - PearsonCoeff(pred, Y_valid[j*batchsize:(j+1)*batchsize], mask_valid[j*batchsize:(j+1)*batchsize])
                running_loss.append(loss.item())
            print("Iter " + str(i) + ", validation loss:", np.mean(running_loss))
            writer.add_scalar('Loss/valid', np.mean(running_loss), i)
            # save model
            torch.save(model.state_dict(), os.path.join(args.weight_path, 'epoch_' + str(i).zfill(4) + '.pt'))


    import pdb;
    pdb.set_trace()

def train_STSIM_M():
    # this function will include STSIM-M and its variations, DISTS is a too different so will use a seperate function
    from steerable.sp3Filters import sp3Filters

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Metric(sp3Filters, device)

    dist_train, ref_train, Y_train, mask_train = prepare_data()

    # get STSIM-M features, not necessary but can accelerate the code
    dist_train = m.STSIM_M(dist_train.double())
    ref_train = m.STSIM_M(ref_train.double())

    model = STSIM_M([82, 10], device).double().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # test set
    image_dir = 'data/scenes_distorted'
    label_file = 'data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    dist_test, ref_test, Y_test, mask_test = dataset.getdata(batchsize)

    m_g = Metric(sp3Filters, device=device)

    # get STSIM-M features, not necessary but can accelerate the code
    dist_test = m_g.STSIM_M(dist_test)
    ref_test = m_g.STSIM_M(ref_test)

    epoch = 500
    for i in range(epoch):
        pred = model(dist_train, ref_train)
        loss = -PearsonCoeff(pred, Y_train, mask_train) # min neg ==> max
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 25 == 0: # train
            print('training iter '+ str(i) +' :', loss.item())
        if i % 50 == 0: # test
            pred = model(dist_test, ref_test)
            loss = -PearsonCoeff(pred, Y_test, mask_test)  # min neg ==> max
            print('validation iter ' + str(i) + ' :', loss.item())

    import pdb;
    pdb.set_trace()

if __name__ == '__main__':
    #train_DISTS()

    train_STSIM_M()