import argparse
import os
import itertools

from utils.parse_config import parse_config
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from metrics.STSIM import *

def generate_triplet(data):
    '''
    Args:
        data: (N,9,82*3)
        batchsize: # of N used for generating batch
    Returns:
    '''
    anchor, pos, neg = [],[],[]
    for (i,j) in itertools.combinations(range(data.shape[1]),2):
        anchor.append(data[:,i])
        pos.append(data[:,j])

        rand_idx1 = (range(data.shape[0]) + np.random.randint(1, data.shape[0], size=data.shape[0]))%data.shape[0]
        assert (range(data.shape[0]) == rand_idx1).sum() == 0

        rand_idx2 = np.random.randint(0,data.shape[1], size=data.shape[0])
        neg.append(data[rand_idx1, rand_idx2])

    #import pdb;pdb.set_trace()
    return torch.cat(anchor), torch.cat(pos), torch.cat(neg)

def train(config):
    import json
    with open(config) as f:
        config = json.load(f)
        print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_folder']):
        os.mkdir(config['weights_folder'])

    dataset_dir = config['dataset_dir']
    evaluation_interval = int(config['checkpoint_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    lr = float(config['lr'])
    epochs = int(config['epochs'])
    alpha = float(config['alpha'])

    # STSIM features
    data = torch.load(dataset_dir).double().to(device)
    # data_train = data[:300]
    # data_valid = data[300:400]
    data_test = data[400:]
    model = STSIM_M([82 * 3, 10], device=device).double().to(device)
    #model.load_state_dict(torch.load(config['weights_path']))  # shifted filterbank on Netflix data 128*128
    #model.to(device).double()
    # test2
    acc = 0
    data_test = data_test.reshape(-1, data_test.shape[2])
    for i in tqdm(range(data_test.shape[0])):
        pred = model(data_test, data_test[i].repeat(data_test.shape[0], 1))
        pred[i] += 1e10
        res = torch.argmin(pred).item()
        if res // 9 == i // 9:
            acc += 1
    acc /= data_test.shape[0]
    print('precision at one (randomized weight):', acc)

    '''
    # test1
    generate_triplet(data_test)
    #anchor_train, pos_train, neg_train = generate_triplet(data_train)
    anchor_test, pos_test, neg_test = generate_triplet(data_test)

    # learnable parameters
    dis_pos = model(anchor_test, pos_test)
    dis_neg = model(anchor_test, neg_test)
    # pred = dis_pos - dis_neg*alpha
    pred = dis_pos - dis_neg
    error = (pred>0).sum().item()
    '''

    model.load_state_dict(torch.load(config['weights_path']))  # shifted filterbank on Netflix data 128*128
    model.to(device).double()
    # test2
    acc = 0
    # data_test = data_test.reshape(-1, data_test.shape[2])
    for i in tqdm(range(data_test.shape[0])):
        pred = model(data_test, data_test[i].repeat(data_test.shape[0],1))
        pred[i] += 1e10
        res = torch.argmin(pred).item()
        if res//9 == i//9:
            acc+=1
    acc /= data_test.shape[0]
    print('precision at one (trained):',acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='weights/STSIM_macro_12282021/config.json', help="path to data config file")
    opt = parser.parse_args()
    print(opt.config)
    train(opt.config)
