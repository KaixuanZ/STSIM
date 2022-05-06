import argparse
import os
import itertools

import matplotlib.pyplot as plt
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
@torch.no_grad()
def threshold(config):
    import json
    with open(config) as f:
        config = json.load(f)
        print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_folder']):
        os.mkdir(config['weights_folder'])

    dataset_dir = config['dataset_dir']

    # STSIM features
    data = torch.load(dataset_dir).double().to(device)

    # model = STSIM_M([82 * 3, 10], device=device).double().to(device)
    model = STSIM_M([5900, 10], device=device).double().to(device)
    model.load_state_dict(torch.load(config['weights_path']))  # shifted filterbank on Netflix data 128*128
    model.to(device).double()

    batch_size = data.shape[0]//5
    dis_pos, dis_neg = [], []
    for i in range(data.shape[0]//batch_size):
        # test1
        anchor_test, pos_test, neg_test = generate_triplet(data[batch_size*i:batch_size*(i+1)])

        # learnable parameters
        # import pdb;pdb.set_trace()
        dis_pos.append(model(anchor_test, pos_test))
        dis_neg.append(model(anchor_test, neg_test))
    dis_pos = torch.cat(dis_pos)
    dis_neg = torch.cat(dis_neg)

    n_neg, bins_neg, _ = plt.hist(dis_neg.cpu().numpy(), 100, density=True, facecolor='r', alpha=0.75, label='different')
    n_pos, bins_pos, _ = plt.hist(dis_pos.cpu().numpy(), 100, density=True, facecolor='g', alpha=0.75, label='identical')
    plt.legend()
    plt.xlabel('metric value')
    plt.ylabel('probability')
    plt.xlim([0,20])
    plt.savefig('tmp2.eps')
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='weights/STSIM_macro_05052022/config.json', help="path to data config file")
    opt = parser.parse_args()
    print(opt.config)
    threshold(opt.config)
