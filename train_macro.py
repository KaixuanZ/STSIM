import argparse
import os
import itertools

from utils.parse_config import parse_config
import torch.optim as optim
import numpy as np

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
    data_train = data[:300]
    data_valid = data[300:400]
    data_test = data[400:]

    generate_triplet(data_train)
    import pdb;pdb.set_trace()
    # learnable parameters
    model = STSIM_M([82*3, 10], device=device).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    valid_perform = []
    for i in range(epochs):
        anchor_train, pos_train, neg_train = generate_triplet(data_train)
        pred = model(anchor_train, pos_train) - model(anchor_train, neg_train) + alpha
        loss = torch.mean(F.relu(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % evaluation_interval == 0:
            print('training iter ' + str(i) + ' :', loss.item())
        if i % evaluation_interval == 0:  # validation
            anchor_valid, pos_valid, neg_valid = generate_triplet(data_valid)
            pred = model(anchor_valid, pos_valid) - model(anchor_valid, neg_valid) + alpha
            loss = torch.mean(F.relu(pred))
            print('validation iter ' + str(i) + ' :', loss.item())
            valid_perform.append(loss.item())
        if i % checkpoint_interval == 0:  # save weights
            torch.save(model.state_dict(), os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))

    idx = valid_perform.index(min(valid_perform))
    print('best model')
    print('epoch:', idx * evaluation_interval)
    print('performance on validation set:', valid_perform[idx])
    config['weights_path'] = os.path.join(config['weights_folder'],
                                          'epoch_' + str(idx * evaluation_interval).zfill(4) + '.pt')

    # save config
    import json
    output_path = os.path.join(config['weights_folder'], 'config.json')
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_STSIM_macro.cfg", help="path to data config file")
    opt = parser.parse_args()
    print(opt)
    config = parse_config(opt.config)
    print(config)
    train(config)
