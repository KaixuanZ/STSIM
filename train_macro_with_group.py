import argparse
import os
import itertools
from tqdm import tqdm
from utils.parse_config import parse_config
import torch.optim as optim
import numpy as np
from utils.dataset_macro import Dataset_wgroup

from metrics.STSIM_VGG import *
from metrics.STSIM import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(config['weights_folder']):
        os.mkdir(config['weights_folder'])

    dataset_dir = config['dataset_dir']
    batch_size = int(config['train_batch_size'])
    evaluation_interval = int(config['checkpoint_interval'])
    checkpoint_interval = int(config['checkpoint_interval'])
    dim = config['dim']
    lr = float(config['lr'])
    epochs = int(config['epochs'])
    alpha = float(config['alpha'])

    import json
    with open('data/grouping_sets.json') as f:
        grouping_sets = json.load(f)
    with open('data/grouping_imgnames.json') as f:
        grouping_imgnames = json.load(f)
    dataset_wg = Dataset_wgroup(dataset_dir, grouping_imgnames, grouping_sets)
    data_generator = torch.utils.data.DataLoader(dataset_wg, batch_size=batch_size, num_workers=12, shuffle=True)

    # learnable parameters
    model = STSIM_VGG(dim, grayscale=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # valid_perform = []
    # import pdb;pdb.set_trace()
    for i in tqdm(range(epochs)):
        running_loss = []
        running_loss1 = []
        running_loss2 = []
        running_loss3 = []
        running_loss4 = []
        for anchor1, pos1, neg, anchor2, pos2, wpos in tqdm(data_generator):
            f_anchor1 = model.forward_once(anchor1.to(device))
            f_pos1 = model.forward_once(pos1.to(device))
            f_neg = model.forward_once(neg.to(device))
            f_anchor2 = model.forward_once(anchor2.to(device))
            f_pos2 = model.forward_once(pos2.to(device))
            f_wpos = model.forward_once(wpos.to(device))
            # import pdb;pdb.set_trace()
            loss1 = F.relu(model(f_anchor1, f_pos1) - model(f_anchor1, f_neg) + alpha)
            loss2 = F.relu(-model(f_anchor2, f_pos2) + model(f_anchor2, f_wpos) - alpha/2)
            loss3 = F.relu(model(f_anchor1, f_pos1) - model(f_anchor1, f_wpos) + alpha)
            loss4 = F.relu(model(f_anchor2, f_pos2) - model(f_anchor2, f_neg) + alpha)
            loss = torch.mean(loss1 + loss2 + loss3 + loss4)
            running_loss.append(loss.item())
            running_loss1.append(loss1.mean().item())
            running_loss2.append(loss2.mean().item())
            running_loss3.append(loss3.mean().item())
            running_loss4.append(loss4.mean().item())
            # import pdb;pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(np.mean(running_loss))
        writer.add_scalar('Loss/train', np.mean(running_loss), i)
        writer.add_scalar('Loss1/train', np.mean(running_loss1), i)
        writer.add_scalar('Loss2/train', np.mean(running_loss2), i)
        writer.add_scalar('Loss3/train', np.mean(running_loss3), i)
        writer.add_scalar('Loss4/train', np.mean(running_loss4), i)
        if i % checkpoint_interval == 0:  # save weights
            torch.save(model.state_dict(), os.path.join(config['weights_folder'], 'epoch_' + str(i).zfill(4) + '.pt'))

    # idx = valid_perform.index(min(valid_perform))
    # print('best model')
    # print('epoch:', idx * evaluation_interval)
    # print('performance on validation set:', valid_perform[idx])
    # config['weights_path'] = os.path.join(config['weights_folder'],
    #                                       'epoch_' + str(idx * evaluation_interval).zfill(4) + '.pt')

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
