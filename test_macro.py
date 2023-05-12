import argparse
import os
import itertools

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics.STSIM_VGG import *
from utils.dataset_macro import Dataset_wgroup


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = 'weights/STSIM_macro_VGG_08132022/epoch_0050.pt'

    import json
    with open('data/grouping_sets.json') as f:
        grouping_sets = json.load(f)
    with open('data/grouping_imgnames.json') as f:
        grouping_imgnames = json.load(f)

    image_dir = '/dataset/MacroTextures3K/'
    # dataset = Dataset(data_dir=image_dir, data_split='train')
    dataset_wg = Dataset_wgroup(image_dir, grouping_imgnames, grouping_sets, test_mode=True)
    batch_size = 16  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset_wg, batch_size=batch_size, num_workers=0, shuffle=False)

    model = STSIM_VGG([5900, 10], grayscale=False).to(device)
    model.load_state_dict(torch.load(weight_path))  # shifted filterbank on Netflix data 128*128
    model.to(device)
    model.eval()

    for imgs in tqdm(data_generator):
        feats = model.forward_once(imgs.to(device))

        pred = model.inference(feats[0].repeat([batch_size,1]), feats)
        import pdb;pdb.set_trace()
