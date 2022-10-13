from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('..')
from metrics.STSIM import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_split='train', format = '.png'):
        self.format = format
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clean_names = lambda x: [i for i in x if i[0] != '.']

        self.dist_img_paths = [os.path.join(data_dir, img) for img in clean_names(os.listdir(data_dir))]
        self.dist_img_paths = sorted(self.dist_img_paths)
        self.m = Metric('SCF', device)

        '''
        if data_split == 'train':
            self.dist_img_paths = self.dist_img_paths[:300]
        if data_split == 'valid':
            self.dist_img_paths = self.dist_img_paths[300:400]
        if data_split == 'test':
            self.dist_img_paths = self.dist_img_paths[400:500]
        '''


    def __len__(self):
        return len(self.dist_img_paths)

    def __getitem__(self, item):
        dist_img_path = self.dist_img_paths[item]
        dist_img = Image.open(dist_img_path)
        dist_img = transforms.ToTensor()(dist_img)

        try:
            assert dist_img.shape[0] == 3
        except:
            if dist_img.shape[0] == 1:
                dist_img = dist_img.repeat(3, 1, 1)
            elif dist_img.shape[0] > 3:
                dist_img = dist_img[:3]

        C,H,W = dist_img.shape
        data = torch.zeros(3*3*C,1,256,256)
        for c in range(C):
            for i in range(3):
                for j in range(3):
                    data[(i*3+j)*3+c,0] = dist_img[c,(H-256)*i//2:(H-256)*i//2+256,(W-256)*j//2:(W-256)*j//2+256]

        # STSIM-M features
        res = self.m.STSIM(data.double().to(self.device))
        return res.reshape(-1,82*3)


if __name__ == "__main__":
    from torch.autograd import Variable

    image_dir = '/dataset/MacroTextures500/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir)

    batch_size = 500  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for X in data_generator:
        X = X.to(device)
        import pdb;

        pdb.set_trace()
        torch.load(X,'tmp.pt')
