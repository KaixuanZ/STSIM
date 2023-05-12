from PIL import Image, ImageOps

import os
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('..')
from metrics.STSIM import *
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_split='train', format = '.png', color_mode=1):
        self.format = format
        self.data_split = data_split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clean_names = lambda x: [i for i in x if i[0] != '.']

        self.dist_img_paths = [os.path.join(data_dir, img) for img in clean_names(os.listdir(data_dir))]
        self.dist_img_paths = sorted(self.dist_img_paths)
        self.m = Metric('SCF', self.device)
        self.color_mode=color_mode

        # import json
        # with open('MacroSyn30000.json', 'w') as json_file:
        #     json.dump(self.dist_img_paths, json_file)
        # import pdb;pdb.set_trace()
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
        '''
        color_mode = 1: concatenate 82 dim features from each channel, feat_dim=82*3=246
        color_mode = 2: grayscale 82 plus means value from each channel, feat_dim=82+3=85
        '''
        if self.color_mode==1:
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
            if self.data_split == 'train':
                C,H,W = dist_img.shape
                data = torch.zeros(3*3*C,1,256,256)
                for c in range(C):
                    for i in range(3):
                        for j in range(3):
                            data[(i*3+j)*3+c,0] = dist_img[c,(H-256)*i//2:(H-256)*i//2+256,(W-256)*j//2:(W-256)*j//2+256]
            if self.data_split == 'test':
                data = dist_img.unsqueeze(1)

            # STSIM-M features
            res = self.m.STSIM(data.double().to(self.device))
            # import pdb;pdb.set_trace()
            return res.reshape(-1,82*3)
        elif self.color_mode==2:
            dist_img_path = self.dist_img_paths[item]
            dist_img = Image.open(dist_img_path)
            dist_img_gray = ImageOps.grayscale(dist_img)
            dist_img_gray = transforms.ToTensor()(dist_img_gray)
            dist_img = transforms.ToTensor()(dist_img)

            try:
                assert dist_img.shape[0] == 3
            except:
                if dist_img.shape[0] == 1:
                    dist_img = dist_img.repeat(3, 1, 1)
                elif dist_img.shape[0] > 3:
                    dist_img = dist_img[:3]
            feat_color = []
            if self.data_split == 'train':
                _, H, W = dist_img_gray.shape
                data = torch.zeros(3 * 3, 1, 256, 256)
                for i in range(3):
                    for j in range(3):
                        data[i * 3 + j, 0] = dist_img_gray[0, (H - 256) * i // 2:(H - 256) * i // 2 + 256,
                                                       (W - 256) * j // 2:(W - 256) * j // 2 + 256]
                        data_color = dist_img[:, (H - 256) * i // 2:(H - 256) * i // 2 + 256,
                                                       (W - 256) * j // 2:(W - 256) * j // 2 + 256]
                        feat_color.append(data_color.mean([1,2]))
            if self.data_split == 'test':
                data = dist_img_gray.unsqueeze(1)

            # STSIM-M features
            res = self.m.STSIM(data.double().to(self.device))
            feat_color = torch.stack(feat_color).to(self.device)
            res = torch.cat([res,feat_color],dim=1)

            return res


if __name__ == "__main__":

    image_dir = '/dataset/MacroTextures3K/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir, data_split='train', color_mode=2) # 82 dim grayscale feature + 3 dim color feature

    batch_size = 100  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    res = []
    for X in tqdm(data_generator):
        X = X.to(device)
        res.append(X)
    import pdb;

    pdb.set_trace()
    torch.save(torch.cat(res),'../data/MacroSyn30000_SCF_dim85.pt')
