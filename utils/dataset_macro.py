from torchvision.io import read_image
from PIL import Image
import random
import os
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('..')
from metrics.STSIM import *
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_split='train', format = '.png'):
        self.format = format
        self.data_split = data_split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clean_names = lambda x: [i for i in x if i[0] != '.']

        self.dist_img_paths = [os.path.join(data_dir, img) for img in clean_names(os.listdir(data_dir))]
        self.dist_img_paths = sorted(self.dist_img_paths)
        self.m = Metric('SCF', self.device)

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

class Dataset_wgroup(torch.utils.data.Dataset):
    def __init__(self, data_dir, grouping_imgnames, grouping_sets, crop_size=256):
        clean_names = lambda x: [i for i in x if i[0] != '.']
        self.data_dir = data_dir
        self.g_imgnames = grouping_imgnames
        self.g_imgpaths = list(self.g_imgnames.keys())
        self.g_imgpaths = [os.path.join(data_dir, n) for n in self.g_imgpaths]

        self.g_sets = grouping_sets

        self.imgpaths = [img for img in clean_names(os.listdir(data_dir))]
        self.imgpaths = set(self.imgpaths) - set(self.g_imgpaths)
        self.imgpaths = [os.path.join(data_dir, path) for path in self.imgpaths]

        self.imgpaths_all = self.imgpaths + self.g_imgpaths
        # import pdb;pdb.set_trace()
        self.generate_offsets()

        # can also use transforms.Compose
        self.transforms = torch.nn.Sequential(
            transforms.RandomCrop(crop_size),
            transforms.Normalize((0.0, 0.0, 0.0), (255.0, 255.0, 255.0)),
        )

    def generate_offsets(self):
        # for neg samples
        self.offsets = torch.randint(1,len(self.imgpaths_all), (len(self.imgpaths),1))
        # print(self.offsets)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, item):
        # anchor1
        img_path = self.imgpaths[item]
        img1 = read_image(img_path).float()
        anchor1 = self.transforms(img1)
        pos1 = self.transforms(img1)

        # negative
        item_neg = (item + self.offsets[item].item()) % len(self.imgpaths_all)
        img_path = self.imgpaths_all[item_neg]
        img_neg = read_image(img_path).float()
        neg = self.transforms(img_neg)

        # anchor2
        img_path = self.g_imgpaths[item%len(self.g_imgpaths)]
        img2 = read_image(img_path).float()
        anchor2 = self.transforms(img2)
        pos2 = self.transforms(img2)

        # weak positive
        weak_pos = []
        for idx in self.g_imgnames[img_path.split('/')[-1]]:
            weak_pos += self.g_sets[str(idx)]
        weak_pos = list( set(weak_pos) - set([img_path.split('/')[-1]]) )
        img_path = weak_pos[self.offsets[item].item() % len(weak_pos)]
        img_path = os.path.join(self.data_dir, img_path)
        img_wpos = read_image(img_path).float()
        wpos = self.transforms(img_wpos)

        return anchor1, pos1, neg, anchor2, pos2, wpos

if __name__ == "__main__":
    import json
    with open('../data/grouping_sets.json') as f:
        grouping_sets = json.load(f)
    with open('../data/grouping_imgnames.json') as f:
        grouping_imgnames = json.load(f)

    image_dir = '/dataset/MacroTextures3K/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = Dataset(data_dir=image_dir, data_split='train')
    dataset_wg = Dataset_wgroup(image_dir, grouping_imgnames, grouping_sets)

    batch_size = 64  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset_wg, batch_size=batch_size, num_workers=20)

    res = []
    # import pdb;pdb.set_trace()
    for i in range(2):
        for anchor1, pos1, neg, anchor2, pos2, wpos in tqdm(data_generator):
            anchor1 = anchor1.to(device)
            pos1 = pos1.to(device)
            neg = neg.to(device)
            anchor2 = anchor2.to(device)
            pos2 = pos2.to(device)
            wpos = wpos.to(device)
        data_generator.dataset.generate_offsets()
        print('end of epoch')