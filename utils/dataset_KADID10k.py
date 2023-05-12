from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, data_split = 'train'):
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_file = os.path.join(data_dir, label_file)

        self.df = pd.read_csv(self.label_file)

        self.ref_img_path = []
        self.dist_img_path = []
        self.label = []
        for i in range(self.df.shape[0]):
            self.ref_img_path.append(os.path.join(self.image_dir, self.df.loc[i]['ref_img']))
            self.dist_img_path.append(os.path.join(self.image_dir, self.df.loc[i]['dist_img']))
            self.label.append((5 - self.df.loc[i]['dmos'])/4)   # tranform DMOS to [0,1], smaller means more similar

        if data_split == 'train':
            self.ref_img_path = self.ref_img_path[:65*125]
            self.dist_img_path = self.dist_img_path[:65*125]
            self.label = self.label[:65*125]
        if data_split == 'valid':
            self.ref_img_path = self.ref_img_path[65*125:73*125]
            self.dist_img_path = self.dist_img_path[65*125:73*125]
            self.label = self.label[65*125:73*125]
        if data_split == 'test':
            self.ref_img_path = self.ref_img_path[73*125:]
            self.dist_img_path = self.dist_img_path[73*125:]
            self.label = self.label[73*125:]


    def __len__(self):
        return len(self.dist_img_path)

    def __getitem__(self, item):
        dist_img_path = self.dist_img_path[item]
        dist_img = Image.open(dist_img_path)
        dist_img = transforms.ToTensor()(dist_img)

        ref_img_path = self.ref_img_path[item]
        ref_img = Image.open(ref_img_path)
        ref_img = transforms.ToTensor()(ref_img)

        y = self.label[item]

        t = ref_img_path.split('/')[-1][1:3]

        return ref_img, dist_img, y, int(t)




if __name__ == "__main__":
    from torch.autograd import Variable

    image_dir = '/dataset/kadid10k/'
    label_file = 'dmos.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir, label_file=label_file)

    batch_size = 10  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for X1, X2, Y, mask in data_generator:
        X1 = Variable(X1.to(device))
        X2 = X2.to(device)
        Y = Y.to(device)
        import pdb;

        pdb.set_trace()
