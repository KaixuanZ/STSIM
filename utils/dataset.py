from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, dist_folder, ref_folder = 'original', format = '.png', maskshape=None):
        self.dist_dir = os.path.join(data_dir, dist_folder)
        self.ref_dir = os.path.join(data_dir, ref_folder)
        self.label_file = os.path.join(data_dir, label_file)
        self.labels = self._getlabels()
        self.format = format

        clean_names = lambda x: [i for i in x if i[0] != '.']
        self.dist_img_paths = [os.path.join(self.dist_dir, img) for img in os.listdir(self.dist_dir)]
        self.dist_img_paths = sorted(clean_names(self.dist_img_paths))
        self.mask = None

        if maskshape is not None:
            self.mask = torch.zeros([1,128,128])
            for i in range(self.mask.shape[1]):
                for j in range(self.mask.shape[2]):
                    self.mask[0,i,j] = (-1)**(i+j)
        else:
            self.mask = None

    def __len__(self):
        return len(self.dist_img_paths)

    def __getitem__(self, item):
        dist_img_path = self.dist_img_paths[item]
        dist_img = Image.open(dist_img_path)
        dist_img = transforms.ToTensor()(dist_img)

        tmp = dist_img_path.split('/')[-1]  #file name
        tmp = tmp.split('.')[0]
        tmp = tmp.split('_')
        t, d = tmp[0], tmp[1] # texture, distortion, n-th data

        y = self.labels[int(d), int(t)]
        ref_img_path = os.path.join(self.ref_dir, t + self.format)
        ref_img = Image.open(ref_img_path)
        ref_img = transforms.ToTensor()(ref_img)

        if self.mask is not None:
            return ref_img*self.mask, dist_img*self.mask, y, int(t)
        else:
            return ref_img, dist_img, y, int(t)


    def _getlabels(self):
        '''
        :param label_file:
        :param id: the id of label matrix
        :return: [i-th distortion, j-th texture, k-th version of label]
        '''
        df = pd.read_excel(self.label_file, header=None)
        #label1 = df.iloc[:9,:10].to_numpy().astype(np.double)
        #label2 = df.iloc[12:21,:10].to_numpy().astype(np.double)
        label3 = df.iloc[31:40,:10].to_numpy().astype(np.double)
        #return np.stack([label1,label2,label3], axis=2)
        return label3

if __name__ == "__main__":
    from torch.autograd import Variable

    image_dir = '/dataset/jana2012/'
    label_file = 'label.xlsx'
    dist_img_folder = 'train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir, label_file=label_file, dist_folder=dist_img_folder)

    batch_size = 1000  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for X1, X2, Y, mask in data_generator:
        X1 = Variable(X1.to(device))
        X2 = X2.to(device)
        Y = Y.to(device)
        import pdb;

        pdb.set_trace()
