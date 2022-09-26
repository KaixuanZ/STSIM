from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, dist, ref = 'original', format = '.png'):
        self.dist_dir = os.path.join(data_dir, dist)
        self.ref_dir = os.path.join(data_dir, ref)
        self.label_file = os.path.join(data_dir, label_file)
        self.labels = self._getlabels()
        self.format = format

        clean_names = lambda x: [i for i in x if i[0] != '.']
        self.dist_img_paths = [os.path.join(self.dist_dir, img) for img in clean_names(os.listdir(self.dist_dir))]
        #self.dist_img_paths = clean_names(self.dist_img_paths)

    def __len__(self):
        return len(self.dist_img_paths)

    def __getitem__(self, item):
        dist_img_path = self.dist_img_paths[item]
        dist_img = Image.open(dist_img_path)
        #print(dist_img_path)
        dist_img = transforms.ToTensor()(dist_img)

        tmp = dist_img_path.split('/')[-1]  #file name
        tmp = tmp.split('.')[0].split('_')
        t, d = tmp[0], tmp[2]
        y = self.labels[int(d), int(0)]

        ref_img_path = os.path.join(self.ref_dir, t + self.format)
        ref_img = Image.open(ref_img_path)
        ref_img = transforms.ToTensor()(ref_img)

        return ref_img, dist_img, y, int(t)


    def _getlabels(self):
        '''
        :param label_file:
        :param id: the id of label matrix
        :return: [i-th distortion, j-th texture, k-th version of label]
        '''
        df = pd.read_excel(self.label_file, header=None)
        label3 = df.iloc[:15,:1].to_numpy().astype(np.double)
        return label3

if __name__ == "__main__":
    from torch.autograd import Variable

    image_dir = '/dataset/databases_jyl/concatenated img/'
    label_file = 'label.xlsx'
    dist_img_folder = 'test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir, label_file=label_file, dist=dist_img_folder)

    batch_size = len(dataset)  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for X1, X2, Y, mask in data_generator:
        X1 = Variable(X1.to(device))
        X2 = X2.to(device)
        Y = Y.to(device)
        import pdb;

        pdb.set_trace()
