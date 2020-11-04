import cv2
import numpy as np
import pandas as pd
import os
import torch

clean_names = lambda x: [i for i in x if i[0] != '.']

class Dataset():
    def __init__(self, image_dir, label_file, device):
        '''
        :param image_dir:
        :param label_file:
        :param device:
        :param label_id: 0,1,2 three version of scores
        '''
        self.image_paths = [os.path.join(image_dir,path) for path in sorted(clean_names(os.listdir(image_dir)))]
        self.label_file = label_file
        self.labels = self.getlabels()  # [i,j,k]
        self.N = len(self.image_paths)
        self.cur = 0
        self.device = device


    def getlabels(self):
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

    def _getdata(self, pair = True):
        '''
        :return: original image, distorted image, label, class of texture
        '''
        img = cv2.imread(self.image_paths[self.cur], 0)/255

        path = self.image_paths[self.cur].split('/')[-1]
        i, j = int(path[0]), int(path[2])   # class of texture, class of distortion
        label = self.labels[j, i]

        self.cur += 1
        if pair:
            H, _ = img.shape
            img1 = img[:H // 2, :]
            img2 = img[H // 2:, :]
            return torch.from_numpy(img1).double().to(self.device), torch.from_numpy(img2).double().to(self.device), torch.tensor(label).to(self.device), torch.tensor(int(i)).to(self.device)
        else:
            return torch.from_numpy(img).double().to(self.device), torch.tensor(label).to(self.device), torch.tensor(int(i)).to(self.device)

    def getdata(self, batchsize, augment = False):
        '''
        return a batch of data
        :param batchsize:
        :param augment: augment = True means using data generated with random seeds
        :return: image data (pairs), label of perceptual distance, mask of texture class
        '''
        X1,X2,Y,mask = [],[],[],[]
        for i in range(batchsize):
            if augment:
                img1, label, m = self._getdata(pair=False)
            else:
                img1, img2, label, m = self._getdata()
                X2.append(img2)

            X1.append(img1)
            Y.append(label)
            mask.append(m)
            if self.cur == self.N:
                self.cur = 0
                break

        # N, C=1, H, W
        X1 = torch.stack(X1)
        # N
        Y = torch.stack(Y)
        mask = torch.stack(mask)

        if augment:
            return X1.unsqueeze(1), Y, mask
        else:
            X2 = torch.stack(X2)
            return X1.unsqueeze(1), X2.unsqueeze(1), Y, mask

def test():
    image_dir = '../data/scenes_distorted'
    label_file = '../data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 1000
    X1, X2, Y, mask = dataset.getdata(batchsize, augment = False)

    import pdb;
    pdb.set_trace()

    image_dir = '../data/training_images'
    dataset = Dataset(image_dir, label_file, device)
    X1, Y, mask = dataset.getdata(batchsize, augment = True)

    import pdb;
    pdb.set_trace()

if __name__ == "__main__":
    test()
