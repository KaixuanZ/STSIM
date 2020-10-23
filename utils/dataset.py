import cv2
import pandas as pd
import os
import torch



class Dataset():
    def __init__(self, image_dir, label_file, device):
        self.image_paths = [os.path.join(image_dir,path) for path in sorted(os.listdir(image_dir))]
        self.labels = self.getlabels(label_file)
        self.N = len(self.image_paths)
        self.cur = 0
        self.device = device


    def getlabels(self, label_file):
        '''
        :param label_file:
        :return: [10 rows, 10 columns], [i-th distortion, j-th texture], last row is the original image (9 distortions + 1 original)
        '''
        df = pd.read_excel(label_file, header=None)
        #return df.iloc[12:21,:10].to_numpy()
        return df.iloc[:9,:10].to_numpy()

    def _getdata(self):
        '''
        :return: original image, distorted image, label
        '''
        img = cv2.imread(self.image_paths[self.cur], 0)/255
        H,_ = img.shape
        img1 = img[:H//2, :]
        img2 = img[H//2:, :]

        path = self.image_paths[self.cur].split('/')[-1]
        i, j = int(path[0]), int(path[2])   # class of texture, class of distortion
        label = self.labels[j, i]

        self.cur += 1
        return torch.from_numpy(img1).double().to(self.device), torch.from_numpy(img2).double().to(self.device), torch.tensor(label).to(self.device)

    def getdata(self, batchsize):
        # return a batch of data
        X1,X2,Y = [],[],[]
        for i in range(batchsize):
            if self.cur == self.N:
                self.cur = 0
                break
            img1, img2, label = self._getdata()
            X1.append(img1)
            X2.append(img2)
            Y.append(label)

        # N, C=1, H, W
        X1 = torch.stack(X1)
        X2 = torch.stack(X2)

        # N
        Y = torch.stack(Y)

        return X1.unsqueeze(1), X2.unsqueeze(1), Y

def test():
    image_dir = '../data/scenes_distorted'
    label_file = '../data/huib_analysis_data_across_distortions.xlsx'
    device = torch.device('cuda:0')
    dataset = Dataset(image_dir, label_file, device)

    batchsize = 32
    X1, X2, Y = dataset.getdata(batchsize)


    import pdb;
    pdb.set_trace()

if __name__ == "__main__":
    test()