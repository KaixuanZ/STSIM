from PIL import Image

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

        # with open('../data/MacroTextures3K.json', 'w') as json_file:
        #     json.dump(self.dist_img_paths, json_file)

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
            data = torch.zeros(3*3,C,256,256)
            for c in range(C):
                for i in range(3):
                    for j in range(3):
                        data[i*3+j,c] = dist_img[c,(H-256)*i//2:(H-256)*i//2+256,(W-256)*j//2:(W-256)*j//2+256]
        if self.data_split == 'test':
            data = dist_img
        # import pdb;
        # pdb.set_trace()
        return data


if __name__ == "__main__":

    image_dir = '/dataset/MacroTextures3K/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(data_dir=image_dir, data_split='test')

    batch_size = 1000  # the actually batchsize <= total images in dataset
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    import pdb;

    pdb.set_trace()
    res = []
    for X in tqdm(data_generator):
        X = X.to(device)
        res.append(X)
    torch.save(torch.cat(res),'../data/MacroSyn30000_SCF.pt')
