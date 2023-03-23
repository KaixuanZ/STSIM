import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from metrics.STSIM import *
from tqdm import tqdm

device = 'cuda'

def test_STSIM(path1, path2, w_size=128, s_size=128):
    img_o = Image.open(path1)
    img_o = transforms.ToTensor()(img_o)
    img_o = img_o[:,:-5,:-3].to(device).unsqueeze(0)
    img_o = F.interpolate(img_o, 1024)

    img_d = Image.open(path2)
    img_d = transforms.ToTensor()(img_d)
    img_d = img_d[:,5:,3:].to(device).unsqueeze(0)
    img_d = F.interpolate(img_d, 1024)

    model = STSIM_M([82,10], mode=0, filter = 'SCF', device = device)
    model.load_state_dict(torch.load('weights/STSIM_01242022_SCF_global_mode0/epoch_0199.pt'))
    model.to(device).double()

    preds = []
    for i in range((img_o.shape[-2]-w_size)//s_size+1):
        for j in range((img_o.shape[-1]-w_size)//s_size+1):
           preds.append(model(img_o[:,:,i*s_size:i*s_size+w_size, j*s_size:j*s_size+w_size], img_d[:,:,i*s_size:i*s_size+w_size, j*s_size:j*s_size+w_size]).item())
    print(np.mean(preds))

w_size = [128,256,512,1024]
path1 = '/home/kaixuan/Downloads/woman_g.tiff'
step_size = 0.5
# step_size = 1
for w in tqdm(w_size):
    print('window size:',w)
    s = int(w*step_size)
    print('step size:',s)

    path2='/home/kaixuan/Downloads/womanJPEG.tiff'
    print('JPEG')
    test_STSIM(path1,path2,w, s)

    path2='/home/kaixuan/Downloads/womanMTC.tiff'
    print('MTC')
    test_STSIM(path1,path2,w, s)


