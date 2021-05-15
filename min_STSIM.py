from metrics.STSIM import *
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
from tqdm import tqdm
from hist_matching import steerable_hist_match
from skimage import exposure
from skimage.exposure import match_histograms

#npImg1 = cv2.imread("data/0001.tiff",0)/255
npImg1 = cv2.imread("data/fg.png",0)/255
ref = npImg1

img1 = torch.from_numpy(npImg1).double().unsqueeze(0).unsqueeze(0)
N,C,H,W = img1.shape
img2 = torch.rand([N,C,H*4,W*4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img1 = img1.double().to(device)
img2 = img2.double().to(device)


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)

m = Metric('SF', device)
#stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2)) ** 2)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)

#optimizer = optim.Adam([img2], lr=0.01)

i = 0
for i in tqdm(range(31)):
    if i%10==0:
        # hist match
        tmp = img2.detach().cpu().squeeze(0).squeeze(0)
        tmp = tmp.numpy()
        tmp = steerable_hist_match(ref, tmp)
        #tmp = match_histograms(tmp, ref, multichannel=False)
        img2 = Variable(torch.from_numpy(tmp).double().unsqueeze(0).unsqueeze(0), requires_grad = True)
    stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2)) ** 2)
    #optimizer = optim.Adam( [img2], lr=max( 0.01/(max(1,i/10)) ,0.005) )
    optimizer = optim.Adam( [img2], lr=0.01 )

    optimizer.zero_grad()
    stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2))**2)
    stsim_loss.backward()
    optimizer.step()
    print(stsim_loss)

    res = img2.detach().cpu().squeeze(0).squeeze(0)
    res = res.numpy()

    res = steerable_hist_match(ref, res)
    cv2.imwrite('data/res/res_' + str(i).zfill(3) + '.png', res*255)

    if i>0:
        delta = np.mean(np.abs(res-res_pre))*255
        print(i,delta)
    res_pre = res

import pdb;
pdb.set_trace()