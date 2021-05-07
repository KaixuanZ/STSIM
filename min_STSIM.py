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

npImg1 = cv2.imread("data/0001.tiff",0)/255
ref = npImg1

img1 = torch.from_numpy(npImg1).double().unsqueeze(0).unsqueeze(0)
img2 = torch.rand(img1.size())

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
for i in tqdm(range(501)):
    if i%10==0:
        # hist match
        tmp = img2.detach().cpu().squeeze(0).squeeze(0)
        tmp = tmp.numpy()
        tmp = steerable_hist_match(ref, tmp)
        #tmp = match_histograms(tmp, ref, multichannel=False)
        img2 = Variable(torch.from_numpy(tmp).double().unsqueeze(0).unsqueeze(0), requires_grad = True)
    stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2)) ** 2)
    optimizer = optim.Adam( [img2], lr=max( 0.01/(max(1,i/10)) ,0.002) )

    if stsim_loss < 0.001:
        break
    optimizer.zero_grad()
    stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2))**2)
    stsim_loss.backward()
    optimizer.step()
    print(stsim_loss)

    if i%100==0:
        res = img2.detach().cpu().squeeze(0).squeeze(0)
        res = res.numpy()

        res = steerable_hist_match(ref, res)
        cv2.imwrite('data/res_' + str(i).zfill(3) + '.png', res*255)
import pdb;
pdb.set_trace()