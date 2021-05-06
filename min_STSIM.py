from metrics.STSIM import *
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("data/fg.png",0)
npImg1 = npImg1.reshape(npImg1.shape[0],npImg1.shape[1],1)

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img1 = img1.double().to(device)
img2 = img2.double().to(device)


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)

m = Metric('SF', device)
# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2))**2)

optimizer = optim.Adam([img2], lr=0.01)

i = 0
while stsim_loss > 0.001 and i<500:
    i+=1
    optimizer.zero_grad()
    stsim_loss = torch.sum((m.STSIM(img1) - m.STSIM(img2))**2)
    print(stsim_loss)
    stsim_loss.backward()
    optimizer.step()

tmp = img2.detach().cpu().squeeze(0).squeeze(0)
tmp = tmp.numpy()
tmp = tmp-tmp.mean()
tmp = tmp/tmp.std()
tmp = tmp*npImg1.std()
tmp = tmp+npImg1.mean()
cv2.imwrite('data/tmp1.png',tmp)

tmp = tmp-npImg1.mean()
tmp = tmp/(tmp.max() - tmp.min())
tmp = tmp*(npImg1.max() - npImg1.min())
tmp = tmp+npImg1.mean()
cv2.imwrite('data/tmp2.png',tmp)
import pdb;
pdb.set_trace()