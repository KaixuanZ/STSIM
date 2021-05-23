from metrics.STSIM import *
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import os
from tqdm import tqdm
from hist_matching import steerable_hist_match
from skimage import exposure
from skimage.exposure import match_histograms

def min_STSIM(features, params, size=(1,1,128,128), output_dir = None):
    img = torch.rand(size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Metric('SF', device)

    img = img.double().to(device)

    features = Variable( features,  requires_grad=False)
    img = Variable( img, requires_grad = True)

    for i in tqdm(range(21)):
        #if i%10==0:
        if i%5==0:
            # hist match
            tmp = img.detach().cpu().squeeze(0).squeeze(0)
            tmp = tmp.numpy()
            tmp = steerable_hist_match(params, tmp)
            img = Variable(torch.from_numpy(tmp).double().unsqueeze(0).unsqueeze(0), requires_grad = True)

        optimizer = optim.Adam( [img], lr=0.5/(max(1,i//5)) )

        optimizer.zero_grad()
        stsim_loss = torch.sum((features - m.STSIM(img))**2) + torch.sum(img**2)
        stsim_loss.backward()
        optimizer.step()
        #print(stsim_loss)

        res = img.detach().cpu().squeeze(0).squeeze(0)
        res = res.numpy()
        #noise = np.random.randn(*res.shape) * res.std() * 0.2
        #res = res + noise
        res = steerable_hist_match(params, res)

        if output_dir is not None:
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            cv2.imwrite(os.path.join(output_dir, str(i).zfill(3) + '.png'), (res+14)*255/44)

        '''
        if i>0:
            delta = np.mean(np.abs(res-res_pre))*255
            print(i,delta)
        res_pre = res
        '''
    return res

if __name__ == '__main__':
    #ref = cv2.imread("data/0001.tiff",0)/255
    #ref = cv2.imread("data/fg.png",0)/255
    #min_STSIM(ref, 'data/res')

    img_o = cv2.imread('data/original.png',0).astype(float)
    img_den = cv2.imread('data/denoised.png',0).astype(float)

    size = 128
    h,w = 400,700
    fg = img_o - img_den

    fg_syn = min_STSIM(fg[h:h+size,w:w+size])
    import pdb;pdb.set_trace()