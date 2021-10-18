'''
try to model the repetitive pattern
'''

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from metrics.STSIM import *
import torch

def _GetPSD1D(psd2D, average=False):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    if average:
        # average all psd2D pixels with label 'r' for 0<=r<=wc
        # NOTE: this will miss power contributions in 'corners' r>wc
        psd1D = ndimage.mean(psd2D, r, index=np.arange(0, min(wc, hc)))
    else:
        # SUM all psd2D pixels with label 'r' for 0<=r<=wc
        # NOTE: this will miss power contributions in 'corners' r>wc
        psd1D = ndimage.sum(psd2D, r, index=np.arange(0, min(wc,hc)))

    return psd1D

def GetPSD1D(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_psd_2d = np.abs(f_shift) ** 2
    return _GetPSD1D(f_psd_2d)

def main(img_path1, img_path2, label, viz=True):
    img1 = cv2.imread(img_path1, 0).astype(float)
    img2 = cv2.imread(img_path2, 0).astype(float)

    fg = img1 - img2
    t = min(-fg.min(), fg.max())
    cv2.imwrite('tmp.png', (fg+t)*255/2/t)
    psd1d_fg = GetPSD1D(fg)
    Gaussian = np.random.normal(0, (fg ** 2).mean() ** 0.5, fg.shape)
    psd1d_Gaussian = GetPSD1D(Gaussian)
    if viz:
        k = 0
        plt.plot(range(k,psd1d_fg.size), psd1d_fg[k:], label=label)
        plt.plot(range(k,psd1d_Gaussian.size), psd1d_Gaussian[k:], label='Gaussian')
        plt.legend()
        plt.show()

def main1(img_path1, label, viz=True):
    fg = cv2.imread(img_path1, 0).astype(float)
    fg = fg - fg.mean()
    h,w = fg.shape
    fg = fg[h//8:-h//8, w//8:-w//8]
    #t = min(-fg.min(), fg.max())
    #cv2.imwrite('tmp.png', (fg+t)*255/2/t)
    psd1d_fg = GetPSD1D(fg)
    Gaussian = np.random.normal(0, (fg ** 2).mean() ** 0.5, fg.shape)
    psd1d_Gaussian = GetPSD1D(Gaussian)
    if viz:
        k = 0
        plt.plot(range(k,psd1d_fg.size), psd1d_fg[k:], label=label)
        plt.plot(range(k,psd1d_Gaussian.size), psd1d_Gaussian[k:], label='Gaussian')
        plt.legend()
        plt.show()

#fg = '/Data/DareDevil/synthesized_H&B/frame_00300.png'
#main1(fg, 'synthesized film grain H&B')

#img_o = '/Data/DareDevil/denoised/frame_00300.png'
#img_den = '/Data/DareDevil/original/frame_00300.png'
#main(img_o, img_den, 'original film grain')

#img_dec = '/Data/DareDevil/decoded/frame_00300.png'
#img_ren = '/Data/DareDevil/renoised_default/frame_00300.png'
#main(img_dec, img_ren, 'synthesized film grain')

#img_dec = '/Data/DareDevil/decoded/frame_00300.png'
#img_ren = '/Data/DareDevil/renoised_04/frame_00300.png'
#main(img_dec, img_ren, 'synthesized film grain')

#img_dec = '/Data/DareDevil/decoded/frame_00300.png'
#img_ren = '/Data/DareDevil/renoised_Daizong/frame300_averaged_binary_n64r4v0.tif'
#main(img_dec, img_ren, 'synthesized film grain n64r4v0')

img_o = '/Data/DareDevil/denoised/frame_00300.png'
img_den = '/Data/DareDevil/renoised_Daizong/frame300_denoised_averaged_binary_n96r4v0.tif'
main(img_o, img_den, 'synthesized film grain n96r4v0')


