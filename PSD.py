'''
try to model the repetitive pattern
'''

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from metrics.STSIM import *
import torch
from scipy.stats import wasserstein_distance

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
    if img.shape[0]!=img.shape[1]:
        # equal height and width, o.w. the radius sum/average doesn't make sense
        size = min(img.shape)
        img = img[:size,:size]
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_psd_2d = np.abs(f_shift) ** 2
    return _GetPSD1D(f_psd_2d)

def main(img_path1, img_path2, label, psd1d_o, viz=True, prefix=""):
    img1 = cv2.imread(img_path1, 0).astype(float)
    img2 = cv2.imread(img_path2, 0).astype(float)
    h, w = 350, 1150
    size = 128

    fg = img1 - img2
    #fg = fg[h:h + size, w:w + size]
    t = min(-fg.min(), fg.max())
    cv2.imwrite('_'.join([prefix,'tmp.png']), (fg+t)*255/2/t)

    fg = fg-fg.mean()
    psd1d_fg = GetPSD1D(fg)
    # Gaussian = np.random.normal(0, (fg ** 2).mean() ** 0.5, fg.shape)
    # psd1d_Gaussian = GetPSD1D(Gaussian)
    if viz:
        k = 0
        plt.plot(range(k,psd1d_fg.size), psd1d_fg[k:], label=label)
        plt.plot(range(k,psd1d_o.size), psd1d_o[k:], label='original film grain')
        # plt.plot(range(k,psd1d_Gaussian.size), psd1d_Gaussian[k:], label='Gaussian')
        plt.legend()
        #plt.show()
        plt.savefig('_'.join([prefix,'tmp1.png']))
        plt.close()

        tmp = medfilt(psd1d_fg, 3)
        tmp = (psd1d_fg - tmp) / tmp
        tmp[0] = 0
        plt.plot(range(k, tmp.size), tmp[k:], label=label)
        plt.savefig('_'.join([prefix,'tmp2.png']))
        plt.close()
    return psd1d_fg

def main1(img_path1, label, psd1d_o, viz=True):
    # for H&B synthesis
    fg = cv2.imread(img_path1, 0).astype(float)
    fg = fg/255*28-14

    h,w = fg.shape
    #fg = fg[h//4:-h//4, w//4:-w//4]
    fg = fg[:1080, :1080]
    #t = min(-fg.min(), fg.max())
    #cv2.imwrite('tmp.png', (fg+t)*255/2/t)
    psd1d_fg = GetPSD1D(fg)
    psd1d_fg[0] = 0
    # Gaussian = np.random.normal(0, (fg ** 2).mean() ** 0.5, fg.shape)
    # psd1d_Gaussian = GetPSD1D(Gaussian)
    if viz:
        k = 0
        plt.plot(range(k,psd1d_fg.size), psd1d_fg[k:], label=label)
        plt.plot(range(k,psd1d_o.size), psd1d_o[k:], label='original film grain')
        # plt.plot(range(k,psd1d_Gaussian.size), psd1d_Gaussian[k:], label='Gaussian')
        plt.legend()
        plt.savefig('tmp1.png')
        plt.close()

        tmp = medfilt(psd1d_fg, 3)
        tmp = (psd1d_fg - tmp) / tmp
        tmp[0] = 0
        plt.plot(range(k, tmp.size), tmp[k:], label=label)
        plt.savefig('tmp2.png')
        plt.close()
    return psd1d_fg

def Wasserstein_dis(psd1, psd2, viz=True, figname='tmp1.png', normalize=True, STSIM_Mf=None):
    if normalize:
        w_dis = wasserstein_distance(psd1 / np.sum(psd1), psd2 / np.sum(psd2))
    else:
        w_dis = wasserstein_distance(psd1, psd2)

    if viz:
        k = 0
        plt.plot(range(k,psd1.size), psd1[k:], label='original')
        plt.plot(range(k,psd2.size), psd2[k:], label='synthesized')
        plt.legend()
        if STSIM_Mf is None:
            plt.title('wasserstein distance={:0.4f}'.format(w_dis))
        else:
            plt.title('w_dist={:0.4f}, STSIM-Mf={:0.4f}'.format(w_dis,STSIM_Mf))
        #plt.show()
        plt.savefig(figname)
        plt.close()
    return w_dis


img_o = '/Data/DareDevil/denoised/frame_00300.png'
img_den = '/Data/DareDevil/original/frame_00300.png'
psd1d_o = main(img_o, img_den, 'original film grain', psd1d_o=None, viz=False)

#fg = '/Data/DareDevil/synthesized_H&B/frame_00300.png'
# fg = '/Data/DareDevil/synthesized_H&B/fg_synthesized_scaled.png'
# psd1d_syn_HB = main1(fg, 'synthesized film grain H&B', psd1d_o)

# print(Wasserstein_dis(psd1d_o, psd1d_syn_HB, figname='tmp6.png', STSIM_Mf=None))


# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_default/frame_00300.png'
# psd1d_syn0 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn0')
#
# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_01/frame_00300.png'
# psd1d_syn1 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn1')
#
# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_02/frame_00300.png'
# psd1d_syn2 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn2')
#
# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_03/frame_00300.png'
# psd1d_syn3 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn3')
#
# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_04/frame_00300.png'
# psd1d_syn4 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn4')

img_dec = '/Data/DareDevil/decoded/frame_00300.png'
img_ren = '/Data/DareDevil/renoised_default_ps64/frame_00300.png'
psd1d_syn5 = main(img_dec, img_ren, 'synthesized film grain', psd1d_o=psd1d_o, prefix='tmp/syn5')

# print(Wasserstein_dis(psd1d_o, psd1d_syn0, figname='tmp1.png', STSIM_Mf=0.1678))
# print(Wasserstein_dis(psd1d_o, psd1d_syn1, figname='tmp2.png', STSIM_Mf=0.2503))
# print(Wasserstein_dis(psd1d_o, psd1d_syn2, figname='tmp3.png', STSIM_Mf=0.2321))
# print(Wasserstein_dis(psd1d_o, psd1d_syn3, figname='tmp4.png', STSIM_Mf=0.2319))
# print(Wasserstein_dis(psd1d_o, psd1d_syn4, figname='tmp5.png', STSIM_Mf=0.1810))
#
# img_dec = '/Data/DareDevil/decoded/frame_00300.png'
# img_ren = '/Data/DareDevil/renoised_Daizong/frame300_averaged_binary_n64r4v0.tif'
# psd1d_syn2 = main(img_dec, img_ren, 'synthesized film grain n64r4v0')
#
# img_o = '/Data/DareDevil/denoised/frame_00300.png'
# img_den = '/Data/DareDevil/renoised_Daizong/frame300_denoised_averaged_binary_n96r4v0.tif'
# psd1d_syn3 = main(img_o, img_den, 'synthesized film grain n96r4v0')
#
