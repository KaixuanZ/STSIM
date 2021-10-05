'''
try to model the repetitive pattern
'''

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.sum(psd2D, r, index=np.arange(0, min(wc,hc)))

    return psd1D

def GetPSD1D(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_psd_2d = np.abs(f_shift) ** 2
    return _GetPSD1D(f_psd_2d)

def try_corr(fg_syn):
    x = range(1,100)
    corrs = [np.corrcoef(fg_syn[:-i].flatten(), fg_syn[i:].flatten())[0, 1] for i in tqdm(x)]
    plt.plot(x, corrs)
    plt.show()
    return corrs


def try_psd1d(fg_syn):
    psd = GetPSD1D(fg_syn)
    plt.plot(range(psd.size), psd)
    plt.show()
    return psd

def try_psd2d(fg_syn):
    f = np.fft.fft2(fg_syn)
    f_shift = np.fft.fftshift(f)
    f_psd_2d = np.abs(f_shift) ** 2
    #f_psd_2d = np.log(f_psd_2d)

    x,y = range(f_psd_2d.shape[1]), range(f_psd_2d.shape[0])
    x,y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, fg_syn)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    return f_psd_2d


#img_dec = 'data/DareDevil/denoised/frame_00300.png'
img_dec = 'data/DareDevil/decoded_av1/frame_00300.png'
#img_ren = 'data/DareDevil/original/frame_00300.png'
img_ren = 'data/DareDevil/renoised_av1/frame_00300.png'

img_dec = cv2.imread(img_dec, 0).astype(float)
img_ren = cv2.imread(img_ren, 0).astype(float)

fg_syn = img_ren - img_dec

#cv2.imwrite('tmp.png', (fg_syn+10)*12)

corrs = try_corr(fg_syn)
#corrs1 = try_corr(fg_syn[:,100])
#corrs2 = try_corr(fg_syn[:,200])
#corrs3 = try_corr(fg_syn[:,300])
#psd1d = try_psd1d(fg_syn[:,420:1920-420])
#psd2d = try_psd2d(fg_syn)
import pdb;pdb.set_trace()