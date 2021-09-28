import cv2
import os
import numpy as np
from min_STSIM import min_STSIM
from tqdm import tqdm
from metrics.STSIM import *
import torch
from scipy.stats import gennorm
import pyrtools as pt
import time

def expand(fg):

    H, W = fg.shape

    print(H, W) # 128 * 128
    # res = np.zeros([H*k,W*k])
    res = np.zeros([H * 10, W * 16])

    k = 4
    edge = 4

    for i in range(10*k):
        for j in range(16*k):
            ii, jj = int(np.random.randint(H-2*edge-H//k)), int(np.random.randint(W-2*edge-W//k))

            src = fg[edge + ii:edge + H//k +ii, edge + jj:edge + W//k +jj]

            src = cv2.flip(src, np.random.randint(2))
            tmp = np.random.randint(4)
            if tmp == 1:
                src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
            elif tmp == 2:
                src = cv2.rotate(src, cv2.ROTATE_180)
            elif tmp == 3:
                src = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #        res[i*H//4:(i+1)*H//4,j*W//4:(j+1)*W//4] = src
            # res[i*H//2:(i+1)*H//2,j*W//2:(j+1)*W//2] = src
            tmp = np.random.randint(4) - 1
            if tmp <2:
                src = cv2.flip(src, tmp)
            res[i * H // k:(i + 1) * H // k, j * W // k:(j + 1) * W // k] = src
    return res

def data_loader(original, denoised, color=0):
    img_o = cv2.imread(original, color).astype(float)
    img_den = cv2.imread(denoised, color).astype(float)

    # img_o = cv2.imread('data/original_DareDevil.png',0).astype(float)
    # img_den = cv2.imread('data/denoised_DareDevil.png',0).astype(float)

    # tax
    # img_o = img_o[10:-10]
    # img_den = img_den[10:-10]

    # law
    # img_o = img_o[102:-102]
    # img_den = img_den[102:-102]

    # tax
    # h,w = 920 , 1750

    # DareDevil
    # h,w = 450,700


    # law
    # h,w = 0,600

    fg = img_o - img_den
    return img_o, img_den, fg

def STSIM_features(fg, metric, mask=None):
    if mask is None:
        fg_patch_torch = torch.from_numpy(fg).double().unsqueeze(0).unsqueeze(0).to(device)
        feature_fg = metric.STSIM(fg_patch_torch)
    else:
        mask_torch = torch.from_numpy(mask).double().unsqueeze(0).unsqueeze(0).to(device)
        fg_torch = torch.from_numpy(fg).double().unsqueeze(0).unsqueeze(0).to(device)
        feature_fg = metric.STSIM(fg_torch, mask_torch)
    return feature_fg

def pyr_params(fg, mask=None):
    if mask is None:
        # subband params
        pyr = pt.pyramids.SteerablePyramidSpace(fg, height=3, order=3)  # 3+2 scales, 3 orientations
        params = {}
        print('estimating subband parameters')
        params['original'] = gennorm.fit(fg.ravel())
        for key in tqdm(pyr.pyr_coeffs):
            params[key] = gennorm.fit(pyr.pyr_coeffs[key].ravel())
    else:
        # subband params
        pyr = pt.pyramids.SteerablePyramidSpace(fg, height=3, order=3)  # 3+2 scales, 3 orientations
        params = {}
        print('estimating subband parameters')
        params['original'] = gennorm.fit(fg[mask].ravel())
        for key in tqdm(pyr.pyr_coeffs):
            k = mask.shape[-1]//pyr.pyr_coeffs[key].shape[-1]
            mask1 = mask[::k,::k]
            params[key] = gennorm.fit(pyr.pyr_coeffs[key][mask1].ravel())
    return params

if __name__ == '__main__':
    original = 'data/DareDevil/original/frame_00300.png'
    denoised = 'data/DareDevil/denoised/frame_00300.png'
    output_dir = 'data/res'
    h, w = 350, 1150
    size = 128

    img_o, img_den, fg = data_loader(original, denoised)
    fg_patch = fg[h:h+size, w:w+size]
    #import pdb;pdb.set_trace()
    '''
    # mask of flat region
    edges = cv2.Canny(img_den.astype(np.uint8), 50, 100)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = ~mask.astype(bool)
    '''

    '''
    from matplotlib import pyplot as plt
    plt.imshow(mask, cmap='gray')
    plt.show()
    import pdb;pdb.set_trace()
    '''

    # look up table
    #LUT = {}
    #for value in set(img_den[mask].ravel()):
    #    LUT[value] = fg[(img_den==value)&mask].std()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Metric('SF', device)

    # STSIM features based on mask
    print('extracting STSIM features')
    feature_fg = STSIM_features(fg_patch, m)

    print('computing subbands statistics')
    params = pyr_params(fg_patch)

    t1 = time.time()
    # synthesize a patch of film grain (size * size)
    print('synthesizing film grain noise')
    fg_syn = min_STSIM(feature_fg, params, output_dir=output_dir, size=(1,1,2*size,2*size))
    res = fg_syn

    t2 = time.time()
    print('time for synthesize:', t2-t1)
    # a larger image by concatenating 32 by 32 patch
    #res = expand(fg_syn)
    #res = expand(fg[450:450+128,700:700+128])
    #res = res[:img_o.shape[0], :img_o.shape[1]]

    #cv2.imwrite('tmp1.png', (fg - fg.min())/(fg.max()-fg.min())*255)
    #cv2.imwrite('tmp2.png', (res - fg.min())/(fg.max()-fg.min())*255)

    #DareDevil
    cv2.imwrite(os.path.join(output_dir,'fg_patch.png'), (fg_patch - fg.min()) / (-fg.min() * 2) * 255)
    cv2.imwrite(os.path.join(output_dir,'fg_frame.png'), (fg - fg.min())/(-fg.min()*2)*255)
    cv2.imwrite(os.path.join(output_dir,'fg_synthesized.png'), (res - fg.min())/(-fg.min()*2)*255)

    '''
    # look up table
    for value in set(img_den[mask].ravel()):
        res[img_den==value] = res[img_den==value]*fg[(img_den==value)&mask].std()/res[(img_den==value)&mask].std()

    #cv2.imwrite('tmp3.png', (res - fg.min())/(fg.max()-fg.min())*255)
    cv2.imwrite('tmp3.png', (res - fg.min())/(-fg.min()*2)*255)
    '''

    import pdb;pdb.set_trace()

