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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def RGB2YUV(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv

def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    return rgb

def expand(fg):

    H, W = fg.shape
    k = 4
    edge = 4
    print(H, W) # 128 * 128
    #res = np.zeros([H*k,W*k])
    res = np.zeros([H * 10, W * 10])

    for i in range(10*k):
        for j in range(10*k):
            ii, jj = int(np.random.randint(H-2*edge-H//k)), int(np.random.randint(W-2*edge-W//k))

            src = fg[edge + ii:edge + H//k +ii, edge + jj:edge + W//k +jj]

            src = cv2.flip(src, np.random.randint(2))

            # tmp = np.random.randint(4)
            # if tmp == 1:
            #     src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
            # elif tmp == 2:
            #     src = cv2.rotate(src, cv2.ROTATE_180)
            # elif tmp == 3:
            #     src = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

            #        res[i*H//4:(i+1)*H//4,j*W//4:(j+1)*W//4] = src
            # res[i*H//2:(i+1)*H//2,j*W//2:(j+1)*W//2] = src
            # tmp = np.random.randint(4) - 1
            # if tmp <2:
            #     src = cv2.flip(src, tmp)
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
        pyr = pt.pyramids.SteerablePyramidSpace(fg, height=3, order=0)  # 3+2 scales, 4 orientations
        params = {}
        print('estimating subband parameters')
        params['original'] = gennorm.fit(fg.ravel())
        for key in tqdm(pyr.pyr_coeffs):
            params[key] = gennorm.fit(pyr.pyr_coeffs[key].ravel())
    else:
        # subband params
        pyr = pt.pyramids.SteerablePyramidSpace(fg, height=3, order=0)  # 3+2 scales, 4 orientations
        params = {}
        print('estimating subband parameters')
        params['original'] = gennorm.fit(fg[mask].ravel())
        for key in tqdm(pyr.pyr_coeffs):
            k = mask.shape[-1]//pyr.pyr_coeffs[key].shape[-1]
            mask1 = mask[::k,::k]
            params[key] = gennorm.fit(pyr.pyr_coeffs[key][mask1].ravel())
    return params

def fg_synthesize(fg, size, iter, output_dir):
    m = Metric('SF', device)

    # STSIM features based on mask
    print('extracting STSIM features')
    feature_fg = STSIM_features(fg, m)

    print('computing subbands statistics')
    params = pyr_params(fg)

    # synthesize a patch of film grain (size * size)
    print('synthesizing film grain noise')
    fg_syn = min_STSIM(feature_fg, params, output_dir=output_dir, size=(1, 1, 2 * size, 2 * size), iter=iter)

    return fg_syn

if __name__ == '__main__':
    original = 'data/DareDevil/original/frame_00300.png'
    denoised = 'data/DareDevil/denoised/frame_00300.png'
    output_dir = 'data/res'
    h, w = 350, 1150
    size = 128

    color = 0
    iter = 0
    if color==1:
        img_o, img_den, fg = data_loader(original, denoised, color)
        fg_patch = fg[h:h + size, w:w + size]
        fg_yuv = RGB2YUV(fg[:,:,::-1])
        res_yuv = []
        fg_patch_yuv = fg_yuv[h:h + size, w:w + size]
        res_yuv = [fg_synthesize(fg_patch_yuv[:,:,c], size, iter, output_dir) for c in range(3)]
        res_yuv = cv2.merge(res_yuv)
        res = YUV2RGB(res_yuv)
        res = res[:,:,::-1]

    elif color==0:
        img_o, img_den, fg = data_loader(original, denoised, color)

        fg_patch = fg[h:h+size, w:w+size]

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
        t1 = time.time()

        res = fg_synthesize(fg_patch, size, iter, output_dir)

        t2 = time.time()
        print('time for synthesize:', t2-t1)
        # a larger image by concatenating 32 by 32 patch
        res = expand(res)
        #res = expand(fg[450:450+128,700:700+128])
        res = res[:img_o.shape[0], :img_o.shape[1]]

        #cv2.imwrite('tmp1.png', (fg - fg.min())/(fg.max()-fg.min())*255)
        #cv2.imwrite('tmp2.png', (res - fg.min())/(fg.max()-fg.min())*255)

    #DareDevil
    import pdb;

    pdb.set_trace()
    cv2.imwrite(os.path.join(output_dir,'fg_patch.png'), (fg_patch - fg.min()) / (-fg.min() * 2) * 255)
    cv2.imwrite(os.path.join(output_dir,'fg_frame.png'), (fg - fg.min())/(-fg.min()*2)*255)
    cv2.imwrite(os.path.join(output_dir,'fg_synthesized.png'), (res - fg.min())/(-fg.min()*2)*255)


    # mask of flat region
    edges = cv2.Canny(img_den.astype(np.uint8), 50, 100)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = ~mask.astype(bool)

    # look up table
    # LUT = {}
    # for value in set(img_den[mask].ravel()):
    #     LUT[value] = fg[(img_den == value) & mask].std()
    for value in set(img_den[mask].ravel()):
        res[img_den==value] = res[img_den==value]*fg[(img_den==value)&mask].std()/res[(img_den==value)&mask].std()


    cv2.imwrite(os.path.join(output_dir,'fg_synthesized_scaled.png'), (res - fg.min())/(-fg.min()*2)*255)


    import pdb;pdb.set_trace()

