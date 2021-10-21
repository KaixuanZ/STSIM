import argparse
import numpy as np
from utils.dataset import Dataset
from utils.parse_config import parse_config

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.stats

import os
from PIL import Image, ImageOps

h,w,size = 350,1150,128
def read_img(path, device, grayscale=True, crop=True):
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    if grayscale:
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
    if crop:
        img = img[:,:,h:h+size,w:w+size]
    return img.to(device).double()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test.cfg", help="path to data config file")
    parser.add_argument("--batch_size", type=int, default=4080, help="size of each image batch")
    opt = parser.parse_args()
    print(opt)

    config = parse_config(opt.config)
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read train config
    import json
    with open(config['train_config_path']) as f:
        train_config = json.load(f)
        print(train_config)

    input_dir = '/dataset/NetflixData/DareDevil_yuv_1920_1080_10bit/'
    path_o = os.path.join(input_dir ,'frame_original/frame_00300.png')
    path_den = os.path.join(input_dir ,'frame_denoised/frame_00300.png')
    path_dec = os.path.join(input_dir ,'frame_decoded/frame_00300.png')
    path_ren_00 = os.path.join(input_dir ,'frame_renoised/frame_00300.png')
    path_ren_01 = os.path.join(input_dir ,'frame_renoised_01/frame_00300.png')
    path_ren_02 = os.path.join(input_dir ,'frame_renoised_02/frame_00300.png')
    path_ren_03 = os.path.join(input_dir ,'frame_renoised_03/frame_00300.png')
    path_ren_04 = os.path.join(input_dir ,'frame_renoised_04/frame_00300.png')

    img_o = read_img(path_o, device)
    img_den = read_img(path_den, device)
    img_dec = read_img(path_dec, device)
    img_ren_00 = read_img(path_ren_00, device)
    img_ren_01 = read_img(path_ren_01, device)
    img_ren_02 = read_img(path_ren_02, device)
    img_ren_03 = read_img(path_ren_03, device)
    img_ren_04 = read_img(path_ren_04, device)

    # test with different model
    from metrics.STSIM import *

    filter = train_config['filter']
    m_g = Metric(filter, device=device)

    model = STSIM_M(train_config['dim'], mode=int(train_config['mode']), filter = filter, device = device)
    model.load_state_dict(torch.load(train_config['weights_path']))
    model.to(device).double()
    pred0 = model(img_o - img_den, img_ren_00 - img_dec)
    pred1 = model(img_o - img_den, img_ren_01 - img_dec)
    pred2 = model(img_o - img_den, img_ren_02 - img_dec)
    pred3 = model(img_o - img_den, img_ren_03 - img_dec)
    pred4 = model(img_o - img_den, img_ren_04 - img_dec)
    print(pred0, pred1, pred2, pred3, pred4)
    import pdb;pdb.set_trace()