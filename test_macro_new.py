from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import os
import json
from tqdm import tqdm
# from metrics.STSIM import *
from metrics.STSIM_VGG import *

clean_names = lambda x: [i for i in x if i[0] != '.']

def extract_feats(model, data_folder = '/home/kaixuan/Downloads/texture10k', device='cuda'):
    imgfiles = clean_names(sorted(os.listdir(data_folder)))

    feats = []
    for f in tqdm(imgfiles[:10]):
        imgpath = os.path.join(data_folder, f)
        img = Image.open(imgpath)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        feat = model.forward_once(img)

        feats.append(feat)

    return torch.cat(feats), imgfiles



if __name__ == '__main__':
    data_folder = '/home/kaixuan/Downloads/texture10k'
    output_dir = 'res'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # model
    device='cuda'
    model = STSIM_VGG([5900,10],grayscale=False).to(device)
    model.load_state_dict(torch.load('weights/STSIM_macro_VGG_05222022/epoch_0100.pt'), strict=False)
    model.to(device).double()

    # return STSIM-VGG featues and the idx-filename
    feats, img_fnames = extract_feats(model, data_folder)

    # save results
    path_pt = os.path.join(output_dir, 'feats.pt')
    path_LUT = os.path.join(output_dir, 'LUT.json')
    torch.save(feats, path_pt)
    with open(path_LUT, 'w') as json_file:
        json.dump(img_fnames, json_file)

    # read results
    feats1 = torch.load(path_pt)
    with open(path_LUT) as f:
        img_fnames1 = json.load(f)
    # you can manually check the results loaded is the same as above
    print(feats.shape)
    print(feats1.shape)

    # compute distance
    dists = model(feats, feats1)    # this should be all 0
    print(dists)

    # compute distance in feature space between image[0] and all the images
    # the smaller the similar, the first element in distance should be 0
    dists = model(feats[0].repeat([10,1]), feats)   # use feats[i] to select different reference image
    print(dists)

    # import pdb;pdb.set_trace()