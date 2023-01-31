# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class STSIM_VGG(torch.nn.Module):
    def __init__(self, dim, grayscale=True):
        super(STSIM_VGG, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))
        self.C = 1e-10

        for param in self.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(dim[0], dim[1])
        if grayscale:
            self.chns = [1,64,128,256,512,512]
        else:
            self.chns = [3,64,128,256,512,512]
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h

        coeffs = [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        f = []
        # single subband statistics
        for c in coeffs:
            mean = torch.mean(c, dim=[2, 3])
            var = torch.var(c, dim=[2, 3])
            f.append(mean)
            f.append(var)

            c = c - mean.unsqueeze(-1).unsqueeze(-1)
            f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[2, 3]) / (var + self.C))
            f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[2, 3]) / (var + self.C))
        # import pdb;pdb.set_trace()
        return torch.cat(f, dim=-1)  # [BatchSize, FeatureSize]

    def forward(self, x, y, require_grad=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        pred = self.linear(torch.abs(feats0 - feats1))  # [N, dim]
        pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod
        # import pdb;pdb.set_trace()
        return torch.sqrt(pred - torch.sum(self.linear.bias**2))  # [N, 1]

def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

if __name__ == '__main__':

    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/r0.png')
    parser.add_argument('--dist', type=str, default='../images/r1.png')
    args = parser.parse_args()
    
    ref = prepare_image(Image.open(args.ref).convert("RGB"))
    dist = prepare_image(Image.open(args.dist).convert("RGB"))
    assert ref.shape == dist.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STSIM_VGG().to(device)
    import pdb;
    pdb.set_trace()
    ref = ref.to(device)
    dist = dist.to(device)
    score = model(ref, dist)
    print(score.item())
    # score: 0.3347

