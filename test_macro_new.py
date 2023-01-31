import torch
import os
# from metrics.STSIM import *
from metrics.STSIM_VGG import *


if __name__ == '__main__':
    device='cuda'
    model = STSIM_VGG([5900,10],grayscale=False).to(device)
    model.load_state_dict(torch.load('weights/STSIM_macro_VGG_05222022/epoch_0100.pt'), strict=False)
    model.to(device).double()
    tmp1 = torch.ones(1,3,100,100).to(device)
    tmp2 = torch.ones(1,3,100,100).to(device)
    # tmp2 = torch.zeros(10,5900).to(device)
    pred = model(tmp1, tmp2)
    import pdb;pdb.set_trace()