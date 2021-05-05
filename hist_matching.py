import cv2
import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms

img1 = cv2.imread("data/fg.png",0)
img2 = np.random.rand(img1.shape[0],img1.shape[1])

# if we want to apply steerable pyramid, we have to rewrite a pytorch version, simoncelli's code only support an older python version

for i in range(10):
    img2 = match_histograms(img2, img1, multichannel=False)
    cv2.imwrite('data/tmp'+ str(i).zfill(3)+ '.png',img2)
import pdb;
pdb.set_trace()