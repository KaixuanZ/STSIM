import cv2
import numpy as np
import pyrtools as pt
from skimage import exposure
from skimage.exposure import match_histograms
from tqdm import tqdm

def steerable_hist_match(ref, image):
    # this step is redundent if they are already histogram matched
    image = match_histograms(image, ref, multichannel=False)

    pyr1 = pt.pyramids.SteerablePyramidSpace(ref, height=3, order=3)    # 3+2 scales, 3 orientations
    pyr2 = pt.pyramids.SteerablePyramidSpace(image, height=3, order=3)    # 3+2 scales, 3 orientations
    for key in pyr1.pyr_coeffs:
        pyr2.pyr_coeffs[key] = match_histograms(pyr2.pyr_coeffs[key], pyr1.pyr_coeffs[key], multichannel=False)
    image = pyr2.recon_pyr()
    image = match_histograms(image, ref, multichannel=False)

    return image

if __name__ == '__main__':
    img1 = cv2.imread("data/fg.png",0)
    img2 = np.random.rand(img1.shape[0],img1.shape[1])

    img2 = match_histograms(img2, img1, multichannel=False)
    for i in tqdm(range(10)):
        img2 = steerable_hist_match(img1, img2)
        cv2.imwrite('data/iter'+str(i).zfill(2)+'.png', img2)

    import pdb;
    pdb.set_trace()