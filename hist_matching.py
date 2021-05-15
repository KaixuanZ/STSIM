import cv2
import numpy as np
import pyrtools as pt
from skimage import exposure
from skimage.exposure import match_histograms
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import gennorm
import matplotlib.pyplot as plt

def steerable_hist_match(ref, image):
    # this step is redundent if they are already histogram matched
    image = match_histograms(image, ref, multichannel=False)

    pyr1 = pt.pyramids.SteerablePyramidSpace(ref, height=3, order=3)    # 3+2 scales, 3 orientations
    pyr2 = pt.pyramids.SteerablePyramidSpace(image, height=3, order=3)    # 3+2 scales, 3 orientations
    for key in pyr1.pyr_coeffs:
        pyr2.pyr_coeffs[key] = match_histograms(pyr2.pyr_coeffs[key], pyr1.pyr_coeffs[key], multichannel=False)

        x = np.linspace(pyr1.pyr_coeffs[key].min(), pyr1.pyr_coeffs[key].max(), 100)
        # mu, sigma = pyr1.pyr_coeffs[key].mean(), (pyr1.pyr_coeffs[key].var())**0.5
        #plt.plot(x, stats.norm.pdf(x, mu, sigma))

        if key in [(0,0), (0,2), (1,0), (1,2), (2,0), (2,2)]:
            beta, loc, scale = gennorm.fit(pyr1.pyr_coeffs[key][1:-1,1:-1])
        else:
            beta, loc, scale = gennorm.fit(pyr1.pyr_coeffs[key])
        plt.plot(x, gennorm.pdf(x, beta, loc, scale))

        plt.hist(pyr1.pyr_coeffs[key].flatten(),bins='auto', density=1)

        plt.title(str(key))
        #plt.show()
        plt.savefig('tmp_0514/' + str(key) + '.png')
        plt.close()

    import pdb;pdb.set_trace()
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