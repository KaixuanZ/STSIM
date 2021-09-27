import numpy as np

def _match_cumulative_cdf(source, tmpl_dist, params):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    # calculate normalized quantiles
    src_quantiles = np.cumsum(src_counts) / source.size
    src_quantiles[-1] -= 1e-5

    interp_a_values = tmpl_dist.ppf(src_quantiles, *params)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def my_match_histograms(image, distribution, params):
    """
    Modified from scipy.exposure.histogram_matching.py
    
    Adjust an image so that its cumulative histogram matches that of distribution.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    distribution : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    multichannel : bool, optional
        Apply the matching separately for each channel. This argument is
        deprecated: specify `channel_axis` instead.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the distribution
        differ.
    distributions
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    '''
    if image.ndim != distribution.ndim:
        raise ValueError('Image and distribution must have the same number '
                         'of channels.')

    if channel_axis is not None:
        if image.shape[-1] != distribution.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'distribution image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    distribution[..., channel])
            matched[..., channel] = matched_channel
    else:
    '''
    matched = _match_cumulative_cdf(image, distribution, params)

    return matched

if __name__ == '__main__':
    import cv2
    import pyrtools as pt
    from scipy.stats import gennorm
    img1 = cv2.imread("../data/fg.png", 0)/255
    img2 = np.random.rand(img1.shape[0], img1.shape[1])

    pyr1 = pt.pyramids.SteerablePyramidSpace(img1, height=3, order=3)  # 3+2 scales, 3 orientations
    pyr2 = pt.pyramids.SteerablePyramidSpace(img2, height=3, order=3)  # 3+2 scales, 3 orientations
    for key in pyr1.pyr_coeffs:

        params = gennorm.fit(pyr1.pyr_coeffs[key].ravel())
        pyr2.pyr_coeffs[key] = my_match_histograms(pyr2.pyr_coeffs[key], gennorm , params)
    image = pyr2.recon_pyr()

    params = gennorm.fit(img1.ravel())
    image = my_match_histograms(img2, gennorm, params)
    import pdb;

    pdb.set_trace()
