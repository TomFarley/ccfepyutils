from copy import copy

import numpy as np
import cv2


def threshold(image, thresh, value=0):
    """Set elements of data bellow the value to threshold_value"""
    mask = np.where(image < thresh)
    out = copy(image)
    out[mask] = value
    return out

def reduce_noise(image, diameter=5, sigma_color=75, sigma_space=75):
    """strong but slow noise filter

    :param: d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed
            from sigmaSpace.
    :param: sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors
            within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of
            semi-equal color.
    :param: sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels
            will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it
            specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace ."""
    i_max = float(np.max(image))
    image = (image * 255.0 / i_max).astype(np.uint8)  # convert frame data to 8 bit bitmap for cv2

    # try:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # except:
    #     pass
    diameter, sigma_color, sigma_space = int(diameter), int(sigma_color), int(sigma_space)
    # image = cv2.guidedFilter(image, image, 3, 9)  # guide, src (in), radius, eps  -- requires OpenCV3
    image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)  # strong but slow noise filter
    # image = cv2.fastNlMeansDenoising(image,None,7,21)

    # try:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # except:
    #     pass
    image = image * i_max / 255.0  # convert frame data to 8 bit bitmap for cv2
    return image

def sharpen(image, ksize_x=15, ksize_y=15, sigma=16, alpha=1.5, beta=-0.5, gamma=0.0):
    """
    ksize – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .
    sigmaX – Gaussian kernel standard deviation in X direction.
    sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX
    :param image:
    :return:
    """
    blured_image = cv2.GaussianBlur(image, (15, 15), 16)
    ## Subtract gaussian blur from image - sharpens small features
    sharpened = cv2.addWeighted(image, alpha, blured_image, beta, gamma)
    return sharpened

def extract_bg(movie, i, window, method='min'):
    """Extract"""
    funcs = {'min': np.min}
    raise NotImplementedError

def extract_fg(movie, i, window, method='min'):
    """ """
    raise NotImplementedError
    return