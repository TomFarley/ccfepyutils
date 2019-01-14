from copy import copy

import numpy as np
import cv2
import matplotlib
from ccfepyutils import batch_mode
import matplotlib
if batch_mode:
    matplotlib.use('Agg')
    print('Using non-visual backend')
else:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import logging

from ccfepyutils.classes.plot import Plot

logger = logging.getLogger(__name__)

def to_grayscale(image):
    image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_out

def to_nbit(image, nbit=8):
    nbit = int(nbit)
    image = np.array(image)
    original_max = float(np.max(image))
    original_type = image.dtype
    new_max = 2**nbit - 1
    new_type = getattr(np, 'uint{:d}'.format(nbit))
    if not image.dtype == new_type:
        image = (image * new_max / original_max).astype(new_type)
    return image, original_max, original_type

def to_original_type(image, original_max, original_type, from_type=8):
    from_max = 2**from_type - 1
    image_out = (image * original_max / from_max).astype(original_type)
    return image_out

def threshold(image, thresh_abs=None, thresh_frac=0.25, fill_value=0):
    """Set elements of data bellow the value to threshold_value"""
    if thresh_abs is not None:
        mask = np.where(image < thresh_abs)
    elif thresh_frac is not None:
        mask = np.where(image < (np.min(image) + thresh_frac * (np.max(image)-np.min(image))))
    out = copy(image)
    out[mask] = fill_value
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
    image, original_max, original_type = to_nbit(image)
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
    image = to_original_type(image, original_max, original_type)
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

def hist_equalisation(image, adaptive=True, clip_limit=2.0, tile_grid_size=(8, 8), apply=True):
    """Apply histogram equalisation to image"""
    if not apply:
        return image
    image_out, original_max, original_type = to_nbit(image, nbit=8)
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image_out = clahe.apply(image_out)
    else:
        image_out = cv2.equalizeHist(image_out)
    image_out = to_original_type(image_out, original_max, original_type)
    return image_out

def gamma_enhance(image, gamma=1.2):
    """Apply gamma enhancement to image"""
    image_out = image
    if gamma not in (None, 1, 1.0):
        image_out = image ** gamma
    return image_out

def adjust_contrast(image, adjust_contrast=1.2):
    image_out = image
    if adjust_contrast not in (None, 1, 1.0):
        image_ceil = 2**np.ceil(np.log2(np.max(image))) - 1
        image_out = (((image/image_ceil - 0.5) * adjust_contrast)+0.5)*image_ceil
        image_out[image_out < 0] = 0
        image_out[image_out > image_ceil] = image_ceil
    return image_out

def adjust_brightness(image, adjust_brightness=1.2):
    image_out = image
    if adjust_brightness not in (None, 1, 1.0):
        image_ceil = 2**np.ceil(np.log2(np.max(image))) - 1
        image_out = image + (adjust_brightness-1)*image_ceil
        image_out[image_out < 0] = 0
        image_out[image_out > image_ceil] = image_ceil
    return image_out

def canny_edge_detection(image, canny_threshold1=50, canny_threshold2=250, canny_edges=True):
    image_out, original_max, original_type = to_nbit(image)
    image_out = cv2.Canny(image_out, canny_threshold1, canny_threshold2, canny_edges)  # 500, 550
    image_out = to_original_type(image_out, original_max, original_type)
    return image_out

def invert_image(image, bit_depth=255):
    convert_to_8bit = True if bit_depth == 255 and np.max(image) > bit_depth else False
    if convert_to_8bit:
        image_out, original_max, original_type = to_nbit(image)
    image_out = bit_depth - image_out
    if convert_to_8bit:
        image_out = to_original_type(image_out, original_max, original_type)
    return image_out

def extract_bg(image, frame_stack, method='min'):
    """Extract slowly varying background from range of frames"""
    funcs = {'min': np.min, 'mean': np.mean}  # TODO: Add scipy.fftpack.fft
    func = funcs[method]
    out = func(frame_stack, axis=0)
    return out
    # assert method in funcs, 'Background extraction method "{}" not supported. Options: {}'.format(method, funcs.keys())
    # limits = movie._frame_range['frame_range']
    # frames = movie.get_frame_list(n, n_backwards=n_backwards, n_forwards=n_forwards, step_backwards=step_backwards,
    #                               step_forwards=step_forwards, skip_backwards=skip_backwards,
    #                               skip_forwards=skip_forwards, limits=limits, unique=unique)

def extract_fg(image, frame_stack, method='min'):
    """Extract rapidly varying forground from range of frames"""
    bg = extract_bg(image, frame_stack, method=method)
    # Subtract background to leave foreground
    out = image - bg
    return out

def add_abs_gauss_noise(image, sigma_frac=0.05, sigma_abs=None, mean=0.0, return_noise=False):
    """ Add noise to frame to emulate experimental random noise. A positive definite gaussian distribution is used
    so as to best model the noise in background subtracted frame data
    """
    if sigma_abs is not None:
        scale = sigma_abs
    else:
        # Set gaussian width to fraction of image intensity range
        scale = sigma_frac * np.ptp(image)
    noise = np.abs(np.random.normal(loc=mean, scale=scale, size=image.shape))
    if not return_noise:
        image = image + noise
        return image
    else:
        return noise

def extract_numbered_contour(mask, number):
    """Extract part of image mask equal to number"""
    out = np.zeros_like(mask)
    out[mask == number] = number
    return out

def contour_info(mask, image=None, extract_number=None, x_values=None, y_values=None):
    """Return information about the contour

    :param mask: 2d array where points outside the contour are zero and points inside are non-zero"""
    if extract_number:
        mask = extract_numbered_contour(mask, extract_number)

    # Dict of information about the contour
    info = {}

    info['npoints'] = len(mask[mask > 0])
    info['ipix'] = np.array(np.where(mask > 0)).T

    # Get the points around the perimeter of the contour
    im2, cont_points, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        cont_points = cont_points[0]
    except IndexError as e:
        raise e

    # Pixel coords of perimeter of contour
    ipix = np.array(cont_points)[:, 0]
    ix = ipix[:, 0]
    iy = ipix[:, 1]
    info['ipix_perim'] = ipix
    info['npoints_perim'] = len(ipix)

    moments = cv2.moments(cont_points)
    try:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    except ZeroDivisionError:
        cx = np.mean(cont_points[:, :, 0])
        cy = np.mean(cont_points[:, :, 1])
    info['centre_of_mass'] = (cx, cy)

    ## Get total extent in x and y directions of contour (rectangle not rotated)
    x, y, bounding_width, bounding_height = cv2.boundingRect(cont_points)
    info['bound_width'] = bounding_width
    info['bound_height'] = bounding_height

    area = cv2.contourArea(cont_points)
    info['area'] = area

    perimeter = cv2.arcLength(cont_points, True)
    info['perimeter'] = perimeter

    hull = cv2.convexHull(cont_points)  # Area of elastic band stretched around point set
    hull_area = cv2.contourArea(hull)
    if area > 0 and hull_area > 0:
        solidity = float(area) / hull_area  # Measure of how smooth the outer edge is
    elif area == 0:  # only one point or line?
        solidity = 1
    else:
        solidity = 0.0
    info['solidity'] = solidity

    logger.debug('COM: ({}, {}), area: {}, perimeter: {}, solidity: {},'.format(cx, cy, area, perimeter, solidity))

    # Get data related to intensities in the image
    if image is not None:  # bug: max_loc giving values outside of image! x,y reversed?
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask=mask)
        info['amp_min'] = min_val  # min and max WITHIN contour (not relative to background)
        info['amp_max'] = max_val
        info['min_loc'] = min_loc
        info['max_loc'] = max_loc

        info['amp_mean'] = np.mean(image[mask.astype(bool)])

    if x_values is not None:
        # TODO: convert pixel values to coordinate values
        raise NotImplementedError
        for key in []:
            info[key+''] =info[key] * x_values
    if y_values is not None:
        raise NotImplementedError

    return info

# def extract_fg(movie, n, method='min', n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1,
#                skip_backwards=0, skip_forwards=0, unique=True, **kwargs):
#     """Extract rapidly varying forground from range of frames"""
#     frame = movie[n][:]
#     bg = extract_bg(movie, n, method=method, n_backwards=n_backwards, n_forwards=n_forwards,
#                     step_backwards=step_backwards, step_forwards=step_forwards,
#                     skip_backwards=skip_backwards, skip_forwards=skip_forwards,
#                     unique=unique)
#     # Subtract background to leave foreground
#     out = frame - bg
#     return out

image_enhancement_functions = {
            'threshold': threshold, 'reduce_noise': reduce_noise, 'sharpen': sharpen, 'threshold': threshold,
             'gamma_enhance': gamma_enhance, 'hist_equalisation': hist_equalisation, 'invert_image': invert_image,
             'canny_edge_detection': canny_edge_detection,
             'extract_bg': extract_bg, 'extract_fg': extract_fg,
             'add_abs_gauss_noise': add_abs_gauss_noise}