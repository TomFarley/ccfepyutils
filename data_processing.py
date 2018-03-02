#!/usr/bin/env python

"""General purpose tools for processing data"""
import itertools
import numbers

import os
from copy import deepcopy

import numpy as np
import scipy as sp
from scipy.signal import butter, filtfilt


def local_maxima(arr, bg_thresh=0.0, neighborhood=(7, 7)):
    """ Return indices of elements in arr that are local maxima over a region the size of neighborhood
    NOTE: Adjacent points with the same local maximum value will all be returned ie there is not necessarily only
    one local maximum point per neighborhood region
    Method: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    import scipy.ndimage.morphology as morphology
    import scipy.ndimage.filters as filters
    # neighborhood = morphology.generate_binary_structure(len(arr.shape),2) # 3x3 matrix == True
    neighborhood = np.ones(neighborhood, dtype=bool)
    ## Maximum_filter sets each pixel to the maximum value of all the pixels in the surrounding footprint (eg immediate sides and diagonals)
    local_max = filters.maximum_filter(arr, footprint=neighborhood)
    ## Equality with arr gives boolian array with true where values are local maxima for their footprint region
    local_max = (local_max == arr)
    ## Need to remove local maximia in background
    background = (arr <= bg_thresh)  # typically mostly False in this case - TODO: consider using (arr < bg_thresh)?
    ## Erosion removes objects smaller than the structure (neighborhood) ie only remains True if all surrounding points are True also
    eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, iterations=1, mask=None,
            border_value=1)  # Extra encapsulating border considered true
    detected_maxima = local_max ^ eroded_background  # local max - 0 ~ local max
    imaxima = np.where(detected_maxima)
    return np.array(imaxima)  # return indices of maxima

def local_minima(arr, bg_thresh=0.0, neighborhood=(6, 6)):
    """ Detect local minima, see local_maxima for comments"""
    import scipy.ndimage.morphology as morphology
    import scipy.ndimage.filters as filters
    # ref: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    neighborhood = np.ones(neighborhood, dtype=bool)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr <= bg_thresh)
    eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    iminima = np.where(detected_minima)
    return np.array(iminima)

def sorted_local_maxima(arr, bg_thresh=0.0, neighborhood=(7, 7), min_max=None):
    ind = local_maxima(arr, bg_thresh=bg_thresh, neighborhood=neighborhood)  # indices of maxima
    ix = ind[1, :]
    iy = ind[0, :]
    z = arr[iy, ix]
    isorted = np.argsort(z)[::-1]  # indices to sort z in decreasing order
    if min_max is not None:  # duplicaton of bg_thresh?
        isorted = np.extract(z[isorted] > min_max, isorted)

    maxima = {'ix': ix[isorted], 'iy': iy[isorted], 'z': z[isorted], 'N': len(isorted)}
    return maxima

def sorted_local_minima(arr, bg_thresh=0.0, neighborhood=(7, 7), max_min=None):
    ind = local_minima(arr, bg_thresh=bg_thresh, neighborhood=neighborhood)  # indices of maxima
    ix = ind[1, :]
    iy = ind[0, :]
    z = arr[iy, ix]
    isorted = np.argsort(z)  # indices to sort z in ascending order
    if max_min is not None:  # duplicaton of bg_thresh?
        isorted = np.extract(z[isorted] < max_min, isorted)

    maxima = {'ix': ix[isorted], 'iy': iy[isorted], 'z': z[isorted], 'N': len(z)}
    return maxima


def running_mean_periodic(series,window):
    """ Compute the running mean of a 1D sequence """
    input = np.asarray(series)
    output = []
    if window % 2 == 0:
        width = window/2
    else:
        width = (window - 1)/2
    for i in np.arange(input.shape[0]):
        if i - width < 0: # lhs of window spans end     [#i###                  ##]
            temp = np.concatenate((input[i-width:],input[0:i+width])) # join over end
        elif i + width > input.shape[0]: # rhs of window spans end   [##                  ###i#]
            temp = np.concatenate((input[i-width:-1],input[0:i + width - input.shape[0]]))
        else:
            temp = input[i-width:i+width]

        output.append(np.mean(temp))

    return np.asarray(output)


def running_mean(series,window):
    """ Compute the running mean of a 1D sequence """
    input = np.asarray(series)
    output = []
    if window % 2 == 0:
            width = window/2
    else:
            width = (window - 1)/2
    for i in np.arange(input.shape[0]-(2*width))+width:
            if i - width < 0:
                    temp = np.concatenate((input[i-width:],input[0:i+width]))
            elif i + width > input.shape[0]:
                    temp = np.concatenate((input[i-width:-1],input[0:i + width - input.shape[0]]))
            else:
                    temp = input[i-width:i+width]

            output.append(np.mean(temp))
    result = np.concatenate((input[0:width],np.asarray(output),input[-width:]))
    return result


def low_pass(signal, order=3, critical_freq=0.05):
    """ Apply butterworth low pass filter
    """
    # Create a lowpass butterworth filter.
    b, a = butter(order, critical_freq)

    # # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
    # # of the filter.
    # zi = lfilter_zi(b, a)
    # z, _ = lfilter(b, a, signal, zi=zi*signal[0])
    #
    # # Apply the filter again, to have a result filtered at an order
    # # the same as filtfilt.
    # z2, _ = lfilter(b, a, z, zi=zi*z[0])

    # Use filtfilt to apply the filter.
    y = filtfilt(b, a, signal)

    return y


def data_split(x, y=None, gap_length=3, data_length=10, av_diff=False, return_longest=False, verbose=True):
    """ Split data at gaps where difference between x data points in much greater than the average/modal difference
    Return indices and values of data in each continuous section (and y values if supplied)"""
    i = np.arange(len(x))
    ## Find the average distace between the x data
    diff = np.diff(x)               # differences between adjacent data
    av_gap = np.mode(diff) if not av_diff else np.average(diff)       # average/modal separation
    ## Get indices of begining of gaps sufficiently greater than the average
    igap = np.nonzero(diff>gap_length*av_gap)[0] # nonzero nested in tuple

    if verbose: print('data_split: {} gap(s) identified: {}'.format(len(igap), igap))

    xsplit = []
    if y is not None:
        ysplit = []
    isplit_all = []
    ## No gap => 1 linear section, 1 gap => 2 linear sections, 2 pags => 3 linear sections etc.
    ## If no gaps, don't split the data
    if len(igap) == 0:
        xsplit.append(x)
        if y is not None:
            ysplit.append(y)
        isplit_all.append(i)
    else:
        ## First set of linear data before first gap
        if igap[0]-0 >= data_length: # Only add data if set is long enough
            isplit = np.arange(0, igap[0]) # begining of data to begining of gap
            xsplit.append(x[isplit])
            if y is not None:
                ysplit.append(y[isplit])
            isplit_all.append(isplit)
        else:
            if verbose: print('data_split: First set exluded as too short')

        ## Deal with linear data that isn't bordered by the ends of the set
        for i in np.arange(1,len(igap)): # if start=stop, loop over empty array -> do nothing when len(ifap)=1
            ## Note: arange doesn't include stop, so len 2 just loops over i=1
            if igap[i]-igap[i-1]+1 >= data_length: # Only add data if set is long enough
                isplit = np.arange(igap[i-1]+1, igap[i]) # end of last gap begining of next gap
                xsplit.append(x[isplit])
                if y is not None:
                    ysplit.append(y[isplit])
                isplit_all.append(isplit)
            else:
                if verbose: print('data_split: Set {} exluded as too short'.format(i))

        ## Last set of linear data after last gap
        if (len(x)-1)-igap[-1]+1 >= data_length: # Only add data if set is long enough
            isplit = np.arange(igap[-1]+1, len(x)-1) # end of last gap to end of data
            xsplit.append(x[isplit])
            if y is not None:
                ysplit.append(y[isplit])
            isplit_all.append(isplit)
        else:
            if verbose: print('data_split: Last set exluded as too short')

    # If return_longest is True, only return longest section of data without gaps, else return all data with gap removed
    ind = np.array([len(x) for x in xsplit]).argmax() if return_longest else np.arange(len(xsplit))
    if y is not None:
        return isplit_all[ind], xsplit[ind], ysplit[ind]
    else:
        return isplit_all[ind], xsplit[ind]


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as e:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def moving_average (values, window):
    """ Return moving average with window length
    """
    weights = np.repeat(1.0, window)/window
    try:
        sma = np.convolve(values, weights, 'valid')
    except:
        pass
    return sma


def correlation(x, y):
    """ Return linear correlation coefficient and line of best fit
    """
    pear_r = np.corrcoef(x, y)[1,0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, np.array(y))[0]
    return pear_r, (m, c)


def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    From scipy cookbook recipe: http://scipy.org/Cookbook/SignalSmooth

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    # print('\n\nIn tf_data.smooth()\n\n')
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]  # original
    s=np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-(window_len+1):-1]]
    #print(len(s))
    assert(len(s) == len(x) + 2*window_len)

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')  # len(y) = len(s)-2 ?
    # return y[window_len-1:-window_len+1]  # original
    y = y[window_len:-window_len]
    assert(len(y) == len(x))
    return y


def pdf(x, min_data_per_bin=10, nbins_max=40, max_resolution=None, density=False, max_1=False):
    """Return distribution in x of peaks in y nsigma above mean"""
    # Todo: add outlier detection to auto pdf
    if len(x) < 2:
        return np.array([]), np.array([])
    nbins = np.floor(len(x) / min_data_per_bin)
    nbins = int(np.round(np.max([8, nbins])))
    nbins = np.min((nbins, nbins_max))

    # Check don't exceed max resolution for x data to avoid jagged beating of pdf
    if max_resolution is None:  # may overestimate max_resolution for small sample sizes
        diffs = np.abs(np.diff(x))
        max_resolution = 2*np.min(diffs[diffs>0])  # twice minimum distance between values to avoid gaps
    x_range = np.ptp(x)
    if (max_resolution is not False) and (x_range/nbins < max_resolution):  # Reduce number of bins
        nbins = int(np.floor(x_range/max_resolution))
    if nbins < 3:
        nbins = 3

    counts, bins = np.histogram(x, bins=nbins, density=density)
    counts = counts / np.max(counts) if max_1 else counts
    return moving_average(bins, 2), counts


def _find_dist_extrema(arr, point, index=True, normalise=False, func=np.argmin):
    """ Find closest point to supplied point in either 1d array, 2d grid or 2xn array of coordinate pairs
    """
    inparr = np.array(arr)
    if len(inparr) == 0:
        return None
    # print('point', point)
    # print('inparr.shape', inparr.shape)
    # print('inparr', inparr)
    shape = inparr.shape
    if isinstance(point, numbers.Number) and len(inparr.shape) == 1:  # if point is a single number take array to be 1d
        if index: return func(np.abs(inparr-point))
        else: return inparr[func(np.abs(inparr-point))]
    elif len(np.array(point).shape) == 1 and len(point) > 1:  # point is a 2D coordinate
        point = np.array(point)
        # Make sure array in two row format
        if shape[1] == 2 and shape[0] > 0:
            inparr = inparr.T
            shape = inparr.shape
        ## 2D coordinates
        if shape[0] == 2 and shape[1] > 0 and len(point) == 2:
            (valx,valy) = point
            normarr = deepcopy(inparr)
            # Treat x and y coordinates as having the same fractional accuracy ie as if dx=dy
            if normalise:
                normarr[0] = (normarr[0]-np.min(normarr[0])) / (np.max(normarr[0]) - np.min(normarr[0]))
                normarr[1] = (normarr[1]-np.min(normarr[1])) / (np.max(normarr[1]) - np.min(normarr[1]))
                valx = (valx-np.min(inparr[0])) / (np.max(inparr[0]) - np.min(inparr[0]))
                valy = (valy-np.min(inparr[1])) / (np.max(inparr[1]) - np.min(inparr[1]))
            ixy = func((((normarr[0,:]-valx)**2.0 + (normarr[1,:] - valy)**2.0)**0.5))
            if index:
                return ixy
            else:
                return inparr[:, ixy]
        ## 3D coordinates
        elif len(shape) == 3 and len(point) == 3:
            # incomplete!
            (valx, valy, valz) = point
            return func((((inparr[:,0]-valx)**2.0 + (inparr[:,1] - valy)**2.0 + (inparr[:,2] - valz)**2.0)**0.5))

        else:
            print('point', point)
            print('inparr', inparr)
            print('inparr.shape', inparr.shape)
            raise RuntimeError('findNearest: Input parameters did not match any anticipated format')
    # Both imp array and point are sets of 2D coordinates
    # Find points with shortest distance between them in the two point clouds
    elif np.array(point).ndim == 2:
        point = np.array(point)
        # Make sure inparr and point arrays have shape (n, 2). If (2, n) transpose first
        if point.shape[0] == 2 and point.shape[1] > 0:
            point = point.T
        if np.array(inparr).shape[0] == 2 and np.array(inparr).shape[1] > 0:
            inparr = inparr.T

        assert np.array(point).shape[1] == 2
        point = np.array(point)

        def distance2(p1, p2):
            # return np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2  # remove sqrt for speed up

        # TODO: Consider incremental/bisection resolution increase when have only one local distance minima
        # Get all combinations of points
        points = np.array([tup for tup in itertools.product(inparr, point)])

        arr0, point0 = points[func([distance2(Pa, Pb) for (Pa, Pb) in points])]
        if index:
            return [np.where(np.bitwise_and(inparr[:, 0] == arr0[0], inparr[:, 1] == arr0[1]))[0][0],
                    np.where(np.bitwise_and(point[:, 0] == point0[0], point[:, 1] == point0[1]))[0][0]]
        else:
            return arr0, point0

    else:
            raise RuntimeError('findNearest: Input array did not match any anticipated format')


def find_nearest(arr, point, index=True, normalise=False):
    """ Find closest point to supplied point in either 1d array, 2d grid or 2xn array of coordinate pairs
    """
    return _find_dist_extrema(arr, point, index=index, normalise=normalise, func=np.argmin)


def find_furthest(arr, point, index=True, normalise=False):
    """ Find furthest point to supplied point in either 1d array, 2d grid or 2xn array of coordinate pairs
    """
    return _find_dist_extrema(arr, point, index=index, normalise=normalise, func=np.argmax)