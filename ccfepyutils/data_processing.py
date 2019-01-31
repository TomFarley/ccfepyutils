#!/usr/bin/env python

"""General purpose tools for processing data"""
import itertools, numbers, logging, os
from copy import deepcopy

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
from ccfepyutils import batch_mode
import matplotlib
if batch_mode:
    matplotlib.use('Agg')
    print('Using non-visual backend')
else:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from ccfepyutils.utils import safe_arange

logger = logging.getLogger(__name__)


def local_maxima(arr, bg_thresh=0.0, neighborhood=(7, 7), elliptical=True):
    """ Return indices of elements in arr that are local maxima over a region the size of neighborhood
    NOTE: Adjacent points with the same local maximum value will all be returned ie there is not necessarily only
    one local maximum point per neighborhood region
    Method: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    import scipy.ndimage.morphology as morphology
    import scipy.ndimage.filters as filters
    # neighborhood = morphology.generate_binary_structure(len(arr.shape),2) # 3x3 matrix == True
    if elliptical:
        import cv2
        neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, neighborhood)
    else:
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

def local_minima(arr, bg_thresh=0.0, neighborhood=(7, 7), elliptical=True):
    """ Detect local minima, see local_maxima for comments"""
    import scipy.ndimage.morphology as morphology
    import scipy.ndimage.filters as filters
    # ref: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    if elliptical:
        import cv2
        neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, neighborhood)
    else:
        neighborhood = np.ones(neighborhood, dtype=bool)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr <= bg_thresh)
    eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    iminima = np.where(detected_minima)
    return np.array(iminima)

def sorted_local_maxima(arr, bg_thresh=0.0, neighborhood=(7, 7), min_max=None, elliptical=True):
    ind = local_maxima(arr, bg_thresh=bg_thresh, neighborhood=neighborhood, elliptical=elliptical)  # indices of maxima
    ix = ind[1, :]
    iy = ind[0, :]
    z = arr[iy, ix]
    isorted = np.argsort(z)[::-1]  # indices to sort z in decreasing order
    if min_max is not None:  # duplicaton of bg_thresh?
        isorted = np.extract(z[isorted] > min_max, isorted)

    maxima = {'ix': ix[isorted], 'iy': iy[isorted], 'z': z[isorted], 'N': len(isorted)}
    return maxima

def sorted_local_minima(arr, bg_thresh=0.0, neighborhood=(7, 7), max_min=None, elliptical=True):
    ind = local_minima(arr, bg_thresh=bg_thresh, neighborhood=neighborhood, elliptical=elliptical)  # indices of maxima
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


def data_split(x, y=None, gap_length=3, data_length=10, gap_length_abs=None,
               av_diff=False, return_longest=False, verbose=True):
    """ Split data at gaps where difference between x data points in much greater than the average/modal difference
    Return indices and values of data in each continuous section (and y values if supplied)"""
    i = np.arange(len(x))
    x = np.array(x)
    diff = np.diff(x)               # differences between adjacent data
    if gap_length_abs is None:
        ## Find the average distace between the x data
        av_gap = stats.mode(diff) if not av_diff else np.average(diff)       # average/modal separation
        gap_length_abs = gap_length*av_gap
    ## Get indices of begining of gaps sufficiently greater than the average
    igap = np.nonzero(diff > gap_length_abs)[0]  # nonzero nested in tuple

    if verbose:
        print('data_split: {} gap(s) identified: {}'.format(len(igap), igap))

    xsplit = []
    if y is not None:
        y = np.array(y)
        ysplit = []
    isplit_all = []
    ## No gap => 1 linear section, 1 gap => 2 linear sections, 2 gaps => 3 linear sections, etc.
    ## If no gaps, don't split the data
    if len(igap) == 0:
        xsplit.append(x)
        if y is not None:
            ysplit.append(y)
        isplit_all.append(i)
    else:
        ## First set of linear data before first gap
        if igap[0]-0 >= data_length:  # Only add data if set is long enough
            isplit = np.arange(0, igap[0])  # begining of data to begining of gap
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
    ind = int(np.array([len(x) for x in xsplit]).argmax()) if return_longest else np.arange(len(xsplit)).astype(int)
    isplit_all, xsplit, ysplit = np.array(isplit_all), np.array(xsplit), np.array(ysplit)
    if y is not None:
        return isplit_all[ind], xsplit[ind], ysplit[ind]
    else:
        return isplit_all[ind], xsplit[ind], None


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


def moving_average (values, window, force_odd=False):
    """ Return moving average with window length
    """
    # Make sure window odd
    if force_odd and (window % 2 != 1):
        window += 1
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
        w = np.ones(window_len, np.int)
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')  # len(y) = len(s)-2 ?
    # return y[window_len-1:-window_len+1]  # original
    y = y[window_len:-window_len]
    assert(len(y) == len(x))
    return y

def auto_correlation(signal, detrend=False, norm=True):
    """Return auto correlation function of signal"""
    signal = np.array(signal)
    if detrend:
        signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[signal.size - 1:]
    if norm:
        autocorr /= np.max(autocorr)
    return autocorr

def pdf(x, nbins=None, bin_edges=None, min_data_per_bin=10, nbins_max=50, nbins_min=3,
        data_per_bin_percentiles=(5, 95), max_resolution=None, density=False, max_1=False, filter_nans=True,
        detect_delta_fuctions=False, av_data_per_delta=10, compensate_for_extrema=True, res_delta_zero=0.1):
    """Return bin edges and counts in PDF for given x data

    detect_delta_fuctions: used to plot sharp deltas rather than triangles when only a few unique x values

    bin_edges, bin_centres, counts = pdf(x, nbins=15)"""
    # TODO: Add nbins_ideal as in blobs.pdf - use to set min_data_per_bin
    # Add forced bin_width option
    assert not ((nbins is not None) and (bin_edges is not None)), 'Only supply one bins argument'
    x = np.array(x)
    if filter_nans:
        try:
            x = x[~np.isnan(x)]
        except:
            pass
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    if (not detect_delta_fuctions) and (len(x) < 2):
        return np.array([]), np.array([]), np.array([])
    if (nbins is None) and (bin_edges is None):
        nbins = np.floor(len(x) / min_data_per_bin)
        nbins = int(np.round(np.max([8, nbins])))
        nbins = np.min((nbins, nbins_max))

        try:
            x_range = np.ptp(x)
        except:
            pass
        av_data_per_unique_value = len(x) / len(set(x))
        contains_deltas = (av_data_per_unique_value > av_data_per_delta) or (x_range == 0)

        # Check don't exceed max resolution for x data to avoid jagged beating of pdf
        # If max_resolution not supplied, estimate it from data
        if (max_resolution is None):
            if x_range == 0 and x[0] == 0:
                # Delta at 0 so no information about data resolution - use kwarg info
                max_resolution = res_delta_zero
            elif detect_delta_fuctions and contains_deltas:
                max_resolution = x[0] / 10000
            elif (x_range > 0):
                # General case of non-delta data. May overestimate max_resolution for small sample sizes
                diffs = np.abs(np.diff(x))
                max_resolution = np.min(diffs[diffs > 0])  # minimum distance between values to avoid gaps
            else:
                # Single delta, not at zero - use 100th of data magnitude as guess
                max_resolution = x[0] / 10000

        if (not detect_delta_fuctions) or (not contains_deltas):
            # Number of bins needed for requested minimum average data per bin
            nbins = np.floor(len(x) / min_data_per_bin)
            # Make sure nbins not below minimum requested
            nbins = int(np.round(np.max([nbins_min, nbins])))

            x_lims = [np.min(x), np.max(x)]
            x_sub_lims = [np.percentile(x, data_per_bin_percentiles[0]), np.percentile(x, data_per_bin_percentiles[1])]
            x_sub_range = x_sub_lims[1] - x_sub_lims[0]
            if compensate_for_extrema:
                # Increase number of bins if extrema are increasing the x range and so unevenly distributing min_data_per_bin
                # Bascially add extra bins to cover areas of x range where effectively empty bins
                try:
                    nbins = int(np.ceil(nbins * (0.01 * (data_per_bin_percentiles[1] - data_per_bin_percentiles[0])) /
                                        ((x_sub_range) / x_range)))
                except ValueError as e:
                    raise e
            if nbins_max is not None:
                # Keep nbins below max requested
                nbins = np.min((nbins, nbins_max))

            if (max_resolution is not False) and (x_range/(nbins-1) < max_resolution):
                # Reduce number of bins if sampling above data resolution
                nbins = int(np.floor(x_range/max_resolution)) + 1
        else:
            # Return sharp deltas rather than triangles when only a few unique x values
            # if x_range == 0:
            #     # Single delta function
            #     nbins = 3
            #     x0 = x[0]
            #     bin_edges = [x0-1.5*max_resolution, x0-0.5*max_resolution, x0+0.5*max_resolution, x0+1.5*max_resolution]
            #     logger.debug('Using bin edges for delta function pdf: {}'.format(bin_edges))
            # else:
            try:
                bin_edges = safe_arange(np.min(x)-1.5*max_resolution, np.max(x)+1.5*max_resolution, max_resolution)
            except Exception as e:
                raise e

    bins = np.sort(bin_edges) if (bin_edges is not None) else nbins
    # logger.info('Bins: {}'.format(bins))
    try:
        counts, bin_edges = np.histogram(x, bins=bins, density=density)
    except Exception as e:
        raise e

    counts = counts / np.max(counts) if max_1 else counts
    bin_centres = moving_average(bin_edges, 2)
    return bin_edges, bin_centres, counts

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


def interp_val(x, y, x0, kind='linear'):
    "Interpolated y value at x0 (scipy.interpolate.interp1d wrapper)"
    ## TODO: add numpy array check
    x = np.array(x)
    y = np.array(y)
    isort = np.argsort(x)  # interp x values must be monotonically increasing
    finterp = interp1d(x[isort], y[isort], kind=kind)  # cubic noisy
    y0 = finterp(x0)  # [0] #???
    return y0


def central_diff(x, y, n=1):
    """ Calculate central difference derivate """
    ## Cannot calc central derivative of first element
    diffs = np.array([float('NaN')])
    for i in 1 + np.arange(len(x) - 2):
        if x[i + 1] == x[i - 1]:  # dx=0
            dydx = float('Inf')
        else:
            dydx = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        diffs = np.append(diffs, dydx)
    ## Cannot calc central derivative of last element
    diffs = np.append(diffs, float('NaN'))

    ## Calculate higher derivatives recursively - are the NaNs ok?
    if n > 1:
        n -= 1
        diffs = central_diff(x, diffs, n=n)

    return diffs


def argrelmax_global(array):
    """ Inedex of glabal array maximum """
    order = len(array)
    imax = argrelmax(array, order=order)[0]
    if len(imax) == 0:
        # print "WARNING: argrelmax_gloabal failed to find a global maximum - trying orders < {}".format(order)
        print("WARNING: argrelmax_gloabal failed to find a global maximum - using max/where".format(order))
        ## I think this happens when two or more max points have the same value - common in low res data
    else:
        return imax

    maximum = np.amax(array)
    imax = np.nonzero(array == maximum)[0]
    print('imax = {}, from max/where method, max = {}'.format(imax[0], maximum))

    # while len(imax) == 0:
    #     try:
    #         order -= 1
    #         imax = argrelmax(array, order = order)[0] # Extract index of global maximum
    #     except IndexError:
    #         raise
    # print 'imax = {}, for order = {}'.format(imax, order)

    return imax[0]

def nsigma(y, nsigma=2.5):
    """Return mu+nsigma*sigma for supplied y"""
    y_mean = np.mean(y)
    y_std = np.std(y)
    out = y_mean + y_std * nsigma
    return out

def get_peaks_argrelmax(y, nsigma=2.5, order=2):
    """ Find peaks in time trace via nsigma devition from mean or from location in 3D filament

    Keyword arguments
    x - x data (eg time points)
    y - y data (eg intensity values)
    nsigma - Std dev threshold for peak to be averaged (default=2.5)
    order - window size for maxima detection
    """

    # Find local maxima in time series nsigma above mean
    y_thresh = deepcopy(y)
    y_thresh[np.where(y_thresh < nsigma(y, nsigma))] = nsigma(y, nsigma)
    ifil = np.argrelmax(y_thresh, order=order)[0]
    ifil = np.round(ifil).astype(np.int)
    return ifil

def conv_diff(y_data, width=51, order=1):
    """ Normalised derivative of data convolved with gaussian """

    x1 = np.linspace(-3, 3, width)  # +/- 3 sigma of gaussian, sampled at width # of points
    gauss0 = np.exp(-x1 ** 2)
    gauss1 = (-2 * x1) * np.exp(-x1 ** 2)
    gauss2 = (4 * x1 ** 2) * np.exp(-x1 ** 2)
    gauss3 = (-8 * x1 ** 3) * np.exp(-x1 ** 2)
    gauss4 = (32 * x1 ** 4) * np.exp(-x1 ** 2)

    # if order == 0:
    #     y1 = gauss0
    # elif order == 1:
    #     y1 = gauss1
    # elif order == 2:
    #     y1 = gauss2
    # elif order == 3:
    #     y1 = gauss3
    # elif order == 3:
    #     y1 = gauss4
    # else:
    #     raise RuntimeError('conv_diff got unexpected order arguement value')

    if order > 0:
        y1 = gauss1

        y_conv = np.r_[y_data[width - 1:0:-1], y_data, y_data[-1:-width:-1]]  # mirror data at edges
        y_conv = np.convolve(y_conv, y1, mode="same")
        y_conv = y_conv[(width - 1):-(width - 1)]  # "-(width-1)" is slight bodge to get right number of elements
        y_conv /= np.amax(np.absolute(y_conv))  # normalise output (not rigourus!)

        y_conv = conv_diff(y_conv, width=width, order=order - 1)

        return y_conv
    else:
        return y_data


def find_linear(x_data, y_data, width=51, gap_length=3, data_length=10,
                tol_type='rel_peak', tol=0.6, plot=False, fig_name=False):
    """ Identify linear sections in experimental data
    x_data, y_data - experimental data
    width - convolution smoothing width
    Returns two lists of arrays containing x data and y data for each linear section
    """
    ## Approach 1:
    # create convolution kernel for calculating the smoothed second order derivative from:
    # http://stackoverflow.com/questions/13691775/python-pinpointing-the-linear-part-of-a-slope
    x1 = np.linspace(-3, 3, width)  # +/- 3 sigma of gaussian, sampled at width # of points

    ## Try normalising this gaussian?

    # db(x1=len(x1))
    norm = np.sum(np.exp(-x1 ** 2)) * (x1[1] - x1[0])  # ad hoc normalization
    ## Twice differentiated normal distribution - not sure about -2 and norm
    y1 = (4 * x1 ** 2 - 2) * np.exp(-x1 ** 2) / width * 8  # norm*(x1[1]-x1[0])
    y2 = (-8 * x1 ** 3) * np.exp(-x1 ** 2) / width * 8  # for third derivative

    ## Add mirroed data at ends
    # db(len(y_data[width-1:0:-1]), len(y_data[-1:-width:-1]))
    y_conv = np.r_[y_data[width - 1:0:-1], y_data, y_data[-1:-width:-1]]
    y_conv2 = np.r_[y_data[width - 1:0:-1], y_data, y_data[-1:-width:-1]]
    ## Calculate second order deriv. through convolution
    y_conv = np.convolve(y_conv, y1, mode="same")
    y_conv2 = np.convolve(y_conv2, y2, mode="same")
    # db( len(y_conv[0:(width-1)]), len(y_conv[-(width):-1]) )
    ## Remove mirrored data at ends
    y_conv = y_conv[(width - 1):-(width - 1)]  # "-(width-1)" is slight bodge to get right number of elements
    y_conv2 = y_conv2[(width - 1):-(width - 1)]  # "-(width-1)" is slight bodge to get right number of elements

    ## Approach 2:
    ## Use home made central diff applied to smoothed fit
    y_smooth = smooth(y_data, window_len=58, window='hamming')
    y_cent = central_diff(x_data, y_smooth, n=2)
    # y_cent *=  np.amax(y_conv) / np.amax(y_cent) # smooth(y_cent, window_len=10)

    # db(len(y_data), len(y_conv), len(y_smooth), len(y_cent))
    # db(y_conv=y_conv)

    ## Find where 2nd derviative goes above tollerance
    if tol_type == 'abs':
        tol = tol  # tol = 1.8e-7
    elif tol_type == 'rel_peak':
        ## Tolerance in 2nd derviative taken relative to maximum in 2nd derivative
        tol = tol * np.amax(y_conv)
    elif tol_type == 'rel_mean':
        tol = tol * np.average(np.absolute(y_conv))

    lims = interp_val(y_conv, x_data, tol)
    # db(lims=lims)
    x_intol, y_intol = tf.extract_2D(x_data, y_data, np.abs(y_conv) <= tol)

    xlinear, ylinear = data_split(x_intol, y_intol, gap_length=gap_length, data_length=data_length)

    # db('{} linear region(s) identified'.format(len(xlinear)))

    if plot:
        # plot data
        fig = plt.figure(fig_name) if fig_name else plt.figure()
        fig.clear()
        ## Plot data
        plt.plot(x_data, y_data, "o", label="noisy data")
        plt.plot(x_data, y_smooth, "x-", label="smoothed data")
        for x, y in zip(xlinear, ylinear):
            plt.plot(x, y, "or", label="linear data")
            plt.axvline(x[0], linestyle='--', color='k', alpha=0.5)
            plt.axvline(x[-1], linestyle='--', color='k', alpha=0.5)
        ## Plot 2nd derivative (shitfted to data)
        shift = np.average(y_data)
        plt.plot(x_data, y_conv + shift, '-', label="conv second deriv", alpha=0.6)
        plt.plot(x_data, y_conv2, '-', label="conv third deriv", alpha=0.6)
        plt.plot(x_data, y_cent + shift, '--', label="central second deriv", alpha=0.6, color=(0.7, 0.2, 0.5))
        # plt.axhline(0)
        plt.axhline(0 + shift, linestyle='--', color='k', alpha=0.5)
        # plt.axhline(tol)
        # plt.axvspan(0,4, color="y", alpha=0.2)
        # plt.axvspan(6,14, color="y", alpha=0.2)
        plt.axhspan(-tol + shift, tol + shift, color="b", alpha=0.4)
        # plt.vlines([0, 4, 6],-10, 10)
        # plt.xlim(-2.5,12)
        # plt.ylim(-2.5,6)
        plt.legend(loc='best')
        plt.show()

    return xlinear, ylinear

if __name__ == "__main__":
    print('*** data_processing.py demo ***')
    x = np.linspace(0,1000,1000)
    y = np.sin(np.deg2rad(x))#+x
    y = y/np.amax(np.absolute(y))
    x = x
    print("interp_val(x, y, 2.3649, kind='linear') = ", end=' ')
    print(interp_val(x, y, 2.3649, kind='linear'))

    fig = plt.figure('conv_diff test_tmp')
    fig.clear()
    plt.plot( x, y, label = 'sin', c='k', lw=3)
    for n in range(4):
        plt.plot(x, conv_diff(y, order=n), label='n={}'.format(n))
    plt.title('conv_diff test_tmp')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    x = np.arange(30)
    s = smooth(x, window_len=5)
    print('lenx:', len(x), len(s))

    pass

