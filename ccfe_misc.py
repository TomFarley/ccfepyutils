#!/usr/bin/env python

"""
Micelanious general purpose functions
"""

import numbers
from Tkinter import Tk
from tkFileDialog import askopenfilename
from matplotlib.widgets import RectangleSelector
import numpy as np
import itertools
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from copy import copy, deepcopy
import sys
import os
import inspect
from collections import Mapping, Container
from sys import getsizeof
from pathlib import Path
try:
    import cpickle as pickle
except ImportError:
    import pickle

def nsigfig(values, sig_figs=1):
    """Round input values to given number of significant digits"""
    arr = np.array(values)
    arr[np.where(arr==0)] = 1
    vfunc = np.vectorize(np.round)
    dps = np.floor(np.log10(np.abs(arr)))
    dps = np.where(dps == -np.inf, np.ones_like(dps), dps)  # Deal with zeros
    dps = ((sig_figs-1) - dps).astype(int)
    out = vfunc(arr, dps)
    return out if safe_len(values) > 1 else np.float64(out)

def match_in(values, reference, index=False, tol=1e-10):
    """Replace values with closest reference values if within tolerance"""
    reference = np.array(reference)
    def closest_index(value):
        diff = np.isclose(value, reference, atol=tol, rtol=1e-20)
        if np.any(diff):
            out = reference[diff][0] if not index else np.nonzero(diff)[0][0]
        else:
            out = np.nan
        return out
    vfunc = np.vectorize(closest_index)
    out = vfunc(values)
    return out


def is_scalar(var):
    """ True if variable is scalar """
    if hasattr(var, "__len__"):
        return False
    else:
        return True

def is_number(s):
    """
    TODO: Test on numbers and strings and arrays
    """
    try:
        n=str(float(s))
        if n == "nan" or n=="inf" or n=="-inf" : return False
    except ValueError:
        try:
            complex(s) # for complex
        except ValueError:
            return False
    return True

def safe_len(var):
    """ Length of variable returning 1 instead of type error for scalars """
    if is_scalar(var): # checks if has atribute __len__
        return 1
    elif len(np.array(var) == np.nan) == 1 and var == np.nan: # If value is NaN return zero length
        return 0
    else:
        return len(var)

def any_equal(object, list):
    """ Return true if object is equal to any of the elements of list
    """
    return np.any([object == l for l in list])

def to_array(obj, silent=True):
    """Return object as an ndarray"""
    try:
        # convert from dataframe
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            obj = obj.values
    except ImportError:
        pass

    if isinstance(obj, (list, tuple)):
        obj = np.array(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = np.array([obj])  # make it indexable
    else:
        try:
            obj = np.array(obj)
        except:
            if not silent:
                print('Could not convert {} to a np.ndarray'.format())
    return obj

def getUserFile(type=""):
    Tk().withdraw()
    filename = askopenfilename(message="Please select "+type+" file:")
    return filename

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


def compare_dict(dict1, dict2, tol=1e-12, top=True):
    """ Recursively check that two dictionaries and all their constituent sub dictionaries have the same numerical
    values. Avoids floating point precision comparison problems.
    """
    assert(isinstance(dict1, dict) and isinstance(dict2, dict))
    from collections import Counter
    if Counter(dict1.keys()) != Counter(dict2.keys()):  # Use counter to ignore order (if objects are hashable)
        print('def compare_numeric_dict: Dictionaries have different keys:\ndict1: {}\ndict2: {}'.format(
            dict1.keys(), dict2.keys()))
        return False

    for key in dict1.keys():
        if isinstance(dict1[key], dict) or isinstance(dict2[key], dict):

            if not (isinstance(dict1[key], dict) and isinstance(dict2[key], dict)):
                print('Dictionaries are different - One value is a dict while the other is not')
                return False
            if compare_dict(dict1[key], dict2[key], top=False) is False:
                return False
        # elif isinstance(dict2[key], dict):
        #     if compare_numeric_dict(dict1, dict2[key], top=False) is False:
        #         return False
        else:
            try:
                if np.abs(dict1[key]-dict2[key]) > tol:  # numerical
                    return False
            except TypeError:
                if dict1[key] != dict2[key]:  # string etc
                    return False
    return True

def is_close(a, b, atol=1e-8, all=False):
    """Return true if elements of a appear in comparison_list (b) within tollerance
    From: http://stackoverflow.com/questions/39602004/can-i-use-pandas-dataframe-isin-with-a-numeric-tolerance-parameter
    """
    import pandas as pd
    if isinstance(a, pd.Series):
        a = a.values[:, None]  # add dimension to compare pairwise
    out = np.isclose(a, b, atol=atol).any(axis=1)
    if all is False:
        return out
    else:
        return np.all(out)


from matplotlib import cbook
class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists,func=None, tolerance=5, offsets=(-20, 20), 
                 template='x: %0.2f\ny: %0.2f', display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be 
            selected. 
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%) 
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless. 
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        self.func = func
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))
        self.ids = []
        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "axI"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='left',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            if self.func is None:
               
                annotation.set_text(self.template % (x, y))        
                annotation.set_visible(True)
                event.canvas.draw()
            else:
                self.func(event,annotation)
  
    def connect(self,fig):
        self.cid = fig.canvas.mpl_connect('pick_event', self) 
                  
    def disconnect(self,fig):
        fig.canvas.mpl_disconnect(self.cid)
	     
    def clear(self,fig):
        for ann in self.annotations.values():
            ann.set_visible(False)
        fig.canvas.draw()
        
        
class ROISelector(object):
    
    def __init__(self,artist):
            self.artist = artist
            self.selector = RectangleSelector(self.artist.axes,self.on_select,
                                       button=3, minspanx=5, minspany=5, spancoords='pixels',
                                       rectprops = dict(facecolor='red', edgecolor = 'red',
                                                        alpha=0.3, fill=True)) # drawtype='box'
            self.coords = []
            
    def on_select(self,click,release):
            x1,y1 = int(click.xdata),int(click.ydata)
            x2,y2 = int(release.xdata),int(release.ydata)
            self.coords =[(x1,y1),(x2,y2)]
            
    def activate(self):
        self.selector.set_active(True)
        
    def deactivate(self):
        self.selector.set_active(False)        
    

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

def file_filter(path, extension='.p', contain=None, not_contain=None):

    for (dirpath, dirnames, filenames) in os.walk(path_in):
        break  # only look at files in path_in
    if extension is not None:
        filenames = [f for f in filenames if f[-len(extension):] == extension]  # eg only pickle files
    if contain is not None:
        if isinstance(contain, basestring):
            contain = [contain]
        for pat in contain:
            filenames = [f for f in filenames if pat in f] # only files with fixed variable
    if not_contain is not None:
        if isinstance(not_contain, basestring):
            not_contain = [not_contain]
        for pat in not_contain:
            filenames = [f for f in filenames if pat not in f] # only files with fixed variable

    fn = filenames[0]

    return filenames


def printProgress (iteration, total, prefix = '', suffix = '', frac=False, t0=None,
                   decimals = 2, nth_loop=1, barLength = 50):
    """
    from http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    if iteration % nth_loop != 0:  # Only print every nth loop to reduce slowdown from printing
        return
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '|' * filledLength + '-' * (barLength - filledLength)
    frac = '{}/{} '.format(iteration, total) if frac else ''
    if t0 is None:
        time = ''
    else:
        t1 = datetime.now()
        t_diff_past = relativedelta(t1, t0)  # time past in loop
        mul = float(total-iteration)/iteration if iteration > 0 else 0
        t_diff_rem = t_diff_past * mul  # estimate of remaining time
        t_diff_past = '({h}h {m}m {s}s)'.format(h=t_diff_past.hours, m=t_diff_past.minutes, s=t_diff_past.seconds)
        if t_diff_rem.hours > 0:  # If expected to take over an hour display date and time of completion
            t_diff_rem = (datetime.now() + t_diff_rem).strftime("(%d/%m/%y %H:%M)")
        else:  # Display expected time remaining
            t_diff_rem = '({h}h {m}m {s}s)'.format(h=t_diff_rem.hours, m=t_diff_rem.minutes, s=t_diff_rem.seconds)
        time = ' {past} -> {remain}'.format(past=t_diff_past, remain=t_diff_rem)

    sys.stdout.write('\r %s |%s| %s%s%s%s %s' % (prefix, bar, frac, percents, '%', time, suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

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


def in_ellipse(point, centre, rx, ry, boundary=True, return_r=False):
    """ Return true if point is within ellipse with centre at centre and semi-major and semi-minor axes rx and ry
    Note rx and ry are radii not diameters
    """
    assert len(point) == 2
    assert len(centre) == 2

    r = ((point[0]-centre[0])/rx)**2 + ((point[1]-centre[1])/ry)**2
    if return_r:
        return r
    from numbers import Number
    if isinstance(r, Number):  # point is a single coord pair
        if r == 1 and boundary:
            return True
        elif r < 1:
            return True
        else:
            return False
    else:  # point is an array of points [x_array, y_array]
        r[r < 1] = True
        r[r > 1] = False
        r[r == 1] = boundary
        return r[:].astype(bool)

def geo_mean_w(x, weights=None, axis=None, **kwargs):
    """ Return weighted geometric mean along given axis
    """
    if weights is None:
        from scipy.stats.mstats import gmean
        return gmean(x, axis=axis, **kwargs)
    else:
        if axis is None:
            x = np.array(x).flatten()
            weights = np.array(weights).flatten()
        assert len(x) == len(weights)
        # Weighted geometric mean of fom values (using +1, -1 method for avoiding infs from log)
        return np.exp(np.sum(weights * np.log(x+1), axis=axis) / np.sum(weights, axis=axis)) - 1

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

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str) or isinstance(0, unicode):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r

def mkdir(dirs, start_dir=None, depth=None, verbose=False):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary. Alternatively, if a depth
    is supplied only the last <depth> levels of directories will be created i.e. the path <depth> levels above must
    pre-exist.
    Inputs:
        dirs 			- Directory path
        start_dir       - Path from which all new directories must branch
        depth           - Maximum levels of directories what will be created for each path in <dirs>
        verbatim = 0	- True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    from pathlib import Path
    # raise NotImplementedError('Broken!')
    if start_dir is not None:
        if isinstance(start_dir, Path):
            start_dir = str(start_dir)
        start_dir = os.path.abspath(start_dir)
        if not os.path.isdir(start_dir):
            print('Directories {} were not created as start directory {} does not exist.'.format(dirs, start_dir))
            return 1

    if isinstance(dirs, Path):
        dirs = str(dirs)
    if isinstance(dirs, basestring):  # Nest single string in list for loop
        dirs = [dirs]
    # import pdb; pdb.set_trace()
    for d in dirs:
        if isinstance(d, Path):
            d = str(d)
        d = os.path.abspath(d)
        if depth is not None:
            depth = np.abs(depth)
            d_up = d
            for i in np.arange(depth):  # walk up directory by given depth
                d_up = os.path.dirname(d_up)
            if not os.path.isdir(d_up):
                print('Directory {} was not created as start directory {} (depth={}) does not exist.'.format(
                    d, d_up, depth))
                continue
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                print('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            os.makedirs(d)
            print('Created directory: ' + d)
        else:
            if verbose:
                print('Directory "' + d + '" already exists')
    return 0

def fails(string):
    """ Return True if evaluating expression produces an error
    """
    try:
        exec(string)
    except:
        return True
    else:
        return False

def test_pickle(obj):
    """Test if an object can successfully be pickled and loaded again
    Returns True if succeeds
            False if fails
    """
    import pickle
    # sys.setrecursionlimit(10000)
    path = 'test.p.tmp'
    if os.path.isfile(path):
        os.remove(path)  # remove temp file
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print('Pickled object')
        with open(path, 'rb') as f:
            out = pickle.load(f)
        print('Loaded object')
    except Exception as e:
        print('{}'.format(e))
        return False
    if os.path.isfile(path):
        print('Pickled file size: {:g} Bytes'.format(os.path.getsize(path)))
        os.remove(path)  # remove temp file
    import pdb; pdb.set_trace()
    print('In:\n{}\nOut:\n{}'.format(out, obj))
    if not isinstance(obj, dict):
        out = out.__dict__
        obj = obj.__dict__
    if compare_dict(out, obj):
        return True
    else:
        return False

def pickle_dump(obj, path, **kwargs):
    """Wrapper for pickle.dump, accepting multiple path formats (file, string, pathlib.Path).
    - Automatically appends .p if not pressent.
    - Uses cpickle when possible.
    - Automatically closes file objects."""
    if isinstance(path, Path):
        path = str(path)

    if isinstance(path, basestring):
        if path[-2:] != '.p':
            path += '.p'
        with open(path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif isinstance(path, file):
        pickle.dump(obj, path, **kwargs)
        path.close()
    else:
        raise ValueError('Unexpected path format')

def pickle_load(path, base=None, **kwargs):
    if isinstance(path, Path):
        path = str(path)

    if base is not None:
        path = os.path.join(base, path)

    if isinstance(path, basestring):
        if path[-2:] != '.p':
            path += '.p'
        with open(path, 'rb') as f:
            try:
                out = pickle.load(f, **kwargs)
            except EOFError as e:
                logger.error('path "{}" is not a pickle file. {}'.format(path, e))
                raise e
    elif isinstance(path, file):
        out = pickle.load(path, **kwargs)
        path.close()
    else:
        raise ValueError('Unexpected path format')
    return out

def data_split(x, y=None, gap_length=3, data_length=10, av_diff=False, return_longest=False, verbose=True):
    """Split data at gaps where difference between x data points in much greater than the average/modal difference
    Return: (indices of continuous sections, continuous sections in x, (continuous sections in y))"""
    import scipy as sp
    i = np.arange(len(x))
    ## Find the average distace between the x data
    diff = np.abs(np.diff(x))  # differences between adjacent data
    av_gap = sp.stats.mode(diff) if not av_diff else np.average(diff)  # average/modal separation
    ## Get indices of begining of gaps sufficiently greater than the average
    igap = np.nonzero(diff > gap_length * av_gap)[0]+1  # nonzero nested in tuple

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
        if igap[0] - 0 >= data_length:  # Only add data if set is long enough
            isplit = np.arange(0, igap[0])  # begining of data to begining of gap
            xsplit.append(x[isplit])
            if y is not None:
                ysplit.append(y[isplit])
            isplit_all.append(isplit)
        else:
            if verbose: print('data_split: First set exluded as too short')

        ## Deal with linear data that isn't bordered by the ends of the set
        for i in np.arange(1, len(igap)):  # if start=stop, loop over empty array -> do nothing when len(ifap)=1
            ## Note: arange doesn't include stop, so len 2 just loops over i=1
            if igap[i] - igap[i - 1] + 1 >= data_length:  # Only add data if set is long enough
                isplit = np.arange(igap[i - 1] + 1, igap[i])  # end of last gap begining of next gap
                xsplit.append(x[isplit])
                if y is not None:
                    ysplit.append(y[isplit])
                isplit_all.append(isplit)
            else:
                if verbose: print('data_split: Set {} exluded as too short'.format(i))

        ## Last set of linear data after last gap
        if (len(x) - 1) - igap[-1] + 1 >= data_length:  # Only add data if set is long enough
            isplit = np.arange(igap[-1] + 1, len(x) - 1)  # end of last gap to end of data
            xsplit.append(x[isplit])
            if y is not None:
                ysplit.append(y[isplit])
            isplit_all.append(isplit)
        else:
            if verbose: print('data_split: Last set exluded as too short')

    # If return_longest is True, only return longest section of data without gaps, else return all data with gap removed
    ind = np.array([len(xi) for xi in xsplit]).argmax() if return_longest else np.arange(len(xsplit))
    if y is not None:
        return isplit_all[ind], xsplit[ind], ysplit[ind]
    else:
        return isplit_all[ind], xsplit[ind]

def exists_lookup(obj, *args):
    """"""
    tmp = copy(obj)
    try:
        for arg in args:
            tmp = tmp[arg]
        return tmp
    except (IndexError, ValueError, KeyError) as e:
        return Exception  # return exception instrad of traceback
    # except Exception as e:
    #     print('Unanticipated error "{}" in exists_lookup'.format(e))

def exists_equal(value, obj, indices):
    """Check if value == obj[indices], returning false (not exception) if indices do not exist
    Indices are recursive and can be for arrays or dicts"""
    if not isinstance(indices, tuple):
        indices = (tuple,)
    obj = exists_lookup(obj, *indices)
    return obj == value

def args_for(func, kwargs, exclude=[]):
    return {k: v for k, v in kwargs.iteritems() if (k in inspect.getargspec(func)[0]) and (k not in exclude)}

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    reference = [1, -345, 0, 23432.5]
    arr = [23432.234, -345.36, 0.0004, 4356256, -0.254, -344.9] #[345.45654, 6.4576, 0.0007562, 4.34534, 0.34534]
    print('match_in...')
    print('match_in: {} -> {}: {}'.format(arr, reference, match_in(arr, reference, tol=0.2)))

    print('nsignif: {} -> {}'.format(arr, nsigfig(arr, 3)))

    plt.figure()
    plt.subplot(2,1,1)
    line1, = plt.plot(range(10), 'ro-')
    plt.subplot(2,1,2)
    line2, = plt.plot(range(10), 'bo-')

    DataCursor([line1, line2])
    # plt.show()

    x = [1,4,7,9,12,16]
    print('moving_average:', moving_average(x, 1))
    print('moving_average:', moving_average(x, 2))
    print('moving_average:', moving_average(x, 3))
    print(find_nearest(x, 11))
    print(find_nearest(x, 11, index=False))

    a, b = [[1,2], [11,2], [2,4], [7,32], [7,1], [8,23], [6,6.5]], [[15,52], [17,23], [12,24], [17,38], [6,6], [10,23]]
    indices = find_furthest(a, b, index=True)
    print(indices)

    print(a[indices[0]], b[indices[1]])
