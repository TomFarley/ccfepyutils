#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
Utility functions used in the filament tracker program
"""
from past.builtins import basestring  # pip install future
import numbers
from pprint import pprint
try:
    from Tkinter import Tk  # python2 freia
    from tkFileDialog import askopenfilename
except:
    from tkinter import Tk  # python3 freia
    from tkinter import filedialog as askopenfilename
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd
import itertools
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from copy import copy, deepcopy
import sys
import os
import re
import inspect
from collections import Mapping, Container
from sys import getsizeof
from pathlib import Path
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger(__name__)

try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

signal_abbreviations = {
    'Ip': "amc_plasma current",
    'ne': "ESM_NE_BAR",  # (electron line averaged density)
    'ne2': "ane_density",  # Gives lower density - what is this?
    'Pnbi': "anb_tot_sum_power",  # Total NBI power
    'Pohm': "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
    'Ploss': "esm_p_loss",  # Total power crossing the separatrix
    'q95': "efm_q_95",  # q95
    'q0': "efm_q_axis",  # q0
    'q952': "EFM_Q95",  # (q95)
    'Da': "ada_dalpha integrated",
    # 'Da-mp': 'ph/s/cm2/sr',
    'sXray': 'xsx/tcam/1',
    'Bphi': 'efm_bphi_rmag',
    'zmag': "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
    'dn_edge': "ADG_density_gradient",
    'Bvac': "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
    'LPr': "arp_rp radius"  # (radial position of the reciprocating probe)
}
signal_sets = {
    'set1': [
        "amc_plasma current",
        "ESM_NE_BAR",  # (electron line averaged density)
        "ane_density",  # Gives lower density - what is this?
        "anb_tot_sum_power",  # Total NBI power
        "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
        "esm_p_loss",  # Total power crossing the separatrix
        "efm_q_95",  # q95
        "efm_q_axis",  # q0
        "EFM_Q95",  # (q95)
        "ada_dalpha integrated",
        'xsx/tcam/1',  # soft xray 1
        'efm_bphi_rmag',
        "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
        "ADG_density_gradient",
        "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
        "arp_rp radius"]   # (radial position of the reciprocating probe)
    }

def get_data(signal, pulse, save_path='~/data/MAST_signals/', save=True, *args, **kwargs):
    """Get data with IDL_bridge getdata if available, else load from pickle store."""
    pulse = int(pulse)
    if signal in signal_abbreviations:
        signal = signal_abbreviations[signal]
    save_path = os.path.expanduser(save_path)
    if save:
        pulse_str = '{pulse:d}'.format(pulse=pulse)
        fn = signal.replace('/', '_').replace(' ', '_')+'.p'
        mkdir(os.path.join(save_path, pulse_str), start_dir=save_path)
    try:
        import idlbridge as idl
        getdata = idl.export_function("getdata")
        d = getdata(signal, pulse, *args, **kwargs)
        if d['erc'] != 0:
            logger.warning('Failed to load data for {}: {}'.format(pulse_str, signal))
        elif save:
            pickle_dump(d, os.path.join(save_path, pulse_str, fn), protocol=2)
            logger.info('Saved data for {}; {} to {}'.format(pulse_str, signal, os.path.join(save_path, pulse_str, fn)))
        return d
    except ImportError:
        try:
            d = pickle_load(os.path.join(save_path, pulse_str, fn))
            return d
        except IOError:
            logger.warning('Cannot locate data for {}:{} in {}'.format(pulse_str, signal, save_path))

def store_mast_signals(signals, pulses, save_path='~/data/MAST_signals/', *args, **kwargs):
    if isinstance(signals, (str, basestring)) and signals in signal_sets:
        signals = signal_sets[signals]
    pulses = make_itterable(pulses)
    signals = make_itterable(signals)
    save_path = os.path.expanduser(save_path)
    assert os.path.isdir(save_path), 'Save path does not exist'
    for pulse in pulses:
        for signal in signals:
            get_data(signal, pulse, save_path=save_path, noecho=1, *args, **kwargs)


def rm_files(path, pattern, verbose=True):
    path = str(path)
    if verbose:
        print('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
    for fn in os.listdir(path):
        if re.search(pattern, fn):
            os.remove(os.path.join(path, fn))
            if verbose:
                print('Deleted file: {}'.format(fn))

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

def sub_range(array, limits, indices=False):
    """Return subset of array that falls within limits

    Arguments:
    array - data to be filtered
    limits - array [min, max] to filter array with
    """
    if limits is None:
        if not indices:
            return array
        else:
            return np.where(array == array)
    assert len(limits) == 2
    ind = np.where(np.logical_and(array >= limits[0], array <= limits[1]))[0]
    if not indices:
        return array[ind]
    else:
        return ind

def extract_intervals(array, intervals, indices=False, concatenate=True):
    out = []
    for interval in intervals:
        out.append(sub_range(array, interval, indices=indices))
    out = np.array(out)
    if concatenate:
        out = np.concatenate(out)
    return out

def replace_in(values, reference, index=False, tol=1e-10):
    """Replace values with closest reference values if within tolerance"""
    reference = to_array(reference)
    def closest_index(value):
        diff = np.isclose(value, reference, atol=tol, rtol=1e-20)
        if np.any(diff):  # get value or index of closest value
            out = reference[diff][0] if not index else np.nonzero(diff)[0][0]
        else:  # Nan if no close value
            out = np.nan
        return out
    vfunc = np.vectorize(closest_index)  # apply to one value at a time
    out = vfunc(to_array(values))
    if is_number(values):
        out = out[0]
    return out

def make_itterable(obj, ndarray=False):
    """If object is a scalar nest it in a list so it can be iterated over
    If ndarray is True, the object will be returned as an array (note avoids scalar ndarrays)"""
    if not hasattr(obj, '__iter__') or isinstance(obj, basestring):
        obj = [obj]
    if ndarray:
        obj = np.array(obj)
    return obj

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
    except TypeError as e:  # eg trying to convert an array
        return False
    return True

def safe_len(var, scalar=1):
    """ Length of variable returning 1 instead of type error for scalars """
    if is_scalar(var): # checks if has atribute __len__
        return scalar
    elif len(np.array(var) == np.nan) == 1 and np.all(np.array(var) == np.nan):
        # If value is [Nan] return zero length # TODO: change to catch [Nan, ..., Nan] ?
        return 0
    else:
        return len(var)

def safe_zip(*args):
    """Return zip iterator even if supplied with scaler values"""
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = make_itterable(arg)
    return zip(*args)

def any_equal(object, list):
    """ Return true if object is equal to any of the elements of list
    """
    return np.any([object == l for l in list])

def to_list(obj):
    """Convert to list, nesting if nessesary"""
    if isinstance(obj, list):
        out = obj
    elif is_scalar(obj):
        out = (obj,)  # number to [num] etc
    else:
        out = list(obj)  # convert tuple, ndarray etc
    return out

def to_array(obj, silent=True):
    """Return object as an itterable ndarray"""
    if isinstance(obj, np.ndarray):
        return obj

    try:
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            obj = obj.values
    except ImportError:
        pass
    if isinstance(obj, (list, tuple)):
        obj = np.array(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = np.array([obj])  # make it indexable
    elif is_number(obj):  # Equivalent to make_itterable
        obj = np.array([obj])
    else:
        try:
            obj = np.array([obj])
        except:
            if not silent:
                print('Could not convert {} to a np.ndarray'.format())
    return obj

def to_arrays(*args, **kwargs):
    """Convert arbitrary number of args to np.ndarrays"""
    out = []
    for arg in args:
        out.append(to_array(arg, **kwargs))
    return tuple(out)

def describe_array(array):
    """Return string containing n elements, mean, std, min, max of array"""
    array = np.array(array).flatten()
    df = pd.DataFrame({'values': array})
    return df.describe()

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

def isclose_within(values, reference, tol=1e-8, all=False):
    """Return true if elements of a appear in comparison_list (b) within tollerance
    From: http://stackoverflow.com/questions/39602004/can-i-use-pandas-dataframe-isin-with-a-numeric-tolerance-parameter
    """
    # import pandas as pd
    values = to_array(values)
    reference = to_array(reference)
    values = np.expand_dims(values, axis=1)  # add dimension to compare pairwise
    out = np.isclose(values, reference, atol=tol).any(axis=1)
    if all is False:
        return out
    else:
        return np.all(out)

def equal_string(obj, string):
    """Return True if obj is equal to string"""
    if isinstance(string, (tuple, list)):  # Check agaist list of strings for match
        return np.any([equal_string(obj, s) for s in string])
    if isinstance(obj, (str, basestring)) and obj == string:
        return True
    else:
        return False

# def assert_raises(func)

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

def pos_path(value):
    """Return True if value is a potential file path else False"""
    if not isinstance(value, string_types):
        return False
    value = os.path.expanduser(value)
    path, fn = os.path.split(value)
    if os.path.isdir(path):
        return True
    else:
        return False

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


def printProgress(iteration, total, prefix='', suffix='', frac=False, t0=None,
                  decimals=2, nth_loop=2, barLength=50):
    """
    from http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration starting at 0 (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    # TODO: convert to class with __call__ (print 0% on __init__) - add to timeline class
    # TODO: Change/add nth_loop to min time between updates
    # TODO: Add compatibility for logger handlers
    # TODO: Make bar optional
    if (iteration % nth_loop != 0) and (
            iteration != total - 1):  # Only print every nth loop to reduce slowdown from printing
        return
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '|' * filledLength + '-' * (barLength - filledLength)
    frac = '{}/{} '.format(iteration, total) if frac else ''
    if t0 is None:
        time = ''
    else:
        t1 = datetime.now()
        t_diff_past = relativedelta(t1, t0)  # time past in loop
        mul = float(total - iteration) / iteration if iteration > 0 else 0
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

def mkdir(dirs, start_dir=None, depth=None, info=None, verbose=False):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary. Alternatively, if a depth
    is supplied only the last <depth> levels of directories will be created i.e. the path <depth> levels above must
    pre-exist.
    Inputs:
        dirs 			- Directory path
        start_dir       - Path from which all new directories must branch
        depth           - Maximum levels of directories what will be created for each path in <dirs>
        info            - String to write to DIR_INFO.txt file detailing purpose of directory etc
        verbatim = 0	- True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    from pathlib import Path
    # raise NotImplementedError('Broken!')
    if start_dir is not None:
        start_dir = os.path.expanduser(str(start_dir))
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
            if info:  # Write file describing purpose of directory etc
                with open(os.path.join(d, 'DIR_INFO.txt'), 'w') as f:
                    f.write(info)
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

def to_csv(x, ys, fn, xheading=None, yheadings=None, description='data'):
    """Quickly and easily save data to csv, with one dependent variable"""
    import pandas as pd
    ys = np.squeeze(ys)
    if ys.shape[0] != len(x) and ys.shape[1] == len(x):
        ys = ys.T
    df = pd.DataFrame(data=ys, index=make_itterable(x), columns=yheadings)
    if xheading:
        df.index.name = xheading
    # df.columns.name = 'tor'
    df.to_csv(str(fn))
    logger.info('Saved {} to: {}'.format(description, str(fn)))

def data_split(x, y=None, gap_length=3, data_length=10, av_diff=False, return_longest=False, verbose=True):
    """Split data at gaps where difference between x data points in much greater than the average/modal difference
    Return: (indices of continuous sections, continuous sections in x, (continuous sections in y))"""
    # TODO: Add option to supply absolute value of min gap size
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
    ## No gap => 1 linear section, 1 gap => 2 linear sections, 2 gaps => 3 linear sections etc.
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

def args_for(func, kwargs, include=(), exclude=(), match_signature=True, named_dict=True, remove=True):
    """Return filtered dict of args from kwargs that match input for func.
    :param - Effectively filters kwargs to return those arguments
    :param - func            - function(s) to provide compatible arguments for
    :param - kwargs          - list of kwargs to filter for supplied function
    :param - exclude         - list of kwargs to exclude from filtering
    :param - match_signature - apply filtering to kwargs based on func call signature
    :param - named_dict      - if kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
    :param - remove          - remove filtered kwargs from original kwargs
    """
    func = make_itterable(func)  # Nest lone function in list for itteration
    kws = {}
    keep = []
    name_args = []
    for f in func:
        keep += inspect.getargspec(f)[0]  # Add arguments for each function to list of arguments to keep
        name_args += '{name}_args'.format(name=f.__name__)
    if match_signature:
        matches = {k: v for k, v in kwargs.items() if (((k in keep) and (k not in exclude)) or (k in include))}
        kws.update(matches)
    if named_dict:  # Look for arguments <function>_args={dict of keyword arguments}
        keep_names = {k: v for k, v in kwargs.items() if (k in name_args)}
        kws.update(keep_names)
    if remove:  # Remove key value pairs from kwargs that were transferred to kws
        for key in kws:
            kwargs.pop(key)
    return kws

def caller_details(level=1):
    """Return (func_name, args, kwargs) of function that called this function"""
    inspect
    raise NotImplementedError

def call_with_kwargs(func, kwargs, exclude=[], match_signature=True, named_dict=True, remove=True, *args):
    """Return output of func called with dict of args from kwargs that match input for func.
    Effectively filters kwargs to return those arguments
    func            - function to provide compatible arguments for
    kwargs          - list of kwargs to filter for supplied function
    exclude         - list of kwargs to exclude from filtering
    match_signature - apply filtering to kwargs based on func call signature
    named_dict      - if kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
    """
    kws = args_for(func, kwargs, exclude=exclude, match_signature=match_signature, named_dict=named_dict)
    output = func(*args, **kws)
    return output

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

def out_path(input, default_path, default_fn, default_extension, path_obj=False):
    """Generate output path given input and variable input.

    Any information missing in input with be replaced with defaults.
    Input can be eg:
    Full path including filename and extension
    Just path, no filename
    Just filename, no path
    Not a string/Path, in which case defaults will be used """
    default_path = os.path.expanduser(default_path)
    try:
        if isinstance(input, Path):
            input = str(input)
        input = os.path.expanduser(input)
        if os.path.isdir(input) and input[-1] != os.sep:  # Make sure a directory ends in slash
            input += os.sep
        # Split up input information
        path, fn = os.path.split(input)
        base, ext = os.path.splitext(fn)
        if len(ext) > 5:  # Avoid full stops in filenames being treated as extensions - 4 char ext len limit
            base += ext
            ext = ''

        # Replace missing information with defaults
        if not path:
            path = default_path
        if not fn:
            fn = os.path.splitext(default_fn)[0]
        if not ext:
            ext = default_extension
        fn = join_with_one('.', fn, ext)
        out = os.path.join(path, fn)

    except AttributeError as e:
        # TODO: allow no extension and extension included in default_fn
        fn = join_with_one('.', default_fn, default_extension)
        out = os.path.join(default_path, fn)
    if path_obj:
        out = Path(out)
    return out

def join_with_one(sep, *args):
    """Join strings with exactly one separator"""
    l = len(sep)
    out = ''
    for i, (arg1, arg2) in enumerate(zip(args, args[1:])):
        arg12 = [arg1, arg2]
        if i == 0:
            # If arg1 ends with separator, remove it
            while arg12[0].endswith(sep):
                arg12[0] = arg12[0][:len(arg12[0])-l]
            out += arg12[0]
        # check if arg2 begins or ends with separator (need to loop over mutable!)
        while arg12[1].startswith(sep):
            arg12[1] = arg12[1][l:]
        while arg12[1].endswith(sep):
            arg12[1] = arg12[1][:len(arg12[1])-l]
        # add to output
        out += sep+arg12[1]
    return out

def python_path(filter=None):
    import os
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        print('PYTHONPATH not set')
        user_paths = []
    if filter:
        filter = os.path.expanduser(path)
        user_paths = [p for p in user_paths if filter in p]
    return user_paths

def locate_file(fn, paths, _raise=True):
    """Return path to file given number of possible paths"""
    for path in paths:
        path = os.path.expanduser(str(path))
        out = os.path.join(path, fn)
        if os.path.isfile(out):
            return out
    if _raise:
        raise IOError('File "{}" is not present in any of the following direcgtories: {}'.format(fn, paths))
    else:
        return None

def return_none():
    return None

def none_filter(old, new):
    """Return new values, replacing None values with corresponding old values"""
    nest = False
    if not (type(old)==type(new) and isinstance(old, (tuple, list))):  # if not both tuple or list, nest in list
        old, new = [old], [new]
        nest = True
    for i, (o, n) in enumerate(zip(old, new)):
        if n is not None:
            old[i] = n
    if nest:
        old = old[0]
    return old

def fwhm2sigma(values):
    fwhm = 2*np.sqrt(2*np.log(2))
    return values / fwhm

def sigma2fwhm(values):
    fwhm = 2*np.sqrt(2*np.log(2))
    return values * fwhm

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    reference = [1, -345, 0, 23432.5]
    arr = [23432.234, -345.36, 0.0004, 4356256, -0.254, -344.9] #[345.45654, 6.4576, 0.0007562, 4.34534, 0.34534]
    print('match_in...')
    print('match_in: {} -> {}: {}'.format(arr, reference, replace_in(arr, reference, tol=0.2)))
    print('isclose_withini: {} -> {}: {}'.format(arr, reference, isclose_within(arr, reference, tol=0.2)))

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
    print('Arrays:')
    pprint(to_arrays(x, a, b, arr))

