#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from datetime import datetime

from ccfepyutils.data_processing import moving_average, find_nearest, find_furthest

"""
Utility functions used in the filament tracker program
"""
from past.builtins import basestring  # pip install future
from pprint import pprint
import string
try:
    from Tkinter import Tk  # python2 freia
    from tkFileDialog import askopenfilename
except:
    from tkinter import Tk  # python3 freia
    from tkinter import filedialog as askopenfilename
import numpy as np
import pandas as pd
from copy import copy
import sys, os, inspect
from collections import Mapping, Container
from sys import getsizeof

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

def make_itterables(*args):
    """Convert multiple input arguments to itterables"""
    # TODO: make compatible with ndarray
    out = []
    for obj in args:
        if not hasattr(obj, '__iter__') or isinstance(obj, basestring):
            obj = [obj]
        # if ndarray:
        #     obj = np.array(obj)
        out.append(obj)
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
    # from numbers import
    try:
        n=str(float(s))
        if n == "nan" or n=="inf" or n=="-inf" :
            return False
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

def safe_arange(start, stop, step):
    """Return array of elements between start and stop, each separated by step.

    Replacement for np.arange that always includes stop.
    Normally np.arange should not include stop, but due to floating point precision sometimes it does, so output is
    unpredictable"""
    n = np.abs(stop - start) / step
    if np.isclose(n, np.round(n)):
        # If n only differs from an integer by floating point precision, round it
        n = int(np.round(n))+1
    else:
        # If n is not approximately an integer, floor it
        n = int(np.floor(n))+1
        stop = start + (n-1)*step
    out = np.linspace(start, stop, n)
    return out

def any_equal(object, list):
    """ Return true if object is equal to any of the elements of list
    """
    return np.any([object == l for l in list])

def is_subset(subset, full_set):
    """Return True if all elements of subset are in fullset"""
    return set(subset).issubset(set(full_set))

def is_in(items, collection):
    items = make_itterable(items)
    collection = make_itterable(collection)
    out = pd.Series(items).isin(collection).values
    return out

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
        if obj.ndim > 0:
            return obj
        else:
            # 0D array - convert to singe element 1D array
            return np.array([float(obj)])

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

def remove_duplicates_from_list(seq):
    """Remove duplicates from list while preserving order (Sets don't preserve order)
    From https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def describe_array(array):
    """Return string containing n elements, mean, std, min, max of array"""
    array = np.array(array).flatten()
    df = pd.DataFrame({'values': array})
    return df.describe()


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
    from matplotlib.widgets import RectangleSelector
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


def printProgress(iteration, total, prefix='', suffix='', frac=False, t0=None,
                  decimals=2, nth_loop=2, barLength=50):
    """
    Based on http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call at start of a loop to create terminal progress bar
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
        if isinstance(t0, float):
            # Convert float time from time.time() (seconds since the Epoch) to datetime
            t0 = datetime.fromtimestamp(t0)
        t1 = datetime.now()
        t_diff_past = relativedelta(t1, t0)  # time past in loop
        mul = float(total - iteration) / iteration if iteration > 0 else 0
        t_diff_rem = t_diff_past * mul  # estimate of remaining time
        t_diff_past = '({h}h {m}m {s}s)'.format(h=t_diff_past.hours, m=t_diff_past.minutes, s=t_diff_past.seconds)
        if t_diff_rem.hours > 0:  # If expected to take over an hour display date and time of completion
            t_diff_rem = (datetime.now() + t_diff_rem).strftime("(%d/%m/%y %H:%M)")
        else:  # Display expected time remaining
            t_diff_rem = '({h}h {m}m {s}s)'.format(h=t_diff_rem.hours, m=t_diff_rem.minutes, s=t_diff_rem.seconds)
        if mul == 0:
            t_diff_rem = '?h ?m ?s'
        time = ' {past} -> {remain}'.format(past=t_diff_past, remain=t_diff_rem)

    sys.stdout.write('\r %s |%s| %s%s%s%s %s' % (prefix, bar, frac, percents, '%', time, suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def in_ellipse(point, centre, rx, ry, angle=0, boundary=True, return_r=False):
    """ Return true if point is within ellipse with centre at centre and semi-major and semi-minor axes rx and ry
    NOTE: rx and ry are radii not diameters
    Equation for rotated ellipse from:
    https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
    """
    assert len(point) == 2
    assert len(centre) == 2

    x, y = point[0], point[1]
    x0, y0 = centre[0], centre[1]
    dx, dy = x-x0, y-y0
    sin, cos = np.sin(angle), np.cos(angle)

    r = ((cos*dx+sin*dy)/rx)**2 + ((sin*dx-cos*dy)/ry)**2
    # r = ((point[0]-centre[0])/rx)**2 + ((point[1]-centre[1])/ry)**2

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


def fails(string):
    """ Return True if evaluating expression produces an error
    """
    try:
        exec(string)
    except:
        return True
    else:
        return False


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
        # Add arguments for each function to list of arguments to keep
        if isinstance(f, type):
            # If a class look at it's __init__ method
            keep += inspect.getargspec(f.__init__)[0]
        else:
            keep += inspect.getargspec(f)[0]
        name_args += ['{name}_args'.format(name=f.__name__)]
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

def positional_args(func):
    """From https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-python-function-receives"""
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    return args   # *args and **kwargs are not required, so ignore them.

def missing_args(func, argdict):
    """Tell what you are missing from your particular dict"""
    return set(positional_args(func)).difference(argdict)

def invalid_args(func, argdict):
    """Check for invalid args, use:"""
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if varkw: return set()  # All accepted
    return set(argdict) - set(args)

def isCallableWithArgs(func, argdict):
    return not missing_args(func, argdict) and not invalid_args(func, argdict)

def caller_details(level=1):
    """Return (func_name, args, kwargs) of function that called this function"""
    inspect
    raise NotImplementedError

def in_freia_batch_mode():
    """Return True if current python interpreter is being run as a batch job (ie no display for plotting etc)"""
    batch_mode = os.getenv('LOADL_ACTIVE', None)
    return batch_mode == 'yes'

def set_windowless_matplotlib_backend():
    try:
        import matplotlib
        matplotlib.use('Agg')
        logger.debug('Set matplotlib backend to "Agg" for batch mode')
    except Exception:
        logger.warning('Failed to switch matplotlib backend to Agg')

def get_methods_class(meth):
    """Get class that defined method
    Taken from:
    stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545"""
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects

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

def return_none():
    return None

def none_filter(old, new, nest=None):
    """Return new values, replacing None values with corresponding old values"""
    # If necessary nest values in list so can iterate over them
    if nest is None:
        # If either old or new is not a tuple or list nest them
        # NOTE: this will fail if single values are passed which are naturally lists or tuples - then use keyword
        nest = (not isinstance(old, (tuple, list))) or (not isinstance(new, (tuple, list)))
    # old, new = make_itterables(old, new)
    if nest:  # and (not (type(old) == type(new)))  # if not both tuple or list, nest in list
        old, new = [old], [new]
    # Make sure old is mutable
    if isinstance(old, tuple):
        old = list(old)
    for i, (o, n) in enumerate(zip(old, new)):
        if n is not None:
            old[i] = n
    if nest:
        old = old[0]
    return old

def class_name(obj):
    import re
    out = re.search(".*\.(\w+)\'", str(obj.__class__)).group(1)
    return out

def fwhm2sigma(values):
    fwhm = 2*np.sqrt(2*np.log(2))
    return values / fwhm

def sigma2fwhm(values):
    fwhm = 2*np.sqrt(2*np.log(2))
    return values * fwhm

def lookup_from_dataframe(df, col, **kwargs):
    """Lookup/extract column value based on other column in a dataframe"""
    if len(kwargs) == 0:
        raise ValueError('No input column values passed to lookup')
    # If requested column is same as input just return input
    if list(kwargs.keys()) == [col]:
        return kwargs[col]
    elif not isinstance(df, pd.DataFrame):
        raise ValueError('Cannot lookup {} for {}. No dataframe passed: df={}.'.format(output, kwargs, df))
    assert col in df.columns
    # Mask of values that satisfy input value criteria
    mask = np.ones(len(df)).astype(bool)
    # TODO: Loop for multiple values per key
    for key, value, in kwargs.items():
        if key not in df.columns:
            raise ValueError('inp="{}" is not a valid column. Options: {}'.format(key, df.columns))
        # Allow negative indexing with 'i'
        if key == 'i' and value < 0:
            value = df[key].values[value]
        values = df[key].values
        if value not in values.astype(type(value)):
            raise ValueError('value {} is not a valid "{}" value. Options: {}'.format(value, key, df[key]))
        value = make_itterable(value)
        mask *= df[key].isin(value)
    new_value = df.loc[mask, col].values
    if len(new_value) == 1:
        new_value = new_value[0]
    return new_value

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    safe_arange(1.32, 1.46, 0.003)


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


def datetime2str(time, format="%y%m%d%H%M%S"):
    string = time.strftime(format)
    return string


def str2datetime(string, format="%y%m%d%H%M%S"):
    time = datetime.strptime(string, format)
    return time


def convert_str_datetime_format(string, format1="%y%m%d%H%M%S", format2="%H:%M:%S %d/%m/%y"):
    time = str2datetime(string, format1)
    string = datetime2str(time, format2)
    return string


def t_now_str(format="compressed", dl=''):
    if format == 'compressed':
        format="%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S"
    elif format == 'natural':
        format="%H:%M:%S %d/%m/%y"
    format = format.format(dl=dl)
    string = datetime2str(datetime.now(), format=format)
    return string


class PartialFormatter(string.Formatter):
    def __init__(self):
        pass

    def get_field(self, field_name, args, kwargs):
        # Handle a key not found
        try:
            val = super(PartialFormatter, self).get_field(field_name, args, kwargs)
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val = ([field_name], field_name)
        return val

    def format_field(self, value, spec):
        # handle an invalid format
        if isinstance(value, list):
            return '{' + '{value}:{spec}'.format(value=value[0], spec=spec) + '}'
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None: return self.bad_fmt
            else: raise