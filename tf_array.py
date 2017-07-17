#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_array.py: Frequently used array operations and wrappers.

Detailed description:

Notes:
    @bug:

Todo:
    @todo: sub_arr(arr, between=[start, end]) using extract

Info:
    @since: 17-06-14
"""

import numpy as np                  # Maths library
import matplotlib.pyplot as plt     # Plotting library

from scipy.optimize import curve_fit                # Curve fitting
from scipy.signal import find_peaks_cwt, argrelmax  # Peak finding
from scipy.interpolate import interp1d              # Interpolation

import itertools    #
# import os           # System directory/file opperations
# import shutil       # High-level file operations
# import re           # Regular expressions

from pprint import pprint   # Pretty printing

## CAN import: tf_debug
## CANNOT import: tf_string
from tf_libs.tf_debug import Debug
import tf_libs.tf_numeric as tf_numeric
# from . import tf_debug.debug_print as dprint
# from . import tf_debug
# from . import tf_string
# import tf_string

db = Debug(1,1,0)

def argsort(seq, reverse=False):
    """ Return indices of sorted (assernding) list """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def check_array(arr, ndarray=True, nest_depth=0, verbatim = False):
    """Check that the supplied array is as expected"""
    if verbatim: print('In: ', type(arr), arr)

    ## Remove nesting lists
    if nest_depth==0:
        while (type(arr) == list) and (len(arr)==1):
            arr = arr[0]

    assert (type(arr) == list or np.ndarray), arr
    assert type(arr[0]) != list, arr

    if ndarray: # Make sure arr is an ndarray
        # if type(arr) == list:
        #     arr = list(itertools.chain(*arr))
        arr = np.array(arr)

    if verbatim:
        print('Out: ', type(arr), arr)

    return arr


def sub_arr(array, lim, con_array = None, min=None, max=None, boundaries=True):
    """Purpose: Extract sub array of values between min and max limits
    arguements:
     array          var     array to take subset of
     lim            var     array containing [min, max]
     boundaries     bool    include boundaries ie <= and >=
    keywords:
     con_array      var     condition array to apply min/max check on
    Outputs:
     array of values in array with indices satisfying min < con_array < max
    Call example: 
     function()

    TODO: implement only using max or min
    """
    array = check_array(array) # check array is a numpy array

    assert lim[1] >= lim[0], 'min > max'

    if con_array == None: # If no separate array supplied use same array for min/max
        con_array = array
    else: 
        assert np.size(con_array) != np.size(array), 'WARNING: size(con_array) != size(array)'

    if boundaries == True:
        sub = np.extract( (con_array>=lim[0]) * (con_array<=lim[1]), array)
    else:
        sub = np.extract( (con_array>lim[0]) * (con_array<lim[1]), array)
    return sub

def extract_2D(arr1, arr2, condition):
    """ Return elements of two arrays where indices match condition """
    inds = np.nonzero(condition)
    return arr1[inds], arr2[inds]

def arr_range(array, var_name=False):
    """ Return numeric range of array as two element array """
    range = np.array([min(array), max(array)])
    if var_name:
        print(var_name, 'range:', range)
    return range

def arr_nearest(array, value, output = 'value', side = 'both', next=0):
    """Element in nd array closest to the scalar value
    Use 'next' to return next nearest values
    @Todo: add nest functionality """
    if side == 'both':
        idx = np.abs(array - value).argmin()
    elif side =='above':
        idx = (array - value)
        idx = np.abs(np.extract(idx>0, idx)).argmin()
    elif side == 'below':
        idx = (array - value)
        idx = np.abs(np.extract(idx<0, idx)).argmin()
    else:
        print('arr_nearest: Invalid side argument')

    if (output == 'value') or (output == 'v'):
        return array.flat[idx]
    elif (output == 'index') or (output == 'i'):
        return idx
    else:
        print('arr_nearest: Invalid output argument:', output)
        print("\t Accepted arguments: 'value', 'v', 'index', 'i'")

def closest_max(x, y, x0, order = 3, output = 'value'):
    """ x value/index of max y value closest to x0 
    Requires arr_nearest 
    """
    imax = argrelmax(y, order = order)[0] # Extract maximum in dI
    i0 = arr_nearest(x, x0, output='index')
    iclose =  arr_nearest(imax, i0, output='value')
    x1 = imax
    
    if (output == 'value') or (output == 'v'):
        return x[iclose]
    elif (output == 'index') or (output == 'i'):
        return iclose
    else:
        print('arr_nearest: Invalid output arguement:', output)
        print("\t Accepted arguments: 'value', 'v', 'index', 'i'")





def tup0(obj):
    """ If obj is a tuple return its first element, else return obj unchanged """
    if type(obj) == tuple:
        return obj[0]
    else:
        return obj


if __name__ == "__main__":
    from test.run_test import test_tf_array
    test_tf_array()




    # x = np.linspace(0,10,101)
    # y = np.linspace(10,30,101)
    #
    # print("arr_range(x, var_name=False) = ", end=' ')
    # print(arr_range(x, var_name=False))
    # print()
    #
    # lim = [2,3.4]
    # print("sub_arr(x, lim, con_array = None, min=None, max=None, boundaries=True) = ", end=' ')
    # print(sub_arr(x, lim, con_array = None, min=None, max=None, boundaries=True))
    # print()
    #
    # print("arr_nearest(x, 2.65467, output = 'value', side = 'both', next=0) = ", end=' ')
    # print(arr_nearest(x, 2.65467, output = 'value', side = 'both', next=0))
    # pass

