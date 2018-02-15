#!/usr/bin/env python

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

__version__ = "1.0.1"
__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"


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


def is_between(vals, lims=(0,1), boundaries=True):
    """ Return true if all elements of val are numerically between lims """

    if safe_len(lims) == 1 and is_number(len): # if lims is a number, check for equality
        lims = [lims,lims]

    if lims[0] > lims[1]: #swap limits to ascending
        lims = [lims[1],lims[0]]

    vals = np.array(vals) # make sure vals is a np array

    if boundaries == True:
        return (vals >= lims[0]).all() and (vals <= lims[1]).all()
    else:
        return (vals > lims[0]).all() and (vals < lims[1]).all()

def of_types(obj, types=(list, tuple), length=-1):
    """ Return true if obj is one of types and has length length. Lengths can be a tuple of lower and upper limits """

    if length == -1: # just check obj is one of types - ignore length
        return isinstance(obj, types)
    else:
        if len(length) == 1:
            length = [int(length),int(length)]

        assert len(length) == 2, 'length has too many elemens'

        return isinstance(obj, (tuple,list)) and is_between(len(obj),length)

def make_iter(obj):
    """ In order to itterate over an object which may be a single item or a tuple of items nest a single item in a
    tuple """

    if hasattr(obj, '__iter__'):
        return obj
    else:
        return (obj,)

def make_lt(obj):
    """ Make list or tuple: In order to itterate over an object which may be a single item or a list/tuple of items
    nest a
    single item in a
    tuple """
    if of_types(obj, types=(list, tuple), length=-1):
        return obj
    else:
        return (obj,)