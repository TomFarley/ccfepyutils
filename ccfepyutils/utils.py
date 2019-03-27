#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
Utility functions used in the filament tracker program
"""
import time
from past.builtins import basestring  # pip install future
from pprint import pprint
import string, re
from datetime import datetime
# try:
#     from Tkinter import Tk  # python2 freia
#     from tkFileDialog import askopenfilename
# except:
#     from tkinter import Tk  # python3 freia
#     from tkinter import filedialog as askopenfilename
import numpy as np
import pandas as pd
from matplotlib import cbook
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
logger.setLevel(logging.DEBUG)

try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

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

def sub_range(array, limits, indices=False, include_limits=True):
    """Return subset of array that falls within limits

    Arguments:
    array - data to be filtered
    limits - array [min, max] to filter array with
    include_limits - whether to include min, max limits in returned range
    """
    if limits is None:
        if not indices:
            return array
        else:
            return np.where(array == array)
    assert len(limits) == 2
    if include_limits:
        ind = np.where(np.logical_and(array >= limits[0], array <= limits[1]))[0]
    else:
        ind = np.where(np.logical_and(array > limits[0], array < limits[1]))[0]
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
        if tol is not None:
            diff = np.isclose(value, reference, atol=tol, rtol=1e-20)
            if np.sum(diff) == 1:
                out = reference[diff][0] if not index else np.nonzero(diff)[0][0]
                return out
            elif not np.any(diff):  # Nan if no close value
                out = np.nan
                return out
        # Find closest of several matches
        ind = np.argmin(np.abs(reference-value))
        out = ind if index else reference[ind]
        return out
    vfunc = np.vectorize(closest_index)  # apply to one value at a time
    out = vfunc(to_array(values))
    if is_number(values):
        out = out[0]
    return out

def make_iterable(obj, ndarray=False, cast_to=None, cast_dict=None, nest_types=None):
    """If object is a scalar nest it in a list so it can be iterated over
    If ndarray is True, the object will be returned as an array (note avoids scalar ndarrays)
    :param cast_to - output will be cast to this type
    :param cast_dict - dict linking input types to the types they should be cast to
    :param nest_types - iterable types that should still be nested (eg dict)"""
    if not hasattr(obj, '__iter__') or isinstance(obj, basestring):
        obj = [obj]
    if (nest_types is not None) and isinstance(obj, nest_types):
        obj = [obj]
    if (cast_dict is not None) and (type(obj) in cast_dict):
        obj = cast_dict[type(obj)](obj)
    if ndarray:
        obj = np.array(obj)
    if isinstance(cast_to, type):
        if cast_to == np.ndarray:
            obj = np.array(obj)
        else:
            obj = cast_to(obj)  # cast to new type eg list
    return obj

def make_iterables(*args):
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

def is_scalar(var, ndarray_0d=True):
    """ True if variable is scalar or string"""
    if isinstance(var, str):
        return True
    elif hasattr(var, "__len__"):
        return False
    elif isinstance(var, np.ndarray) and var.ndim == 0:
        return ndarray_0d
    else:
        return True

def is_number(s, cast_string=False):
    """
    TODO: Test on numbers and strings and arrays
    """
    # from numbers import
    if (not cast_string) and isinstance(s, string_types):
        return False
    try:
        n=str(float(s))
        if n == "nan" or n=="inf" or n=="-inf" :
            return False
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False
    except TypeError as e:  # eg trying to convert an array
        return False
    return True

def is_numeric(value):
    """Return True if value is a number or numeric array object, else False"""
    if isinstance(value, bool):
        numeric = False
    else:
        try:
            sum_values = np.sum(value)
            numeric = is_number(sum_values, cast_string=False)
        except TypeError as e:
            numeric = False
    return numeric

def str_to_number(string, cast=None, expect_numeric=False):
    """ Convert string to int if integer, else float. If cannot be converted to number just return original string
    :param string: string to convert number
    :param cast: type to cast output to eg always float
    :return: number
    """
    if isinstance(string, (int, float)):
        # leave unchanged
        return string
    if isinstance(string, str) and ('_' in string):
        # Do not convert strings with underscores and digits to numbers
        out = string
    else:
        try:
            out = int(string)
        except ValueError as e:
            try:
                out = float(string)
            except ValueError as e:
                out = string
    if isinstance(cast, type):
        out = cast(out)
    if not isinstance(out, (int, float)) and expect_numeric:
        raise ValueError('Input {string} could not be converted to a number'.format(string))
    return out

def ndarray_0d_to_scalar(array):
    out = array
    if isinstance(array, np.ndarray) and array.ndim == 0:
        out = array.item()
        # dtype = array.dtype
        # if 'float' in str(dtype):
        #     out = float(array)
        # elif 'int' in str(dtype):
        #     out = int(array)
        # elif '<U' in str(dtype):
        #     out = str(array)
        # elif dtype == object:
        #     raise NotImplementedError
        # else:
        #     raise NotImplementedError(array.dtype)
    return out

def equivalent_numpy_python_type(value, to_type='python', none_type=type(None),
                                 datetime_type=pd._libs.tslibs.timestamps.Timestamp, raise_on_other=False):
    """Return equivalent numpy/python data datatype for given type"""
    value_type = value if isinstance(value, type) else type(value)
    # type_dict = {np.str_: str, np.bool_: bool, np.integer: int, np.floating: float, type(None): none_type}
    type_dict = {str: np.str_, bool: np.bool_, int: np.integer, float: np.floating,
                 type(None): none_type, datetime: datetime_type}

    for python_type, numpy_type in type_dict.items():
        if issubclass(value_type, (python_type, numpy_type)):
            break
    else:
        if raise_on_other:
            raise NotImplementedError(f'Type {value}')
        else:
            # Return type unchanged
            return value_type
    if to_type == 'numpy':
        out_type = numpy_type
    elif to_type == 'python':
        out_type = python_type
    else:
        raise ValueError(numpy_type)
    return out_type

def cast_numpy_python(value, to_type='python'):
    """Cast to equivalent numpy/python data datatype for given input type"""
    new_type = equivalent_numpy_python_type(value, to_type=to_type)
    new_value = new_type(value)
    return new_value

def input_timeout(prompt='Input: ', timeout=1, raise_on_timeout=False, yes_no=False, default_yes=True):
    import sys, select, time
    # print(prompt, end='')
    print(prompt)
    t0 = time.time()
    i, o, e = select.select([sys.stdin], [], [], timeout*60)
    t1 = time.time()
    if (t1-t0) / 60 > timeout:
        message = 'Input timed out after {} mins'.format(timeout)
        print(message)
        if raise_on_timeout:
            raise IOError(message)
    if yes_no:
        if i.lower() in ('y', 'yes'):
            i = True
        elif i.lower() in ('n', 'no'):
            i = False
        elif i == '':
            i = default_yes
        else:
            raise ValueError('Input "{}" not recognised yes/no option'.format(i))
    return i

def safe_len(var, scalar=1, all_nan=0, none=0, ndarray_0d=0, exclude_nans=False, **kwargs):
    """ Length of variable returning 1 instead of type error for scalars """
    # logger.debug(var)
    if var is None:
        return none
    elif isinstance(var, np.ndarray) and var.ndim == 0:
        return ndarray_0d
    elif is_scalar(var):  # checks if has atribute __len__ etc
        return scalar
    elif kwargs and var.__class__.__name__ in kwargs:
        return kwargs[var.__class__.__name__]
    else:
        assert hasattr(var, '__len__')
        try:
            if (len(np.array(var)) == np.sum(np.isnan(np.array(var)))):
                # If value is [Nan, Nan, ...] return zero length
                return all_nan
        except TypeError as e:
            pass
        if exclude_nans:
            var = np.array(var)
            nan_mask = np.isnan(var)
            return len(var[~nan_mask])
        else:
            return len(var)

def safe_zip(*args):
    """Return zip iterator even if supplied with scaler values"""
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = make_iterable(arg)
    return zip(*args)

def safe_arange(start, stop, step):
    """Return array of elements between start and stop, each separated by step.

    Replacement for np.arange that DOES always include stop.
    Normally np.arange should not include stop, but due to floating point precision sometimes it does, so output is
    unpredictable"""
    n = np.abs((stop - start) / step)
    if np.isclose(n, np.round(n)):
        # If n only differs from an integer by floating point precision, round it
        n = int(np.round(n))+1
    else:
        # If n is not approximately an integer, floor it
        n = int(np.floor(n))+1
        stop = start + (n-1)*step
    out = np.linspace(start, stop, n)
    return out

def safe_isnan(value, false_for_non_numeric=True):
    """Return false rather than throwing an error if the input type is not numberic"""
    try:
        out = np.isnan(value)
    except TypeError as e:
        if false_for_non_numeric:
            out = False
        else:
            raise e
    return out

def safe_in(item, iterable, in_strings=True):
    """Return True if item in iterable, returning False if iterable is not iterable"""
    try:
        if (not in_strings) and isinstance(iterable, str):
            out = item in [iterable]
        else:
            out = item in iterable
    except TypeError as e:
        out = False
    return out

def cast_safe(obj, new_type):
    """Cast obj to new_type, nesting if necessary, or change dtype for arrays"""
    assert isinstance(new_type, type)
    if (new_type in (tuple, list, np.ndarray)) and is_scalar(obj):
        out = new_type([obj])
    elif is_scalar(obj):
        out = new_type(obj)
    elif new_type in (tuple, list, np.ndarray):
        out = new_type(obj)
    elif isinstance(obj, np.ndarray):
        out = obj.astype(new_type)
    else:
        raise NotImplementedError('Cast safe not implemented for new_type: {}, obj: {}'.format(new_type, obj))
    return out


def ceil_cast(value, cast_type=int):
    """Return ceil cast to int

    Avoids unnecessary rounding up due to floating point representation"""
    scalar = is_scalar(value)

    # Avoid unnecessary rounding up due to floating point representation
    out = to_array(value)
    mask = np.isclose(out, np.round(out))
    out[mask] = np.round(out[mask])

    out = np.ceil(out)

    if scalar:
        out = out[0]

    if cast_type is not None:
        out = cast_safe(out, cast_type)
    return out

def round_cast(value, cast_type=int):
    """Round input and return cast to int"""
    out = np.round(value)
    out = cast_safe(out, cast_type)
    return out

def any_equal(object, list):
    """ Return true if object is equal to any of the elements of list
    """
    return np.any([object == l for l in list])

def is_subset(subset, full_set):
    """Return True if all elements of subset are in fullset"""
    return set(subset).issubset(set(full_set))

def is_in(items, collection):
    items = make_iterable(items)
    collection = make_iterable(collection)
    out = pd.Series(items).isin(collection).values
    return out

def is_in_str(sub_strings, string):
    assert isinstance(string, str)
    sub_strings = make_iterable(sub_strings)
    out = np.zeros_like(sub_strings, dtype=bool)
    for i, sub_str in enumerate(sub_strings):
        out[i] = sub_str in string
    return out

def is_in_fuzzy(patterns, options, contains=True, ignore_case=True):
    """Return string elements of options that partially match a pattern ignoring case"""
    matches_all = []
    options = make_iterable(options)
    args = [re.IGNORECASE] if ignore_case else []
    for pattern in make_iterable(patterns):
        assert isinstance(pattern, str)
        if contains:
            r = re.compile(pattern, *args)
            matches = list(filter(r.search, options))
        else:
            # Add end characters for exact match
            r = re.compile(r'^'+pattern+r'$', *args)
            matches = list(filter(r.match, options))
        if matches:
            matches_all += matches
    # Remove any duplicates due to multiple patterns matching a given option
    # matches_all = list(set(matches_all))
    return matches_all

def similarity_difflib(reference, other):
    import difflib
    similarity = difflib.SequenceMatcher(a=reference, b=other).ratio()
    return similarity

def similarity_Levenshtein(reference, option):
    import Levenshtein
    similarity = Levenshtein.ratio(reference, option)
    return similarity

def similarilty_to(reference, options, return_type='order', n_return=None,
                   similarity_threshold=None, similarity_measure='difflib'):
    """Compare strings and rank in order of similarity"""
    assert return_type in ('order', 'values', 'similarity')
    assert similarity_measure in ('difflib',)
    if similarity_measure == 'difflib':
        similarity_func = similarity_difflib
    else:
        raise ValueError('similarity_measure "{}" not recognised'.format(similarity_measure))
    reference = str(reference)
    options = to_array(options)
    similarities = np.zeros(len(options))
    for i, option in enumerate(options):
        option = str(option)
        similarity = similarity_func(reference.lower(), option.lower())
        similarities[i] = similarity
    ordering = np.argsort(similarities)[::-1]

    if return_type == 'order':
        out = ordering
    elif return_type == 'values':
        out = options[ordering]
    elif return_type == 'similarity':
        out = similarities
    else:
        raise ValueError('Return type "{}" not recognised'.format(return_type))

    if similarity_threshold is not None:
        # Only return values above similarity threshold
        mask = similarities > similarity_threshold
        out = out[mask]
    if n_return is not None:
        out = out[:n_return]
    return out

def to_list(obj):
    """Convert to list, nesting if nessesary"""
    if isinstance(obj, list):
        out = obj
    elif isinstance(obj, (np.ndarray, tuple, set)):
        out = list(obj)  # convert tuple, ndarray etc
    else:
        out = [obj]  # eg float to [float] etc
    # elif is_scalar(obj):
    #
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
    elif is_number(obj):  # Equivalent to make_iterable
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
                logger.debug('Dictionaries are different - One value is a dict while the other is not')
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

def compare_objs(obj1, obj2):
    import xarray as xr
    if type(obj1) != type(obj2):
        return False
    if isinstance(obj1, dict):
        return compare_dicts(obj1, obj2)
    elif isinstance(obj1, (list, tuple)):
        return compare_lists(obj1, obj2)
    elif isinstance(obj1, (np.ndarray, pd.Series, xr.DataArray)):
        return compare_arrays(obj1, obj2)
    elif isinstance(obj1, pd.DataFrame):
        return compare_dataframes(obj1, obj2)
    else:
        return obj1 == obj2

def compare_dicts(dict1, dict2):
    equal = True
    for key, value in dict1.items():
        if key not in dict2:
            equal = False
        else:
            if not compare_objs(dict1[key], dict2[key]):
                equal = False
    return equal

def compare_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    equal = True
    for a, b in zip(list1, list2):
        if not compare_objs(a, b):
            equal = False
    return equal

def compare_arrays(array1, array2):
    # TODO: Match nans
    if array1.shape != array2.shape:
        out = False
    elif array1.dtype != array2.dtype:
        out = False
    elif np.all(array1 == array2):
        out = True
    elif array1.dtype == np.dtype(object):
        # Handle objects that are not equal to themselves eg nan, None
        out = True
        for item1, item2 in zip(array1, array2):
            if item1 == item2:
                sub_out = True
            elif (is_numeric(item1) and np.isnan(item1)) and (is_numeric(item2) and np.isnan(item2)):
                sub_out = True
            elif (item1 is None) and (item2 is None):
                sub_out = True
            else:
                sub_out = False
            out *= sub_out
    else:
        # equal_mask = array1 == array2
        # Handle floating point imprecision
        equal_mask = np.isclose(array1, array2)
        both_nan = np.isnan(array1) & np.isnan(array2)
        out = equal_mask | both_nan
        out = np.all(out)
    return bool(out)

def compare_dataframes(df1, df2):
    # TODO: Match nans
    if df1.shape != df2.shape:
        out = False
    elif (not np.all(df1.index == df2.index)):
        out = False
    elif (not np.all(df1.columns == df2.columns)):
        out = False
    elif (not np.all(df1.dtypes == df2.dtypes)):
        out = False
    elif np.dtype(object) not in df1.dtypes.values:
        # Basic datatypes
        equal_mask = df1 == df2
        both_nan = df1.isnull() & df2.isnull()
        out = equal_mask | both_nan
        out = np.all(out)
    else:
        # Compare columnwise
        out = True
        for col in df1.columns:
            array1 = df1[col].values
            array2 = df2[col].values
            out &= compare_arrays(array1, array2)
    return bool(out)

def compare_dataframes_details(df1, df2, df1_name='df_1', df2_name='df_2',
                               include_missing_in_diffs=True, log_difference=True, raise_on_difference=False):
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert np.all(df1.columns.isin(df2.columns)), 'Dataframes must have matching columns'
    summary = {'same': [], 'different': [], 'missing': [], 'added': [], 'identical': False}

    # Loop over df2 finding items unique to df2
    for item in df2.index:
        if item not in df1.index:
            summary['added'].append(item)
        else:
            if compare_dataframes(df2.loc[[item]], df1.loc[[item]]):
                summary['same'].append(item)
            else:
                summary['different'].append(item)

    # Loop over df1 finding items unique to df1
    for item in df1.index:
        if item not in df2.index:
            summary['missing'].append(item)

    if len(summary['same']) != len(df2):
        different_items = summary['different'] + summary['added']
        if include_missing_in_diffs:
            different_items += summary['missing']
        cols_1 = {col: '{}_{}'.format(col, df1_name) for col in df1.columns}
        cols_2 = {col: '{}_{}'.format(col, df2_name) for col in df2.columns}
        different_items_in_df1 = [item for item in different_items if item in df1.index]
        df_diffs = copy(df1.loc[different_items_in_df1]).rename(columns=cols_1)
        df_diffs = df_diffs.reindex(different_items)
        for col in cols_2:
            df_diffs[cols_2[col]] = df2[col]
        message = 'Dataframe comparison; Same: {}, Different: {}, Missing: {}\n{}'.format(
                len(summary['same']), summary['different'], summary['missing'], df_diffs)
        # print(df_diffs)
        if raise_on_difference:
            raise ValueError(message)
        if log_difference:
            logger.info(message)
    else:
        df_diffs = None
        summary['identical'] = True

    return summary, df_diffs

def isclose_within(values, reference, tol=1e-8, all=False, return_values=False):
    """Return vool_array/bool (if all=True) if elements of 'values' appear in 'reference' comparison_list within tollerance
    From: http://stackoverflow.com/questions/39602004/can-i-use-pandas-dataframe-isin-with-a-numeric-tolerance-parameter
    """
    # import pandas as pd
    scalar_value = is_scalar(values)
    values = to_array(values)
    reference = to_array(reference)
    values = np.expand_dims(values, axis=1)  # add dimension to compare pairwise
    out = np.isclose(values, reference, atol=tol).any(axis=1)
    if all:
        # Return single boolean
        out = np.all(out)
    elif return_values:
        # Return values that are close
        out = values.flatten()[out]
    else:
        # Return indices of values that are close
        pass
    if scalar_value and not is_scalar(out):
        out = out[0]
    return out

def equal_string(obj, string):
    """Return True if obj is equal to string"""
    if isinstance(string, (tuple, list)):  # Check agaist list of strings for match
        return np.any([equal_string(obj, s) for s in string])
    if isinstance(obj, (str, basestring)) and obj == string:
        return True
    else:
        return False

# def assert_raises(func)

def printProgress(iteration, total, prefix='', suffix='', frac=False, t0=None,
                  decimals=2, nth_loop=2, barLength=50, flush=True):
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
    if flush:
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

def argsort(itterable):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    try:
        out = sorted(range(len(itterable)), key=itterable.__getitem__)
    except TypeError as e:
        itterable = [str(val) for val in itterable]
        out = sorted(range(len(itterable)), key=itterable.__getitem__)
    return out

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
    #TODO: Include positional arguments!
    func = make_iterable(func)  # Nest lone function in list for itteration, TODO: Handle itterable classes
    kws = {}
    keep = []  # list of argument names
    name_args = []
    for f in func:
        # Add arguments for each function to list of arguments to keep
        if isinstance(f, type):
            # If a class look at it's __init__ method
            keep += list(inspect.signature(f.__init__).parameters.keys())
        else:
            keep += list(inspect.signature(f).parameters.keys())
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

def ask_input_yes_no(message, suffix=' ([Y]/n)? ', message_format='{message}{suffix}', default_yes=True,
                     batch_mode_default=True, sleep=0.1):
    """Ask yes/no question to raw input"""
    if in_freia_batch_mode():
        return batch_mode_default
    if default_yes is False:
        suffix = ' (y/[N])? '
    if sleep:
        # Make sure logging output has time to clear before prompt is printed
        time.sleep(sleep)
    question = message_format.format(message=message, suffix=suffix)
    answer = input(question)
    accept = ['y', 'yes']
    if default_yes:
        accept.append('')
    if answer.lower() in accept:
        out = True
    else:
        out = False
    return out

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
    # old, new = make_iterables(old, new)
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

def lookup_from_dataframe(df, col, _raise_on_missing=True, **kwargs):
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
        if not isclose_within(value, values.astype(type(value))):
            if _raise_on_missing:
                raise ValueError('value {} is not a valid "{}" value. Options: {}'.format(value, key, df[key]))
            else:
                return None
        value = make_iterable(value)
        mask *= isclose_within(df[key].values, value)
    new_value = df.loc[mask, col].values
    if len(new_value) == 1:
        new_value = new_value[0]
    return new_value

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

def has_inflection(data):
    """Return True if 1D data has point of inflection ie increasing and decreasing sections"""
    diff = np.diff(data)
    if (np.sum(diff > 0) > 0) and (np.sum(diff < 0) > 0):
        return True
    else:
        return False

def dataframe_description_str(df, **kwargs):
    description = df.describe()
    annotation_str = ['{:5s}: {:10s}'.format(item, '{:0.3g}'.format(value)) for item, value in description.items()]
    annotation_str = '\n'.join(annotation_str)
    return annotation_str

def append_to_df_index(df, new_indices, default_value=np.nan, default_type=None, sort_index=True):
    """Return dataframe extended by new indicies"""
    new_indices = copy(new_indices.remove_unused_levels())
    df_append = pd.DataFrame(default_value, index=new_indices, columns=df.columns, dtype=default_type)
    out = df.append(df_append)
    if sort_index:
        out.sort_index(axis=0)
    return out

class PartialFormatter(string.Formatter):
    """NOTE: Can use double braces for this! '{{no key passed to this}}'.format() -> '{no key passed to this}"""
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


class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""

    def __init__(self, artists, func=None, tolerance=5, offsets=(-20, 20),
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
                self.func(event, annotation)

    def connect(self, fig):
        self.cid = fig.canvas.mpl_connect('pick_event', self)

    def disconnect(self, fig):
        fig.canvas.mpl_disconnect(self.cid)

    def clear(self, fig):
        for ann in self.annotations.values():
            ann.set_visible(False)
        fig.canvas.draw()


class ROISelector(object):
    from matplotlib.widgets import RectangleSelector
    def __init__(self, artist):
        self.artist = artist
        self.selector = RectangleSelector(self.artist.axes, self.on_select,
                                          button=3, minspanx=5, minspany=5, spancoords='pixels',
                                          rectprops=dict(facecolor='red', edgecolor='red',
                                                         alpha=0.3, fill=True))  # drawtype='box'
        self.coords = []

    def on_select(self, click, release):
        x1, y1 = int(click.xdata), int(click.ydata)
        x2, y2 = int(release.xdata), int(release.ydata)
        self.coords = [(x1, y1), (x2, y2)]

    def activate(self):
        self.selector.set_active(True)

    def deactivate(self):
        self.selector.set_active(False)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ccfepyutils.data_processing import moving_average, find_nearest, find_furthest

    safe_arange(1.32, 1.46, 0.003)
    l = [[4,5], [1,1], [2,1], [1,2]]
    argsort(l)

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