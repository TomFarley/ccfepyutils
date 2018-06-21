#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging, os, itertools, re, inspect, configparser, time
from collections import defaultdict, OrderedDict
from datetime import datetime
from copy import copy, deepcopy
from pathlib import Path
from logging.config import fileConfig

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sp

from ccfepyutils.utils import args_for
from ccfepyutils.data_processing import data_split

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_threshold(data, thresh_method='sigma', thresh_value=2.5):
    """Return amplitude threshold value with given method"""
    data = np.array(data).flatten()
    if thresh_value == 'abs':
        thresh = thresh_value
    elif thresh_method == 'sigma':
        thresh = np.mean(data) + thresh_value * np.std(data)
    elif thresh_method == 'mean':
        thresh = thresh_value * np.mean(data)
    elif thresh_method == 'percentile':
        thresh = np.percentile(data, thresh_value)
    elif thresh_method is None:
        thresh = np.min(data)
    else:
        raise ValueError('thresh_type = "{}" not recognised'.format(thresh_method))
    return thresh




def locate_peaks(data, peak_method='argrelmax', thresh_method='sigma', thresh_value=2.5, exclude_ends=True, **kwargs):
    """
    Return indices and coordinates of peaks in signal
    :param data: xr.DataArray - 1D dataset in which to locate peaks
    :param peak_method: str - Peak detection method
    :param thresh_method: str - Peak amplitude threshold method
    :param thresh_value: - Peak amplitude threshold value
    :param exclude_ends: bool - Exclude one sided local maxima at ends of dataset
    :param kwargs: Peak detection method specific keyword arguemnts
    :return: (peak_indices, peak_coor_values)
    """
    funcs = {'argrelmax': locate_peaks_argrelmax, 'delta': locate_peaks_delta, 'cwt': locate_peaks_cwt,
             'kirk16': locate_peaks_kirk16}
    assert peak_method in funcs.keys(), '{} not in peak_method options: {}'.format(peak_method, funcs.keys())
    assert isinstance(data, xr.DataArray)
    assert data.ndim == 1, 'data is {}D. Peak detection requires 1D signal'.format(data.ndim)
    info = {}
    y = data.values
    x = data.coords[data.dims[0]].values
    func = funcs[peak_method]
    kws = args_for(func, kwargs)
    i_peaks = func(y, x=x, **kws)  # np.array(i_peaks).astype(int)
    x_peaks = x[i_peaks]
    y_peaks = y[i_peaks]
    if exclude_ends and len(i_peaks) > 0:
        i_peaks = i_peaks[(i_peaks != 0) & (i_peaks != len(y))]
    if thresh_method is not None:
        # Only keep peaks below the threshold level
        thresh = get_threshold(data, thresh_method, thresh_value)
        info['thresh_method'] = thresh_method
        info['thresh_value'] = thresh_value
        info['thresh'] = thresh
        i_filter = y_peaks >= thresh
        i_peaks, x_peaks, y_peaks = i_peaks[i_filter], x_peaks[i_filter], y_peaks[i_filter]

    return i_peaks, x_peaks, y_peaks, info

def locate_peaks_argrelmax(data, order=2, **kwargs):
    """ Find peaks above threshold level in 1D trace

    Keyword arguments
    data - y data (eg intensity values)
    order - window size for maxima detection
    """
    # Find local maxima in data
    data = np.array(data)
    assert data.ndim == 1
    ifil = sp.signal.argrelmax(data, order=order)[0]
    ifil = np.round(ifil).astype(np.int)
    return ifil


def locate_peaks_delta(x, y, delta='auto', nsigma=2.5):
    """Use matlab ported peakdet funtion to locate peaks"""
    raise NotImplementedError
    from elzar.tools.peakdetect import peakdet
    if equal_string(delta, 'auto'):
        delta = 0.70 * np.std(y)

    max, min = peakdet(y, delta)
    if len(max) == 0:  # No maxima found
        return np.array([])
    imax = max[:, 0].astype(int)  # indices of maxima
    vmax = max[:, 1]  # values of maxima
    isigma = vmax > self.nsigma(y, nsigma)
    return imax[isigma]


def locate_peaks_kirk16(x, y, min_peak_diff=0.25, window_width=3):
    """Find peaks surrounded by minima that are sufficiently low
    Method used in Kirk2016
    x - x data (eg time points)
    y - y data (eg intensity values)
    min_peak_diff - Minimum difference in hgihts of surrounding peaks for a minima to be accepted (default=0.25)
    """
    # TODO: loop over peaks in decending order of intensity and apply to fixed width window
    imax_all = argrelmax(y, order=1)[0]
    imin_all = argrelmin(y, order=1)[0]
    imin = []
    for i in imin_all:
        try:
            il = imax_all[np.argmin(i - imax_all[imax_all < i])]  # maxima to left of current min
            ir = imax_all[np.argmin(imax_all[imax_all > i] - i)]  # maxima to right of current min
        except ValueError as e:  # if at end of data no neighbouring maxima, include minima
            imin.append(i)  # add local minima to list
            continue
        l = y[il]  # value of left maxima
        r = y[ir]  # value of right maxima
        m = y[i]  # value of minima
        min_max = np.min((l, r))  # value of the smaller of the two adjacent maxima
        # Check the minimum drop caused by the local minima is sufficiently large relative to its neighbouring maxim
        if min_max - m >= (min_peak_diff * np.abs(r - l)):
            imin.append(i)  # add local minima to list

    # Only keep the largest local maxima between each remaining minima
    imin = np.array(imin)
    imax = []
    for il, ir in zip(imin, imin[1:]):
        igap = np.where((il < imax_all) * (imax_all < ir))[0]
        if len(igap) > 1:
            max_y = np.max(y[imax_all[igap]])
            for i in igap:
                if y[imax_all[i]] == max_y:
                    break
        if len(igap) == 1:
            i = igap[0]
        imax.append(i)

    return imin, imax


def locate_peaks_cwt(x, y, width_range=None, domain_range=[0.05, 0.2], centre=True):
    """Return indices of peaks in data
    x - x data
    y - y data
    width range - range of possible widths of peaks in x units
    domain range - if no width_range, determin width_range based of fractions of x domain
    centre - return location of maxima in peak, rather than centre of mass of peak"""
    from scipy.signal import find_peaks_cwt
    # Generate width range based on domain range
    if width_range is None:
        x_extent = np.ptp(x)
        # find peaks spanning fraction of x domain
        width_range = [domain_range[0] * x_extent, domain_range[1] * x_extent]
    # Calcualte range of peak widths in index space
    dx = x[1] - x[0]
    iwidth = np.arange(np.floor(width_range[0] / dx), np.ceil(window_range[1] / dx))
    ind_peak = find_peaks_cwt(y, iwidth)  # index space widths
    # Find position of peak maximuim. Find_peaks_cwt returns centre of mass of peaks - could be far from max for
    # skewed peaks
    if centre:
        # Could make this more intelegent...
        ind_peak = ind_max[np.argmin(np.abs(ind_max - np.max(ind_peak)))]  # local maxima closest to peak centre
    return ind_peak

def conditional_average(x, y, x_peaks, window_width, return_average=True):
    # TODO: implement i_peaks options
    if len(x_peaks) == 0:
        return [], []
    peak_shapes = []
    for x_centre in x_peaks:
        if x_centre - window_width >= np.min(x) and x_centre + window_width <= np.max(x):  # check not too close to boarders of data
            mask = (x > x_centre - window_width) * (x < x_centre + window_width)
            peak_shapes.append(y[mask])
    x_window = np.linspace(-window_width, window_width, len(peak_shapes[0]))
    if return_average:
        peak_shapes = np.mean(peak_shapes, axis=0)
    return x_window, peak_shapes

def level_crossing(x, y, thresh):
    """Identify sections of data above and below threshold"""
    mask = y >= thresh
    out = dict()
    out['x_above'], out['y_above'] = x[mask], y[mask]
    out['x_below'], out['y_below'] = x[~mask], y[~mask]
    out['i_above'], out['x_above'], out['y_above'] = data_split(out['x_above'], y=out['y_above'], 
                                                                gap_length=1, data_length=1, return_longest=False)
    out['i_below'], out['x_below'], out['y_below'] = data_split(out['x_below'], y=out['y_below'], 
                                                                gap_length=1, data_length=1, return_longest=False)
    out['len_above'] = [len(i) for i in out['i_above']]
    out['len_below'] = [len(i) for i in out['i_below']]
    return out