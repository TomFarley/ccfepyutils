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
import matplotlib
from ccfepyutils import batch_mode
from ccfepyutils.mpl_tools import set_matplotlib_backend
set_matplotlib_backend(batch_mode, non_visual_backend='Agg', visual_backend='Qt5Agg')

import matplotlib.pyplot as plt

from ccfepyutils.utils import make_iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract(data_array, squeeze=True, **kwargs):
    """ Extract subset of data and perform opperations over axes.

    :param data_array: xarray.DataArray to extract data from
    :param squeeze: Remove dimensions of length 1 in output
    :param kwargs: Values of dimensions to select:
                        <dim_name> = <array>  - extract data at these coordinate values
                        <dim_name>_range = [min, max]  - extract data within coordinate interval
                        <dim_name>_<func> = True  - apply <func> along given dimension eg mean, std, max, cumsum etc
    :return: DataArray containing extracted subset of data
    """
    kws = copy(kwargs)
    # Select the requested data
    for arg in data_array.coords.keys():
        if data_array[arg].ndim == 0:
            # Can't filter single values coords
            continue
        if (arg in data_array.coords) and (arg not in data_array.dims):
            data_array = data_array.swap_dims({data_array.coords[arg].dims[0]: arg})
        if arg in kws:
            # Deal with inexact matches due to floating point
            values = make_iterable(kws[arg])
            mask = np.zeros_like(data_array.coords[arg], dtype=bool)
            for v in values:
                mask += np.isclose(data_array.coords[arg], v)
            if np.sum(mask) == 0:
                raise ValueError('None of the values "{}" are in the coordinate "{}": {}'.format(
                        values, arg, data_array.coords[arg]))
            data_array = data_array.loc[{arg: mask}]
            kws.pop(arg)
        arg_range = '{}_range'.format(arg)
        if arg_range in kws:
            limits = kws[arg_range]
            assert len(limits) == 2, 'Range must have two elements'
            mask = (limits[0] <= data_array.coords[arg]) * (data_array.coords[arg] <= limits[1])
            data_array = data_array.loc[{arg: mask}]
            kws.pop(arg_range)
    # Collapse dimensions by taking average or standard deviation etc along axis
    for kw in copy(kws):
        for arg in data_array.coords.keys():
            m = re.match('{}_(\w+)'.format(arg), kw)
            if m:
                func = m.groups()[0]
                if hasattr(data_array, func):
                    if arg not in data_array.dims:
                        data_array = data_array.swap_dims({data_array.coords[arg].dims[0]: arg})
                    data_array = getattr(data_array, func)(dim=arg)
                    kws.pop(kw)
    if len(kws) > 0:
        raise ValueError('Keyword arguments {} not recognised'.format(kws))
    if squeeze:
        # Remove redudanct dimensions with length 1
        data_array = data_array.squeeze()
    return data_array

if __name__ == '__main__':
    pass