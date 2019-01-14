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
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# from nested_dict import nested_dict

from ccfepyutils.utils import make_iterable, remove_duplicates_from_list, is_subset, get_methods_class
from ccfepyutils.classes.plot import Plot

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Timeline(object):
    """ """

    def __init__(self, bar_length=50):
        self._df = pd.DataFrame(np.nan, columns=['time', 'status', 'loop_index'])
        self._t0 = datetime.now()
        self._loops = OrderedDict()
        self._bar_length = bar_length
        pass

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}>'.format(class_name, None)
    
    def start_loop(self, item, n, prefix = '', suffix = ''):
        decimals = 2
        raise NotImplementedError

    def iterate_loop(self, item):
        nth_loop = 2
        raise NotImplementedError

    def end_loop(self, item):
        raise NotImplementedError
    
    


if __name__ == '__main__':
    t = Timeline()

    t.start('test1')
    for i in np.arange(10):
        t.add_iteration('test1')
        time.sleep(2)
    t.end('test1')

