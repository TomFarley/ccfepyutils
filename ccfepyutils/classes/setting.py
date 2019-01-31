#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os, itertools, gc, re, inspect, configparser, abc, numbers, time, shutil
from collections import OrderedDict
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import ccfepyutils
from ccfepyutils.classes.state import State, in_state
from ccfepyutils.utils import make_iterable, remove_duplicates_from_list, is_subset, get_methods_class, t_now_str, \
    to_list, ask_input_yes_no
from ccfepyutils.io_tools import mkdir, filter_files_in_dir, delete_file, attempt_n_times
from ccfepyutils.netcdf_tools import dict_to_netcdf, netcdf_to_dict

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

settings_dir = ccfepyutils.settings_dir


# TODO: Fix deepcopy of Setting objects
class Setting(abc.ABC):
    # __metaclass__ = abc.ABCMeta
    type = None

    def __init__(self, settings, item):
        self._settings = settings
        self._item = item
        try:
            self._df = self._settings._df.loc[[item], :]
        except KeyError as e:
            # SettingList does not have a single item
            self._df = self._settings._df

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'>", str(self.__class__)).groups()[0]
        return '<{}: {}={}>'.format(class_name, self._item, str(self))

    def __str__(self):
        return str(self.value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._settings._df.loc[self._item, :]
        else:
            if item in self._settings.column_sets_names:
                item = self._settings.column_sets_names[item]
            return self._settings._df.loc[self._item, item]

    def __setitem__(self, item, value):
        if item not in self._settings.columns:
            raise ValueError('Item "{}" is not a setting column'.format(item))
        self._settings._df.loc[self._item, item] = value

    def __iter__(self):
        raise NotImplementedError

    @property
    def columns(self):
        return self._df.columns.values

    @property
    def value(self):
        return self.type(self._df.loc[self._item, self.value_column])


class SettingStr(Setting, str):
    type = str
    value_column = 'value_str'

    def __new__(cls, settings, item):
        value = str(settings._df.loc[item, cls.value_column])
        return str.__new__(cls, value)

    def __init__(self, settings, item):
        str.__init__(self)
        Setting.__init__(self, settings, item)

    @property
    def value(self):
        val = super().value
        if val == '*None*':
            val = None
        return val

    def format(self, *args, **kwargs):
        return self.value.format(*args, **kwargs)


class SettingInt(Setting, int):
    type = int
    value_column = 'value_num'

    def __new__(cls, settings, item):
        value = int(settings._df.loc[item, cls.value_column])
        return int.__new__(cls, value)

    def __init__(self, settings, item):
        int.__init__(self)
        Setting.__init__(self, settings, item)


class SettingFloat(Setting, float):
    type = float
    value_column = 'value_num'

    def __new__(cls, settings, item):
        value = float(settings._df.loc[item, cls.value_column])
        return float.__new__(cls, value)

    def __init__(self, settings, item):
        float.__init__(self)
        Setting.__init__(self, settings, item)


class SettingBool(Setting, int):
    type = bool
    value_column = 'value_num'

    def __new__(cls, settings, item):
        value = bool(settings._df.loc[item, cls.value_column])
        return int.__new__(cls, value)

    def __init__(self, settings, item):
        int.__init__(self)
        Setting.__init__(self, settings, item)

    def __bool__(self):
        return bool(self._settings._df.loc[self._item, 'value_num'])


# TODO: Move new and init into Setting base class!

class SettingList(Setting, list):  # TODO: implement SettingsList  !
    type = list

    # value_column = 'value_num'
    # def __new__(cls, settings, item):
    #     value = settings.list_item_values(settings, item)
    #     return list.__new__(cls, value)

    def __init__(self, settings, item):
        list.__init__(self)
        Setting.__init__(self, settings, item)
        value = settings.list_item_values(settings, item)
        self += value

    def __str__(self):
        return str(self.value)

    def __iter__(self):
        for x in self.value:
            yield x

    def __getitem__(self, item):
        return self.value[item]

    def __contains__(self, item):
        return (item in self.value)

    def __len__(self):
        return len(self.value)

    def __eq__(self, other):
        return self.value == other

    @property
    def value(self):
        return self.type(self._settings.list_item_values(self._settings, self._item))