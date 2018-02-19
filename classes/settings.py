#!/usr/bin/env python
import itertools
from abc import ABC
import gc
import re
import inspect
import configparser

import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict
from datetime import datetime
import time
from copy import copy, deepcopy
import os
import shutil
from pathlib import Path
from netCDF4 import Dataset

from ccfepyutils.classes.state import State, in_state
from ccfepyutils.utils import make_itterable, remove_duplicates_from_list, is_subset
from ccfepyutils.io_tools import mkdir
from ccfepyutils.netcdf_tools import dict_to_netcdf, netcdf_to_dict, set_netcdf_atrribute, dataframe_to_netcdf

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## TODO: Load from config file
# TODO: Create base .ccfetools/settings/ directory structure if doesn't exist
settings_dir = os.path.expanduser('~/.ccfetools/settings/')


# TODO: Fix deepcopy of Setting objects
class Setting(ABC):
    type = None
    
    def __init__(self, settings, item):
        self._settings = settings
        self._item = item
        self._df = self._settings._df.loc[[item], :]

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}={}>'.format(class_name, self._item, str(self))

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

# TODO: Move new and init into Setting base class

class SettingList(Setting, list):  # TODO: implement SettingsList  !
    type = list
    # value_column = 'value_num'
    def __new__(cls, settings, item):
        value = list(settings._df.loc[item, cls.value_column])
        return int.__new__(cls, value)

    def __init__(self, settings, item):
        list.__init__(self)
        Setting.__init__(self, settings, item)


class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes

    Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. the name of the class your settings are for
        - The name is a label for a particular set of settings for the application e.g. my_default_settings

    Settings are saved for easy reuse in future. By default these files are saved in '~/.ccfetools/settings'. A SettingsLogFile is generated for each application which records the
    names and modification dates etc. of all settings for that application. When accessed a SettingsType object of the
    appropriate type is returned, providing access to setting meta data via indexing.
    
    NOTES:
        - The name of a setting must be unique for each named set of settings.
        - Settings files are only saved automatically when they are accessed. Use settings.save() to save manually.

    Usage examples:
        - Load/create a settings object (will return an existing instance if already exists):
            settings = Settings.get(<my_amplication>)
        - Add a setting item:
            settings('my_setting', value, name='My settings', description='Value of ...')  # keywords set metadata
        - Access a setting:
            setting = settings['my_setting']
        - View set of columns in settings file eg. 'info' columns
            settings.view('info')
            
        - List applications with existing saved settings:
            Settings.get_applications()
        - List existing named sets of settings for an application:
            Settings.get_settings()
    """


    instances = nested_dict()
    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}

    state_table = {'core': {
                       'init': ['modified', 'loading'],
                       'loaded': ['accessing', 'modifying'],
                       'modified': ['modifying', 'accessing', 'saving'],
                       'saved': ['accessing', 'modifying']},
                   'transient': {
                       'loading': ['loaded', 'modifying', 'accessing'],
                       'modifying': ['modified', 'accessing'],
                       'saving': ['saved', 'modifying'],
                       'accessing': ['saving', 'modifying', 'modified', 'saved']}
                   }
    column_sets = {'value': [('value', str), ('value_str', str), ('value_num', float)],
                   'info': [('name', str), ('description', str), ('symbol', str), ('unit', str)],
                   'type': [('float', bool), ('int', bool), ('bool', bool), ('str', bool)],
                   'io': [('fn_str', str), ('priority', float)],
                   'meta': [('setting', bool), ('runtime', bool), ('order', int)],
                   'repr': [('name', str), ('value', str), ('description', str)]}  # plotting?

    def __init__(self, application, name):
        """Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. my_code
        - The name is a label for a particular set of settings for the application e.g. my_default_settings
        """
        # TODO: Fix deepcopy of Settings objects
        self._reset_settings_attributes()
        self.call_table = {'modifying': {'enter': [self._block_protected, self.check_consistency]},
                           'modified': {'accessing': [self.save]}}
        self.state = State(self, self.state_table, 'init', call_table=self.call_table)
        assert isinstance(application, str)
        assert isinstance(name, str)
        assert (application, name) not in self.instances.keys_flat(), ('Setting object {}:{} already exists.\n'
                'Use Settings.get(application, name) to ensure there is only one instance'.format(
                application, name))
        self.application = application  # property sets logfile
        self.name = name
        self._column_sets = Settings.column_sets   # Groups of columns with similar purposes
        self._column_sets_names = self.column_sets_names  # TODO: properly implement saving and loading of column sets

        self.instances[application][name] = self

        if self.file_exists:
            self.load()
        else:
            self.init()

    def _reset_settings_attributes(self):
        self.state = None
        self.call_table = None
        self._application = None
        self._name = None
        self.log_file = None
        self._column_sets = None
        self._column_sets_names = None

    @classmethod
    def get(cls, application=None, name=None):
        if application is None:
            raise ValueError('No application supplied to Settings.get().\nExisting applications settings: {}'.format(
                            cls.existing_applications()))
        if name is None:
            raise ValueError('No settings set name supplied to Settings,get().\n'
                             'Existing settings for application "{}": {}'.format(
                            application, cls.existing_settings(application)))
        if (application, name) in cls.instances.keys_flat():
            return cls.instances[application][name]
        else:
            return Settings(application, name)

    @in_state('init', 'modified')
    def init(self):
        columns = self.ordered_column_names
        types = self.column_sets_types
        self._df = pd.DataFrame({key: [] for key in columns})#, dtype=types)  # Initialise empty dataframe
        # Set the types of each column
        type_dict = [dict(v) for k,v in self.column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        self._df.loc[:, :] = self._df.astype(type_dict)
        # self.df.loc[:, [self.column_sets_names['type']]] = False

    def view(self, cols='repr', order=None, ascending=True):
        """Return dataframe containing a subst of columns, with items ordered as requried"""
        if cols == 'all':
            col_set = self.ordered_column_names
        else:
            col_set = []
            cols = make_itterable(cols)
            for col in cols:
                if col in self.column_sets_names.keys():
                    col_set += self.column_sets_names[col]
                else:
                    assert col in self.columns, 'Column not found: {}'.format(col)
                    col_set += [col]
        if order is None:
            df = self._df
        elif order == 'alphabetical':
            df = self._df.sort_index(ascending=ascending)
        elif order == 'custom':
            df = self._df.sort_values('order', ascending=ascending)
        out = df.loc[:, col_set]
        return out

    def view_str(self, cols='repr', order=None, ascending=True):
        """Return string containing a subst of columns, with items ordered as requried and index replaced with name"""
        df = self.view(cols=cols, order=order, ascending=ascending)
        if 'name' in df.columns:
            columns = list(df.columns)
            out = df.reset_index().loc[:, columns].to_string(index=False, justify='left')
        else:
            out = df.to_string(index=True, justify='left')
        return out

    @in_state('modifying', 'modified')
    def __call__(self, item, value=None, create_columns=False, **kwargs):
        """Set value of setting
        :item: name of setting to change
        :value: value to set the item to
        :kwargs: Use to set non-value columns"""
        # Store a list as an enumerated set of items
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                item_i = '{}:{}'.format(item, i)
                self(item_i, v, create_columns=create_columns, **kwargs)
            return
        df = self._df
        # new = item not in list(self.items)
        if item not in list(self.items):  # add item setting type columns appropriately
            self.add_item(item, value)
        elif value is not None:  # set value given we already know type
            col = self.get_value_column(self._df, item)
            df.loc[item, col] = value
            df.loc['item', 'value'] = str(value)
            logger.debug('Existing item of {} set {}={}'.format(repr(self), col, value))
        for k, v in kwargs.items():
            if k in self.columns:
                df.loc[item, k] = v
            elif create_columns:
                df.loc[item, k] = v
                logger.info('Added column {} to {}'.format(k, repr(self)))
            else:
                raise IndexError('{} is not a valid Settings column. Possible values: {}'.format(k, self.columns))
        cols = self.column_sets_names['type']
        df.loc[:, cols] = df.loc[:, cols].fillna(False)
        # df.loc[nan_types, cols] = False
        # self.state('modified')
        # self.log_file.updated(self.name)
        # self.save()

    @in_state('accessing')
    def __getitem__(self, item):
        # If item is a list, unpack it's values into a list
        if self.list_items(item):
            return self.list_items(item)
        if self.get_func_name_args(item):
            return self.get_func_name_args(item)

        df = self._df
        # Check if item is in df index or is the setting name
        if item in df.index:
            pass
        elif item in df['name']:
            mask = df['name'] == item  # boolian mask where name == 'item'
            if np.sum(mask) == 1:
                item = df.index[mask]
            else:
                raise ValueError('Setting item name {} is not unique: {}'.format(item, df.loc[mask, 'value']))
        else:
            raise AttributeError('Item "{}" is not in {}'.format(item, repr(self)))
        # Save data if it has been modified before access
        if self.state == 'modified':
            pass  # TODO: save on access
            # self.save()

        # Isolate settings values
        setting = df.loc[item]
        np_true = np.bool_(True)  # np bool array does not use python True/False instances
        # Return appropriate type of settings object
        if setting['float'] is np_true:
            out = SettingFloat(self, item)
        elif setting['int'] is np_true:
            out = SettingInt(self, item)
        elif setting['bool'] is np_true:
            out = SettingBool(self, item)
        else:
            out = SettingStr(self, item)
        return out

    def __contains__(self, item):
        if item in self._df.index:
            return True
        else:
            return False

    @in_state('modifying', 'modified')
    def __setitem__(self, item, value):
        self(item, value)

    @in_state('modifying', 'modified')
    def add_item(self, item, value):
        assert item not in list(self.items), 'Item {} already exists'.format(item)
        type_to_col = {'bool': 'value_num', 'int': 'value_num', 'float': 'value_num', 'str': 'value_str'}
        if isinstance(value, bool):
            dtype = 'bool'
        elif isinstance(value, numbers.Integral):
            dtype = 'int'
        elif isinstance(value, numbers.Real):
            dtype = 'float'
        elif isinstance(value, str):
            dtype = 'str'
        elif value is None:
            value = '*None*'
            dtype = 'str'
        else:
            raise TypeError('Setting {}={} of type {} is not supported.'.format(item, value, type(value)))
        df = self._df
        df.loc[item, [type_to_col[dtype], dtype]] = value, True
        other_types = list(set([k for k in type_to_col if k != dtype]))
        df.loc[item, other_types] = False
        df.loc[item, 'order'] = len(df) - 1
        df.loc[item, 'runtime'] = False  # default to False
        df.loc[item, 'setting'] = False  # default to False
        df.loc[item, 'value'] = str(value)
        df.loc[item, 'name'] = item  # Default name to item key

        logger.info('Added new item to settings: {} = {}'.format(item, value))

    @in_state('modifying', 'modified')
    def add_items(self, items):
        for item, value in items.items():
            self.add_item(item, value)

    @in_state('modifying', 'modified')
    def rename_item(self, old_name, new_name):
        """Rename setting key"""
        assert old_name in self._df.index
        self._df = self._df.rename({old_name: new_name}, axis='index')

    @in_state('modifying', 'modified')
    def delete_items(self, items):
        """Remove item(s) from settings"""
        assert items in self.items, 'Items {} not in {}'.format(items, repr(self))
        items = make_itterable(items)
        self._df = self._df.drop(items)
        logger.info('Deleted items {} from settings: {}'.format(items, repr(self)))

    @in_state('modifying', 'modified')
    def append_item(self, name, values={'value': []}, categories=[], create_cols=True):
        """Add item with an already existing name to settings.

        The item key will have a number appended to the end in format <name>::2 etc"""
        # TODO: implement settings append
        raise NotImplementedError

    @in_state('loading', 'loaded')
    def load(self):
        assert self.file_exists
        with Dataset(self.fn_path) as root:
            # self.__dict__.update(netcdf_to_dict(root, 'meta'))  # redundant info as stored in logfile
            self._column_sets_names = netcdf_to_dict(root, 'column_sets_names')
        self._df = xr.open_dataset(self.fn_path, group='df').to_dataframe()
        self.log_file.loaded(self.name)
        self.check_consistency()
 
    @in_state('modifying')
    def from_config(self, fn):
        assert os.path.isfile(os.path.expanduser(fn)), 'Config file does not exist: {}'.format(fn)
        config = configparser.ConfigParser()
        config.read(fn)
        modified = False
        for section, item in config.items():
            if len(item) == 0:
                continue
            for key, value in item.items():
                name = '{}::{}'.format(section, key)
                # If no change required, skip
                if name in self and self[name] == value:
                    continue
                self[name] = value
                modified = True
        if modified:
            self.state('modified')
    
    @in_state('saving', 'saved')
    def save(self, state_transition=None):
        if not self.file_exists:
            self.create_file()
            return
        # Don't resave if already saved
        if self.state.previous_state == 'saved' or self.state.previous_state == 'modifying':
            self.state.reverse()
            return 
        meta = {'t_created': self.t_created, 't_modified': self.t_modified, 't_accessed': self.t_accessed}
        try:
            #TODO: delete/overwrite previous values? Use with?
            os.remove(self.fn_path)
            root = Dataset(self.fn_path, "w", format="NETCDF4")
            dict_to_netcdf(root, 'meta', meta)
            dict_to_netcdf(root, 'column_sets_names', self.column_sets_names)
            root.close()
            self._df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')

        except PermissionError as e:
            logger.exception('Unable to write to file')
        except:
            root.close()
            logger.exception('Failed to update Settings File for application "{app}": {path}'.format(
                    app=self.application, path=self.fn_path))
        else:
            logger.info('Updated/saved Settings File for application "{app}": {path}'.format(
                    app=self.application, path=self.fn_path))
            self.log_file.updated(self.name)

    def create_file(self):
        assert not self.file_exists, 'Settings file already exists: {}'.format(self.fn_path)
        # meta = {key: self.__dict__[key] for key in ['_t_created', '_t_modified', '_t_accessed']}
        self.log_file.created(self.name)
        meta = {'t_created': self.t_created, 't_modified': self.t_modified, 't_accessed': self.t_accessed}
        if not os.path.isdir(self.path):
            mkdir(self.path, depth=1)
        try:
            with Dataset(self.fn_path, "w", format="NETCDF4") as root:
                dict_to_netcdf(root, 'meta', meta)
                dict_to_netcdf(root, 'column_sets_names', self.column_sets_names)
                pass
            self._df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')

        except PermissionError as e:
            logger.exception('Unable to write to file')
        except:
            if self.file_exists:
                self.delete_file(force=True)
            logger.exception('Failed to create Settings File for application "{app}": {path}'.format(
                app=self.application, path=self.fn_path))
        else:
            logger.info('Created SettingsFile for application "{app}", name {name}: {path}'.format(
                        app=self.application, name=self.name, path=self.fn_path))
        self.state('saved')

    def delete_file(self, force=False):
        """Delete the configurations file for the current application"""
        if not force:
            print('Are you sure you want to delete the settings file:\n{}'.format(self.fn_path))
            out = input('(Y)/n')
            if not (out.lower() in ('y', '')):
                return
        os.remove(self.fn_path)
        logger.warning('Deleted settings file: {}'.format(self.fn_path))

    @classmethod
    def get_copy(cls, application=None, original=None, new_name=None):
        """Get new settings object based on an existing 'original' template"""
        old_settings = cls.get(application, original)
        assert new_name is not None, 'Name required for new settings set'
        new_settings = old_settings.copy(new_name)
        del old_settings
        return new_settings

    def copy(self, new_name):
        """Copy internal values to new settings set name"""
        new_settings = Settings(self.application, new_name)
        new_settings._df.loc[:, :] = self._df.loc[:, :]
        new_settings._column_sets_names = copy(self.column_sets_names)
        new_settings.save()
        return new_settings

    def new_time(self, format='natural'):
        # TODO: update with config file
        # Find times of exisiting settings for application
        format_str = self.t_formats[format]
        if self.file_exists:
            while t_now_str(format=format_str) in self.log_file:
                time.sleep(1.0)
        return t_now_str(format=format_str)

    def set(self, item, value, ignore=[None]):
        """Set setting provided it is not an ignore value"""
        if value in ignore:
            return
        # No change, so do nothing
        if item in self and self[item] == value:
            return
        self[item] = value

    def check_consistency(self):
        """Checks on consistency of dataframe"""
        # TODO: Check format of columns
        # Make sure each item only has one type
        unique_type = self.view('type').astype(int).sum(axis=1) == 1
        if not all(unique_type):
            raise ValueError('Inconsistent types:\n{}'.format(self.view('type').loc[~unique_type]))
        # If columns in the class column_sets are missing from the dataframe, add them
        if is_subset(self.columns, self.ordered_column_names):
            self._add_missing_columns()
        if True:
            self._reset_column_types()

    @in_state('modifying', 'modified')
    def _add_missing_columns(self):
        """Add empty collumns where missing from class defined column sets"""
        null_types = {int: 0, str: '', float: 0.0, bool: False}
        if is_subset(self.columns, self.ordered_column_names):
            for tup in self.column_sets.values():
                for col, type in tup:
                    if col not in self.columns:
                        self._df.loc[:, col] = null_types[type]
                        logger.warning('Added missing column "{}" to {}'.format(col, repr(self)))

    def _reset_column_types(self):
        """Set datatypes of dataframe columns"""
        # TODO: remove nans, etc
        type_dict = [dict(v) for k, v in self.column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        self._df.loc[:, :] = self._df.astype(type_dict)
        pass
        # raise NotImplementedError

    # def add_column(self, value):
    #     if value in self.columns:
    #         return
    #     self.df[value] = np.nan  # self.df.index
    #
    # def add_columns(self, values):
    #     for value in values:
    #         self.add_column(value)

    @classmethod
    def get_logfile(cls, application):
        """Get saved settings names for given application

        :param application: application settings are for
        :return:
        """
        log = SettingsLogFile(application)
        if log.file_exists:
            return log
        else:
            logger.warning('No saved settings for {}'.format(application))
            return None
    
    @classmethod
    def existing_applications(cls):
        """Get list of all applications with saved settings"""
        return SettingsLogFile.get_applications()

    @classmethod
    def existing_settings(cls, application):
        """Get list of all saved settings names for given application"""
        return SettingsLogFile(application).names

    def list_items(self, item):
        """Return df items that are part of the item list. Return False if not a list."""
        r = re.compile(r'^{}:\d'.format(item))
        newlist = list(filter(r.match, self.items))
        if len(newlist) > 0:
            return [self[i].value for i in newlist]
        else:
            return False

    def item_is_func_arg(self, item):
        raise NotImplementedError

    def set_func_args(self, func, func_name=None, kwargs=None):
        raise NotImplementedError

    def get_func_name_args(self, func_name):
        """Return df items that are arguments for function with func_name. Return False if not a list."""
        r = re.compile(r'^{}::.*'.format(func_name))
        newlist = list(filter(r.match, self.items))
        if len(newlist) > 0:
            return {key: self[key].value for key in newlist}
        else:
            return False

    def get_func_args(self, func, func_name=None, **kwargs):
        """Get arguments for function from settings object"""
        if func_name is None:
            func_name = func.__name__
        args, kws = [], {}
        sig = inspect.signature(func)
        for i, kw in enumerate(sig.parameters.values()):
            name = kw.name
            setting = '{func}::{arg}'.format(func=func_name, arg=name)
            if setting in self:
                kws[name] = self[setting].value
            if setting in kwargs:
                kws[name] = kwargs[name]
        return kws

    @classmethod
    def get_value_column(cls, df, item):
        """Return name of column item value is in"""
        if any(df.loc[item, ['bool', 'int', 'float']]):
            col = 'value_num'
        elif df.loc[item, 'str']:
            col = 'value_str'
        else:
            raise ValueError('Setting {} has no type!'.format(item))
        return col

    def update_from_dataframe(self, df):
        for item, values in df.iterrows():
            if 'value' in values:
                value = values['value']
            else:
                col = self.get_value_column(df, item)
                value = values[col]

            kwargs = {}
            for col, col_value in values.items():
                if col in self.columns:
                    kwargs[col] = col_value
            self(item, col_value, **kwargs)

    
    def _block_protected(self):
        """Block modificaton of a protected file"""
        if (self.name in self.log_file) and (self.log_file[self.name]['protected']):
            raise RuntimeError('Cannot modify protected settings file {}!'.format(repr(self)))
    
    def __str__(self):
        # TODO: set ordering
        df = self.view()
        return '{}:\n{}'.format(repr(self)[1:-1], str(df))

    def __repr__(self):
        return '<Settings: {app};{name}, {state}>'.format(app=self._application, name=self.name, state=self.state)

    def __del__(self):
        self.instances[self.application].pop(self.name)
        # super().__del__()

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        assert isinstance(value, str)
        self._application = value
        self.log_file = SettingsLogFile(self.application)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        elif self.state == 'init':
            return None
        else:
            return self.t_modified

    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        self._name = value

    @property
    def t_created(self):
        return self.log_file._df.loc[self.name, 'created']

    @property
    def t_modified(self):
        if self.log_file is not None:
            return self.log_file._df.loc[self.name, 'modified']
        else:
            return None

    @property
    def t_accessed(self):
        return self.log_file._df.loc[self.name, 'loaded']

    @property
    def items(self):
        return self._df.index.values

    @property
    def columns(self):
        return list(self._df.columns.values)

    @property
    def column_sets_names(self):
        names = {key: [v[0] for v in value] for key, value in self._column_sets.items()}
        return names

    @property
    def ordered_column_names(self):
        names = [value for value in self.column_sets_names.values()]
        names = list((itertools.chain.from_iterable(names)))
        names = [name for name in remove_duplicates_from_list(names)]
        return names

        # columns = list(itertools.chain.from_iterable(self._column_sets.values()))
        # types = [value[1] for value in columns]
        # columns = [value[0] for value in columns]

    @property
    def column_sets_types(self):
        types = {key: [v[1] for v in value] for key, value in self._column_sets.items()}
        return types

    @property
    def path(self):
        """Path to settings files"""
        ## TODO: Load from config file
        return os.path.expanduser('~/.ccfetools/settings/{}/'.format(self.application))


    @property
    def fn(self):
        """Filename of current settings file"""
        assert self.name is not None
        return 'settings-{app}-{name}.nc'.format(app=self.application, name=self.name)


    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)


    @property
    def file_exists(self):
        """Return True if a settings file with the current application and name already exists else False"""
        return os.path.isfile(self.fn_path)

    # @property
    # def t_modified(self):
    #     return convert_str_datetime_format(self._t_modified, format1=self.t_compressed_format)


class SettingsLogFile(object):
    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}
    def __init__(self, application):
        """ """
        assert isinstance(application, str)
        self.application = application

    def __getitem__(self, name):
        """Get log information for setting set"""
        assert name in self.names
        return self._df.loc[name, :]

    def __call__(self, name, cols=None):
        """Get log information for setting set, specifying columns"""
        if cols is None:
            return self[name]
        else:
            cols = make_itterable(cols)
            return self._df.loc[name, cols]

    def init(self):
        # TODO: add safety checks
        self._df = pd.DataFrame({'modified': [], 'loaded': [], 'created': [],
                                'mod_count': [], 'load_count': [], 'load_count_total': [],
                                'protected': []})  # Initialise empty dataframe
        self._df.index.name = 'name'
        self.save()

    def save(self):
        gc.collect()# Fix problems with netcdf file handles persisting
        exists = self.file_exists
        # Don't save the log file if it is empty
        if not exists and len(self.names) == 0:
            return
        mkdir(self.path, depth=1)
        try:
            self._df.to_xarray().to_netcdf(self.fn_path, mode='w')
            if not exists:
                logger.info('Created SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
            else:
                logger.debug('Updated SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
        except PermissionError as e:
            logger.warning('Trouble updating {}. Deleting and refreshing file'.format(repr(self)))
            try:
                self.delete_file(True)
                self._df.to_xarray().to_netcdf(self.fn_path, mode='w')
                logger.warning('Successfully updated {} after refresh'.format(repr(self)))
            except PermissionError as e:
                logger.exception('Unable to write to file!')

    def load(self):
        """Load settings logfile for current application"""
        if not self.file_exists:
            self.init()
            return False
        self._df = xr.open_dataset(self.fn_path).to_dataframe()
        return True
    
    def created(self, name, time=None, overwrite=True):
        """Update time settings file for settings name was created"""
        df = self._df
        if name in self.names and (not overwrite or df.loc[name, 'protected']):
            raise RuntimeError('Cannot overwrite values for setting {}'.format(name))
        if time is None:
            time = t_now_str('natural')
        df.loc[name, ['created', 'modified', 'loaded']] = time
        df.loc[name, ['mod_count', 'load_count', 'load_count_total']] = 0
        df.loc[name, 'protected'] = False
        pass
    
    def loaded(self, name, time=None):
        """Update time settings was last accessed"""
        if name not in self._df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self._df.loc[name, 'loaded'] = time
        self._df.loc[name, 'load_count'] += 1  # since modification
        self._df.loc[name, 'load_count_total'] += 1  # since creation

    def updated(self, name, time=None):
        """Update the time string of a named settings configuration"""
        if name not in self._df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self._df.loc[name, 'modified'] = time
        self._df.loc[name, 'mod_count'] += 1
        self._df.loc[name, 'load_count'] = 0
        self.save()

    def rename(self, old, new):
        """Rename a settings configuration and update existing instances"""
        self._df.rename(index={old: new})
        self.save()

    def backup(self):
        """Backup current application settings configurations file to backup folder"""
        self.save()
        # TODO: Load path setttings from config file
        backup_path = '~/.ccfetools/settings/backups/'
        backup_fn = 'settings_backup-{app}-{time}.nc'.format(app=self.application, time=t_now_str())
        fn_path = os.path.join(backup_path, backup_fn)
        shutil.copyfile(self.fn_path, fn_path)
        logger.info('Created backup of SettingsLogFile for application "{app}": {path}'.format(
                app=self.application, path=fn_path))

    def delete_file(self, force=False):
        """Delete the configurations file for the current application"""
        if not force:
            print('Are you sure you want to delete the settings log file:\n{}'.format(self.fn_path))
            out = input('Y/n')
            if not (out.lower() in ('y', '')):
                return
        os.remove(self.fn_path)
        logger.warning('Deleted settings log file: {}'.format(self.fn_path))

    def print(self):
        print(self)

    @ classmethod
    def get_applications(cls):
        """Get list of all applications with saved settings"""
        applications = next(os.walk(settings_dir))[1]
        return sorted(applications)

    def __repr__(self):
        return '<SettingsLogFile: {app}({l})>'.format(app=self.application, l=len(self._df))

    def toggle_protected(self, name):
        assert name in self.names
        self._df.loc[name, 'protected'] = ~self._df.loc[name, 'protected']
        logger.info('Set protected state of {} to {}'.format(name, self._df.loc[name, 'protected']))

    def __str__(self):
        # tmp = copy(self.df)
        # tmp.loc[:, 'modified'] = [convert_str_datetime_format(s, format2=self.t_display_format) for s in tmp['modified']]
        columns = ['modified', 'loaded']
        string = u'{}\n{}'.format(repr(self)[1:-1], str(self._df[columns]))
        return string

    def __contains__(self, item):
        if (item in self.times) or (item in self.names):
            return True
        else:
            return False

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value
        self.load()

    @property
    def path(self):
        """Path to settings log files"""
        out = Path(settings_dir) / self.application
        return str(out)

    @property
    def fn(self):
        """Filename of current settings file"""
        return 'settings_log-{app}.nc'.format(app=self.application)

    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)

    @property
    def names(self):
        return list(self._df.index.values)

    @property
    def times(self):
        return self._df['modified']

    @property
    def file_exists(self):
        """Return True if this log file already exists"""
        return os.path.isfile(self.fn_path)

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

if __name__ == '__main__':
    s = Settings('test', 'default')
    print(s)
    s.add_columns(['Description', 'I/O', 'Precedence', 'Representation'])
    print(s)
    s.add_item('path', '~')

    # print(s.columns)
    # print(s.items)
    print(s)
    pass