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

from ccfepyutils.classes.state import State
from ccfepyutils.utils import mkdir, make_itterable
from ccfepyutils.netcdf_tools import dict_to_netcdf, netcdf_to_dict, set_netcdf_atrribute, dataframe_to_netcdf

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

## TODO: Load from config file
# TODO: Create base .ccfetools/settings/ directory structure if doesn't exist
settings_dir = os.path.expanduser('~/.ccfetools/settings/')

class Setting(ABC):
    def __init__(self, settings, item):
        self._settings = settings
        self._item = item
        self._df = self._settings.df.loc[[item], :]

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}={}>'.format(class_name, self._item, str(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._settings.df.loc[self._item, :]
        else:
            if item in self._settings.column_sets_names:
                item = self._settings.column_sets_names[item]
            return self._settings.df.loc[self._item, item]

    def __setitem__(self, item, value):
        if item not in self._settings.columns:
            raise ValueError('Item "{}" is not a setting column'.format(item))
        self._settings.df.loc[self._item, item] = value

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
        value = str(settings.df.loc[item, cls.value_column])
        return str.__new__(cls, value)

    def __init__(self, settings, item):
        str.__init__(self)
        Setting.__init__(self, settings, item)

class SettingInt(Setting, int):
    type = int
    value_column = 'value_num'
    def __new__(cls, settings, item):
        value = int(settings.df.loc[item, cls.value_column])
        return int.__new__(cls, value)

    def __init__(self, settings, item):
        int.__init__(self)
        Setting.__init__(self, settings, item)

class SettingFloat(Setting, float):
    type = float
    value_column = 'value_num'
    def __new__(cls, settings, item):
        value = float(settings.df.loc[item, cls.value_column])
        return float.__new__(cls, value)

    def __init__(self, settings, item):
        float.__init__(self)
        Setting.__init__(self, settings, item)

class SettingBool(Setting, int):
    type = bool
    value_column = 'value_num'
    def __new__(cls, settings, item):
        value = bool(settings.df.loc[item, cls.value_column])
        return int.__new__(cls, value)

    def __init__(self, settings, item):
        int.__init__(self)
        Setting.__init__(self, settings, item)

    def __bool__(self):
        return bool(self._settings.df.loc[self._item, 'value_num'])


class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes"""
    instances = nested_dict()
    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}

    state_table = {'init': ['modified', 'saved'],
                   'modified': ['saved'],
                   'saved': ['modified'],
                   'loaded': ['modified']
                   }
    column_sets = {'value': [('value_str', str), ('value_num', float)],
                   'info': [('name', str), ('symbol', str), ('description', str), ('unit', str)],
                   'io': [('fn_str', str), ('priority', float)],
                   'type': [('float', bool), ('int', bool), ('bool', bool), ('str', bool)],
                   'meta': [('runtime', bool)],
                   'repr': [('value_str', str), ('value_num', float), ('name', str), ('description', str)]}  # plotting?

    def __init__(self, application, name):
        """Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. my_code
        - The name is a label for a particular set of settings for the application e.g. my_default_settings
        """
        assert isinstance(application, str)
        assert isinstance(name, str)
        assert (application, name) not in self.instances.keys_flat(), ('Setting object {}:{} already exists. '
                'Use Settings.get(application, name) to ensure there is only one instance'.format(
                application, name))
        self.log_file = None
        self.application = application  # property sets logfile
        self.name = name
        self.state = State(self, self.state_table, 'init')
        self._column_sets = Settings.column_sets   # Groups of columns with similar purposes

        self.instances[application][name] = self
        self.call_table = {'modified': {'exit': [self.save]}}
        # self.modified()  # update t_modified and log_file
        if self.file_exists:
            self.load()
        else:
            self.init()

    @classmethod
    def get(cls, application, name):
        if (application, name) in cls.instances.keys_flat():
            return cls.instances[application][name]
        else:
            return Settings(application, name)

    def __str__(self):
        # TODO: set ordering
        cols = [self.column_sets_names[key] for key in ['value', 'info']]
        cols = list(itertools.chain.from_iterable(cols))
        return '{}:\n{}'.format(repr(self)[1:-1], str(self.df[cols]))

    def __repr__(self):
        return '<Settings: {app};{name}, {state}>'.format(app=self._application, name=self.name, state=self.state)

    def init(self):
        columns = list(set(itertools.chain.from_iterable(self._column_sets.values())))
        types = [value[1] for value in columns]
        columns = [value[0] for value in columns]
        self.df = pd.DataFrame({key: [] for key in columns})#, dtype=types)  # Initialise empty dataframe
        # Set the types of each column
        type_dict = [dict(v) for k,v in self.column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        self.df = self.df.astype(type_dict)
        # self.df.loc[:, [self.column_sets_names['type']]] = False
        self.state('modified')

    def __call__(self, item, value=None, create_columns=False, **kwargs):
        """Set value of setting
        :item: name of setting to change
        :value: value to set the item to
        :kwargs: Use to set non-value columns"""
        df = self.df
        # new = item not in list(self.items)
        if item not in list(self.items):  # add item setting type columns appropriately
            self.add_item(item, value)
        elif value is not None:  # set value given we already know type
            if any(df.loc[item, ['bool', 'int', 'float']]):
                df.loc[item, 'value_num'] = value
            elif df.loc[item, 'str']:
                df.loc[item, 'value_str'] = value
            else:
                raise ValueError('Setting {} has no type!'.format(item))
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
        self.log_file.updated(self.name)
        self.save()

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
        return self.log_file.df.loc[self.name, 'created']

    @property
    def t_modified(self):
        return self.log_file.df.loc[self.name, 'modified']

    @property
    def t_accessed(self):
        return self.log_file.df.loc[self.name, 'loaded']

    @property
    def items(self):
        return self.df.index.values

    @property
    def columns(self):
        return self.df.columns.values

    @property
    def column_sets_names(self):
        names = {key: [v[0] for v in value] for key, value in self._column_sets.items()}
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

    def read_file(self):
        raise NotImplementedError
        return df

    def load(self):
        assert self.file_exists
        with Dataset(self.fn_path) as root:
            # self.__dict__.update(netcdf_to_dict(root, 'meta'))  # redundant info as stored in logfile
            self._column_sets_names = netcdf_to_dict(root, 'column_sets_names')
        self.df = xr.open_dataset(self.fn_path, group='df').to_dataframe()
        self.state('saved')
        self.log_file.loaded(self.name)

    def from_config(self, fn):
        assert os.path.isfile(os.path.expanduser(fn)), 'Config file does not exist: {}'.format(fn)
        config = configparser.ConfigParser()
        config.read(fn)
        for section, item in config.items():
            if len(item) == 0:
                continue
            for key, value in item.items():
                name = '{}::{}'.format(section, key)
                # If no change required, skip
                if name in self and self[name] == value:
                    continue
                self[name] = value

    def save(self, state_transition=None):
        if not self.file_exists:
            self.create_file()
            return

        meta = {'t_created': self.t_created, 't_modified': self.t_modified, 't_accessed': self.t_accessed}
        try:
            #TODO: delete/overwrite previous values? Use with?
            os.remove(self.fn_path)
            root = Dataset(self.fn_path, "w", format="NETCDF4")
            dict_to_netcdf(root, 'meta', meta)
            dict_to_netcdf(root, 'column_sets_names', self.column_sets_names)
            root.close()
            self.df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')

        except PermissionError as e:
            logger.exception('Unable to write to file')
        except:
            root.close()
            logger.exception('Failed update Settings File for application "{app}": {path}'.format(
                    app=self.application, path=self.fn_path))
        else:
            logger.debug('Updated/saved Settings File for application "{app}": {path}'.format(
                    app=self.application, path=self.fn_path))
            self.state('saved')

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
            self.df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')

        except PermissionError as e:
            logger.exception('Unable to write to file')
        except:
            if self.file_exists:
                self.delete_file(force=True)
            logger.exception('Failed to create Settings File for application "{app}": {path}'.format(
                app=self.application, path=self.fn_path))
        else:
            logger.info('Created Settings File for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))

    def delete_file(self, force=False):
        """Delete the configurations file for the current application"""
        if not force:
            print('Are you sure you want to delete the settings file:\n{}'.format(self.fn_path))
            out = input('Y/n')
            if not (out.lower() in ('y', '')):
                return
        os.remove(self.fn_path)
        logger.warning('Deleted settings file: {}'.format(self.fn_path))

    def new_time(self, format='natural'):
        # TODO: update with config file
        # Find times of exisiting settings for application
        format_str = self.t_formats[format]
        if self.file_exists:
            while t_now_str(format=format_str) in self.log_file:
                time.sleep(1.0)
        return t_now_str(format=format_str)

    def __getitem__(self, item):
        self.check_consistency()

        df = self.df
        # Check if item is in df index or is the setting name
        if self.state == 'modified':
            self.save()
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
        if item in self.df.index:
            return True
        else:
            return False

    def __setitem__(self, item, value):
        self(item, value)

    def set(self, item, value, ignore=[None]):
        """Set setting provided it is not an ignore value"""
        if value in ignore:
            return
        # No change, so do nothing
        if item in self and self[item] == value:
            return
        self[item] = value

    def view(self, cols='repr'):
        if cols in self.column_sets_names.keys():
            cols = self.column_sets_names[cols]
        return self.df.loc[:, cols]

    def check_consistency(self):
        """Checks on consistency of dataframe"""
        # TODO: Check format of columns
        # Make sure each item only has one type
        unique_type = self.view('type').astype(int).sum(axis=1) == 1
        if not all(unique_type):
            raise ValueError('Inconsistent types:\n{}'.format(self.view('type').loc[~unique_type]))
        if True:
            self._reset_solumn_types()

    def _reset_solumn_types(self):
        """Set datatypes of dataframe columns"""
        # TODO: remove nans, etc
        type_dict = [dict(v) for k, v in self.column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        self.df = self.df.astype(type_dict)
        pass
        # raise NotImplementedError

    def add_column(self, value):
        if value in self.columns:
            return
        self.df[value] = np.nan  # self.df.index

    def add_columns(self, values):
        for value in values:
            self.add_column(value)

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
        else:
            raise TypeError('Setting {}={} of type {} is not supported.'.format(item, value, type(value)))
        self.df.loc[item, [type_to_col[dtype], dtype]] = value, True
        other_types = list(set([k for k in type_to_col if k != dtype]))
        self.df.loc[item, other_types] = False

        self.check_consistency()
        logger.info('Added new item to settings: {} = {}'.format(item, value))

    def add_items(self, items):
        raise NotImplementedError

    def rename_item(self, old_name, new_name):
        """Rename setting key"""
        assert old_name in self.df.index
        self.df = self.df.rename({old_name: new_name}, axis='index')

    def delete_items(self, items):
        """Remove item(s) from settings"""
        assert items in self.items, 'Items {} not in {}'.format(items, repr(self))
        items = make_itterable(items)
        self.df = self.df.drop(items)
        logger.info('Deleted items {} from settings: {} = {}'.format(items, repr(self)))


    def append_item(self, name, values={'value': []}, categories=[], create_cols=True):
        """Add item with an already existing name to settings.

        The item key will have a number appended to the end in format <name>::2 etc"""
        # TODO: implement settings append
        raise NotImplementedError

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

    def set_func_args(self, func, func_name=None, kwargs=None):
        raise NotImplementedError

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


class SettingsLogFile(object):
    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}
    def __init__(self, application):
        """ """
        assert isinstance(application, str)
        self.application = application

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
        return self.df.index

    @property
    def times(self):
        return self.df['modified']

    def create(self):
        # TODO: add safety checks
        self.df = pd.DataFrame({'modified': [], 'loaded': [], 'created': [],
                                'mod_count': [], 'load_count': [], 'load_count_total': [],
                                'protected': []})  # Initialise empty dataframe
        self.df.index.name = 'name'
        self.save()

    def save(self):
        exists = self.file_exists
        mkdir(self.path, depth=1)
        gc.collect()
        try:
            self.df.to_xarray().to_netcdf(self.fn_path, mode='w')
            if not exists:
                logger.info('Created SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
        except PermissionError as e:
            logger.warning('Trouble updating {}. Deleting and refreshing file'.format(repr(self)))
            try:
                self.delete_file(True)
                self.df.to_xarray().to_netcdf(self.fn_path, mode='w')
            except PermissionError as e:
                logger.exception('Unable to write to file')

    @property
    def file_exists(self):
        """Return True if this log file already exists"""
        return os.path.isfile(self.fn_path)

    def load(self):
        """Load settings logfile for current application"""
        if not self.file_exists:
            self.create()
            return False
        self.df = xr.open_dataset(self.fn_path).to_dataframe()
        return True
    
    def created(self, name, time=None, overwrite=True):
        """Update time settings file for settings name was created"""
        df = self.df
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
        if name not in self.df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self.df.loc[name, 'loaded'] = time
        self.df.loc[name, 'load_count'] += 1  # since modification
        self.df.loc[name, 'load_count_total'] += 1  # since creation

    def updated(self, name, time=None):
        """Update the time string of a named settings configuration"""
        if name not in self.df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self.df.loc[name, 'modified'] = time
        self.df.loc[name, 'mod_count'] += 1
        self.df.loc[name, 'load_count'] = 0
        self.save()

    def rename(self, old, new):
        """Rename a settings configuration and update existing instances"""
        self.df.rename(index={old: new})
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

    def __repr__(self):
        return '<SettingsLogFile: {app}({l})>'.format(app=self.application, l=len(self.df))

    def __str__(self):
        # tmp = copy(self.df)
        # tmp.loc[:, 'modified'] = [convert_str_datetime_format(s, format2=self.t_display_format) for s in tmp['modified']]
        columns = ['modified', 'loaded']
        string = u'{}\n{}'.format(repr(self)[1:-1], str(self.df[columns]))
        return string

    def __contains__(self, item):
        if (item in self.times) or (item in self.names):
            return True
        else:
            return False

    def print(self):
        print(self)

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