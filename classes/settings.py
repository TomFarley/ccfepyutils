#!/usr/bin/env python

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

from ccfepyutils.classes.state import State

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)


class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes"""
    instances = nested_dict()
    time_format = "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl='')
    state_table = {'init': ['modified', 'saved'],
                   'modified': ['saved'],
                   'saved': ['modified'],
                   'accessed': ['modified']
                   }

    def __init__(self, application, name=None):
        """ """
        assert isinstance(application, str)
        assert isinstance(name, str)
        assert (application, name) not in self.instances.keys_flat(), 'Setting object {}:{} already exists'.format(
                application, name)
        self.log_file = None
        self.application = application
        self.name = name
        self.state = State(self, self.state_table, 'init')
        self._t_created = None
        self._t_modified = None

        self.instances[application][name] = self
        self.call_table = {'modified': {'exit': [self.save]}}
        self.set_t_created()
        self.modified()
        if self.file_exists:
            self.load()
            self.state('saved')
        else:
            self.df = pd.DataFrame({'value': []})  # Initialise empty dataframe
            self._column_sets = {}  # Groups of columns with similar purposes
            self.state('modified')

    def __str__(self):
        # TODO: set ordering
        return repr(self.df)

    def __repr__(self):
        return '<Settings: {app};{name}, {state}>'.format(app=self._application, name=self.name, state=self.state)

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
    def items(self):
        return self.df.index.values

    @property
    def columns(self):
        return self.df.columns.values

    @property
    def path(self):
        """Path to settings files"""
        ## TODO: Load from config file
        return os.path.expanduser('~/.ccfetools/setting/{}/'.format(self.application))


    @property
    def fn(self):
        """Filename of current settings file"""
        assert self.name is not None
        return 'settings-{app}-{name}.hdf'.format(app=self.application, name=self.name)


    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)


    @property
    def file_exists(self):
        """Return True if a settings file with the current application and name already exists else False"""
        return os.path.isfile(self.fn_path)

    @property
    def t_modified(self):
        return convert_str_datetime_format(self._t_modified, format1=self.time_format)

    def read_file(self):
        raise NotImplementedError
        return df

    def load(self):
        assert os.path.isfile(self.fn_path)
        raise NotImplementedError
        self.state('saved')

    def save(self, state_transition=None):

        raise NotImplementedError

    def new_time(self):
        # TODO: update with config file
        # Find times of exisiting settings for application
        if self.file_exists:
            while t_now_str(format=self.time_format) in self.log_file:
                time.sleep(1.0)
        return t_now_str(format=self.time_format)

    def set_t_created(self):
        self._t_created = self.new_time()

    def set_t_modified(self):
        self._t_modified = self.new_time()

    def modified(self):
        """Set modified time string"""
        self._t_modified = self.new_time()
        self.log_file.update(self.name, self._t_modified)

    def add_column(self, value):
        assert value not in self.columns
        self.df[value] = self.df.index

    def add_columns(self, values):
        for value in values:
            self.add_column(value)


    def add_item(self, name, values={'value': []}, create_cols=True):
        assert name not in self.items, 'Item {} already exists'.format(name)
        if not isinstance(values, dict):
            values = {'value': values}
        if not create_cols:
            raise NotImplementedError  # Check all column values already exist

        series = pd.Series(values, name=name)
        self.df = self.df.append(series)
        # raise NotImplementedError

    def add_items(self, items):
        raise NotImplementedError


class SettingsLogFile(object):
    t_display_format = "%H:%M:%S %d/%m/%y"
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
        ## TODO: Load from config file
        return os.path.expanduser('~/.ccfetools/settings/')

    @property
    def fn(self):
        """Filename of current settings file"""
        return 'settings-{app}.nc'.format(app=self.application)

    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)

    @property
    def names(self):
        return self.df.index

    @property
    def times(self):
        return self.df['names']

    def create(self):
        # TODO: add safety checks
        self.df = pd.DataFrame({'time': []})  # Initialise empty dataframe
        self.df.index.name = 'name'
        self.save()

    def save(self):
        exists = os.path.isfile(self.fn_path)
        try:
            self.df.to_xarray().to_netcdf(self.fn_path)
            if not exists:
                logger.info('Created SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
        except PermissionError as e:
            logger.exception('Unable to write to file')

    def load(self):
        """Load settings logfile for current application"""
        if not os.path.isfile(self.fn_path):
            self.create()
            return False
        self.df = xr.open_dataset(self.fn_path).to_dataframe()
        return True

    def update(self, name, time):
        """Update the time string of a named settings configuration"""
        self.df.loc[name] = time
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

    def delete(self):
        """Delete the configurations file for the current application"""
        os.remove(self.fn_path)

    def __repr__(self):
        return '<SettingsLogFile: {app}({l})>'.format(app=self.application, l=len(self.df))

    def __str__(self):
        tmp = copy(self.df)
        tmp.loc[:, 'time'] = [convert_str_datetime_format(s, format2=self.t_display_format) for s in tmp['time']]
        string = u'{}\n{}'.format(repr(self), repr(tmp))
        return string

    def __contains__(self, item):
        if (item in self.times) or (item in self.names):
            return True
        else:
            return False

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

def t_now_str(format="%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S"):
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