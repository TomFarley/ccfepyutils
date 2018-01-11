#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict
from datetime import datetime
import time
from copy import deepcopy
import os

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
        self.application = application
        self.name = name
        self.state = State(self, self.state_table, 'init')
        self.t_created = None
        self.t_modified = None
        self.log_file = None

        self.instances[application][name] = self
        self.call_table = {'modified': {'exit': [self.save]}}
        self.set_t_created()
        self.modified()
        if self.file_exists:
            self.load()
            self.state('saved')
        else:
            self.df = pd.DataFrame({'value': []})  # Initialise empty dataframe
            self.state('modified')

    def __str__(self):
        # TODO: set ordering
        return str(self.df)

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


            while self.datetime2str(datetime.now()) in log.index:
                time.sleep(1.0)
        return self.datetime2str(datetime.now())

    def set_t_created(self):
        self.t_created = self.new_time()

    def set_t_modified(self):
        self.t_modified = self.new_time()

    def modified(self):
        """Set modified time string"""

    def datetime2str(self, time):
        string = time.strftime(self.time_format)
        return string

    def str2datetime(self, string):
        time = datetime.strptime(string, self.time_format)
        return time

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
    def __init__(self, application):
        """ """
        assert isinstance(application, str)
        self.application = application

    @property
    def path(self):
        """Path to settings log files"""
        ## TODO: Load from config file
        return os.path.expanduser('~/.ccfetools/setting/')

    @property
    def fn(self):
        """Filename of current settings file"""
        return 'settings-{app}.hdf'.format(app=self.application)

    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)

    def read(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

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