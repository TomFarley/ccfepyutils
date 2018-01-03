#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict

from copy import deepcopy

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)


class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes"""
    instances = nested_dict()

    def __init__(self, application, name='default'):
        """ """
        assert isinstance(application, str)
        assert isinstance(name, str)
        assert (application, name) not in self.instances.keys_flat(), 'Setting object {}:{} already exists'.format(
                application, name)
        self._application = application
        self._name = name

        self.instances[application][name] = self

        self.df = pd.DataFrame({'value': []})  # Initialise empty dataframe

    def __str__(self):
        # TODO: set ordering
        return str(self.df)

    @property
    def items(self):
        return self.df.index.values

    @property
    def columns(self):
        return self.df.columns.values

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


if __name__ == '__main__':
    s = Settings('test', 'default')
    print(s)
    s.add_columns(['Description', 'I/O', 'Precedence', 'Representation'])
    print(s)
    s.add_item('path', '~')

    # print(s.columns)
    # print(s.items)
    print(s)