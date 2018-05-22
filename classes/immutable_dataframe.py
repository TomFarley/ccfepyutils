#!/usr/bin/env python

""" 
Taken from https://stackoverflow.com/questions/24928306/pandas-immutable-dataframe#
"""

import logging, os, itertools, re, inspect, configparser, time
import hashlib
import warnings

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class _ReadOnly(object):

    def __init__(self, obj, extraFilter=tuple()):

        self.__dict__['_obj'] = obj
        self.__dict__['_d'] = None
        self.__dict__['_extraFilter'] = extraFilter
        m = hashlib.md5()
        m.update(bytes(str(obj), 'utf-8'))
        hash = m.hexdigest()
        hash_int = int(hash, 16)
        self.__dict__['_hash'] = hash_int  # int().hexdigest(), 16)

    @staticmethod
    def _cloak(obj):
        try:
            hash(obj)
            return obj
        except TypeError:
            return _ReadOnly(obj)

    def __getitem__(self, value):

        return _ReadOnly._cloak(self._obj[value])

    def __setitem__(self, key, value):

        raise TypeError(
            "{0} has a _ReadOnly proxy around it".format(type(self._obj)))

    def __delitem__(self, key):

        raise TypeError(
            "{0} has a _ReadOnly proxy around it".format(type(self._obj)))

    def __getattr__(self, value):

        if value in self.__dir__():
            return _ReadOnly._cloak(getattr(self._obj, value))
        elif value in dir(self._obj):
            raise AttributeError("{0} attribute {1} is cloaked".format(
                type(self._obj), value))
        else:
            raise AttributeError("{0} has no {1}".format(
                type(self._obj), value))

    def __setattr__(self, key, value):

        raise TypeError(
            "{0} has a _ReadOnly proxy around it".format(type(self._obj)))

    def __delattr__(self, key):

        raise TypeError(
            "{0} has a _ReadOnly proxy around it".format(type(self._obj)))

    def __dir__(self):

        if self._d is None:
            self.__dict__['_d'] = [
                i for i in dir(self._obj) if not i.startswith('set')
                and i not in self._extraFilter]
        return self._d

    def __repr__(self):

        return self._obj.__repr__()

    def __call__(self, *args, **kwargs):

        if hasattr(self._obj, "__call__"):
            return self._obj(*args, **kwargs)
        else:
            raise TypeError("{0} not callable".format(type(self._obj)))

    def __hash__(self):

        return self._hash

    def __eq__(self, other):

        try:
            return hash(self) == hash(other)
        except TypeError:
            if isinstance(other, list):
                try:
                    return all(zip(self, other))
                except:
                    return False
            return other == self


class DataFrameProxy(_ReadOnly):

    EXTRA_FILTER = ('drop', 'drop_duplicates', 'dropna')

    def __init__(self, *args, **kwargs):

        if (len(args) == 1 and
                not len(kwargs) and
                isinstance(args, pd.DataFrame)):

            super(DataFrameProxy, self).__init__(args[0],
                DataFrameProxy.EXTRA_FILTER)

        else:

            super(DataFrameProxy, self).__init__(pd.DataFrame(*args, **kwargs),
                DataFrameProxy.EXTRA_FILTER)



    def sort(self, inplace=False, *args, **kwargs):

        if inplace:
            warnings.warn("Inplace sorting overridden")

        return self._obj.sort(*args, **kwargs)

if __name__ == '__main__':
    from types import MappingProxyType
    from pandas.util import hash_pandas_object

    df = pd.DataFrame.from_dict({'hi': 'bye', 'good': 'bad', 'yes': 'no'}, orient='index')
    df2 = DataFrameProxy(df)
    df3 = MappingProxyType(df.to_dict())
    d = df.to_dict()
    df4 = tuple((k, d[k]) for k in sorted(d.keys()))
    df5 = hash_pandas_object(df)
    print(hash(df2))
    print(repr(df2))
    print(df4)
    h = hashlib.new('sha1')
    h.update(bytes(str(df5), 'utf-8'))
    hash_id = h.hexdigest()
    print(hash_id)

    # print(hash_pandas_object(df))