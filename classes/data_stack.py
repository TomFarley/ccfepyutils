#!/usr/bin/env python

import xarray as xr
import numpy as np
from collections import defaultdict, OrderedDict

from copy import deepcopy

from elzar.tools.utils import isclose_within, find_nearest

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)



class Slice(object):

    def __init__(self, stack, dim, value):
        assert issubclass(stack, Stack)
        self.stack = stack
        self.dim = stack.dim2xyz(dim)
        self.value = stack.closest_coord(dim, value)

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def plot(self):
        pass

class Stack(object):
    """Class for 3d data composed of a set of 2d data frames"""
    slice_class = Slice
    xyz2num = {'x': 1, 'y': 2, 'z': 0}

    def __init__(self, x, y, z, values=None, name=None):
        #TODO: convert param objects to dict or vv
        self._x = x
        self._y = y
        self._z = z
        self._values = values
        self.name = name
        self.data = None
        # If not initialised here, the xarray will be initialised when it is first accessed
        if values is not None:
            self.set_data()

        if values is not None:
            self.set_data()

    @property
    def dims(self):
        return [i['name'] for i in (self._z, self._x, self._y)]

    @property
    def coords(self):
        return OrderedDict(((i['name'], i['values']) for i in (self._z, self._x, self._y)))

    @property
    def shape(self):
        return tuple((len(i['values']) for i in (self._z, self._x, self._y)))

    @property
    def dim_xyz(self):
        return OrderedDict(((d['name'], x) for (d, x) in zip((self._z, self._x, self._y), ('x', 'y', 'z'))))

    def _init_xarray(self, values=None, x=None, y=None, z=None):
        self._values, self._x, self._y, self._z = none_filter((self._values, self._x, self._y, self._z),
                                                              (values, x, y, z))
        if values is None:
            values = np.empty(self.shape) * np.nan
        logger.info('coords: {}'.format(coords))
        self.data = xr.DataArray(values, coords=self.coords, dims=self.dims)

    def set_data(self, values=None, reset=False):
        """Ready xarray for data access"""
        self._values = none_filter(self._values, values)
        assert self._values is not None, 'Cannot initialise xarray without data values'
        if self.data is None or reset:
            self._init_xarray()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        raise NotImplementedError

    def dim2xyz(self, key):
        if key in ['x', 'y', 'z']:
            return key
        elif key in self.dims:
            return self.dim_xyz[key]
        else:
            raise ValueError('Stack dimension {} is not a valid dimension name: {}'.format(key, self.dim_xyz))

    def dims2xyz(self, input):
        if isinstance(input, dict):
            out = {}
            for key, value in input.items():
                out[self.dim2xyz(key)] = value
        else:
            out = []
            for key in input:
                out.append(self.dim2xyz(key))
        return out

    def closest_coord(self, dim, value, tol=None):
        """Return value in coordinates closest to value"""
        #TODO: implement tol
        dim = self.dim2xyz(dim)
        value = find_nearest(value, self.coords[dim])
        return value


    def mask(self, **kwargs):
        """Return mask corresponding to supplied coordiante values"""
        # Convert dimension names to x, y, z
        kwargs = self.dims2xyz(kwargs)
        input = {'x': None, 'y': None, 'z': None}.update(kwargs)
        # Replace 'average' arguments with None's in order to return all values for later averaging
        for key, value in input.items():
            if value == 'average':
                input[key] = None
        coords = self.coords

        masks = {c: ~np.isnan(coords[c]) if input[c] is None else isclose_within(coords[c], input[c])
                 for c in ['x', 'y', 'z']}
        return masks


    def extract(self, ):

        raise NotImplementedError

    def slice(self, **kwargs):
        assert len(kwargs) == 1
        dim, value = kwargs.keys()[0], kwargs.values()[0]
        return self.slice_class(self, dim, value)






def return_none():
    return None

def none_filter(old, new):
    """Return new values, replacing None values with corresponding old values"""
    for i, (o, n) in enumerate(zip(old, new)):
        if n is not None:
            old[i] = n
    return old

if __name__ == '__main__':
    coords = {'x': defaultdict(return_none, name='R'),
                   'y': defaultdict(return_none, name='tor'),
                   'z': defaultdict(return_none, name='t')}
    coords2 = deepcopy(coords)
    coords2 = coords2
    coords2['x']['values'] = np.linspace(1.36, 1.42, 3)
    coords2['y']['values'] = np.linspace(-0.8, 0.8, 5)
    coords2['z']['values'] = np.linspace(0.217, 0.218, 2)

    coords = coords2
    stack = Stack(coords['x'], coords['y'], coords['z'])
    logger.info(repr(stack))
    pass