#!/usr/bin/env python

import xarray as xr
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers

from copy import deepcopy

from ccfepyutils.utils import isclose_within, find_nearest, make_itterable
from ccfepyutils.classes.plot import Plot
from classes.utils import return_none, none_filter

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)



class Slice(object):

    def __init__(self, stack, dim, value):
        assert issubclass(type(stack), Stack)
        self.stack = stack
        self.dim = stack.xyz2dim(dim)
        self.xyz = stack.dim2xyz(dim)
        self.idim = stack.xyz2num[self.xyz]
        self.iother = [0, 1, 2]
        self.iother.pop(self.idim)  # dimensions remaining in slice
        self.dim_other = [self.stack.dims[i] for i in self.iother]
        self.value = stack.closest_coord(dim, value)

    @property
    def data(self):
        # TODO cache xarray
        return self.stack.data.loc[{self.dim:self.value}]

    @property
    def df(self):
        dim_other = self.dim_other
        coords = [self.stack.coord_obj(dim) for dim in dim_other]
        values = self.data.values.T
        # if reverse_y:
        #     values = values[::-1, :]
        df = pd.DataFrame(values, index=coords[1]['values'], columns=coords[0]['values'])
        df.index.name = coords[1]['name']
        df.columns.name = coords[0]['name']
        return df

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return '<Slice: {name}[{dim}={value}]'.format(name=self.stack.name, dim=self.dim, value=self.value)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.data.values[item]

    def plot(self, **kwargs):
        # TODO: Add automatic axis labeling once parameter class is complete
        Plot(self.data.values, **kwargs)
        return Plot


class Stack(object):
    """Class for 3d data composed of a set of 2d data frames"""
    slice_class = Slice
    xyz2num = {'x': 0, 'y': 1, 'z': 2}  # axis index value of each coordinate
    num2xyz = {v: k for k, v in xyz2num.items()}  # reverse lookup direction

    def __init__(self, x, y, z, values=None, name=None, quantity=None, stack_axis='x'):
        #TODO: convert param objects to dict or vv
        self.data = None
        self.x_obj = x
        self.y_obj = y
        self.z_obj = z
        self._values = np.array(values)
        self.name = name
        self.quantity = quantity
        self._stack_axis = self.dim2xyz(stack_axis)
        # If not initialised here, the xarray will be initialised when it is first accessed
        if values is not None:
            self.set_data()

        self._check_types()

    def _check_types(self):
        """Check internal attributes are all valid types"""
        # TODO replace with property setters and update method?
        assert isinstance(self._x, (dict,))
        assert isinstance(self._y, (dict,))
        assert isinstance(self._z, (dict,))
        assert isinstance(self._values, np.ndarray)
        assert isinstance(self._name, (str, type(None)))
        assert self._stack_axis in ['x', 'y', 'z']

    @property
    def x_obj(self):
        return self._x

    @x_obj.setter
    def x_obj(self, value):
        assert isinstance(value, (dict,))
        # TODO: assert contains relevent keys with appropriate formats ideally via param class
        self._x = value


    @property
    def y_obj(self):
        return self._y

    @y_obj.setter
    def y_obj(self, value):
        assert isinstance(value, (dict,))
        # TODO: assert contains relevent keys with appropriate formats ideally via param class
        self._y = value


    @property
    def z_obj(self):
        return self._z

    @z_obj.setter
    def z_obj(self, value):
        assert isinstance(value, (dict,))
        # TODO: assert contains relevent keys with appropriate formats ideally via param class
        self._z = value

    @property
    def x(self):
        return self.x_obj['values']

    @x.setter
    def x(self, value):
        # assert
        self.x_obj['values'] = value

    @property
    def y(self):
        return self.y_obj['values']

    @y.setter
    def y(self, value):
        # assert
        self.y_obj['values'] = value

    @property
    def z(self):
        return self.z_obj['values']

    @z.setter
    def z(self, value):
        # assert
        self.z_obj['values'] = value

    @property
    def name(self):
        """Name identifying stack instance"""
        if self._name is not None:
            return self._name
        else:
            return 'Stack{}'.format(id(self))

    @name.setter
    def name(self, value):
        """Name identifying stack instance"""
        assert isinstance(value, (str, type(None)))
        self._name = value
        self._init_xarray(refresh_only=True)

    @property
    def quantity(self):
        """Quantity of values in data"""
        if self._quantity is not None:
            return self._quantity
        else:
            return 'Values'

    @quantity.setter
    def quantity(self, value):
        """Quantity of values in data"""
        assert isinstance(value, (str, type(None)))
        self._quantity = value
        self._init_xarray(refresh_only=True)

    @property
    def xyz_order(self):
        """Index ordering of x, y and z coords"""
        return tuple(self.num2xyz[i] for i in (0, 1, 2))

    @property
    def coord_objs(self):
        """Coordinate objects ordered in index order"""
        xyz2obj = {'x': self._x, 'y': self._y, 'z': self._z}
        return tuple(xyz2obj[xyz] for xyz in self.xyz_order)

    def coord_obj(self, key):
        xyz2obj = {'x': self._x, 'y': self._y, 'z': self._z}
        key = self.dim2xyz(key)
        assert key in xyz2obj.keys()
        return xyz2obj[key]

    @property
    def dims(self):
        """Names of coordinate dimensions"""
        return [i['name'] for i in self.coord_objs]

    @property
    def coords(self):
        """Coordinate values"""
        return OrderedDict(((i['name'], i['values']) for i in self.coord_objs))

    @property
    def shape(self):
        return tuple((len(i['values']) for i in self.coord_objs))

    @property
    def dim_xyz(self):
        """Dict linking dimension names to x, y, z equivalent"""
        return OrderedDict(((d['name'], x) for (d, x) in zip(self.coord_objs, ('x', 'y', 'z'))))

    @property
    def xyz_dim(self):
        """Dict linking x, y, z to dimension name"""
        return OrderedDict(((x, d['name']) for (d, x) in zip(self.coord_objs, ('x', 'y', 'z'))))

    def _init_xarray(self, values=None, x=None, y=None, z=None, refresh_only=False):
        if refresh_only and self.data is None:
            return 
        self._values, self._x, self._y, self._z = none_filter((self._values, self._x, self._y, self._z),
                                                              (values, x, y, z))
        if self._values is None:
            self._values = np.empty(self.shape) * np.nan
        self.data = xr.DataArray(self._values, coords=self.coords, dims=self.dims, name=self.quantity)

    def set_data(self, values=None, reset=True):
        """Ready xarray for data access"""
        self._values = none_filter(self._values, values)
        assert self._values is not None, 'Cannot initialise xarray without data values'
        if self.data is None or reset:
            self._init_xarray()

    def loc(self, *args, **kwargs):
        self._init_xarray()
        if self.data is not None:
            return self.data.loc[args]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        """Return data slices from the stack of data"""
        self._init_xarray()
        if item in make_itterable(self.coord_obj(self._stack_axis)['values']):
            return Slice(self, self._stack_axis, item)
        else:
            raise NotImplementedError
        if self.data is not None:
            return self.data

    def __repr__(self):
        if self.data is None:
            return '<Stack, {}>'.format(self.dims)
        else:
            return '<Stack, {}>'.format(repr(self.data))

    def __str__(self):
        raise NotImplementedError

    def xyz2dim(self, key):
        if key in self.dims:
            return key
        elif key in ['x', 'y', 'z']:
            return self.xyz_dim[key]
        else:
            raise ValueError('Stack dimension {} is not a valid dimension name: {}'.format(key, self.xyz_dim))

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
        dim = self.xyz2dim(dim)
        value = find_nearest(self.coords[dim], value, index=False)
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


class Movie(Stack):
    # TODO: Load compatibilities from config file
    compatibities = dict((
        ('MAST', ['SA1.1']),
    ))

    def __init__(self, pulse=None, machine=None, camera=None):
        self.pulse = pulse
        self.machine = machine
        self.camera = camera

        x = defaultdict(return_none, name='t')
        y = defaultdict(return_none, name='pix_x')
        z = defaultdict(return_none, name='pix_y')
        quantity = defaultdict(return_none, name='pix_intensity')

        super(Movie, self).__init__(x, y, z, quantity=quantity, values=None, name=None, stack_axis='x')
        self.load_movie(pulse, machine, camera)

    @property
    def pulse(self):
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        assert isinstance(value, numbers.Number)
        self._pulse = value

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, value):
        assert value in self.compatibities
        self._machine = value

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        assert value in self.compatibities[self.machine]
        self._camera = value

    def load_movie(self, pulse=None, machine=None, camera=None):
        raise NotImplementedError

if __name__ == '__main__':
    coords = {'y': defaultdict(return_none, name='R'),
              'z': defaultdict(return_none, name='tor'),
              'x': defaultdict(return_none, name='t')}
    coords2 = deepcopy(coords)
    coords2 = coords2
    coords2['x']['values'] = np.linspace(0.217, 0.218, 2)
    coords2['y']['values'] = np.linspace(1.36, 1.42, 3)
    coords2['z']['values'] = np.linspace(-0.8, 0.8, 5)


    coords = coords2
    stack = Stack(coords['x'], coords['y'], coords['z'])
    xv, yv, zv = np.meshgrid(stack.x, stack.y, stack.z, indexing='ij')
    values = xv * yv**2 * np.sin(zv*np.pi/0.8)
    stack.set_data(values)
    logger.info(repr(stack))

    # slice = stack.loc(R=1.36)
    slice = stack[0.217]
    print(stack[0.217])
    slice.df
    slice.plot(show=True)
    pass