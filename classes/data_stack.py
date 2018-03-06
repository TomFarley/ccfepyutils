#!/usr/bin/env python

import xarray as xr
import numpy as np
import pandas as pd
# import multiprocessing as mp
import concurrent.futures as cf
from collections import defaultdict, OrderedDict

from copy import deepcopy

from ccfepyutils.utils import isclose_within, make_itterable, class_name, args_for
from ccfepyutils.data_processing import find_nearest
from ccfepyutils.classes.plot import Plot
from ccfepyutils.utils import return_none, none_filter, lookup_from_dataframe

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

pool = cf.ThreadPoolExecutor(max_workers=None)

class Slice(object):

    def __init__(self, stack, dim, value, roi=None):
        assert issubclass(type(stack), Stack)
        self.stack = stack
        self.dim = stack.xyz2dim(dim)
        self.xyz = stack.dim2xyz(dim)
        self.idim = stack.xyz2num[self.xyz]
        self.iother = [0, 1, 2]
        self.iother.pop(self.idim)  # dimensions remaining in slice
        self.dim_other = [self.stack.dims[i] for i in self.iother]
        self.value = stack.closest_coord(dim, value)
        self.roi = roi

    @property
    def data(self):
        # TODO cache xarray
        # TODO: use roi
        if not self.roi:
            return self.stack._data.loc[{self.dim: self.value}]
        else:
            raise NotImplementedError

    @data.setter
    def data(self, value):
        if not self.roi:
            self.stack._data.loc[{self.dim: self.value}] = value
        else:
            raise NotImplementedError

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
    
    @property
    def shape(self):
        return self.stack.slice_shape

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return '<Slice: {name}[{dim}={value}]'.format(name=self.stack.name, dim=self.dim, value=self.value)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        # if isinstance(item, slice):
        #     # Expand the slice object using range()
        #     # to a maximum of eight items.
        # return [self[x] for x in range(*n.indices(8))]
        return self.data.values[item]

    def __setitem__(self, key, value):
        """Set slice data in Stack xarray"""
        self.data.values[key] = value

    def plot(self, ax=None, **kwargs):
        # TODO: Add automatic axis labeling once parameter class is complete
        # NOTE: data is stored in format x, y whereas imshow and contourf expect arrays ordered (y, x)
        kws = {'mode': 'contourf', 'cmap': 'viridis', 'show': False, 'transpose': True}
        kws.update(kwargs)
        show = args_for(Plot.show, kws)
        axes = self.stack.slice_axes_names
        values = self.stack.slice_axes_values
        x = values[0]
        y = values[1]
        z = self.data.values
        plot = Plot(x, y, z, ax=ax, num=repr(self), show=False, xlabel=axes[0], ylabel=axes[1], **kws)
        # plot.set_axis_labels(axes[0], axes[1])
        plot.show(**show)
        return plot


class Stack(object):
    """Class for 3d data composed of a set of 2d data frames"""
    slice_class = Slice
    xyz2num = {'x': 0, 'y': 1, 'z': 2}  # axis index value of each coordinate
    num2xyz = {v: k for k, v in xyz2num.items()}  # reverse lookup direction

    def __init__(self, x, y, z, values=None, name=None, quantity=None, stack_axis='x'):
        #TODO: convert param objects to dict or vv
        self._reset_stack_attributes()
        self.x_obj = x  # dict or Param object containing at least name and values of x coordinate
        self.y_obj = y  # dict or Param object containing at least name and values of y coordinate
        self.z_obj = z  # dict or Param object containing at least name and values of z coordinate
        self._values = np.array(values) if values is not None else None  # 3D data indexed by x, y and z
        self.name = name  # Name of stack object
        self.quantity = quantity  # Quantity data represents eg intensity
        self.set_stack_axis(stack_axis)  # Axis along which to consider the data a stack of data slices

        # If not initialised here, the xarray will be initialised when it is first accessed
        if values is not None:
            self.set_data()

        self._slices = {}  # Cache of slice objects

        # self.set_stack_axis()
        self._check_types()  # Check consistency of input types etc
        self.initialise_meta()

    def _reset_stack_attributes(self):
        """Set all Stack class attributes to None"""
        self._data = None  # xarray containing data values
        self._values = None
        self._meta = None  # Dataframe of meta data related to each slice

        self._name = None  # Name of stack
        self._quantity = None  # Quantity represented by data

        self._window = None  # Subwindow of stack axis to consider TODO: implement stack window
        self._roi = None  # Region of interest (sub-window each slice) TODO: implement stack roi
        self._gui = None  # GUI window instance for navigating the data TODO: implement stack gui
        
        self._name = None
        self._stack_axis = None
        self._slices = None
        self._x = None
        self._y = None
        self._z = None
        
    def _check_types(self):
        """Check internal attributes are all valid types"""
        # TODO replace with property setters and update method?
        assert isinstance(self._x, (dict,))
        assert isinstance(self._y, (dict,))
        assert isinstance(self._z, (dict,))
        assert isinstance(self._values, (np.ndarray, type(None)))
        assert isinstance(self._name, (str, type(None)))
        assert self._stack_axis in ['x', 'y', 'z']

    def initialise_meta(self, columns=()):
        """Initialise meta data dataframe with coordinate values"""
        if self.stack_axis_values is not None:
            self._meta = pd.DataFrame({self.stack_axis: self.stack_axis_values})

    def set_stack_axis(self, coord):
        """Change stack axis coordinate"""
        coord = self.dim2xyz(coord)  # Axis along which to consider the data a stack of data slices
        self._stack_axis = coord
        # raise NotImplementedError

    def coord_obj(self, key):
        """Return coordiante object"""
        xyz2obj = {'x': self._x, 'y': self._y, 'z': self._z}
        key = self.dim2xyz(key)
        assert key in xyz2obj.keys()
        return xyz2obj[key]

    def _init_xarray(self, values=None, x=None, y=None, z=None, refresh=False):
        """Set up xarray for main Movie (not enhanced Movie)"""
        # If already set up and not needing refreshing, skip
        if self._data is not None and not refresh:
            return
        if not self.coord_obj_values_set:
            raise ValueError('Cannot initialise xarray without coordinate values')
        self._values, self._x, self._y, self._z = none_filter((self._values, self._x, self._y, self._z),
                                                              (values, x, y, z))
        self._fill_values()
        self._data = xr.DataArray(self._values, coords=self.coords.values(), dims=self.dims, name=self.quantity)
        logger.debug('Initialised xarray values for {}'.format(repr(self)))

    def _fill_values(self):
        """Called by Stack when data is accessed to ensure self._values is not empty"""
        if self._values is None:
            logger.debug('Setting xarray data values for {}'.format(repr(self)))
            self._values = np.empty(self.shape) * np.nan
            logger.debug('Generated zeroed data to fill xarray for {}'.format(repr(self)))

    def set_dimensions(self, x=None, y=None, z=None):
        """Set xarray coordinate dimensions"""
        self._x, self._y, self._z = none_filter((self._x, self._y, self._z), (x, y, z))
        self._check_types()  # Check consistency of input types etc
        return self

    def set_data(self, values=None, reset=True):
        """Ready xarray for data access"""
        self._values = none_filter(self._values, values)
        assert self._values is not None, 'Cannot initialise xarray without data values'
        if self.data is None or reset:
            self._init_xarray()
        return self

    def loc(self, *args, **kwargs):
        self._init_xarray()
        if self._data is not None:
            return self._data.loc[args]

    def lookup_slice_index(self, index):
        values = self.coord_obj(self._stack_axis)['values']
        if index not in values:
            if index in np.arange(len(values)):
                index = values[index]
            else:
                raise IndexError('Index "{}" is not a stack axis coordinate. Stack axis coords: {}'.format(item, values))
        return index

    def get_slice(self, item):
        """Get slice object at given stack coord index"""
        # Check item is valid stack coord index
        values = make_itterable(self.coord_obj(self._stack_axis)['values'])
        if not item in values:
            raise ValueError('Item {} not in stack coordinate values: {}'.format(item, values))

        # Return existing instance or create new
        if item in self._slices.keys():
            return self._slices[item]
        else:
            return self.create_slice(item)

    def create_slice(self, item, raw=False):
        if item in make_itterable(self.coord_obj(self._stack_axis)['values']):
            slice = self.slice_class(self, self._stack_axis, item)
            self._slices[item] = slice
            return slice
        else:
            raise NotImplementedError

    def remove_slice(self, item):
        if item in self._slices.keys():
            self._slices.pop(item)

    def extract_contiguous_chunk(self, x='all', y='all', z='all'):
        """Extract chunk of data from stack within coordinate ranges"""
        x_all = self.x_obj['values']
        y_all = self.y_obj['values']
        z_all = self.z_obj['values']
        if x == 'all':
            x = [np.min(x_all), np.max(x_all)]
        if y == 'all':
            y = [np.min(y_all), np.max(y_all)]
        if z == 'all':
            z = [np.min(z_all), np.max(z_all)]
        ix = x[0] < x_all < x[1]
        iy = y[0] < y_all < y[1]
        iz = z[0] < z_all < z[1]
        out = self.df.loc[{self.xyz2dim(xyz): values for xyz, values in zip(['x', 'y', 'z'], [ix, iy, iz])}]
        return out

    def __call__(self, **kwargs):
        assert len(kwargs) > 0, 'Stack.__call__ requires keyword arg meta data to select frame'
        item = self.lookup(self.stack_dim, **kwargs)
        return self.__getitem__(item)

    def __getitem__(self, item):
        """Return data slices from the stack of data"""
        self._init_xarray()
        item = self.lookup_slice_index(item)
        return self.get_slice(item)

    def __repr__(self):
        if self._data is None:
            return '<Stack, {}>'.format(self.dims)
        else:
            return '<Stack, {}>'.format(repr(self._data))

    def __str__(self):
        return '{}\n{}'.format(repr(self), self._data)

    def xyz2dim(self, key):
        if key in self.dims:
            return key
        elif key in ['x', 'y', 'z']:
            return self.xyz_to_dim[key]
        else:
            raise ValueError('Stack dimension {} is not a valid dimension name: {}'.format(key, self.xyz_to_dim))

    def dim2xyz(self, key):
        if key in ['x', 'y', 'z']:
            return key
        elif key in self.dims:
            return self.dim_to_xyz[key]
        else:
            raise ValueError('Stack dimension {} is not a valid dimension name: {}'.format(key, self.dim_to_xyz))

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
        #TODO: use meta
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

    def extract(self):

        raise NotImplementedError

    def slice(self, **kwargs):
        assert len(kwargs) == 1
        dim, value = kwargs.keys()[0], kwargs.values()[0]
        return self.slice_class(self, dim, value)

    def window(self):
        """Wibndow the data so return letterboxed subset of data perp to stack axis"""
        raise NotImplementedError

    def lookup(self, output='slice_coord', **kwargs):
        """Return meta data value corresponding to supplied meta data input
        :param value: value of inp to look up corresponding out value for
        :param inp: coordinate type of 'value'
        :param out: output meta data type to look up
        :return: out value"""
        if output == 'slice_coord':
            output = self.stack_dim
        new_value = lookup_from_dataframe(self._meta, output, **kwargs)
        return new_value

    def get(self, var, **kwargs):  # TODO: Need to extend to multiple kwargs
        """ Return value of var in self.frames_meta for which each keyword value is satisfied
        Example call signature
        self.lookup('n', t=0.2615)  # Return frame number of frame with time stamp 't'=0.2615 s
        """
        # TODO: add meta dataframe tied to stack axis
        raise NotImplementedError
        if len(kwargs) == 0:
            return None
        assert len(kwargs) == 1
        assert var in self.frames_meta.columns, 'Lookup variable "{}" not recognised. frameHistory.lookup requries ' \
                                                'one of the following metadata: {}'.format(var,
                                                                                           self.frames_meta.columns)
        for key, value in kwargs.iteritems():
            try:
                assert key in self.frames_meta.columns.tolist(), 'index varaible "{}" not recognised. ' \
                                                                 'frameHistory.lookup requries one of the following ' \
                                                                 'metadata: {}'.format(var,
                                                                                       self.frames_meta.columns)
            except:
                pass
            series = self.frames_meta[self.frames_meta[key] == value][var]
            # idx1.intersection(idx2)

        if len(series.values) == 1:
            return series.values[0]
        else:
            return series.values

    @property
    def data(self):
        """Main data xarray"""
        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(value, xr.DataArray)
        self._data = value

    @property
    def dims(self):
        """Names of coordinate dimensions"""
        return [i['name'] for i in self.coord_objs.values()]

    @property
    def coords(self):
        """Coordinate values"""
        return OrderedDict(((i['name'], i['values']) for i in self.coord_objs.values()))

    @property
    def shape(self):
        """Shape of 3D data stack xarray"""
        if self.coord_obj_values_set:
            return tuple((len(i['values']) for i in self.coord_objs.values()))
        else:
            return None

    @property
    def slice_shape(self):
        """"""
        if self.coord_obj_values_set:
            slice_shape = tuple(len(values) for values in self.slice_axes_values)
            return slice_shape
        else:
            return None

    @property
    def stack_axis_length(self):
        """"""
        if self.coord_obj_values_set:
            return len(self.stack_axis_values)
        else:
            return None

    @property
    def dim_to_xyz(self):
        """Dict linking dimension names to x, y, z equivalent"""
        return OrderedDict(((d['name'], x) for (d, x) in zip(self.coord_objs, ('x', 'y', 'z'))))

    @property
    def xyz_to_dim(self):
        """Dict linking x, y, z to dimension name"""
        return OrderedDict(((x, d['name']) for (d, x) in zip(self.coord_objs.values(), ('x', 'y', 'z'))))

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
            return '{}{}'.format(class_name(self), id(self))

    @name.setter
    def name(self, value):
        """Name identifying stack instance"""
        assert isinstance(value, (str, type(None)))
        # refresh = self._quantity is not None
        self._name = value
        # self._init_xarray(refresh=True)

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
        assert isinstance(value, (str, type(None))), '{}: {}'.format(value, type(value))
        # refresh = self._quantity is not None
        self._quantity = value
        # self._init_xarray(refresh=refresh)

    @property
    def xyz_order(self):
        """Index ordering of x, y and z coords"""
        return tuple(self.num2xyz[i] for i in (0, 1, 2))

    @property
    def stack_axis(self):
        """Name of xyz coordinate along stack axis"""
        return self._stack_axis

    @property
    def stack_dim(self):
        """Name of coordinate dimension along stack axis"""
        return self.xyz2dim(self._stack_axis)

    @property
    def slice_axes(self):
        """Names of coordinates in a slice ie perpendicular to the stack axis"""
        out = list(self.xyz_order)
        out.pop(out.index(self.stack_axis))
        return out

    @property
    def slice_dims(self):
        """Names of coordinates in a slice ie perpendicular to the stack axis"""
        out = self.dims
        out.pop(out.index(self.stack_dim))
        return out

    @property
    def stack_axis_values(self):
        """Coordinate values along stack axis"""
        return self.coord_objs[self.stack_axis]['values']

    @property
    def slice_axes_values(self):
        """Coordinate values along slice axes"""
        axes = self.slice_axes
        return [self.coord_obj(coord)['values'] for coord in axes]

    @property
    def slice_axes_names(self):
        """Coordinate values along slice axes"""
        axes = self.slice_axes
        return [self.coord_obj(coord)['name'] for coord in axes]

    @property
    def coord_objs(self):
        """Coordinate objects ordered in index order"""
        xyz2obj = {'x': self._x, 'y': self._y, 'z': self._z}
        return OrderedDict(((xyz, xyz2obj[xyz]) for xyz in self.xyz_order))

    @property
    def coord_obj_values_set(self):
        return all((self._x['values'] is not None, self._y['values'] is not None, self._z['values'] is not None))
    
    @property
    def meta(self):
        return self._meta

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