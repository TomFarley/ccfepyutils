#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
""" Object for holding grid of 'integrated' field line parameter information (at the mid-plane) from which filaments
can be identified etc.
"""
import os
batch_mode = os.getenv('LOADL_ACTIVE', None)  # check if running as batch process -> change backend etc
import sys
import re
import inspect
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
# logger.propagate = False

if batch_mode == 'yes':
    try:
        plt
        print('plt already imported - cant switch to Agg backend')
    except NameError:
        import matplotlib
        matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
        print('Matplotlib backend: {}'.format(matplotlib.get_backend()))
    cython = True
    print('*** Batch process: {} ***\n'.format(batch_mode))

# logger = logging.getLogger('elzar')
print('Created logger:', logger, __name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
try:
    import cpickle as pickle
    print('Using cpickle')
except ImportError:
    import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

# from filaments import user_settings
# from figure_of_merit import FigureOfMerit
from ccfepyutils.utils import compare_dict
#
# from pyEquilibrium.equilibrium import equilibrium
from pyFastcamTools.frameHistory import frameHistory
from pyFastcamTools.Frames import Frame

try:
    import pandas as pd
    from natsort import natsorted
except ImportError:
    sys.path.append(os.path.expanduser('/home/tfarley/.local/lib/python2.7/site-packages/'))
    import pandas as pd
    from natsort import natsorted
    print('Using tfarley .local version of pandas, natsort')
pd.set_option('display.width', 190)  # wider print output in terminal (without wrapping)
# idx = pd.IndexSlice

from elzar import ElzarSettings
settings = ElzarSettings()  # settings options set in ~/elzar/settings/settings.py

class GridLimits(object):
    """ Class to define one dimension of a grid in a given parameter
    'exta' is a dictionary of information that can be useful for coordinate transformations etc
    """
    def __init__(self, **kwargs):
        from collections import OrderedDict
        self.dict = OrderedDict([('name', None), ('min', None), ('max', None), ('step', None), ('N', None),
                                 ('values', None), ('extra', {})])
        if len(kwargs) > 0:
            self.set(**kwargs)

    def set(self, name=None, min=None, max=None, step=None, N=None, values=None, extra=None):
        """ Set internal vlaues
        """
        import inspect
        (args, varargs, keywords, locals_) = inspect.getargvalues(inspect.currentframe())
        for arg in args:
            if locals_[arg] is not None and arg != 'self':
                self.dict[arg] = locals_[arg]
        self.update()

    def update(self):
        """ Update parameters so they are consistent. Priority given in order:
        1) min, max, N
        2) min, max, step
        3) values
        """
        from decimal import Decimal
        if (self['min'] is not None) and (self['max'] is not None):
            min_ = Decimal(str(self['min']))
            max_ = Decimal(str(self['max']))
            if (self['N'] is not None):
                assert np.isclose(self['N'], np.round(self['N'])), \
                    'Number of points in grid limit must be an interger. N = {}'.format(self['N'])
                N = Decimal(str(self['N']))
                step = (max_ - min_) / (N-1)
            elif (self['step'] is not None):
                step = Decimal(str(self['step']))
                N = Decimal(np.ceil(float((max_ - min_) / step))+1)
                max_ = min_ + (N-1) * step  # in case of remainder update max
                if N == 0:
                    N = 1
                    max_ = min_
            else:
                msg = 'Grid limits object (name={}) does not have enough information to be set up: {}'.format(
                                                                                            self['name'], self.dict)
                logger.warning(msg)
                raise ValueError, msg
                return
            values = np.linspace(float(min_), float(max_), int(N))
        elif self['values'] is not None:
            values = self['values']
            min_ = Decimal(str(np.min(values)))
            max_ = Decimal(str(np.max(values)))
            N = len(values)
            step = (max_ - min_) / (N-1) if N > 1 else 0.0
            if np.all(values != np.linspace(float(min_), float(max_), N)):
                logger.warning('Values passed to GridLimits object (name={}) are not regularly spaced: {}'.format(
                                                                                            self['name'], self.dict))
        else:
            msg = 'Grid limits object (name={}) does not have enough information to be set up: {}'.format(
                                                                                            self['name'], self.dict)
            logger.warning(msg)
            raise ValueError, msg
            # return
        self['min'] = float(min_)
        self['max'] = float(max_)
        self['N'] = int(N)
        self['step'] = float(step)
        self['values'] = values
        assert np.all([k is not None for k in ['min', 'max']]), 'GridLims are incomplete'

    def __getattr__(self, item):
        """ Whenever an attribute/method is not found in FilSet it will delagate to this function
        """
        keys = []
        if item.startswith('__') and item.endswith('__') and item not in keys:
            return super(GridLimits, self).__getattr__(item)
        return getattr(self.dict, item)

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, item, value):
        self.dict[item] = value

    def __repr__(self):
        if self['N'] is None:
            return '<GridLimits:EMPTY>'
        out = deepcopy(self.dict)
        out.pop('values', None)
        out = ', '.join(['{}: {}'.format(k, v) for k, v in out.iteritems()])
        out = ('<GridLimits: '+out+'>').replace("u'", "").replace('u"', "")  # Get rid of unicode prefixes
        return out

class GridRes(object):
    """ Class for defining grids with multiple parameters per axis
    """
    def __init__(self, **kwargs):
        self.name = None
        self.xlim = None
        self.ylim = None
        self.x2 = None
        self.y2 = None
        self.x2_trans = None
        self.y2_trans = None
        self.x2_name = None
        self.y2_name = None
        self.resolution = [None, None]
        self.resolution_updated = False
        if len(kwargs) > 0:
            self.set_grid(**kwargs)
        # self.set_values(values)

    # def __getattr__(self, item):
    #     """ Whenever an attribute/method is not found in FilSet it will delagate to this function
    #     """
    #     return getattr(self.__dict__, item)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __repr__(self):
        return '<GridRes; name: {name}, x: {x}, y: {y}, x2:{x2}, y2:{y2}>'.format(name=self.name,
                                        x=self.xlim, y=self.ylim, x2=self.x2_name, y2=self.y2_name)

    def set_grid(self, lookup=None, x=None, y=None, x2=None, y2=None, name=None):
        self.x2_trans = x2
        self.y2_trans = y2
        self.name = name
        if (lookup is not None):
            resolution_dict = settings(resolution=lookup)
            resolution = resolution_dict['resolution']
            self.name = resolution
            # Check if this resolution has previously been used and has not changed since last time
            path = settings.path['pickle'] / 'resolution/'
            fn = 'Rphi_resolution-{}.p'.format(resolution)
            # Check if this resolution setting has been used before (ie a pickle file already exists)
            # import pdb; pdb.set_trace()
            if fn in [x[2] for x in os.walk(str(path))][0]:
                res_old = pickle.load((path/fn).open('rb'))  # Load previous version of this resolution
                if not compare_dict(resolution_dict, res_old):  # Check if they are identical
                    print('Resolution "{}" changed. Updating picked reference resolution dictionary.\n'
                          'Old: {}\nNew: {}.'.format(resolution, res_old, resolution_dict))
                    with (path/fn).open('wb') as f:
                        pickle.dump(resolution_dict, f)
                    self.resolution_updated = True
                else:  # identical - no change
                    self.resolution_updated = False
                    logging.debug('Resolution: "{}" has not changed'.format(resolution))
            else:  # first time this resolution has been used - create a new pickle file
                with (path/fn).open('wb') as f:
                    pickle.dump(resolution_dict, f)
                self.resolution_updated = True
            try:
                x = deepcopy(resolution_dict['xlim'])
                y = deepcopy(resolution_dict['ylim'])
            except KeyError:
                raise KeyError('Grid resolution setting {} doesnt contain necessary "xlim" and "ylim" keys: {}'.format(
                    lookup, resolution_dict))
            if ('extra' in x.keys()) and ('trans' in x['extra']):
                self.x2_trans = x['extra']['trans']
            if ('extra' in y.keys()) and ('trans' in y['extra']):
                self.y2_trans = y['extra']['trans']

        if (x is not None) and (y is not None):
            if isinstance(x, GridLimits):
                self.xlim = x
            elif isinstance(x, dict):
                self.xlim = GridLimits()
                self.xlim.set(**x)
            elif isinstance(x, (np.ndarray, list, pd.Series)):
                self.xlim = GridLimits()
                self.xlim.set(values=np.array(x))
            else:
                raise ValueError('Cannot recognise format of y input: {}'.format(y.__class__))
            if isinstance(y, GridLimits):
                self.ylim = y
            elif isinstance(y, dict):
                self.ylim = GridLimits()
                self.ylim.set(**y)
            elif isinstance(y, (np.ndarray, list, pd.Series)):
                self.ylim = GridLimits()
                self.ylim.set(values=np.array(y))
            else:
                raise ValueError('Cannot recognise format of x input: {}'.format(x.__class__))
        else:
            raise ValueError('Cannot generate grid without "lookup" or "xlim" and "ylim" arguments. Locals: {}'.format(
                locals()))

        self.update()

    def update(self):
        if self.xlim is not None:
            self.x = self.xlim['values']
            self.resolution = (len(self.x), self.resolution[1])
        if self.ylim is not None:
            self.y = self.ylim['values']
            self.resolution = (self.resolution[0], len(self.y))

        if self.x2_trans is not None:
            self.transform('x', self.x2_trans)
        if self.y2_trans is not None:
            self.transform('y', self.y2_trans)

        self.x_name = self.xlim['name']
        self.y_name = self.ylim['name']

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(repr(self))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        ## Plot points in grid
        xv, yv = np.meshgrid(self.x, self.y)
        ax.plot(xv, yv, 'k.', ms=0.6, alpha=0.6)

    def transform(self, axis, func):
        """ Applies a transformation to an axis using function 'func' and places the result in self.<axis>2 """
        # import pdb; pdb.set_trace()
        assert axis == 'x' or axis == 'y'
        other = 'y' if axis == 'x' else 'x'
        if (callable(func) or isinstance(func, basestring)):
            args = []
            kwargs = {}
        else:
            (func, args, kwargs) = func  # unpack arguements
        if isinstance(func, basestring):
            func = self.__getattribute__(func)
        if '2' in func.__name__:  # Use function name to name axes
            name, name2 = func.__name__.split('2')
            getattr(self, axis+'lim')['name'] = name
            setattr(self, axis+'2_name', name2)
        kws = {k: v for k, v in self[axis+'lim']['extra'].iteritems() if k in inspect.getargspec(func)[0]}
        kwargs.update(kws)
        self[axis+'2'] = pd.DataFrame(func(self[axis], self[other], *args, **kwargs),
                                      index=self.x.flatten(), columns=self.y.flatten())
        self[axis + '2'].index.name = self.xlim['name']
        # self[axis + '2'].index.name = self.xlim['name']

    def convert(self, v1, v2, func, **kwargs):
        """ Convert input values to second coordinate system using properties of current grid instance where
        nessesary """
        if isinstance(func, basestring):
           func = self.__getattribute__(func)

        # Automatically get keywords from gridlim objects (eg. phi0)
        kwargs.update({k: v for k, v in self['xlim']['extra'].iteritems() if k in inspect.getargspec(func)[0]})
        kwargs.update({k: v for k, v in self['ylim']['extra'].iteritems() if k in inspect.getargspec(func)[0]})
        out = func(v1, v2, **kwargs)
        # Remove nested lists
        if hasattr(out, '__len__'):
            out = np.diag(out)
        if hasattr(out, '__len__') and len(out) == 1:
            while hasattr(out, '__len__'):
                out = out[0]
        return out

    def x_to(self, x, y, name, **kwargs):
        import inspect
        if self.x_name == name:
            return y
        elif self.x_name+'2'+name in [m[0] for m in inspect.getmembers(GridRes, predicate=inspect.ismethod)]:
            return getattr(self, self.x_name+'2'+name)(x, y, **kwargs)
        else:
            raise ValueError('No method for converting {}'.format(self.x_name+'2'+name))

    def y_to(self, y, x, name, **kwargs):
        import inspect
        if self.y_name == name:
            return y
        elif self.y_name+'2'+name in [m[0] for m in inspect.getmembers(GridRes, predicate=inspect.isfunction)]:
            return getattr(self, self.y_name+'2'+name)(y, x, **kwargs)
        else:
            raise ValueError('No method for converting {}'.format(self.y_name+'2'+name))


    @staticmethod
    def tor2phi(tor, R, deg=True, differential=False, phi0=None, recentred=False):
        """ Convert from tor coordinates (R*phi) to phi. Takes 1D inputs and returms 2D output. """
        if isinstance(tor, pd.DataFrame):
            tor = tor.values
        if isinstance(R, pd.DataFrame):
            R = R.values
        if phi0 is not None and phi0 > 2*np.pi:
            phi0 = np.deg2rad(phi0)
        if not differential:  # for calculating absolute positions etc
            if (phi0 is None):
                raise ValueError('Cannot perform absolute tor2phi conversion without phi0 value')
            # Subtract middle tor value so scalling by R produces ~trapesium not ~parallelogram
            midtor = np.mean(tor) if phi0 is None else 0.0
            phi0 = phi0 if phi0 is not None else (midtor / R[0])
            torv, Rv,  = np.meshgrid(tor-midtor, R)  # dtor relative to line of phi0 (centre of grid)
            # phi in rads unique for each R, tor: dphi + centre_phi
            phi = (torv / Rv) + phi0

        else:  # for calculating widths, grid spacing etc
            torv, Rv,  = np.meshgrid(tor, R)  # dtor, R
            phi = (torv / Rv)

        if deg:
            phi = np.rad2deg(phi)
        return phi

    @staticmethod
    def phi2tor(phi, R, deg=True, differential=False, phi0=None, recentre=True):
        """ Calculate tor coordinates for given R and phi coords. Takes 1D inputs and returms 2D output. """
        # import pdb; pdb.set_trace()
        if isinstance(phi, pd.DataFrame):
            phi = phi.values
        if isinstance(R, pd.DataFrame):
            R = R.values
        if deg:  # radians to degrees
            phi = np.deg2rad(copy(phi))
            phi0 = np.deg2rad(phi0) if (phi0 is not None) else None

        if not differential:  # for calculating absolute positions etc
            if (phi0 is None):
                raise ValueError('Cannot perform absolute phi2tor conversion without phi0 value')
            elif phi0 is None:
                phi0 = np.mean(phi)
            # Get phi along 'centre' line of grid in rad
            # Create 2D grids with phi set relative to 'centre' line
            phiv, Rv,  = np.meshgrid(phi-phi0, R)
            tor = (Rv * phiv)  # tor deviation from 'centre' line in meters
            if not recentre:
                # Get tor values along centre line and add them onto recentred grid
                mid_tor = (phi0 * R)
                tor += mid_tor
        else:  # for calculating widths etc
            phiv, Rv,  = np.meshgrid(phi, R)
            tor = Rv * phiv

        if len(tor) == 1:  # Remove nested arrays for single values
            tor = tor[0][0]

        return tor

    def in_grid(self, x=None, y=None, x_border=0.0, y_border=0.0, return_border=False):
        """ Return True(s) if point(s) within grid limits """
        xlim = self.xlim
        ylim = self.ylim

        border = {'x': x_border, 'y': y_border}
        border['right'] = xlim['max'] - border['x']
        border['left'] = xlim['min'] + border['x']
        border['top'] = ylim['max'] - border['y']
        border['bot'] = ylim['min'] + border['y']

        if return_border:
            return border

        assert x is not None
        # import pdb; pdb.set_trace()
        import numbers
        x = np.array([x]) if isinstance(x, numbers.Number) else np.array(x).flatten()  # flatten if converting series

        return ((x >= border['left']) & (x <= border['right']) &
                (y >= border['bot']) & (y <= border['top']))