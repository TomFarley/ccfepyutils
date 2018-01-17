#!/usr/bin/env python

import itertools
import numpy as np
from numpy import sin, cos, arctan, tan, sqrt
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import inspect
from copy import copy, deepcopy
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

from ccfepyutils.utils import make_itterable, is_scalar, safe_len, safe_zip
from ccfepyutils.classes.plot import Plot
from ccfepyutils.classes.plot import plot_ellipses

class Ellipses(object):
    """Class for converting between different geometric descriptions of an ellipse.
    Need 3 parameters to define ellipse shape and orrientation + 2 for position
    Ellipse axes    - Major axis/diameter, minor axis/diameter, angle between major radius and x direction anticlockwise
    Axis extent     - Lengths along x & y axes through ellipse centre (Method used in Zweeben 2016)
    Extreema extent - Bounding box dimensions (axis aligned)
    Focii           -
    """
    conventions = {'ellipse_axes': ['major', 'minor', 'angle'],
                   'bounding_box': ['x_boundbox', 'y_boudbox', 'angle'],
                   'axes_extent': ['x_axis', 'y_axis', 'angle'],
                   'el_ax_sigma2hwhm': ['major_sigma2hwhm', 'minor_sigma2hwhm', 'angle_sigma2hwhm']}
    def __init__(self, arg1, arg2, arg3, convention='ellipse_axes', x=None, y=None, degrees=False, half_widths=False):
        assert safe_len(arg1) == safe_len(arg2) == safe_len(arg3)
        self.convention = convention.lower()
        self.params = {key: None for key in itertools.chain.from_iterable(self.conventions.values())}  # initialise
        self.x = x
        self.y = y
        self.half_widths = half_widths
        self.n = safe_len(arg1, scalar=0)  # number of ellipses
        params = self.params

        if convention.lower() not in self.conventions:
            raise NotImplementedError

        for key, arg in zip(self.conventions[convention], (arg1, arg2, arg3)):
            if is_scalar(arg):
                params[key] = arg
            else:
                params[key] = np.array(arg)

        if (params['angle'] is not None) and degrees:
            params['angle'] = np.deg2rad(params['angle'])

    def convert(self):
        """Calcualte parameters for other conventions"""
        convention = self.convention
        params = self.params
        args = [params[key] for key in self.conventions[convention]]
        for c in self.conventions:
            if c == convention:  # Don't convert to same type
                continue
            try:
                methods = dict(inspect.getmembers(Ellipses, predicate=inspect.ismethod))
                method = '{}_to_{}'.format(convention, c)
                c_params = self.conventions[c]
                params[c_params[0]], params[c_params[1]], params[c_params[2]] = methods[method](*args)
            except KeyError as e:
                logger.warning('Conversion {} not implemented'.format(method))
                pass

        # if convention == 'ellipse_axes':
        #     params['x_axis'], params['y_axis'], angle = self.ellipse_axes_to_axis_extent(
        #             params['major'], params['minor'], params['angle'])
        #     params['x_boundbox'], params['y_boundbox'], angle = self.ellipse_axes_to_bounding_box(
        #             params['major'], params['minor'], params['angle'])
        # else:
        #     raise NotImplementedError

    @classmethod
    def ellipse_axes_to_axis_extent(self, major, minor, angle):
        """Calculate 2 x coordinates ellipse intercepts coord axes centred around centre"""
        a = (cos(angle)/major)**2
        b = (sin(angle)/minor)**2
        dx = 2 / sqrt(a + b)
        c = (sin(angle)/major)**2
        d = (cos(angle)/minor)**2
        dy = 2 / sqrt(c + d)
        return dx, dy, angle

    @classmethod
    def ellipse_axes_to_bounding_box(self, major, minor, angle):
        """Calculate bounding box dimensions"""
        tx = arctan(-minor*tan(angle) / major)
        dx = 2 * (major*cos(tx)*cos(angle) - minor*sin(tx)*sin(angle))
        ty = arctan(-minor / (tan(angle)*major))
        dy = 2 * (major*cos(ty)*sin(angle) + minor*sin(tx)*cos(angle))
        return dx, dy, angle

    @classmethod
    def ellipse_axes_to_el_ax_sigma2hwhm(cls, major, minor, angle):
        sigma2hwhm = np.sqrt(2*np.log(2))
        major *= sigma2hwhm
        minor *= sigma2hwhm
        return major, minor, angle

    @property
    def position(self):
        return self.x, self.y

    @property
    def position_safe(self):
        """Return position (x, y) of centre of ellipse. Return (0, 0) if not provided."""
        x, y = self.x, self.y
        centre = [x, y]
        for i, v in enumerate(centre):
            if v is None and self.n > 0:
                centre[i] = np.full(self.n, 0)
            elif v is None and self.n == 0:
                centre[i] = 0
        return centre

    def get(self, convention, nested=True):
        """Get values for any convension"""
        assert convention in self.conventions
        values = [self.params[k] for k in self.conventions[convention]]
        if np.any([v is None for v in values]):
            self.convert()
            values = [self.params[k] for k in self.conventions[convention]]
        if nested:
            values = [np.array([v]) if not hasattr(v, '__iter__') else v for v in values]
        return values

    @property
    def ellipse_axes(self):
        convention = 'ellipse_axes'
        values = [self.params[k] for k in self.conventions[convention]]
        if np.any(values == None):
            self.convert()
            values = [self.params[k] for k in self.conventions[convention]]
        return values

    @property
    def bounding_box(self):
        convention = 'bounding_box'
        values = [self.params[k] for k in self.conventions[convention]]
        if np.any(values == None):
            self.convert()
            values = [self.params[k] for k in self.conventions[convention]]
        return values

    @property
    def axes_extent(self):
        convention = 'axes_extent'
        values = [self.params[k] for k in self.conventions[convention]]
        if np.any(values == None):
            self.convert()
            values = [self.params[k] for k in self.conventions[convention]]
        return values

    def plot(self, ax=None, show=False, **kwargs):
        major, minor, angle = self.get('ellipse_axes', nested=True)
        x, y = self.position
        if ax is None:
            plot = Plot()
            ax = plot.ax
        plot_ellipses(ax, major, minor, angle, x=x, y=y, **kwargs)
        if show:
            plt.show()

    def inside(self, x, y):
        """Return True if points inside ellipses"""
        # TODO: move in_ellipse function here
        raise NotImplementedError

    def grid_2d(self, x, y, amps, convention='ellipse_axes'):
        """Return tilted 2D elliptic gaussians on x, y grid for each ellipse
        :param x - x grid axis values
        :param y - y grid axis values
        :param amps - amplitudes of each 2d gaussian"""
        RR, TT = np.meshgrid(x, y)
        x_centre, y_centre = self.position_safe
        dx, dy, angle = self.get(convention)    # Get ellipse dimensions
        if is_scalar(amps):  # Make sure same length as ellipse info
            amps = np.full_like(dx, amps)

        out = []
        for x0, y0, dx0, dy0, angle0, amp0 in safe_zip(x_centre, y_centre, dx, dy, angle, amps):
            
            a = (np.cos(angle0) * np.cos(angle0) / (2.0 * dx0 * dx0)) + (
            np.sin(angle0) * np.sin(angle0) / (2.0 * dy0 * dy0))
    
            b = (-np.sin(2.0 * angle0) / (4.0 * dx0 * dx0)) + (np.sin(2.0 * angle0) / (4.0 * dy0 * dy0))
    
            c = (np.sin(angle0) * np.sin(angle0) / (2.0 * dx0 * dx0)) + (
            np.cos(angle0) * np.cos(angle0) / (2.0 * dy0 * dy0))
    
            exponent = - (a * (RR.T - x0) * (RR.T - x0) + 2.0 * b * (RR.T - x0) * (TT.T - y0) + c * (
            TT.T - y0) * (TT.T - y0))
            grid = amp0 * np.exp(exponent)
            out.append(grid)

        return np.array(out)

    def superimpose_2d(self, x, y, amps, convention='ellipse_axes'):
        grids = self.grid_2d(x, y, amps, convention=convention)
        return np.sum(grids, axis=0)

    def plot_3d(self, x, y, amps, convention='ellipse_axes', mode='surface3D', show=True, **kwargs):
        grids = self.grid_2d(x, y, amps, convention=convention)
        for grid in grids:
            Plot(x, y, grid, mode=mode, show=show, **kwargs)

    def plot_3d_superimposed(self, x, y, amps, convention='ellipse_axes', mode='surface3D', show=True, **kwargs):
        grid = self.superimpose_2d(x, y, amps, convention=convention)
        Plot(x, y, grid, mode=mode, show=show, **kwargs)



if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-6, 6, 100)
    Ellipses(2, 3, 1).plot_3d_superimposed(x, y, 1)