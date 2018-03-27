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

from ccfepyutils.utils import make_itterable, make_itterables, is_scalar, safe_len, safe_zip, describe_array, args_for
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
        self.x = x if is_scalar(x) else np.array(x)
        self.y = y if is_scalar(y) else np.array(y)
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
        return np.array(values)

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

    def plot(self, ax=None, show=False, equal_aspect=False, deg=True, color='k', **kwargs):
        major, minor, angle = self.get('ellipse_axes', nested=True)
        x, y = self.position
        if ax is None:
            plot = Plot()
            ax = plot.ax
        if self.half_widths:  # Convert to full widths for plotting
            major, minor = 2*major, 2*minor

        if deg:
            rad = np.deg2rad(angle)
        else:
            angle, rad = np.rad2deg(angle), angle

        # For rotation to be correct need to work in equal aspect ratio
        if equal_aspect and any(angle > 0):
            x_range = np.ptp(ax.get_xlim())
            y_range = np.ptp(ax.get_ylim())
            aspect_ratio = x_range / y_range

            sin, cos = np.sin(rad)**2, np.cos(rad)**2
            major = major * np.sqrt(1*cos + sin * (1/aspect_ratio))
            minor = minor * np.sqrt(1*cos - sin * aspect_ratio)
            # yy, y, dy = [v * aspect_ratio for v in (yy, y, dy)]

        plot_ellipses(ax, major, minor, angle, x=x, y=y, color=color, **kwargs)
        if show:
            plt.show()

    def inside(self, x, y):
        """Return True if points inside ellipses"""
        # TODO: move in_ellipse function here
        raise NotImplementedError

    def grid_2d(self, x_grid, y_grid, amp, convention='ellipse_axes', combine=True):
        """Return tilted 2D elliptic gaussians on x, y grid for each ellipse
        :param x - x grid axis values
        :param y - y grid axis values
        :param amps - amplitudes of each 2d gaussian"""
        x, y = self.position_safe
        dx, dy, angle = self.get(convention)    # Get ellipse dimensions
        grid = elliptic_gaussian_grid(x_grid, y_grid, x, y, dx, dy, angle, amp, full_widths=(not self.half_widths),
                               combine=combine)
        return grid
        # if not self.half_widths:  # Convert to half widths for gaussian calculation
        #     dx, dy = 0.5*dx, 0.5*dy
        # if is_scalar(amps):  # Make sure same length as ellipse info
        #     amps = np.full_like(dx, amps)
        #
        # # To avoid large negative expoenents which go to zero in fp precision, rescale axes - numerical problems
        # dxy_extrema = np.min(np.concatenate((dx, dy)))
        # # multiplier = 10**(-np.log10(dxy_extrema))  # Rescale minima to 1
        # multiplier = 10**(-np.round(np.log10(dxy_extrema)))  # Rescale minima to order 1
        # x_centre, y_centre, dx, dy, xx, yy = [v * multiplier for v in (x_centre, y_centre, dx, dy, xx, yy)]
        #
        # out = []
        # # TODO: switch to single vectorised opperation for efficiency
        # for x0, y0, dx0, dy0, angle0, amp0 in safe_zip(x_centre, y_centre, dx, dy, angle, amps):
        #     angle0 = np.deg2rad(-angle0)  # switch to anticlockwise rotation like in matplotlib etc
        #     sin2, cos2 = np.sin(angle0)**2, np.cos(angle0)**2
        #     sin_2a, cos_2a = np.sin(2*angle0), np.cos(2*angle0)
        #
        #     # Equation of elliptic 2d Gaussian: f(x,y) = A exp(- (a(x-x_o)^2 + 2b(x-x_o)(y-y_o) + c(y-y_o)^2)
        #     a = (cos2 / (2.0 * dx0**2)) + (sin2 / (2.0 * dy0**2))
        #     b = (-sin_2a / (4.0 * dx0**2)) + (sin_2a / (4.0 * dy0**2))
        #     c = (sin2 / (2.0 * dx0**2)) + (cos2 / (2.0 * dy0**2))
        #     exponent = - (a*(xx - x0)**2 + 2.0*b*(xx - x0)*(yy - y0) + c*(yy - y0)**2)
        #     grid = amp0 * np.exp(exponent)
        #
        #     #tmp: unrotated gaussian
        #     grid_tmp = amp0 * np.exp(-(((xx - x0)**2)/(2.0 * dx0**2) + ((yy - y0)**2)/(2.0 * dy0**2)))
        #     from ccfepyutils.plotly_tools import plotly_surface
        #     # plotly_surface(xx, yy, grid)
        #     # Plot(xx, yy, grid, show=True, mode='contourf')#.to_plotly()
        #
        #     out.append(grid)
        #
        # return np.array(out)

    def superimpose_2d(self, x, y, amps, convention='ellipse_axes'):
        # TODO: remove, redundant
        grids = self.grid_2d(x, y, amps, convention=convention)
        return np.sum(grids, axis=0)

    def plot_3d(self, x, y, amps, convention='ellipse_axes', mode='surface3D', show=True, **kwargs):
        grids = self.grid_2d(x, y, amps, convention=convention)
        for grid in grids:
            Plot(x, y, grid, mode=mode, show=show, **kwargs)

    def plot_3d_superimposed(self, x, y, amps, convention='ellipse_axes', mode='surface3D', show=True, outlines=True,
                             **kwargs):
        kws = args_for(plot_ellipses, kwargs, remove=True)
        grid = self.superimpose_2d(x, y, amps, convention=convention)
        plot = Plot(x, y, grid, mode=mode, show=False, **kwargs)
        if outlines:
            self.plot(plot.ax(), **kws)
        plot.show(show)
        return plot

def elliptic_gaussian_grid(x_grid, y_grid, x, y, dx, dy, angle=0, amp=1, full_widths=False, combine=True, deg=True,
                           equal_aspect=True):
    """Return 2D grid with values set to 2d gaussian values"""
    """Return tilted 2D elliptic gaussians on x, y grid for each ellipse
    :param x_grid - x grid axis values
    :param y_grid - y grid axis values
    :param x - gaussian x coordinates
    :param amps - amplitudes of each 2d gaussian"""
    # TODO: Ideally enable some kind of periodic boundary conditions for y axis
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        xx, yy = np.meshgrid(x_grid, y_grid)
    else:
        xx, yy = x_grid, y_grid
    if full_widths:  # Convert to half widths for gaussian calculation
        dx, dy = 0.5 * dx, 0.5 * dy
    # Make sure inputs same length itterables
    x, y, dx, dy, angle, amp = make_itterables(x, y, dx, dy, angle, amp)
    if is_scalar(amp):  # Make sure same length as ellipse info
        amp = np.full_like(dx, amp)
    if is_scalar(angle):  # Make sure same length as ellipse info
        angle = np.full_like(dx, angle)
    if deg:  # degress to radians
        angle = np.deg2rad(angle)

    # For rotation to be correct need to work in equal aspect ratio
    if equal_aspect and any(angle > 0):
        x_range = np.ptp(x_grid)
        y_range = np.ptp(y_grid)
        aspect_ratio = x_range/y_range
        yy, y, dy = [v * aspect_ratio for v in (yy, y, dy)]

    # To avoid large negative expoenents which go to zero in fp precision, rescale axes - numerical problems
    dxy_extrema = np.min(np.concatenate((dx, dy)))
    # multiplier = 10**(-np.log10(dxy_extrema))  # Rescale minima to 1
    multiplier = 10 ** (-np.round(np.log10(dxy_extrema)))  # Rescale minima to order 1
    x, y, dx, dy, xx, yy = [v * multiplier for v in (x, y, dx, dy, xx, yy)]

    out = []
    # TODO: switch to single vectorised opperation for efficiency
    for x0, y0, dx0, dy0, angle0, amp0 in safe_zip(x, y, dx, dy, angle, amp):
        angle0 = -angle0  # switch to anticlockwise rotation like in matplotlib etc
        sin2, cos2 = np.sin(angle0) ** 2, np.cos(angle0) ** 2
        sin_2a, cos_2a = np.sin(2 * angle0), np.cos(2 * angle0)

        # Equation of elliptic 2d Gaussian: f(x,y) = A exp(- (a(x-x_o)^2 + 2b(x-x_o)(y-y_o) + c(y-y_o)^2)
        a = (cos2 / (2.0 * dx0 ** 2)) + (sin2 / (2.0 * dy0 ** 2))
        b = (-sin_2a / (4.0 * dx0 ** 2)) + (sin_2a / (4.0 * dy0 ** 2))
        c = (sin2 / (2.0 * dx0 ** 2)) + (cos2 / (2.0 * dy0 ** 2))
        exponent = - (a * (xx - x0) ** 2 + 2.0 * b * (xx - x0) * (yy - y0) + c * (yy - y0) ** 2)
        grid = amp0 * np.exp(exponent)

        # tmp: unrotated gaussian
        grid_tmp = amp0 * np.exp(-(((xx - x0) ** 2) / (2.0 * dx0 ** 2) + ((yy - y0) ** 2) / (2.0 * dy0 ** 2)))
        from ccfepyutils.plotly_tools import plotly_surface
        # plotly_surface(xx, yy, grid)
        # Plot(xx, yy, grid, show=True, mode='contourf')#.to_plotly()

        out.append(grid)
    out = np.array(out)
    # Supperimpse gaussians onto one grid
    if combine and out.ndim == 3:
        out = np.sum(out, axis=0)
    return out


if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-6, 6, 100)
    Ellipses(2, 3, 1).plot_3d_superimposed(x, y, 1)