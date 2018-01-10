#!/usr/bin/env python

import itertools
import numpy as np
from numpy import sin, cos, arctan, tan, sqrt
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

from ccfepyutils.utils import make_itterable

class Ellipses(object):
    """Class for converting between different geometric descriptions of an ellipse.
    Need 3 parameters to define ellipse shape and orrientation + 2 for position
    Ellipse axes    - Major axis/diameter, minor axis/diameter, angle between major radius and x direction
    Axis extent     - Lengths along x & y axes through ellipse centre (Method used in Zweeben 2016)
    Extreema extent - Bounding box dimensions (axis aligned)
    Focii           -
    """
    conventions = {'ellipse_axes': ['major', 'minor', 'angle'],
                   'bounding_box': ['x_boundbox', 'y_boudbox', 'angle'],
                   'axes_extent': ['x_axis', 'y_axis', 'angle']}
    def __init__(self, arg1, arg2, arg3, convention='ellipse_axes', x=None, y=None, degrees=False):
        self.convention = convention
        self.params = {key: None for key in itertools.chain.from_iterable(self.conventions.values())}  # initialise
        self.x = x
        self.y = y
        params = self.params

        if convention.lower() not in self.conventions:
            raise NotImplementedError

        for key, arg in zip(self.conventions[convention], (arg1, arg2, arg3)):
            params[key] = arg

        if (params['angle'] is not None) and degrees:
            params['angle'] = np.deg2rad(params['angle'])

    def convert(self):
        """Calcualte parameters for other conventions"""
        convention = self.convention
        params = self.params
        if convention.lower() == 'ellipse_axes':
            params['x_axis'], params['y_axis'], angle = self.ellipse_axes_to_axis_extent(
                    params['major'], params['minor'], params['angle'])
            params['x_boundbox'], params['y_boundbox'], angle = self.ellipse_axes_to_bounding_box(
                    params['major'], params['minor'], params['angle'])
        else:
            raise NotImplementedError

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

    def plot(self):
        # TODO: Add plot method for ellipses
        raise NotImplementedError