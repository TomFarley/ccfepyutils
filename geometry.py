#!/usr/bin/env python

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
    def __init__(self, arg1, arg2, arg3, convention='ellipse_axes', x=None, y=None, degrees=False):
        self.convention = convention
        self.params = {}
        
        params = self.params
        if convention.lower() == 'ellipse_axes':
            params['major'] = arg1
            params['minor'] = arg2
            params['angle'] = arg3 if not degrees else np.deg2rad(arg3)

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
        return self.params['major'], self.params['minor'], self.params['angle']

    @property
    def boundingbox(self):
        return self.params['x_boundbox'], self.params['y_boundbox'], self.params['angle']

    @property
    def axes_extent(self):
        return self.params['x_extent'], self.params['y_extent'], self.params['angle']