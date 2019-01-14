#!/usr/bin/env python
"""Iterator class for selecting x so as to best capture detail in y(x)
Tom Farley, 08-2017

TODO:
- replace parametric spline interpolation

"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import scipy.interpolate
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import argrelmax, argrelmin

class ParameterScan(object):
    """Iterator class for selecting x so as to best capture detail in y(x)
    Usage example:
       ps =  ParameterScan([0,100], n_linear=4, n_refine=6, n_spread=2, n_extreema=2)
       for x in ps:
          y = # your code goes here and returns y(x)
          ps.add_data(x, y)
       ps.plot_data()  # plot results of the scan y(x)
       x, y = ps.data  # return results of the scan y(x)
    """
    def __init__(self, lims, n_linear=4, n_refine=6, n_spread=2, n_extreema=2, extrema_weight=10, func='linear',
                 n_model=1000, extrema_frac=0.25, pinpoint_extrema_weight=2, order=1):
        """
        n_refine                 - Number of points to intelligently evaluate over interval (not including interval limits)
        extrema_weight           - Importance of refining extrema relative to evaluating evenly along the curve
        n_linear                 - Number of points to linearly evaluate across interval (excluding limits)
        n_refine                 - Number of points to assign automatically based on extrema_frac (
        n_spread                 - Number of points to assign to filling gaps at end of refinement
        n_extreema               - Number of points to assign to further refining extrema at end of refinement
        func                     - Not implemented - anticipated distribution of data
        n_model                  - Number of high resolution points to evalute along fitted spline curve
        extrema_frac             - Proportion of range over which feature must be global maxima to be refined
        pinpoint_extrema_weight  - Relative importance of pinpointing extrema location over filling gaps around extrema
        order                    - Order of spline fit
        """
        assert len(lims) == 2, 'lims must contain two elements: [min, max]'
        self.lims = lims  # Limits over which parameter scan will be performed
        self.n_linear = n_linear  # Number of points to space linearly over interval (not including interval limits)
        self.n_refine = n_refine  # Number of points to evaluate over interval (not including interval limits)
        self.n_spread = n_spread
        self.n_extreema = n_extreema
        self._extrema_frac = extrema_frac  # proportion of x range extrema must be global to to be refined
        self._extrema_weight = extrema_weight
        # Number of times smaller than largest gap in data distance from extrema can be to attempt to further pinpoint
        #  extrema rather than bisect interval in data expected to contiain the extrema
        self.pinpoint_extrema_weight = pinpoint_extrema_weight
        self._func = func
        self.n_model = n_model  # Number of parametric points to evaluate along spline fit to data
        self._u_model = np.linspace(0, 1, self.n_model)  # parametric coords for spline evaluation between lims
        self._x_model = np.array([])
        self._y_model = np.array([])
        self._model_state = None
        self._x = np.array([])
        self._y = np.array([])
        self._tck = None
        self._u = np.array([])
        self.order = order  # order used for spline fitting
        self._current_type = 'None'  # Current type of point being added: linear/gap/extrema_pinpoint/extrema_gap

        self.func = func

    @property
    def x(self):
        isort = np.argsort(self._x)
        return self._x[isort]  # make sure x and y returned in order of ascending x

    @property
    def y(self):
        isort = np.argsort(self._x)
        return self._y[isort]  # make sure x and y returned in order of ascending x

    @property
    def n_data(self):
        return len(self._x)

    @property
    def n_scan(self):
        return 2 + self.n_linear + self.n_refine + self.n_spread + self.n_extreema

    @property
    def data(self):
        return (self.x, self.y)

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        if value == 'linear':
            value = lambda x, m, c: m*x + c
        return value

    def __iter__(self):
        """Iterate over points in scan, adding next point each time, based on previous points"""
        assert self.n_linear+self.n_refine+self.n_scan+self.n_extreema > 0, 'Non-zero input argument required'
        self._current_type = 'limits'
        yield self.lims[0]
        yield self.lims[1]

        # Return points evenly (linearly) spaced between limits
        if self.n_linear is not None:
            x_linear = np.linspace(self.lims[0], self.lims[1], self.n_linear+2)[1:-1]
            self._current_type = 'linear'
            for n in np.arange(self.n_linear):
                yield x_linear[n]

        #  First refine data over interval 'intelligently' choosing between interval bisection and extrema indent
        for n in np.arange(self.n_refine):
            x_gap = self.get_gap_point()  # x coord bisecting largest gap (ie spline length) in data
            x_extr = self.get_extreema_point()  # coord that should best pin down lead well defined extrema in data
            if x_extr is not None:  # if there are extrema pick between gap and extrema
                xdata_gap = self._x[np.argmin(np.abs(self._x - x_gap))]
                xdata_extr = self._x[np.argmin(np.abs(self._x - x_extr))]

                u_x_gap = self.closest_u(x_gap)
                u_x_extr = self.closest_u(x_extr)
                u_xdata_gap = self.closest_u(xdata_gap)
                u_xdata_extr = self.closest_u(xdata_extr)

                l_gap = self.dist_along_spline(self._tck, u_x_gap, u_xdata_gap)
                l_extr = self.dist_along_spline(self._tck, u_x_extr, u_xdata_extr)

                if l_gap > self._extrema_weight * l_extr:
                    self._current_type = 'gap'
                    print('Filling largest gap in data')
                    yield x_gap
                else:
                    self._current_type = 'extrema'
                    print('Refining extrema position')
                    yield x_extr
            else:
                print('Filling largest gap in data')
                self._current_type = 'gap'
                yield x_gap

            # dx_gap = np.abs(xdata_gap - x_gap)
            # dx_extr = np.abs(xdata_extr - x_extr)

        # After 'intelligent' refinement, further bisect largest intervals requested number of times
        for n in np.arange(0, self.n_spread):
            assert len(self._x) == 2+self.n_linear+self.n_refine+n, 'New data must be added each loop using .add_data()  to update the model'
            assert len(self._y) == len(self._x) == 2+self.n_linear+self.n_refine+n, 'New x and y data must be added of equal length'
            x_new = self.get_gap_point()
            print('Filling largest gap in data')
            yield x_new
        # Finally refine extrema with requested number of points
        for n in np.arange(self.n_extreema):
            x_new = self.get_extreema_point()
            if x_new is None:
                print('No extrema detected to refine - skipping')
                break
            print('Refining extrema position')
            yield x_new

    def add_data(self, x, y):
        """Add data to model to inform selection of next point in scan"""
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        # Update spline fit with new data
        if self.n_data > 1:
            self._tck, self._u = interpolate.splprep([self.x, self.y], s=0, k=np.min((len(self._x) - 1, self.order)))

    @property
    def model_data(self):
        if self._model_state != self.n_data:
            if self._tck is not None:
                self._x_model, self._y_model = interpolate.splev(self._u_model, self._tck)
            else:
                self._x_model, self._y_model = np.array([]), np.array([])
            self._model_state = self.n_data
        return self._x_model, self._y_model

    def get_gap_point(self):
        """Return x coord bisecting gap between existing data points with largest spline curve length between points"""
        # Fit cubic spline to existing data
        self._tck, u = interpolate.splprep([self.x, self.y], s=0, k=np.min((len(self._x) - 1, self.order)))

        spline = np.array(interpolate.splev(self._u_model, self._tck))  # evaluate spline on fine u
        l_fine = np.sqrt(np.sum(np.diff(spline.T, axis=0) ** 2, axis=1))  # distance between adjacent fine points
        cum_len = np.append([0], np.cumsum(l_fine))
        full_length = cum_len[-1]
        # print('length of curve: {}'.format(full_length))
        l = [cum_len[np.argmin(np.abs(self._u_model - ui))] for ui in u]  # distance along spline line at data points
        i = np.argmax(np.diff(l))  # index of data point starting longest spline segment
        # parametric coord along spline fit at midpoint between largest gap
        u_new = self._u_model[np.argmin(np.abs(cum_len - np.mean((l[i], l[i + 1]))))]

        l_new = cum_len[[np.argmin(np.abs(cum_len - np.mean((l[i], l[i + 1]))))]]  # tmp

        # Find u values at equidistant lengths along the fitted spline
        # l_new = np.linspace(0, full_length, n)
        # u_new = [u_new[np.argmin(np.abs(cum_len - l))] for l in l_new]

        # unew = np.arange(0, 1.01, 0.01)
        # Evaluate x and y values at centre of largest gap in data
        x_new, y_new = interpolate.splev(u_new, self._tck)

        return x_new

    def get_extreema_point(self):
        """Return x coordinate that best refines information about location of extrema in data
        If poorest constrained extrema already has a point very close to its expected location, bisect the
        gap in the data that should contain the extrema, else return x coord where extrema is expected to be"""
        x, y = self.model_data
        order = int(np.ceil(self._extrema_frac * self.n_model))  # window size for extrema select

        x_new = x[argrelmax(y, order=order)]  # local maxima
        x_new = np.append(x_new, x[argrelmin(y, order=order)])  # add local minima

        if len(x_new) == 0:
            return None  # No extrema meet chriteria of being global maxima over _extrema_frac proportion of data range

        spline = np.array(interpolate.splev(self._u_model, self._tck))  # evaluate spline on fine u
        l_fine = np.sqrt(np.sum(np.diff(spline.T, axis=0) ** 2, axis=1))  # distance between adjacent fine points
        cum_len = np.append([0], np.cumsum(l_fine))  # distance along fine spline
        full_length = cum_len[-1]
        # print('length of curve: {}'.format(full_length))
        l = [cum_len[np.argmin(np.abs(self._u_model - ui))] for ui in self._u]  # distance along spline line at data points

        imodel_new = [self.closest_model_index(x_newi) for x_newi in x_new]  # indices of extrema in model (ie fine)
        l_new = cum_len[imodel_new]  # distances of extrema points along line
        dl = [np.abs(cum_len - li)[np.argmin(np.abs(cum_len - li))] for li in l_new]  # distance of extrema points from data points
        i = np.argmax(dl)  # index of extrema furthest from existing datapoints
        if dl[i] > 0.5/self.pinpoint_extrema_weight * np.max(np.diff(l)):  # if distance from point is more than
            x_new = x_new[i]  # take extrema furthest from any existing datapoint
            self._current_type = 'extrema pinpoint'
        else:  # all extrema are too close to existing points - take midpoint near extrema
            # Find existing points either side of x_new[i]
            u = self._u_model[imodel_new[i]]
            u_sides = self._u[np.argsort(np.abs(self._u - u))]  # u of two closest data points
            if u_sides[0] - u < np.mean(np.diff(self._u_model)):  # extrema effectively falls on existing point
                u_bound = u_sides[0]  # existing boundary point
                if u_sides[1] - u_bound > 0:  # don't bisect smallest gap - bisect largest gap on other side of u_bound
                    u_sides = [u_bound, u_sides[u_sides-u_bound < 0][0]]
                else:
                    u_sides = [u_bound, u_sides[u_sides - u_bound > 0][0]]
            else:
                u_sides = u_sides[0:2]
            x, y = self.equi_dist_points_on_spline(self._tck, np.min(u_sides), np.max(u_sides), n=3)
            x_new = x[1]  # middle point of three returned
            self._current_type = 'extrema gap'
        # dx = [np.abs(self._x - xi)[np.argmin(np.abs(self._x - xi))] for xi in x_new]
        # x_new = x_new[np.argmax(dx)]
        return x_new

    def closest_u(self, x):
        """Return parametric coord of existing data point in model closest to supplied x coord"""
        return self._u_model[np.argmin(np.abs(self._x_model - x))]

    def closest_model_index(self, x):
        """Return index of existing data point in model closest to supplied x coord"""
        return np.argmin(np.abs(self._x_model - x))

    @staticmethod
    def equi_dist_points_on_spline(tck, u = 0, v = 1, n=3, n_fine=None):
        """Find x and y values that best define curve"""
        n_fine = 10 * n if n_fine is None else n_fine
        u_fine = np.linspace(u, v, n_fine)
        spline = np.array(interpolate.splev(u_fine, tck))  # evaluate spline on fine u
        l_fine = np.sqrt(np.sum(np.diff(spline.T, axis=0) ** 2, axis=1))  # distance between adjacent fine points
        cum_len = np.cumsum(l_fine)  # cumulate distance along spline curve at each u
        full_length = cum_len[-1]

        # Find u values at equidistant lengths along the fitted spline
        l_new = np.linspace(0, full_length, n)
        u_new = [u_fine[np.argmin(np.abs(cum_len - l))] for l in l_new]

        # unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(u_new, tck)
        return out

    @staticmethod
    def dist_along_spline(tck, u = 0, v = 1, n=1000):
        """ Return the distance from u to v along the spline"""
        spline = np.array(interpolate.splev(np.linspace(u, v, n), tck))
        lengths = np.sqrt(np.sum(np.diff(spline.T, axis=0)**2, axis=1))
        return np.sum(lengths)

    @staticmethod
    def dist_along_linear(interp_func, u = 0, v = 1, n=1000):
        """ Return the distance from u to v along the spline"""
        spline = np.array(interpolate.splev(np.linspace(u, v, n), tck))
        lengths = np.sqrt(np.sum(np.diff(spline.T, axis=0)**2, axis=1))
        return np.sum(lengths)

    def predict(self):
        raise NotImplementedError
        f = curve_fit(self.func, np.arange(self.n_spread))

    def plot_data(self, xlabel='x', ylabel='y'):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, label='Scan results')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()


if __name__ == '__main__':
    pass