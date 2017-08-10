from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import scipy.interpolate
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import argrelmax, argrelmin

class ParameterScan(object):
    """Itterator for selecting x so as to best capture detail in y(x)
    """
    def __init__(self, lims, n_linear=0, n_refine=0, n_spread=0, n_extreema=0, extrema_weight=10, func='linear', n_model=1000,
                 extrema_frac=0.25, pinpoint_extrema_weight=2, order=2):
        """
        n_refine                 - Number of points to intelligently evaluate over interval (not including interval limits)
        extrema_weight           -
        n_spread                 -
        n_extreema               -
        func                     -
        n_model                  -
        extrema_frac             - Proportion of range over which feature must be global maxima to be refined
        pinpoint_extrema_weight  -
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
                    print('Filling gap')
                    yield x_gap
                else:
                    self._current_type = 'extrema'
                    print('Refining extrema')
                    yield x_extr
            else:
                print('Filling gap')
                self._current_type = 'gap'
                yield x_gap

            # dx_gap = np.abs(xdata_gap - x_gap)
            # dx_extr = np.abs(xdata_extr - x_extr)

        # After 'intelligent' refinement, further bisect largest intervals requested number of times
        for n in np.arange(0, self.n_spread):
            assert len(self._x) == 2+self.n_linear+self.n_refine+n, 'New data must be added each loop using .add_data()  to update the model'
            assert len(self._y) == len(self._x) == 2+self.n_linear+self.n_refine+n, 'New x and y data must be added of equal length'
            x_new = self.get_gap_point()
            yield x_new
        # Finally refine extrema with requested number of points
        for n in np.arange(self.n_extreema):
            x_new = self.get_extreema_point()
            if x_new is None:
                break
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
        print('length of curve: {}'.format(full_length))
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
        f = curve_fit(self.func, np.arange(self.n_spread))
    
    @staticmethod
    def demo(func, lims, noise=0.05, time=30, **kwargs):
        """Demo class on known function to demonstrate capabilties"""
        plt.ion()  # interactive mode
        # Fine points describing true function we trying to sample and best represent
        ps = ParameterScan(lims, **kwargs)
        dt = (time-2.) / (ps.n_scan)

        # Generate noisy data


        # Find points that would be returned used simple linear scan
        linear_scan = np.linspace(lims[0], lims[1], ps.n_scan)
        linear_scan = [linear_scan, func(linear_scan)]
        # Add random guassian 'experiemntal' noise to measurement
        linear_scan[1] = linear_scan[1] + np.random.randn(len(linear_scan[1]))*noise*np.mean(linear_scan[1])
        noisy_interp = scipy.interpolate.interp1d(linear_scan[0], linear_scan[1])

        fine = np.linspace(lims[0], lims[1], 1000)
        fine = [fine, func(fine)]

        art_true, = plt.plot(fine[0], fine[1], label='True distribution')
        # plt.plot(fine[0][(0, -1)], fine[1][(0, -1)], label='Interval limits')
        # plt.show(block=False)
        # plt.draw()
        # sleep(3)
        plt.legend(loc='best')
        plt.pause(1)
        art_linear, = plt.plot(linear_scan[0], linear_scan[1], 'o-', label='Linear scan results')
        plt.legend(loc='best')
        # plt.draw()
        plt.pause(1)
        # sleep(3)
        # plt.show()
        # plt.clear()
        art_true.set_visible(False)
        art_linear.set_visible(False)

        model = ps.model_data  # fitted spline data
        art_model, = plt.plot(model[0], model[1], '--r', label='model')  # spline curve
        art_results, = plt.plot(ps._x[:-1], ps._y[:-1], '.k', label='results')  # black dots for points evaluated
        art_next = plt.axvline(np.mean(ps.lims), color='k', ls='--')  # vertical line marking where next point will be
        art_text = plt.annotate('', xy=(0.05, 0.95), xycoords=("axes fraction"))
        art_next.set_visible(False)
        for x in ps:
            art_next.set_xdata(x)
            art_next.set_visible(True)
            art_text.set_text(ps._current_type)  # Update label for type of point being added
            plt.pause(dt)
            art_next.set_visible(False)
            # y = func(x) + np.random.randn()*noise*np.mean(linear_scan[1])  # update with experimental/sim data + noise
            y = noisy_interp(x)  # update with experimental/sim data + noise
            ps.add_data(x, y)
            model = ps.model_data
            art_model.set_data(model)  # update spline fit data
            art_results.set_data(ps._x[:], ps._y[:])  # mark previous points (excluding new point)
            # plt.plot(ps._x[-1], ps._y[-1], 'og', ms=10, label='results')
            plt.legend(loc='best')
            # plt.show()
            # plt.draw()
            # sleep(2)

        art_results.set_visible(False)
        art_model.set_visible(False)
        art_final, = plt.plot(ps.x, ps.y, 'o-g', label='model')  # final set of fitted points
        art_linear.set_visible(True)
        art_true.set_visible(True)
        plt.legend(loc='best')
        plt.pause(15)
        # input()

        print('Demo finished!')

def spike(x):
    x = x -0.9
    return 100*np.exp(-(((x/0.7)**2+5*x+1))) + 3*x - np.exp(x/3.) #+ 4*np.sin(x)

def dec_exp(x):
    y = np.exp(-2*x)
    return y

def wavy(x):
    y = np.cos(x) - x**2 / 40
    return y

if __name__ == '__main__':
    lims = [-10, 10]
    func = spike
    # func = dec_exp
    func = wavy
    # x = np.linspace(lims[0], lims[1], 1000)
    # plt.plot(x, func(x))
    # plt.show()
    # ParameterScan.demo(func, lims, time=10, n_linear=0, n_refine=12, n_spread=2, n_extreema=2,
    #                    noise=0.2, order=1, extrema_frac=0.05)
    ParameterScan.demo(wavy, lims, time=10, n_linear=5, n_refine=10, n_spread=2, n_extreema=2,
                       noise=0.2, order=1, extrema_frac=0.05)

    # x = np.array([ 2.,  1.,  1.,  2.,  2.,  4.,  4.,  3.])
    # y = np.array([ 1.,  2.,  3.,  4.,  2.,  3.,  2.,  1.])
    x = np.array([0.1, 0.18, 0.3, 0.99, 2, 2.1, 3, 4, 7, 10])
    y = np.sin(x)
    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_fine = np.sin(x_fine)
    plt.plot(x, y, 'x', label='raw')
    plt.plot(x_fine, y_fine, '-', label='True')

    # t = np.arange(x.shape[0], dtype=float)
    # t /= t[-1]
    # nt = np.linspace(0, 1, 100)
    # x1 = scipy.interpolate.spline(t, x, nt)
    # y1 = scipy.interpolate.spline(t, y, nt)
    # plt.plot(x1, y1, label='range_spline')
    #
    # t = np.zeros(x.shape)
    # t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    # t = np.cumsum(t)
    # t /= t[-1]
    # x2 = scipy.interpolate.spline(t, x, nt)
    # y2 = scipy.interpolate.spline(t, y, nt)
    # plt.plot(x2, y2, label='dist_spline')

    # Return the distance from u to v along the spline
    def dist_along_spline(tck, u = 0, v = 1, N = 1000):
        spline = np.array(interpolate.splev(np.linspace(u, v, N), tck))
        lengths = np.sqrt(np.sum(np.diff(spline.T, axis=0)**2, axis=1))
        return np.sum(lengths)

    n = 10

    tck, u = interpolate.splprep([x, y], s=0)
    print('u original: {}'.format(u))

    u_fine = np.linspace(0, 1, 1000)
    spline = np.array(interpolate.splev(u_fine, tck))
    l_fine = np.sqrt(np.sum(np.diff(spline.T, axis=0) ** 2, axis=1))
    cum_len = np.cumsum(l_fine)
    full_length = cum_len[-1]
    # full_length = dist_along_spline(tck)

    l_new = np.linspace(0, full_length, n)
    u_new = [u_fine[np.argmin(np.abs(cum_len-l))] for l in l_new]

    print('length of curve: {}'.format(full_length))
    # unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(u_new, tck)

    # plt.plot(out[0], np.sin(out[0]), label='True')
    plt.plot(out[0], out[1], '*', label='para spline')
    plt.legend(loc='best')

    plt.show()

    print('finished!')