#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""Class for performing quick fits to data and plotting them"""
import inspect
import logging

import numpy as np
from scipy.signal import argrelmax, find_peaks_cwt
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

from inference_tools.gp_tools import GpRegressor
from ccfepyutils.ccfe_const import functions  # fix import path ****
from ccfepyutils.utils import is_scalar, sub_range, args_for  # fix import path ****
from ccfepyutils.plotUtils import repeat_color  # fix import path ****

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

class Fitter(object):
    """Class for fitting to data

    Example usages:
    Fit
        # This will fit an exponential to the data to the right of the maximium in the data and plot it to ax
        Fitter(x, y, ax=ax).plot_fit('exp', window='max->')
    """
    def __init__(self, x, y, z=None, ax=None):
        # TODO: Implement y_errors (and x_errors?)
        ind = np.argsort(x)  # Make sure x data ordered
        self.x = x[ind]
        self.y = y[ind]
        self.z = z[ind] if z is not None else None  # Not implemented...
        self.ax = ax

    def get_fig_ax(self, ax):
        if isinstance(ax, matplotlib.axes.Axes):
            return ax.figure, ax
        elif self.ax is not None:
            return self.ax.figure, self.ax
        else:
            fig, ax = plt.subplots()
            return fig, ax

    def get_envelope(self, y_err=None):
        """Get error envelope around points"""
        if y_err is None:
            y_err = np.full_like(self.y, 0.1*self.y)
            # y_err = np.full_like(self.y, 0.1*np.mean(self.y))
        if is_scalar(y_err):
            y_err = np.repeat(y_err, len(self.y))
        try:
            gpr = GpRegressor(self.x, self.y, y_err=y_err)
        except np.linalg.LinAlgError as e:
            logger.warning(e)
            return [], [], None
        x = np.linspace(np.min(self.x), np.max(self.x), 200)
        y, sigma = gpr(x)
        return x, y, sigma

    def data(self, window=None):
        x, y = self.x, self.y
        if window is None:
            return x, y
        if isinstance(window, (str, unicode)):
            if window == 'max->':
                ind = np.arange(np.argmax(y), len(y))
            elif window == '<-max':
                ind = np.arange(0, np.argmax(y)+1)
            elif window == 'rightpeak->':
                window_sizes = [0.3, 0.5]  # Spatial scale range of peaks
                # local maxima on similar spatial scale to peaks
                ind_high = np.where(y > 0.5*np.max(y))
                ind_max = argrelmax(y, order=int(len(x)*window_sizes[0]*0.5))[0]
                if len(ind_max) == 0:
                    ind_max = argrelmax(y, order=2)[0]
                    if len(ind_max) == 0:
                        ind_max = np.array([np.argmax(y)])
                ind_max = np.intersect1d(ind_max, ind_high)  # Only take maxima that are high enough
                if len(ind_max) == 0:
                    ind_max = np.array([np.argmax(y)])
                try:
                    # find peaks spanning 0.1 - 0.6 of the x domain
                    ind_peak = find_peaks_cwt(y, np.arange(int(len(x)*window_sizes[0]), int(len(x)*window_sizes[1])))  # index space widths
                    ind = ind_max[np.argmin(np.abs(ind_max-np.max(ind_peak)))]  # local maxima closest to peak centre
                except ValueError:  # find_peaks_cwt failed
                    ind = np.max(ind_max)
                ind = np.arange(ind, len(y))
        elif (not is_scalar(window)) and len(window) == 2:
            ind = sub_range(x, window, indices=True)
        return x[ind], y[ind]

    def fit(self, func, p0=None, window=None, **kwargs):
        """Fit function to data
        func - function to fit, can be predefined string
        p0 - initial guess parameters for function
        window - window limits of x data range to fit to
        """
        if self.z is not None:
            raise NotImplementedError('3D fitting not implemented yet')

        x, y = self.data(window=window)
        if isinstance(func, string_types):
            if func in functions.keys():
                func = functions[func]
            elif func == 'auto':
                func, popt, pcov, chi2r = self.auto_fit()
            else:
                raise ValueError('Function name "{}" not recognised. \nOptions: {}'.format(func, functions.keys()))
        if func is True:
            func, popt, pcov, chi2r = self.auto_fit()
        try:
            kws = args_for(curve_fit, kwargs)
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=int(1e5), **kws)
        except RuntimeError as e:  # Failed to converge
            logger.debug('Failed to converge: {}'.format(e))
            return func, None, None, None
        except TypeError as e:  # Improper input: degrees of freedom must not exceed n_data
            logger.debug('Improper input: degrees of freedom must not exceed n_data: {}'.format(e))
            return func, None, None, None
        except ValueError as e:
            logger.debug('Improper input: function is not behaving correctly: {}'.format(e))
            raise e
            return func, None, None, None
        except Exception as e:
            logger.debug('Unexpected error in fitting: {}'.format(e))
            raise e

        y0 = func(x, *popt)
        # chi2 = np.abs(np.sum((y-y0)**2/y0))  # correct version?
        # chi2 = np.abs(np.sum((y-y0)**2/y))  # should be y0?
        chi2 = np.abs(np.sum((y-y0)**2))  # should be y0?
        chi2r = chi2 / len(popt)

        logger.debug('{} fitted with chi2: {:0.0f}, chi2r: {:0.0f}, popt: {}'.format(func.__name__, chi2, chi2r, popt))
        return func, popt, pcov, chi2r

    def auto_fit(self, funcs=('linear', 'exp', 'normal', 'lognormal'), p0s=None, window=None, ax=None):
        """Fit fuction with lowest chi^2"""
        if p0s is None:
            p0s = [None for func in funcs]
        best = (None, [1], None, 1e99)
        for func, p0 in zip(funcs, p0s):
            func, popt, pcov, chi2r = self.fit(func, p0=p0, window=window)
            if chi2r is None:
                continue
            elif chi2r*len(popt) < best[3]*len(best[1]):
                best = (func, popt, pcov, chi2r)
        func, popt, pcov, chi2r = best
        chi2 = chi2r * len(popt)
        logger.debug('Best autofit: {} fitted with chi2={:0.0f}, chi2r: {:0.0f}, popt: {}'.format(
            func.__name__, chi2, chi2r, popt))
        return best

    def plot_fit(self, func='auto', p0=None, window=None, extrapolate=[None, None], ax=None, show=False,
                 fit_label='{Func} fit', color='repeat-10', **kwargs):
        """Plot fit
        Returns output from fit
        """
        if func is None:  # If None don't plot
            return
        if ax is None:
            fig, ax = self.get_fig_ax(ax)

        func, popt, pcov, chi2r = self.fit(func, p0=p0, window=window, **kwargs)
        if popt is None:
            logger.warning('Cannot plot fit. Failed to fit function {}'.format(func.__name__))
            return

        x, y = self.data(window=window)  # get windowed x data
        xmin = np.min(x) if extrapolate[0] is None else np.min(np.append(x, extrapolate[0]))
        xmax = np.max(x) if extrapolate[1] is None else np.max(np.append(x, extrapolate[1]))
        x = np.linspace(xmin, xmax, 200)
        y = func(x, *popt)

        func_name = func.__name__.replace('_', ' ')  # Function name
        popt_dict = {'p{}'.format(i+1): p for i, p in enumerate(popt)}
        label = fit_label.format(func=func_name, Func=func_name.capitalize(), chi2r=chi2r, chi2=chi2r*len(popt),
                                 **popt_dict)
        if isinstance(color, string_types) and 'repeat' in color:
            color = repeat_color(color, ax=ax)

        try:  # first try all keyword arguments in order to include **kwargs to ax.plot(**kwargs)
            ax.plot(x, y, ls='--', label=label, color=color, **kwargs)
        except:
            kws = args_for(ax.plot, kwargs)
            ax.plot(x, y, ls='--', label=label, color=color, **kws)
        if show:
            plt.show()
        return func, popt, pcov, chi2r

    def plot_envelope(self, ax=None, y_err=None, show=True, env_label='{Func} fit ($\chi_r^2=${chi2r:0.0f})', nsigma=3, **kwargs):
        """Plot error envelope around data"""
        fig, ax = self.get_fig_ax(ax)
        x, y, sigma = self.get_envelope(y_err=y_err)
        if sigma is None:
            return
        ax.fill_between(x, y-sigma*nsigma, y+sigma*nsigma, alpha=0.3, label='Error envelope ${}\sigma$'.format(nsigma))
        ax.plot(x, y, ls='--', lw=0.5, color='grey', alpha=0.3)
        for i in np.arange(1, int(nsigma+1)):
            ax.plot(x, y+i*sigma, ls='--', lw=0.5, color='grey', alpha=0.3)
            ax.plot(x, y-i*sigma, ls='--', lw=0.5, color='grey', alpha=0.3)

    def plot(self, ax=None, data='scatter', envelope=True, fit=True, p0=None, legend=True, xlabel='x', ylabel='y',
             show=True, fit_window=None, **kwargs):
        """Quick access to plotting methods"""
        fig, ax = self.get_fig_ax(ax)

        if envelope:
            nsigma = 3
            self.plot_envelope(ax=ax, **kwargs)

        if data:
            if data == 'scatter':
                ax.scatter(self.x, self.y, label='data', s=6)

        if fit:
            if (fit is True) or isinstance(fit, (tuple, list)):
                func, popt, pcov, chi2r = self.auto_fit()
            elif callable(fit):  # fit is a function instance
                func, popt, pcov, chi2r = self.fit(fit, p0=p0)
            # elif fit in functions:
            #     func = functions[func]
            #     func, popt, pcov, chi2r = self.fit(func, p0=p0)
            elif isinstance(fit, (str, unicode)):
                func, popt, pcov, chi2r = self.fit(fit, p0=p0)
            else:
                raise ValueError('Function {} not recognised/supported'.format(func))

            self.plot_fit(func, p0=p0, ax=ax, window=fit_window, show=False)

        if legend:
            leg = ax.legend(loc='best', fancybox=True, title=None)
            leg.draggable()
            leg.get_frame().set_alpha(0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()

        if show:
            plt.show()

        return ax

if __name__ == '__main__':
    func_name = 'exp'
    # func_name = 'linear'
    func = functions[func_name]
    p_in = np.random.uniform(0, 4, len(inspect.getargspec(func).args)-1)
    # p_in = (3.4, 2.1, 4.6, 1.9)

    logger.info('Using fuction {} with params {}'.format(func_name, p_in))

    x_true = np.linspace(0, 10, 200)
    y_true = func(x_true, *p_in)

    size = 100
    x = np.sort(np.random.uniform(0, 10, size))

    y = func(x, *p_in)
    sigma = 0.25 * np.mean(y)
    y_err = (np.random.randn(size) * sigma)
    y += y_err

    error_estimate = np.full_like(y, sigma) * 1
    fig, ax = plt.subplots()
    ax.plot(x_true, y_true, lw=1, alpha=0.8, label='Function')
    fit = Fitter(x, y, ax=ax)
    fit.plot(fit=func_name, envelope=True, y_err=error_estimate, show=False)
    plt.show()

    # out = fit.plot_fit(func)
    #
    # ax.plot(x, y, label='data')
    # ax.plot()
    pass
