#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.utils import iteritems
"""Class for performing quick fits to data and plotting them"""
import inspect
import logging

import numpy as np

from ccfepyutils.ccfe_const import FitFunction


def sample_lognormal(mean=0.0, sigma=1.0, scale=1.0, offset=0.0, size=None):
    return scale * np.random.lognormal(mean=mean, sigma=sigma, size=size) + offset

def lognormal(x, mean=0.0, stdev=1.0, amp=1.0, c=0.0):
    return amp/(x * stdev * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mean)**2/(2*stdev**2)) + c

lognormal = FitFunction('lognormal', lognormal,
                        name='lognormal',
                        fit_params=['mean', 'stdev'],
                        equation='amp/(x * stdev * sqrt(2*pi)) * exp(-(log(x)-mean)^2/(2*stdev^2)) + c')

functions = {'lognormal': lognormal}
sample_functions = {'lognormal': sample_lognormal}

if __name__ == '__main__':
    from ccfepyutils.classes.plot import Plot
    xrange = [0.0001, 3]
    x = np.linspace(xrange[0], xrange[1], 1000)
    plot = Plot(x, x, show=False)
    lognormal.plot([0, 4], mean=0.0, stdev=1, ax=plot.ax(), show=True)