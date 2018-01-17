#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from copy import copy, deepcopy
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

from ccfepyutils.classes.plot import Plot

def subfigures_demo():
    x = np.linspace(-3, 17, 100)
    y1 = np.exp(-x) + np.random.normal(0, 1, len(x))
    y2 = np.sin(x)
    xlabel='some value []'
    ylabel='some other value []'

    # Standart matplotlib approach (11 lines)
    fig = plt.figure()           #  --|
    ax1 = fig.add_subplot(211)   #    |--- always do this
    ax2 = fig.add_subplot(212)   #  __|
    ax1.scatter(x, y1, s=4, label='data')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.plot(x, y2, ls='--')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()

    # Using the new Plot class (2 lines)
    plot = Plot(x, y1, fit='exp', mode='scatter', s=4, label='data', xlabel=xlabel, ylabel=ylabel, axes=(2,1))
    plot.plot(x, y2, ls='--', xlabel=xlabel, ylabel=ylabel, ax=2, show=True)
    pass

def surface_demo():
    xlabel='x [m]'
    ylabel='y [m]'
    x, y = np.linspace(0, 10, 100), np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)
    centre = [4, 2]
    z = 3*np.exp(-((xx-centre[0])/3)**2-((yy-centre[1])/5)**2) - 2

    # Standart matplotlib approach (7 lines)
    from mpl_toolkits.mplot3d import Axes3D  # very annoying having to remember to import this!
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(x, y)
    ax.plot_surface(xx, yy, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    # Using the new Plot class (1 line)
    # Note can pass 1d x and y arrays
    Plot(x, y, z, mode='surface3D', xlabel=xlabel, ylabel=ylabel, show=True)


if __name__ == '__main__':
    subfigures_demo()
    surface_demo()
