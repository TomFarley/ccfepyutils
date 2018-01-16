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
    y1 = np.exp(0.2*x)-3
    y2 = np.sin(x)

    plot = Plot(x=x, y=y1, axes=(2,1), xlabel='some value', ylabel='some other value')
    plot.plot(x=x, y=y2, ax=2, show=True)



if __name__ == '__main__':
    subfigures_demo()
