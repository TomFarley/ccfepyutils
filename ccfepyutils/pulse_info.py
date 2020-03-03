#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging, os, itertools, re, inspect, configparser, time
from collections import defaultdict, OrderedDict
from datetime import datetime
from copy import copy, deepcopy
from pathlib import Path
from logging.config import fileConfig

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
from ccfepyutils import batch_mode
import matplotlib
if batch_mode:
    matplotlib.use('Agg')
    print('Using non-visual backend')
else:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def separatrix_raidal_position(pulse=None, time=None, gfile=None, machine='MAST', z=0.0, outboard=True,
                               scheduler=True, psi_n=1.0, **kwargs):
    """Return radial position of separatrix (default at midplane) for given gfile/pulse"""
    from pyEquilibrium.equilibrium import equilibrium
    if gfile is not None:
        eq = equilibrium()
        eq.load_geqdsk(gfile=gfile, **kwargs)
    elif pulse is not None and time is not None:
        if scheduler:
            eq = equilibrium(device=machine, shot=pulse, time=time, **kwargs)
        else:
            from ccfepyutils.classes.gfile_selector import GFileSelector
            gfs =GFileSelector('default')
            gfile = gfs.get_gfile_fn(pulse, time, machine=machine, allow_scheduler_efit=False)
            eq = equilibrium()
            eq.load_geqdsk(gfile=os.path.join(*gfile), **kwargs)
    else:
        raise ValueError('Insufficient inputs. gfile | pulse + time')
    fs = eq.get_fluxsurface(psi_n)
    if 0.0 in fs.Z:
        i = np.where(fs.Z == z)[0]
        r = fs.R[i]
        if outboard:
            r = r[r > eq.axis.r]
            r = np.min(r)
        else:
            r = r[r < eq.axis.r]
            r = np.max(r)
    else:
        raise NotImplementedError('Interpolation required')
    return r


if __name__ == '__main__':
    gfile = '/home/tfarley/data/magnetic_equilibria/29840/kinetic_EFIT/run01/g_p29840_t0.23900'
    r = separatrix_raidal_position(gfile=gfile)
    print(r)
    # r = separatrix_raidal_position(pulse=29840, time=0.23900, scheduler=True)
    # print(r)
    r = separatrix_raidal_position(pulse=29840, time=0.23900, scheduler=False)
    print(r)
