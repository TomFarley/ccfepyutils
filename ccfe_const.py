#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import inspect

""" tf_const.py: Physical constants and function forms

Detailed description:

Notes:
    @bug:

Todo:
    @todo: update gaussian formula
    @todo:

Info:
    @since: 18/09/2015
"""

from collections import defaultdict
import numpy as np
from numpy import exp as exp0
from numpy import log as log0
from numpy import sqrt as sqrt0
from numpy import pi as pi0
import pandas as pd

from ccfepyutils.utils import args_for, to_list

## CAN import:    Debug, simple
## CANNOT import:
# import tf_libs.tf_simple as tf_simple
# import tf_libs.tf_debug as tf_debug
# import tf_libs.tf_string as tf_string

__version__ = "1.0.1"
__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
# __all__ = []

# db = tf_debug.Debug(1,1,0)

""" Physical constants: 
## Physical constants
e   = 1.602176565e-19   # Elementary charge [C]
k_B = 1.3806488e-23     # Boltzman constant = R/N_A [J/K]
N_A = 6.02214129e23     # Avagadros constant [None]
amu = 1.66053886e-27    # Atomic mass unit [kg]
eps_0 = 8.8541878e-12   # Vacuum permittivity (epsilon_0) [F/m]
pi  = 3.1415926535898   # pi 

## Conversion factors
K   = 11604.505         # eV to K conversion factor = e/k_B [K/eV]

## Specific constants
M_Ar = 39.948
"""

## Physical constants
e   = 1.602176565e-19   # Elementary charge [C]
m_e = 9.10938291e-31    # Electron mass [kg]
k_B = 1.3806488e-23     # Boltzman constant = R/N_A [J/K]
N_A = 6.02214129e23     # Avagadros constant [None]
amu = 1.66053886e-27    # Atomic mass unit [kg]
eps_0 = 8.8541878e-12   # Vacuum permittivity (epsilon_0) [F/m]
pi  = 3.1415926535898   # pi 

## Conversion factors
eV2K   = 11604.505         # eV to K conversion factor = e/k_B [K/eV]

## Specific constants
M_Ar = 39.948           # Atomic mass of atomic/molecular Argon [amu]

class FitFunction(object):
    """Class for fit functions

    Wrapper for standard functions with additional meta data"""
    instances = defaultdict(None)  # Dict of FitFunction instances
    fitter = None
    def __init__(self, key, func, name=None, name_short=None, description=None, equation=None, eqn_latex=None,
                 symbol='y', constraints=None, fit_params=None, n_fit_params_min=3, p0=None):
        assert callable(func)
        assert key not in self.instances, 'FitFunction with key {} already exists: {}'.format(key, self.instances)
        signature = inspect.getfullargspec(func)
        self.key = key  # Key to look up this FitFunction instance with
        self.func = func  # Callable to evaluate
        self._name = name  # Name for labels etc
        self._name_short = name_short  # Abreviated/less verbose name for labels
        self._description = description  # Verbose description of the function
        self._equation = equation  # String showing implementation of equation
        self._eqn_latex = eqn_latex  # Latex string showing implementation of equation
        self._ndim = len(signature.args) - len(signature.defaults)
        self._dependent_vars = signature.args[:self._ndim]  # Names of dependent variables
        self._params = signature.args[self._ndim:]  # Names of parameters
        self._n_params = len(self._params)  # Number of parameters
        self._fit_params = self._params[:self._n_fit_params_min] if fit_params is None else fit_params  # Parameters to fit
        self._n_fit_params_min = n_fit_params_min  # Number of fit parameters
        self._n_fit_params_max = len(self._fit_params)  # Number of fit parameters
        self._p0 = signature.defaults if p0 is None else p0  # Default initial guess values for each parameter
        self._symbol = symbol  # Symbol representing function
        self._constraints = constraints  # Constraints on possible parameter values eg strictly sigma > 0 for fitting
        

        self.instances[key] = self

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def name(self):
        if self._name is None:
            return ''
        else:
            return self._name

    @property
    def description(self):
        if self._description is None:
            return ''
        else:
            return self._description

    @property
    def p0(self):
        df = pd.DataFrame({'value': self._p0}, index=self._params)
        df.index.name = 'param'
        return df

    def func_defn(self, print_=True):
        lines = inspect.getsourcelines(self.func)
        body = "".join(lines[0])
        if print_:
            print(body)
        return body

    @classmethod
    def avail(self):
        """Print out list of available FitFunction instances"""
        for key, f in self.instances.items():
            print('{}: {}, {}'.format(key, f.name, f.description))

    @classmethod
    def get(cls, key):
        """Get an existing FitFunction instance"""
        return cls.instances[key]

    def plot(self, annotate=True, *args, **kwargs):
        """Plot the function with supplied values"""
        from ccfepyutils.classes.plot import Plot  # avoid cyclic import
        # Convert args in the form of ranges to linspace arrays
        args = to_list(args)
        for i, arg in enumerate(args):
            if len(arg) == 2:
                args[i] = np.linspace(arg[0], arg[1], 500)
        kws = args_for(self.func, kwargs)
        out = self.func(*args, **kws)  # get output from function

        kws = args_for(Plot.plot, kwargs, include=Plot.args)
        arg = {k: v for k, v in zip('xyz', to_list(args) + [out])}  # collate x, y, z values as nessesary
        if annotate:  # auto add axis labels
            labels = {k: v for k, v in zip(['xlabel', 'ylabel', 'zlabel'], to_list(self._dependent_vars)+[self._symbol])}  # collate x, y, z values as nessesary
            kws.update(labels)
        kws.update(arg)
        plot = Plot(**kws)
        return plot

## Functions
def poly(x, coefs, str_eqn = False):
    """ Polynomial function of order len(args)-1
    Return: arg1 + arg2 x + arg3 x^2 + arg4 x^3 + ..."""

    # str_eqn=kwargs.pop('str_eqn', False)

    # if not args:
    #     raise('poly requires at least one arguement')

    # db(args=args)
    sum = 0
    pow = 0
    eqn = []
    for pow, coef in enumerate(tf_simple.make_iter(coefs)[::-1]):
        sum += coef * x**pow
        if str_eqn:
            if coef == 0: # Don't add 0x^2 to the string etc
                pass
            elif pow > 1:
                eqn.insert(0,'{:0.3g}x^{}'.format(coef,pow))
            elif pow == 1:
                eqn.insert(0,'{:0.3g}x'.format(coef))
            else:
                eqn.insert(0,'{:0.3g}'.format(coef)) # no x^0 for constant
    str_eqn = '' + ' + '.join(eqn) # join x terms separated by +s

    if not str_eqn:
        return sum # just numerical output
    else:
        return sum, str_eqn

def linear(x, m, c):
    """y = m*x + c"""
    return m*x +c

def exp(x, lamda, k):
    """y = e^( -(x-k)/lamda )"""
    return np.exp(-(x-k)/lamda)

def exp_c(x, lamda, k, c):
    """y = e^( -(x-k)/lamda ) + c"""
    return np.exp(-(x-k)/lamda) + c

def exp_a_c(x, A, lamda, k, c):
    """y = e^( -(x-k)/lamda ) + c"""
    return A * np.exp(-(x-k)/lamda) + c

def exp_polly(x, a, m, c, str_eqn=False):
    """ y = a * e^( m ) + c where exp, a and c can all be polynomials in x """

    val_a, str_a = poly(x, a, str_eqn=True)
    val_m, str_m = poly(x, m, str_eqn=True)
    val_c, str_c = poly(x, c, str_eqn=True)
    val = val_a * np.exp(val_m) + val_c

    if not str_eqn:
        return val
    else:

        if not ('x' in str_a):
            str_eqn = str_a + 'exp(' + str_m + ') + ' +str_c
        else:
            str_eqn = '(' + str_a + ')exp(' + str_m + ') + ' + str_c

        return val, str_eqn

def gaussian(x, A, mu, sigma, c):
    """ Gaussian distribution with centre mu and width sigma """

    return A * (1/sqrt0(2*pi0*sigma**2)) * exp0(-(x-mu)**2/(2*sigma**2)) + c
    # return exp(x, A, (1/(2*sigma),0,-mu/(2*sigma)), 0)

def gaussian_upright(x, A, mu, sigma, c):
    """ Gaussian distribution with centre mu and width sigma """
    if A <= 0:  # for purpose of curve fitting return inf if have negative amplitude
        return 1e50
    return A * 1/sqrt0(2*pi0*sigma**2) * exp0(-(x-mu)**2/(2*sigma**2)) + c

def lognormal(x, A, mu, sigma, c):
    """y = A / (sigma*sqrt(2*pi)*(x)) * exp(-(ln(x)-mu)**2/(sigma*sqrt(2)))"""
    return (A / (sigma*sqrt0(2*pi0)*x)) * exp0(-(log0(x)-mu)**2/(2*sigma**2)) + c

def distributions():
    def delta(value, size=1):
        if size > 1:
            value = np.repeat(value, size)
        return value

    def lognormal(mean=0.0, sigma=1.0, scale=1.0, offset=0.0, size=None):
        return scale * np.random.lognormal(mean=mean, sigma=sigma, size=size) + offset

    def normal(mean=0.0, sigma=1.0, scale=1.0, size=None):
        return scale * random.normal(loc=mean, scale=sigma, size=size)

    def decaying_exponential(lamda=1.0, scale=1.0, offset=0.0, size=None):
        """decaying_exponential with max value, scale, at position offset"""
        return (scale * lamda) * random.exponential(scale=lamda, size=size) + offset

    def trunc_exponential(size=None):
        f = np.random.uniform(size=size) * 1e9
        return np.log(f + 1.0) / np.max(np.log(f + 1.0))

    def gaussian_2d(x_range, y_range, x, y, sigma_x, sigma_y, amp, angle=0, deg=True):
        if angle != 0:
            return elliptic_gaussian(x_range, y_range, x, y, sigma_x, sigma_y, amp, angle)
        else:
            xx, yy = np.meshgrid(x_range, y_range)  # 2D gaussian
            return amp * (np.exp(-((xx.T - x) ** 2.0) / (2 * sigma_x * sigma_x)) *
                          np.exp(-((yy.T - y) ** 2.0) / (2 * sigma_y * sigma_y)))

    def elliptic_gaussian(x_range, y_range, x, y, sigma_x, sigma_y, amp, angle, deg=True):
        """Generate tilted 2D ellipse on R tor grid"""
        if deg:
            angle = np.deg2rad(angle)
        RR, TT = np.meshgrid(x_range, y_range)

        a = (np.cos(angle) * np.cos(angle) / (2.0 * sigma_x * sigma_x)) + (
            np.sin(angle) * np.sin(angle) / (2.0 * sigma_y * sigma_y))

        b = (-np.sin(2.0 * angle) / (4.0 * sigma_x * sigma_x)) + (np.sin(2.0 * angle) / (4.0 * sigma_y * sigma_y))

        c = (np.sin(angle) * np.sin(angle) / (2.0 * sigma_x * sigma_x)) + (
            np.cos(angle) * np.cos(angle) / (2.0 * sigma_y * sigma_y))

        exponent = - (a * (RR.T - x) * (RR.T - x) + 2.0 * b * (RR.T - x) * (TT.T - y) + c * (TT.T - y) * (TT.T - y))

        return amp * np.exp(exponent)

functions = {'poly': poly, 'linear': linear, 'exp': exp, 'exp_c': exp_c, 'exp_a_c': exp_a_c, 'normal':
    gaussian, 'gaussian': gaussian, 'gaussian_upright': gaussian_upright, 'lognormal': lognormal}



func_obs = {'lognormal': lognormal}

if __name__ == "__main__":
    print("e = ", e)