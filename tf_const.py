#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
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

import numpy as np
## CAN import:    Debug, simple
## CANNOT import:
import tf_libs.tf_simple as tf_simple
import tf_libs.tf_debug as tf_debug
import tf_libs.tf_string as tf_string

__version__ = "1.0.1"
__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
# __all__ = []

db = tf_debug.Debug(1,1,0)

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

def exp(x, a, m, c, str_eqn=False):
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

def gauss(x, A, mu, sigma, str_eqn=False):
    """ Gaussian distribution with centre mu and width sigma """
    ## @todo: update gaussian formula
    return exp(x, A, (1/(2*sigma),0,-mu/(2*sigma)), 0, str_eqn=str_eqn)

if __name__ == "__main__":
    print("e = ", e)