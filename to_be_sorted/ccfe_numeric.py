#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_numeric.py: Frequently used numeric functions.

Detailed description:

Notes:
    @bug:

Todo:
    @todo:

Info:
    @since: 18/09/2015
"""

import numpy as np
import matplotlib.pyplot as plt


## CAN import: tf_debug
## CANNOT import: tf_array, tf_string

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"



def range_of(list):
    """ Return the numerical range of a numeric list
    Inputs:
        list:   numeric (python) list or numpy array
    Outputs:
        float:  numeric range of input list 
    """
    return np.abs(max(list) - min(list))

def frac_range(list, frac):
    """ Return the numerical value at a given fraction of the way 
    through the range of a numeric list ie:
    min + (max - min) * frac 

    Inputs:
        list:   numeric (python) list or numpy array
        frac:   float specifying where in range to return value
    Outputs:
        float:  numeric range of input list 
    """
    return min(list) + (max(list) - min(list)) * frac 

if __name__=='__main__':
    pass

