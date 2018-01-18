#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.utils import iteritems
"""Class for performing quick fits to data and plotting them"""
import inspect
import logging

import numpy as np

class Parameters(object):

    def __init__(self):
        raise NotImplementedError


class Parameter(str):

    def __new__(cls, value, meta):
        obj = str.__new__(cls, value)
        obj.meta = meta
        # raise NotImplementedError
        return obj
        # Key (unique identifier)
        #
        # Short name
        # Long name
        # Description
        # Unit

class Quantities(object):
    """ """

    def __init__(self):
        raise NotImplementedError

class Quantity(str):
    """ """

    def __new__(cls, value, meta):
        obj = str.__new__(cls, value)
        obj.meta = meta
        # raise NotImplementedError
        return obj

class Unit(str):

    def __new__(cls, value, meta):
        obj = str.__new__(cls, value)
        obj.meta = meta
        # raise NotImplementedError
        return obj


if __name__ == '__main':
    p = Parameter('voltage', 'V')
    pass