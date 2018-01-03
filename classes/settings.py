#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers

from copy import deepcopy

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)


class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes"""
    instances = {}

    def __init__(self, application, name):
        """ """
        self._application = application
        self._name = name

        self.instances