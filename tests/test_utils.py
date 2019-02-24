#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from copy import deepcopy

__author__ = 'tfarley'

import unittest
import os, time
import pickle
import inspect
import numpy as np
from collections import defaultdict

import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestUtils(unittest.TestCase):

    def setUp(self):
        import ccfepyutils

    def test_similarilty_to(self):
        logger.info('** Running test_similarilty_to')
        from ccfepyutils.utils import similarilty_to

        options = ['greetings001.py', 'hell1.txt', '57356385478', 'bye002.txt', None, 'hello.txt', 3254]

        reference = 'hello.txt'
        out = similarilty_to(reference, options, return_type='values', n_return=None,
                   similarity_threshold=None, similarity_measure='difflib')
        self.assertTrue(np.all(out == np.array(['hello.txt', 'hell1.txt', 'bye002.txt', 'greetings001.py', None,
                                        3254, '57356385478'])))
        out = similarilty_to(reference, options, return_type='order', n_return=None,
                   similarity_threshold=None, similarity_measure='difflib')
        self.assertTrue(np.all(out == np.array([5, 1, 3, 0, 4, 6, 2])))
        out = similarilty_to(reference, options, return_type='similarity', n_return=None,
                   similarity_threshold=None, similarity_measure='difflib')
        self.assertTrue(np.all(np.isclose(out, np.array([0.16666667, 0.88888889, 0., 0.52631579, 0.15384615, 1., 0.]))))

        out = similarilty_to(reference, options, return_type='values', n_return=3,
                             similarity_threshold=None, similarity_measure='difflib')
        self.assertTrue(np.all(out == np.array(['hello.txt', 'hell1.txt', 'bye002.txt'])))
        pass

def suite():
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestUtils('test_similarilty_to'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())