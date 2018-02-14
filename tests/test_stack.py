#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from copy import deepcopy

__author__ = 'tfarley'

import unittest
import os
import pickle
import inspect
import numpy as np
from collections import defaultdict
from ccfepyutils.classes.data_stack import Stack, Slice

import logging
from logging.config import fileConfig, dictConfig
fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestStack(unittest.TestCase):

    def setUp(self):
        self.coords = {'x': defaultdict(return_none, name='R'),
                       'y': defaultdict(return_none, name='tor'),
                       'z': defaultdict(return_none, name='t')}
        self.coords2 = deepcopy(self.coords)
        coords2 = self.coords2
        coords2['x']['values'] = np.linspace(1.36, 1.42, 3)
        coords2['y']['values'] = np.linspace(-0.8, 0.8, 5)
        coords2['z']['values'] = np.linspace(0.217, 0.218, 2)
        # logger.info('Set up coords values')
        # self.coords = {'x': {'name': 'R'},
        #                'y': {'name': 'tor'},
        #                'z': {'name': 't'}}
        # self.coords.setdefault(return_none)

    def test_init_empty(self):
        logger.info('** Running test_init_empty')
        coords = self.coords
        stack = Stack(coords['x'], coords['y'], coords['z'])
        self.assertTrue(stack._data is None)  # Make sure data xarray has not been initialised yet (only on access)
        with self.assertRaises(ValueError):
            slice = stack[0]
        # self.assertTrue(stack._data is not None)  # Make sure data xarray has not been initialised yet (only on access)
        logger.info(repr(stack))
        pass

    def test_init_empty2(self):
        logger.info('** Running test_init_empty2')
        coords = self.coords2
        logger.debug('coords2 = {}'.format(coords))
        stack = Stack(coords['x'], coords['y'], coords['z'])
        logger.info(repr(stack))
        pass

    def test_loc(self):
        logger.info('** Running test_loc')
        coords = self.coords2
        stack = Stack(coords['x'], coords['y'], coords['z'])
        slice = stack.loc(R=1.36)
        logger.debug('slice = {}'.format(slice))

    def test_meta(self):
        logger.info('** Running test_meta')
        # coords = self.coords2
        # stack = Stack(coords['x'], coords['y'], coords['z'])
        # slice = stack.loc(R=1.36)
        # logger.debug('slice = {}'.format(slice))




def suite():
    print('Setting test suit')
    suite = unittest.TestSuite()

    suite.addTest(TestStack('test_init_empty'))
    suite.addTest(TestStack('test_init_empty2'))
    suite.addTest(TestStack('test_loc'))
    suite.addTest(TestStack('test_meta'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())