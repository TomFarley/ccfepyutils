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
# fileConfig('../logging_config.ini')
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
        self.values = np.arange(3*5*2).reshape((3, 5, 2))
        self.stack = Stack(coords2['x'], coords2['y'], coords2['z'], stack_axis='t', values=self.values,
                           name='test_stack')
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
        self.assertTrue(stack._data is None)  # Make sure data xarray has not been initialised yet (only on access)
        pass

    def test_init_empty2(self):
        logger.info('** Running test_init_empty2')
        coords = self.coords2
        stack = Stack(coords['x'], coords['y'], coords['z'], stack_axis='t')
        self.assertTrue(np.all(np.isnan(stack[0.218].data.values)))
        stack.set_data(self.values)
        self.assertTrue(np.all(~np.isnan(stack[0.218].data.values)))
        pass

    def test_loc(self):
        logger.info('** Running test_loc')
        coords = self.coords2
        stack = self.stack
        slice = stack.loc(R=1.36)
        self.assertTrue(np.all(slice.values == np.array([[0, 1],[2, 3],[4, 5],[6, 7],[8, 9]])))
        pass

    def test_meta(self):
        logger.info('** Running test_meta')
        # coords = self.coords2
        # stack = Stack(coords['x'], coords['y'], coords['z'])
        # slice = stack.loc(R=1.36)
        # logger.debug('slice = {}'.format(slice))

    def test_extract_contiguous_chunk(self):
        logger.info('** Running test_extract_contiguous_chunk')
        stack = self.stack
        ds = stack.extract_contiguous_chunk(R=[1.32, 1.40], tor=[0, 0.8], t=[0.217, 0.217])
        self.assertTrue(np.all(ds.values == np.array([[[4], [6], [8]], [[14], [16], [18]]])))
        pass

    def test_extract(self):
        logger.info('** Running test_extract')
        stack = self.stack
        # Time varying radial slice
        da = stack.extract(tor=-0.4, t_range=[0.216, 0.217])
        self.assertLess(np.sum(da.shape), np.sum(stack.data.shape))

        da = stack.extract(tor=-0.4, t_mean=True)
        self.assertEqual(da.shape, (3,))




def suite():
    print('Setting test_tmp suit')
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