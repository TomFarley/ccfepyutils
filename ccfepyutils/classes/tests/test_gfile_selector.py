#!/usr/bin/env python

""" 
Author: T. Farley
"""
import unittest
import logging, os, itertools, re, inspect, configparser, time
from collections import defaultdict
from copy import deepcopy, copy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ccfepyutils.classes.gfile_selector import GFileSelector

class TestStack(unittest.TestCase):

    def setUp(self):
        self.selector = GFileSelector('default')

    def test_store_gfile_info(self):
        logger.info('** Running test_get_peaks')
        self.selector.store_gfile_info(29840, scheduler=True)
        self.assertTrue(len(self.selector.store) > 0)
        pass
    
    def test_get_gfile_fn(self):
        logger.info('** test_get_gfile_fn')
        path, fn = self.selector.get_gfile_fn(29840, 0.214)
        self.assertEqual(fn, 'g_p29840_t0.21400')
        path, fn = self.selector.get_gfile_fn(29840, 0.212, current_file=(path, fn), dt_switch_gfile=0.01)
        self.assertEqual(fn, 'g_p29840_t0.21400')
        path, fn = self.selector.get_gfile_fn(29840, 0.272, current_file=(path, fn), dt_switch_gfile=0.01)
        self.assertEqual(fn, 'g_p29840_t0.27200')
        pass





def suite():
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestStack('test_get_peaks'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())