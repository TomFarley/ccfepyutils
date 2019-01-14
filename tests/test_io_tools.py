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

class TestIoTools(unittest.TestCase):

    def setUp(self):
        import ccfepyutils

    def test_filter_files(self):
        logger.info('** Running test_filter_files')
        from ccfepyutils.io_tools import filter_files
        fns = ['hello.txt', 'bye002.txt', 'greetings001.py']
        out = filter_files(fns, '\w+({n}).\w+', n=[1,2])
        self.assertEqual(out, {1: 'greetings001.py', 2: 'bye002.txt'})
        pass

    def test_filter_files_in_dir(self):
        logger.info('** Running test_filter_files_in_dir')
        from ccfepyutils.io_tools import filter_files_in_dir
        path = os.path.dirname(__file__)
        path_bellow = os.path.join(path, '..')

        out = filter_files_in_dir(path, 'test({name})\.py', group_keys=['name'], name=['_io_tools', 'suite'])
        reference = {'_io_tools': 'test_io_tools.py', 'suite': 'testsuite.py'}
        self.assertEqual(out, reference, 'Pattern substiution, depth=0')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1)
        reference = {'/home/tfarley/repos/ccfepyutils/tests':
                         {0: 'test_io_tools.py', 1: 'testsuite.py', 2: 'test_stack.py'}}
        self.assertEqual(out, reference, 'depth=1')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[1e6, None])
        self.assertEqual(out, reference, 'Test modified_range since')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[None, 0])
        self.assertEqual(out, reference, 'Test modified_range before now')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[None, 1e6])
        self.assertEqual(out, {}, 'Test modified_range before eon')

        pass



def suite():
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestIoTools('test_filter_files'))
    suite.addTest(TestIoTools('test_filter_files_in_dir'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())