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
from pathlib import Path

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
        path_bellow = str(Path(os.path.join(path, '..')).resolve())

        out = filter_files_in_dir(path, 'test({name})\.py', group_keys=['name'], name=['_io_tools', 'suite'])
        reference = {'_io_tools': 'test_io_tools.py', 'suite': 'testsuite.py'}
        self.assertEqual(out, reference, 'Pattern substiution, depth=0')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1)
        reference = {'/home/tfarley/repos/ccfepyutils/tests':
                         {0: 'test_gfile_selector.py',
                           1: 'test_settings.py',
                           2: 'test_io_tools.py',
                           3: 'testsuite.py',
                           4: 'test_utils.py',
                           5: 'test_movie.py',
                           6: 'test_stack.py'}
                         }
        self.assertEqual(out, reference, 'depth=1')

        # Modified_range = [n_days_old_min, n_days_old_max]
        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[None, 1e6])
        self.assertEqual(out, reference, 'Test modified_range since')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[0, None])
        self.assertEqual(out, reference, 'Test modified_range before now')

        out = filter_files_in_dir(path_bellow, 'test.*\.py', depth=1, modified_range=[1e6, None])
        self.assertEqual(out, {}, 'Test modified_range before eon')

        pass

    def test_split_path(self):
        from ccfepyutils.io_tools import split_path
        path = '/home/tfarley/files/my_file.txt'
        parts, fn = split_path(path, include_fn=True)
        expected_value = ['/', 'home', 'tfarley', 'files', 'my_file.txt']
        self.assertEqual(parts, expected_value)

        path = Path('/home/tfarley/files/my_file.txt')
        parts, fn = split_path(path, include_fn=True)
        self.assertEqual(parts, expected_value)

        parts, fn = split_path(path, include_fn=False)
        expected_value = ['/', 'home', 'tfarley', 'files']
        self.assertEqual(parts, expected_value)


        path = '/home/tfarley/files'
        parts, fn = split_path(path)
        expected_value = ['/', 'home', 'tfarley', 'files']
        self.assertEqual(parts, expected_value)

    def test_insert_subdir_in_path(self):
        from ccfepyutils.io_tools import insert_subdir_in_path
        path = '/home/tfarley/files/my_file.txt'
        new_path = insert_subdir_in_path(path, 'my_subdir', position=-1, create_dir=False)
        self.assertEqual(new_path, '/home/tfarley/files/my_subdir/my_file.txt')



def suite():
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestIoTools('test_filter_files'))
    suite.addTest(TestIoTools('test_filter_files_in_dir'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())