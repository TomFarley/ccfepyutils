#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from copy import deepcopy

__author__ = 'tfarley'

import unittest

import numpy as np

from ccfepyutils.classes.movie import Movie
from ccfepyutils.utils import safe_len
from ccfepyutils.io_tools import pickle_load

import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestMovie(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        m = Movie(pulse=29852, machine='MAST', camera='SA1.1')
        self.assertTrue(isinstance(m, Movie))
        pass

    def test_frame_data(self):
        m = Movie(pulse=29852, machine='MAST', camera='SA1.1', frames=np.arange(12000, 12010))

        frame = m[12000]
        data_expected = pickle_load('./test_data/movie_data/frame_data_MAST_29852_SA1.1_12000.p')
        data = frame.data
        self.assertTrue(np.all(data == data_expected))

    def test_enhance(self):
        m = Movie(pulse=29852, machine='MAST', camera='SA1.1', frames=np.arange(12000, 12010))
        m.enhance(['extract_fg', 'reduce_noise', 'sharpen'])
        frame = m[12000]
        data_expected = pickle_load('./test_data/movie_data/frame_data_MAST_29852_SA1.1_12000_enhanced.p')
        data = frame.data
        self.assertTrue(np.all(data == data_expected))

def suite():
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestMovie('test_init'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())