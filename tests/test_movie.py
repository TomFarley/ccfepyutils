#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from copy import deepcopy

__author__ = 'tfarley'

import unittest

import numpy as np

from ccfepyutils.classes.movie import Movie
from ccfepyutils.utils import safe_len
from ccfepyutils.io_tools import pickle_load, pickle_dump

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

    def setUp(self):
        self.pulse = 29852
        self.machine = 'MAST'
        self.camera = 'SA1.1'
        self.start_frame = 13
        self.end_frame = 16
        self.nframes = self.end_frame - self.start_frame + 1
        pass

    def test_init(self):
        logger.info('** Running test_init')
        movie = Movie(self.pulse, self.machine, self.camera)
        with self.assertRaises(ValueError):
            Movie(29852, 'LHD-U', 'SA1.1')

    def test_set_frames(self):
        logger.info('** Running test_init')
        movie = Movie(self.pulse, self.machine, self.camera)
        movie.set_frames(start_frame=self.start_frame, end_frame=self.end_frame)
        self.assertTrue(movie._frame_range_info_user['nframes'] == self.nframes)
        movie.set_frames(start_frame=self.start_frame, nframes_user=self.nframes)
        self.assertTrue(movie._frame_range_info_user['frame_range'][1] == self.end_frame)

        with self.assertRaises(ValueError):
            movie.set_frames(end_frame=20)

        with self.assertRaises(AssertionError):
            movie.set_frames(start_frame=100, end_frame=20)

    def test_load_movie_data(self):
        logger.info('** Running test_init')
        movie = Movie(self.pulse, self.machine, self.camera)
        movie.set_frames(start_frame=self.start_frame, end_frame=self.end_frame)
        movie.load_movie_data()
        pass

    def test_index(self):
        logger.info('** Running test_init')
        movie = Movie(self.pulse, self.machine, self.camera)
        movie.set_frames(start_frame=self.start_frame, end_frame=self.end_frame)
        frame = movie[self.start_frame]
        pass

    def test_lookup(self):
        logger.info('** Running test_lookup')
        movie = Movie(self.pulse, self.machine, self.camera)
        movie.set_frames(start_frame=self.start_frame, end_frame=self.end_frame)
        t = movie.lookup('t', i=0)
        pass

    def test_enhance(self):
        logger.info('** Running test_enhance')
        movie = Movie(self.pulse, self.machine, self.camera)
        movie.set_frames(start_frame=self.start_frame, end_frame=self.end_frame)
        movie.enhance('threshold', frames='all')
        movie[self.start_frame].plot()

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
        data = frame.data
        data_expected = pickle_load('./test_data/movie_data/frame_data_MAST_29852_SA1.1_12000_enhanced.p')
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