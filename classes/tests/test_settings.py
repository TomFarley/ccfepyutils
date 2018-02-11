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

import logging
# from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from ccfepyutils.classes.settings import Settings, SettingsLogFile
# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestSettings(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_logfile(self):
        logger.info('** Running test_get_logfile')
        log = Settings.get_logfile('test')
        self.assertTrue(isinstance(log, SettingsLogFile))
        pass

def suite():
    print('Setting test suit')
    suite = unittest.TestSuite()

    suite.addTest(TestSettings('test_saved_settings'))
    # suite.addTest(TestStack('test_init_empty2'))
    # suite.addTest(TestStack('test_loc'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())