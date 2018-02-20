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
import pandas as pd
from collections import defaultdict

import logging
# from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ccfepyutils.classes.settings import Settings, SettingsLogFile
# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestSettings(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_logfile(self):
        logger.info('** Running test_get_logfile')
        log = Settings.get_logfile('settings_test')
        self.assertIsInstance(log, (SettingsLogFile, type(None)))
        pass

    def test_call(self):
        logger.info('** Running test_call')
        settings = Settings.get('settings_test', 'test1')
        value = id(settings)
        item = 'my_value_'+str(value)[:4]
        settings(item, value, description=str(value))
        self.assertEqual(settings[item], value)
        logger.warning(settings[item]['description'])
        logger.warning(str(value))
        print(settings[item]['description'])
        print(str(value))
        self.assertEqual(settings[item]['description'], str(value))
        pass

    def test_view(self):
        logger.info('** Running test_get_logfile')
        settings = Settings.get('settings_test', 'test1')
        self.assertIsInstance(settings, Settings)
        self.assertIsInstance(settings.view(), pd.DataFrame)
        pass

    def test_delete_file(self):
        logger.info('** Running test_delete_file')
        settings = Settings.get('settings_test', 'tmp')
        settings['tmp'] = 'tmp'
        settings.save()
        settings.delete_file(force=True)

def suite():
    print('Setting test suit')
    suite = unittest.TestSuite()

    suite.addTest(TestSettings('test_saved_settings'))
    # suite.addTest(TestStack('test_init_empty2'))
    # suite.addTest(TestStack('test_loc'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())