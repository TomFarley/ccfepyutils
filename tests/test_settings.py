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

from ccfepyutils.classes.settings import Settings
from ccfepyutils.utils import safe_len

import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# pwd = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))

def return_none():
    return None

class TestSettings(unittest.TestCase):

    def setUp(self):
        pass

    def test_get(self):
        logger.info('** Running test_get')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')
        pass

    def test_is_item_in(self):
        logger.info('** Running test_is_item_in')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')

        out = s.is_item_in('item1', 'Settings_unittests_1', 'unittests_1_2')
        self.assertTrue(out)

        out = s.is_item_in('item1', 'Settings_unittests_1', 'unittests_1_4')
        self.assertFalse(out)

    def test_is_item_equal_in(self):
        logger.info('** Running test_is_item_equal_in')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')

        out = s.is_item_equal_in('item1', 'Settings_unittests_1', 'unittests_1_2')
        self.assertEqual(out, (True, ('identical', )))

        out = s.is_item_equal_in('item1', 'Settings_unittests_1', 'unittests_1_3')
        self.assertEqual(out, (False, ('not equal', ('hello', 'goodbye'))))

        out = s.is_item_equal_in('item1', 'Settings_unittests_1', 'unittests_1_4')
        self.assertEqual(out, (False, ('missing', )))

        out = s.is_item_equal_in('item1', 'Settings_unittests_1', 'unittests_1_5')
        # series1 = pd.Series(np.nan, index=['description'], dtype=object)
        self.assertTrue(out[0])
        self.assertEqual(out[1][0], 'columns differ')

    def test_compare_to(self):
        logger.info('** Running test_compare_to')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')

        out = s.compare_to('Settings_unittests_1', 'unittests_1_1')
        self.assertTrue(out[0])

        out = s.compare_to('Settings_unittests_1', 'unittests_1_3')
        self.assertFalse(out[0])

    def test_compare_log_times(self):
        logger.info('** Running test_compare_log_times')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')
        out = s.compare_log_times('Settings_unittests_1', 'unittests_1_2')
        self.assertEqual(out, 'same age')

    def test_update_from_other(self):
        logger.info('** Running test_update_from_other')
        s = Settings.get('Settings_unittests_1', 'unittests_1_1')
        s_tmp = Settings.get('Settings_unittests_1', 'tmp')
        s_tmp.update_from_other('Settings_unittests_1', 'unittests_1_1', only_if_newer=False,
                                update_values=False, update_columns=False, save=False)
        self.assertTrue(s._df.equals(s_tmp._df))

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
    print('Setting test_tmp suit')
    suite = unittest.TestSuite()

    suite.addTest(TestSettings('test_get'))
    suite.addTest(TestSettings('test_init_empty2'))
    suite.addTest(TestSettings('test_is_item_in'))
    suite.addTest(TestSettings('test_is_item_equal_in'))
    suite.addTest(TestSettings('test_compare_to'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())