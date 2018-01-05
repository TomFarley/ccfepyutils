

import logging
from logging.config import fileConfig
import os, inspect

def script_directory(level=0):
    fn = os.path.abspath(inspect.getfile(inspect.currentframe()))  # Path to this file
    path = os.path.dirname(fn)
    return path
this_dir = script_directory()

print('__init__: {}'.format(this_dir))
fn_log = os.path.join(this_dir, 'logging_config.ini')  # change to using file in ~/.ccfetools ?

try:
    assert os.path.isfile(fn_log)
    fileConfig(fn_log)
except KeyError as e:
    print('Failed to load logger settings from {}\nKeyError: {}'.format(fn_log, e))
except AssertionError as e:
    print('Failed to load logger settings from {}\nFile does not exist {}'.format(fn_log, e))
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logger.info('Created logger: {}'.format(logger))