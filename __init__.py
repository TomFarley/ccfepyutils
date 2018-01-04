

import logging
from logging.config import fileConfig
import os
import sys

print('__init__: {}'.format(os.getcwd()))
fn_log = 'logging_config.ini'
try:
    assert os.path.isfile(fn_log)
    fileConfig(fn_log)
except KeyError as e:
    print('Failed to load logger settings from {}\nKeyError: {}'.format(fn_log, e))
except AssertionError as e:
    print('Failed to load logger settings from {}\nFile does not exist {}'.format(fn_log, e))
logging.getLogger(__name__).addHandler(logging.NullHandler())