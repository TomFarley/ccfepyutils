

import logging
from logging.config import fileConfig
import os, inspect
import pandas as pd

def script_directory(level=0):
    fn = os.path.abspath(inspect.getfile(inspect.currentframe()))  # Path to this file
    path = os.path.dirname(fn)
    return path
this_dir = script_directory()

fn_log = os.path.join(this_dir, 'logging_config.ini')  # change to using file in ~/.ccfetools ?

try:
    assert os.path.isfile(fn_log)
    fileConfig(fn_log)
except KeyError as e:
    print('Failed to load logger settings from {}\nKeyError: {}'.format(fn_log, e))
except AssertionError as e:
    print('Failed to load logger settings from {}\nFile does not exist {}'.format(fn_log, e))

# Update pandas display settings
# Double max width of displayed output in terminal so doesn't wrap over many lines
pd.set_option("display.width", 160)  # TODO: Use None when in ipython terminal - auto size?
# Double max column display width to display long descriptive strings in columns
pd.set_option("display.max_colwidth", 80)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

logger.debug('Created logger: {}'.format(logger))