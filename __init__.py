

import logging
from logging.config import fileConfig
import os
import sys
print(os.getcwd())
fn_log = '../logging_config.ini'
try:
    fileConfig(fn_log)
except:
    print('Failed to load logger settings from {}'.format(fn_log))
logging.getLogger(__name__).addHandler(logging.NullHandler())