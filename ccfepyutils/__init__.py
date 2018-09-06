import logging
from logging.config import fileConfig
import os, inspect, shutil
import pandas as pd

batch_mode = os.getenv('LOADL_ACTIVE', 'no')
job_name = os.getenv('LOADL_JOB_NAME', None)
execution_mode = os.getenv('LOADL_STEP_TYPE', None)
in_docker = os.getenv('IN_DOCKER', 'False')
import matplotlib
if (batch_mode == 'yes') or (in_docker == 'True'):
    matplotlib.use('Agg')
    print('In batch mode')
else:
    matplotlib.use('Qt5Agg')
    # matplotlib.use('Qt4Agg')
    pass

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
pd.options.display.max_rows = 999

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

logger.debug('Created logger: {}'.format(logger))

# TODO: get base path from config file?
settings_dir = os.path.expanduser('~/.ccfetools/settings/')

from ccfepyutils.build_dir_struct import check_ccfepyutils_dir_struct
template_settings_dirs = os.path.abspath(os.path.join(this_dir, '../template_settings/values'))

try:
    check_ccfepyutils_dir_struct(template_settings_dirs=template_settings_dirs, settings_dir=settings_dir)
except FileNotFoundError as e:
    settings_dir = os.path.abspath(os.path.join(this_dir, '.ccfetools/settings'))

# Make most commonly used classes available directly from ccfepyutils import
from ccfepyutils.classes.settings import Settings
from ccfepyutils.classes.composite_settings import CompositeSettings
from ccfepyutils.classes.plot import Plot
from ccfepyutils.classes.fitter import Fitter
from ccfepyutils.classes.gfile_selector import GFileSelector
from ccfepyutils.classes.movie import Movie