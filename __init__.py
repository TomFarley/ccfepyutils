

import logging
from logging.config import fileConfig
fileConfig('../logging_config.ini')
logging.getLogger(__name__).addHandler(logging.NullHandler())