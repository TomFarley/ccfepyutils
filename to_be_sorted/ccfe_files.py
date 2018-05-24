#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger(__name__)

def rm_files(path, pattern, verbose=True):
    path = str(path)
    if verbose:
        print('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
    for fn in os.listdir(path):
        if re.search_items(pattern, fn):
            os.remove(os.path.join(path, fn))
            if verbose:
                print('Deleted file: {}'.format(fn))