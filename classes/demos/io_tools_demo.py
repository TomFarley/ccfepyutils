#!/usr/bin/env python

""" 
Author: T. Farley
"""
import logging

import numpy as np

from ccfepyutils.io_tools import locate_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def demo_locate_file():
    paths = ['~/data/synth_frames/{data_set}', '~/data/synth_frames/{machine}/{data_set}']
    fns = ['Frame_{n}.p', 'Frame_data_{n}.npz']

    path_kws = {'data_set': 'for_tom', 'machine': 'MAST'}
    fn_kws = {'n': 0}

    path, fn = locate_file(paths, fns, path_kws=path_kws, fn_kws=fn_kws,
                          return_raw_path=False, return_raw_fn=True, _raise=True, verbose=True)
    logging.info('Located "{}" in {}'.format(fn, path))

if __name__ == '__main__':
    demo_locate_file()