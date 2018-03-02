#!/usr/bin/env python

import os
import numpy as np

from ccfepyutils.classes.settings import Settings
from ccfepyutils.classes.composite_settings import CompositeSettings


def settings():
    # Settings.get_logfile('test').delete_file(force=True)
    settings = Settings.get('Settings_demo', 'default')
    print()
    print(settings)
    print()
    settings('path', '~/my_path')
    settings('pulse', 29852, name='Pulse number')
    settings('processed', True, description='Whether data has been processed using ...')
    settings('itterations', 213, name='cycle itterations', description='awsome item')
    print()
    print(settings)
    print()

    print(settings.log_file)
    path_setting = settings['path']

    settings.delete_file(force=True)

    # s = Settings.get('Movie', 'repeat')
    # s = Settings.get('Elzar_checkpoints', 'repeat')
    s = Settings.get('Movie_range', 'repeat')
    # s = Settings.get('Elzar', 'config')
    s['end frame']
    print(s.view_str())

def composite_settings():
    cs = CompositeSettings('Elzar', 'config')
    print(cs.view())
    s = cs.get_settings_for_item('gfile')
    s.set_column('function', 'Invertor')

    print(cs.view())
    pass

def debug():
    s = Settings.collect('Elzar', 'template')
    s.set_column('runtime', False, items=['Invertor_resolution'], apply_to_groups=True)
    s.set_column('runtime', False, items=['gfile', 'calcam_calib', 'Movie_source', 'QuadMinEllipseDetector', 'FlMatrixInverter', 'FlMatrixInverter'], apply_to_groups=True)
    s.set_column('runtime', False, items=['gfile', 'inversion_corrector_type', 'Enhancer_settings', 'invertor_type', 'detector_type'], apply_to_groups=True)
    pass

if __name__ == '__main__':
    debug()
    # composite_settings()
    # settings()

    pass