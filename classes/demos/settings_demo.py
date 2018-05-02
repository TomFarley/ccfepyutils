#!/usr/bin/env python

import os
import numpy as np

from ccfepyutils.classes.settings import Settings
from ccfepyutils.classes.composite_settings import CompositeSettings


def settings():
    # Settings.get_logfile('test_tmp').delete_file(force=True)
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

def set_column():
    s = Settings.collect('Elzar', 'template')
    s.view(cols=['repr', 'runtime'])
    s.set_column('runtime', False, items=['Invertor_resolution', 'gfile'], apply_to_groups=True)  # can mix groups/items
    s.view(cols=['repr', 'runtime'])

def debug():
    s = Settings.collect('Elzar', 'template')
    pass

if __name__ == '__main__':
    # debug()
    # composite_settings()
    settings()

    pass