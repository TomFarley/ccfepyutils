#!/usr/bin/env python

import os
import numpy as np

from ccfepyutils.classes.settings import Settings

if __name__ == '__main__':
    # Settings.get_logfile('test').delete_file(force=True)
    settings = Settings('Settings_demo', 'default')
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

    s = Settings.get('Enhancer', 'default')
    rn = s['reduce_noise']

    pass