#!/usr/bin/env python

import os
import numpy as np

from ccfepyutils.classes.settings import Settings

if __name__ == '__main__':
    # Settings.get_logfile('test').delete_file(force=True)
    s = Settings('settings_test', 'default')
    print(s)
    s('path', '~')
    s('itterations', 213, name='cycle itterations', description='awsome item')
    print(s)

    # print(s.columns)
    # print(s.items)
    print(s)
    print(s.log_file)
    path_setting = s['path']

    s2 = Settings('settings_test', 'test1')
    path_setting[:]
    s2('path', '~')
    # s('path', '~')
    s2('itterations', 213, name='cycle itterations', description='awsome item')

    # for i in np.arange(100):
    #     print(i, os.fstat(i))


    pass