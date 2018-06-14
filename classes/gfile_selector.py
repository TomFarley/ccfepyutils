#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging, os, itertools, re, inspect, configparser, time
from collections import defaultdict, OrderedDict
from datetime import datetime
from copy import copy, deepcopy
from pathlib import Path
from logging.config import fileConfig

import numpy as np
import pandas as pd

from ccfepyutils.classes.settings import Settings
from ccfepyutils.io_tools import filter_files_in_dir, locate_file
from ccfepyutils.utils import none_filter, str_to_number

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GFileSelector(object):
    """ """

    def __init__(self, settings, fix_gfile=None, start_gfile=None, **kwargs):
        # TODO generalise to other origins other than scheduler and default
        self.fixed_gfile = fix_gfile  # User supplied gfile to always return
        self.last_gfile = start_gfile
        self.settings = Settings.get('GFileSelector', settings)
        self.settings.update_from_dict(kwargs)
        self.store = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=['n', 't']),
                                  columns=['fn', 'i_path', 'scheduler', 'n', 't'])

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}>'.format(class_name, None)

    def get_gfile_fn(self, pulse=None, time=None, allow_scheduler_efit=None, machine='MAST', current_file=None,
                     dt_switch_gfile=None, default_to_last=True):
        if self.fixed_gfile is not None:
            return self.fixed_gfile
        elif (pulse is None) or (time is None):
            if default_to_last:
                return self.last_gfile
            else:
                raise ValueError('"get_gfile_fn" requires "pulse" and "time" values if default_to_last is False and '
                                 'fixed_gfile is None')
        s = self.settings
        store = self.store
        allow_scheduler_efit = none_filter(s['allow_scheduler_efit'].value, allow_scheduler_efit)
        dt_switch_gfile = none_filter(s['dt_switch_gfile'].value, dt_switch_gfile)
        if pulse not in self.store['n']:
            self.store_gfile_info(pulse, machine=machine)
        if pulse not in self.store['n']:
            raise IOError('No gfiles located for pulse "{}"'.format(pulse))
        if current_file is not None:
            assert len(current_file) == 2, '{}'.replace(current_file)
            current_path, current_fn = current_file
            icurrent_path = self.path_to_path_index(current_path)
            mask = (store['fn'] == current_fn) * (store['i_path'] == icurrent_path)
            t_current = store.loc[mask, 't']
            assert len(t_current) == 1
            t_current = t_current.values[0]
            if np.abs(t_current - time) < dt_switch_gfile:
                self.last_gfile = current_file
                return (current_file)
        store = store.loc[pulse]
        closest = store.iloc[np.argmin((store['t'] - time).abs().values)]
        assert len(closest) == 1
        fn_closest = closest['fn']
        path_closest = self.path_index_to_path(closest['i_path'], scheduler=closest['scheduler'])
        self.last_gfile = (path_closest, fn_closest)
        return (path_closest, fn_closest)


    def store_gfile_info(self, pulse, machine='MAST', scheduler=False):
        s = self.settings
        store = self.store
        if scheduler:
            for i_path, path in enumerate(s['scheduler_gfile_paths']):
                path = path.format(pulse=pulse, machine=machine)
                for ifn_pattern, fn_pattern in enumerate(s['scheduler_gfile_fn_formats']):
                    files = filter_files_in_dir(path, fn_pattern, group_keys=('pulse', 'time'),
                                            pulse='(\d{5})', gfile_time='([.\d]+)', raise_on_missing_dir=False)
                    for key, value in files.items():
                        key = tuple([str_to_number(k) for k in key])
                        store.loc[key, ['fn', 'i_path', 'scheduler', 'n', 't']] = [value, i_path, True,
                                                                                           key[0], key[1]]
        for i_path, path in enumerate(s['kinetic_gfile_paths']):
            path = path.format(pulse=pulse, machine=machine)
            for ifn_pattern, fn_pattern in enumerate(s['kinetic_gfile_fn_formats']):
                # fn_re = re.sub('\{pulse[^_]*\}', '(\d{5})', fn_pattern)
                # fn_re = re.sub('\{gfile_time[^_]*\}', '([.\d]+)', fn_re)
                files = filter_files_in_dir(path, fn_pattern, group_keys=('pulse', 'time'),
                                            pulse='(\d{5})', gfile_time='([.\d]+)', raise_on_missing_dir=False)
                for key, value in files.items():
                    key = tuple([str_to_number(k) for k in key])
                    store.loc[key, ['fn', 'i_path', 'scheduler', 'n', 't']] = [value, i_path, False, key[0], key[1]]


        self.store.sort_index()

    def save_scheduler_gfile(self, pulse, time, machine='MAST', fn_format=None, path=None):
        from pyEquilibrium import equilibrium as eq
        if fn_format is None:
            fn = self.settings['scheduler_gfile_fn_formats'][0]
        if path is None:
            path = self.settings['scheduler_gfile_paths'][0]

        e = eq.equilibrium(device=machine, shot=pulse, time=time)
        fn = fn_format.format(pulse=pulse, time=time, machine=machine)
        path = path.format(pulse=pulse, time=time, machine=machine)
        fn_path = os.path.join(path, fn)
        e.dump_geqdsk(fn_path)
        logger.info('Saved gfile "{}" to {}'.format(fn, path))

    def path_index_to_path(self, i, scheduler=False):

        paths = 'kinetic_gfile_paths' if not scheduler else 'scheduler_gfile_paths'
        try:
            path = self.settings[paths].value[int(i)]
            return path
        except IndexError as e:
            raise e
        raise ValueError

    def path_to_path_index(self, path):
        path = os.path.expanduser(path)
        for paths in ['kinetic_gfile_paths', 'scheduler_gfile_paths']:
            try:
                i = [os.path.expanduser(v) for v in self.settings[paths].value].index(path)
                return i
            except IndexError as e:
                pass
        raise ValueError

if __name__ == '__main__':
    pass