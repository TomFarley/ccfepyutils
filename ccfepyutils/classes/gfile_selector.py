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

# from ccfepyutils.classes.settings import Settings
from ccfepyutils.io_tools import filter_files_in_dir, locate_file, mkdir
from ccfepyutils.utils import none_filter, str_to_number, safe_arange

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GFileSelector(object):
    """ """
    try:
        import idam
        idam.setHost("idam1")
    except:
        logger.warning("Idam module not available for saving new scheduler gfiles")
        idam = False

    def __init__(self, settings, fix_gfile=None, start_gfile=None, **kwargs):
        from setpy import Settings
        # TODO generalise to other origins other than scheduler and default
        # self.settings = Settings.get('GFileSelector', settings)
        self.settings = Settings('gfileselector', settings)
        self.settings.update_from_dict(kwargs)
        self.store = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=['pulse', 't']),
                                  columns=['fn', 'i_path', 'scheduler', 'pulse', 'n', 't'])
        # User supplied gfile to always return if not None
        self.fixed_gfile = none_filter(self.settings['fixed_gfile'].value, fix_gfile)
        # Remember previous gfile for easy recall
        self.last_gfile = none_filter(self.fixed_gfile, start_gfile)

        self.settings['gfile'] = os.path.basename(self.fixed_gfile[1]) if self.fixed_gfile is not None else '*None*'

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}>'.format(class_name, None)

    def get_gfile_fn(self, pulse=None, time=None, allow_scheduler_efit=None, machine='MAST', current_file=None,
                     dt_switch_gfile=None, default_to_last=True, raise_on_inexact_match=True):
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
        pulse = int(pulse)
        time = float(time)
        allow_scheduler_efit = none_filter(s['allow_scheduler_efit'].value, allow_scheduler_efit)
        dt_switch_gfile = none_filter(s['dt_switch_gfile'].value, dt_switch_gfile)
        if pulse not in self.store['n']:
            # If available files for pulse haven't been loaded, load them
            self.store_gfile_info(pulse, machine=machine, scheduler=allow_scheduler_efit)
        if ((allow_scheduler_efit) and (self.idam) and
                ((pulse not in self.store['n']) or (np.min(np.abs(store.loc[pulse, 't']-time)) > dt_switch_gfile))):
            # If scheduler gfile hasn't previously been saved in desired time inverval save the closest gfile for
            # requested time
            self.save_scheduler_gfile(pulse, time, machine)
            self.store_gfile_info(pulse, machine=machine, scheduler=allow_scheduler_efit)

        if pulse not in self.store.index.get_level_values('pulse'):
            raise IOError('No gfiles located for pulse "{}" (allow_scheduler_efit={})'.format(
                pulse, allow_scheduler_efit))
        if current_file is not None:
            assert len(current_file) == 2, '{}'.format(current_file)
            current_path, current_fn = current_file
            icurrent_path = self.path_to_path_index(current_path)
            mask = (store['fn'] == current_fn) * (store['i_path'] == icurrent_path)
            t_current = store.loc[mask, 't']
            assert len(t_current) == 1
            t_current = t_current.values[0]
            if np.abs(t_current - time) < dt_switch_gfile:
                # No change - previous (current) file is still within time window
                self.last_gfile = current_file
                return (current_file)
        store = store.loc[pulse]
        # Get file closest in time to that requested
        closest = store.iloc[[np.argmin((store['t'] - time).abs().values)]]
        assert len(closest) == 1
        fn_closest = closest['fn'].values[0]
        path_closest = self.path_index_to_path(closest['i_path'], scheduler=closest['scheduler'].all()).format(
            pulse=pulse, machine=machine)
        new_file = (path_closest, fn_closest)
        t_diff = closest['t'].values[0] - time

        if (np.abs(t_diff) >= dt_switch_gfile) and (not np.isclose(np.abs(t_diff), dt_switch_gfile)):
            message = 'Closest available gfile is outside of desired time window {} +/- {}: {}'.format(
                time, dt_switch_gfile, new_file)
            if raise_on_inexact_match:
                raise ValueError(message)
            elif time != 0:
                logger.warning(message)

        logger.debug('Closest gfile at t={} changed from \n"{}" to \n"{}"'.format(time, current_file, new_file))
        self.last_gfile = new_file
        self.settings['gfile'] = new_file[1]
        return new_file


    def store_gfile_info(self, pulse, machine='MAST', scheduler=False):
        """Add information about available gfiles to self.store for requested (pulse, machine)"""
        #TODO: Consolidate code
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
                        store.loc[key, ['fn', 'i_path', 'scheduler', 'pulse', 't']] = [value, i_path, True,
                                                                                           key[0], key[1]]
        located_file_keys = []
        for i_path, path in enumerate(s['kinetic_gfile_paths']):
            path = path.format(pulse=pulse, machine=machine)
            for ifn_pattern, fn_pattern in enumerate(s['kinetic_gfile_fn_formats']):
                # fn_re = re.sub('\{pulse[^_]*\}', '(\d{5})', fn_pattern)
                # fn_re = re.sub('\{gfile_time[^_]*\}', '([.\d]+)', fn_re)
                files = filter_files_in_dir(path, fn_pattern, group_keys=('pulse', 'time'),
                                            pulse=r'(\\d{5})', gfile_time=r'([.\\d]+)', raise_on_missing_dir=False)
                for key, value in files.items():
                    if key not in located_file_keys:
                        # If file has already been located don't overwrite with file at same t in lower priority dir
                        store.loc[key, ['fn', 'i_path', 'scheduler', 'pulse', 't']] = [value, i_path, False, key[0], key[1]]
                        located_file_keys.append(key)

        store = store.sort_index()
        self.store['pulse'] = store.astype({'pulse': int})['pulse']

    def save_scheduler_gfile(self, pulse, time, machine='MAST', fn_format=None, path=None):
        """Save scheduler gfile to file for access without idam"""
        if not self.idam:
            logging.warning('IDAM is not availble for saving gfiles')
            return (None, None, None)
        from pyEquilibrium import equilibrium as eq
        if fn_format is None:
            fn_format = self.settings['scheduler_gfile_fn_formats'][0]
        if path is None:
            path = self.settings['scheduler_gfile_paths'][0]
        e = eq.equilibrium(device=machine, shot=pulse, time=time)
        if not e._loaded:
            logger.error('Failed to load equilibrium gfile for pulse {}, time {}'.format(pulse, time))
            return (None, None, None)
        gfile_time = e._time
        if gfile_time != time:
            logger.info('Closest scheduler equilibrium time to {:0.6f} s is {:0.5f} s (diff {:0.3} ms)'.format(
                time, gfile_time, (gfile_time-time)*1e3))
        fn = fn_format.format(pulse=pulse, gfile_time=gfile_time, machine=machine)
        path = os.path.expanduser(path.format(pulse=pulse, gfile_time=gfile_time, machine=machine))
        fn_path = os.path.join(path, fn)
        if os.path.isfile(fn_path):
            logger.debug('Closest scheduler equilibrium file to time {:0.6f} s already exists for {:0.5f} s'.format(
                time, gfile_time))
            return fn, path, gfile_time
        mkdir(path, depth=3)
        e.dump_geqdsk(fn_path)
        if os.path.isfile(fn_path):
            logger.info('Saved gfile "{}" to {}'.format(fn, path))
        else:
            raise IOError('Failed to produce scheduler gfile: {}'.format(fn))
        return fn, path, gfile_time

    def save_scheduler_gfiles_in_twin(self, pulse, t_win, machine='MAST', dt_switch_gfile=None, fn_format=None,
                                      path=None, max_workers=16):
        import concurrent.futures
        s = self.settings
        dt_switch_gfile = none_filter(s['dt_switch_gfile'], dt_switch_gfile)
        fn_format = none_filter(s['scheduler_gfile_fn_formats'][0], fn_format)
        path = none_filter(s['scheduler_gfile_paths'][0], path)
        fn, path, t0 = self.save_scheduler_gfile(pulse, t_win[0], machine=machine, fn_format=fn_format, path=path)
        times = list(safe_arange(t0, t_win[1], dt_switch_gfile))

        # Remove times that already have files
        existing_times = []
        for t0 in deepcopy(times):
            fn0 = fn_format.format(pulse=pulse, gfile_time=t0, machine=machine)
            path0 = os.path.expanduser(path.format(pulse=pulse, gfile_time=t0, machine=machine))
            fn_path = os.path.join(path0, fn0)
            if os.path.isfile(fn_path):
                existing_times.append(times.pop(times.index(t0)))
        nt = len(times)
        if nt == 0:
            logger.debug('Gfiles already exist for requested time window for pulse {}: {}'.format(
                pulse, existing_times))
            return
        inputs = list(zip([pulse]*nt, times, [machine]*nt, ))
        logging.info('Getting pulse {} gfiles for {} times ({} already exist): {}'.format(
            pulse, nt, len(existing_times), times))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Dict linking future objects to times
            # executor.map(self.save_scheduler_gfile, inputs)
            future_to_t = {executor.submit(self.save_scheduler_gfile, *inputs[i]): t0 for (i, t0) in enumerate(times)}
            # As each process completes set and save the data for the appropriate frame
            for future in concurrent.futures.as_completed(future_to_t):
                exception = future.exception()
                if exception:
                    logger.info('Exception from future: ({}){} == None: {}, {}'.format(type(exception),
                                                                                       exception, exception is None,
                                                                                       exception == None))
                    logger.exception('Analysis of frame {n} failed\n{error}'.format(n=future_to_t[future],
                                                                                    error=future.exception()))
                    # continue
                t0 = future_to_t[future]
                try:
                    logger.info('Saved gfiles for time {}: {}'.format(t0, future.result()))
                except Exception as e:
                    raise e
        logging.info('Saved pulse {} gfiles for times: {}'.format(pulse, times))


    def path_index_to_path(self, i, scheduler=False):

        paths = 'kinetic_gfile_paths' if not scheduler else 'scheduler_gfile_paths'
        try:
            path = os.path.expanduser(self.settings[paths].value[int(i)])
            return path
        except IndexError as e:
            raise e
        raise ValueError

    def path_to_path_index(self, path):
        path = os.path.expanduser(path)
        path = re.sub('/\d{5}/', '/{pulse}/', path)
        for paths in ['kinetic_gfile_paths', 'scheduler_gfile_paths']:
            index_paths = [os.path.expanduser(v) for v in self.settings[paths].value]
            if path in index_paths:
                i = index_paths.index(path)
                return i
        raise ValueError('Path {} is not in settings options: {}'.format(path, index_paths))

if __name__ == '__main__':
    gs = GFileSelector('scheduler')
    # gs.save_scheduler_gfiles_in_twin(29840, [0.10601, 0.25001], machine='MAST', dt_switch_gfile=0.005, max_workers=1)
    # gs.save_scheduler_gfiles_in_twin(29991, [0.12890, 0.14690], machine='MAST', dt_switch_gfile=0.005, max_workers=1)
    # gs.save_scheduler_gfiles_in_twin(29991, [0.182, 0.195], machine='MAST', dt_switch_gfile=0.005, max_workers=1)
    gs.save_scheduler_gfiles_in_twin(29991, [0.255, 0.272], machine='MAST', dt_switch_gfile=0.005, max_workers=1)
    # gs.save_scheduler_gfiles_in_twin(29852, [0.255, 0.272], machine='MAST', dt_switch_gfile=0.005, max_workers=1)