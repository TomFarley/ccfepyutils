#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os, itertools, gc, re, inspect, configparser, abc, numbers, time, shutil
from collections import OrderedDict
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import ccfepyutils
from ccfepyutils.classes.state import State, in_state
from ccfepyutils.utils import make_iterable, remove_duplicates_from_list, is_subset, get_methods_class, t_now_str, \
    to_list, ask_input_yes_no, safe_isnan, str2datetime
from ccfepyutils.io_tools import mkdir, filter_files_in_dir, delete_file, attempt_n_times
from ccfepyutils.netcdf_tools import dict_to_netcdf, netcdf_to_dict
from ccfepyutils.classes.setting import Setting, SettingList, SettingBool, SettingFloat, SettingInt, SettingStr

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

settings_dir = ccfepyutils.settings_dir

class Settings(object):
    """Object to store, save, load and interact with collections of settings for other classes

    Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. the name of the class your settings are for
        - The name is a label for a particular set of settings for the application e.g. my_default_settings

    Settings are saved for easy reuse in future. By default these files are saved in '~/.ccfetools/settings'. A SettingsLogFile is generated for each application which records the
    names and modification dates etc. of all settings for that application. When accessed a SettingsType object of the
    appropriate type is returned, providing access to setting meta data via indexing.

    Template settings spec
    - Always have reference to template (possibily self)
    - When settings created base them of template if exists by default (copy each setting)
    - When settings loaded, supplement with missing settings from template (copy individual setting)
    - When setting added, ask if should add to template if new (copy setting to template)
    - When setting deleted from template ask if should be deleted from all dependent settings
    New methods required:
    - copy_item_to(application, [name], preserve_order=True, prompt=False) -
    -/ is_item_in(item, application, [name]) - True/False
    -/ is_item_equal_in(item, application, [name]) - True/False, (details)
    -/ compare_to(application, name) - Same value, different value, missing in self, missing in other
    -/ compare_log_times(application, name, type='modified')
    - update_from_other(application, name, only_if_newer=True, add_missing_items=True, update_values=False)
    - update_other(only_if_newer=True, add_missing_items=True, update_values=False)
    -/ set_template()
    - self.items: Inlclude base list indices in returned values

    - Update init to compare template modification date to current modification date and update from template as appro


    NOTES:
        - The name of a setting must be unique for each named set of settings.
        - Settings files are only saved automatically when they are accessed. Use settings.save() to save manually.

    Usage examples:
        - Load/create a settings object (will return an existing instance if already exists):
            settings = Settings.get(<my_amplication>)
        - Add a setting item:
            settings('my_setting', value, name='My settings', description='Value of ...')  # keywords set metadata
        - Access a setting:
            setting = settings['my_setting']
        - View set of columns in settings file eg. 'info' columns
            settings.view('info')
            
        - List applications with existing saved settings:
            Settings.get_applications()
        - List existing named sets of settings for an application:
            Settings.get_settings()
    """

    # index = pd.MultiIndex(levels=[[], []], labels=[[], []], names=['application', 'name'])
    instances = {}

    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}

    state_table = {'core': {
                       'init': ['modified', 'loading'],
                       'loaded': ['accessing', 'modifying'],
                       'modified': ['modifying', 'accessing', 'saving'],
                       'saved': ['accessing', 'modifying', 'saving']},
                   'transient': {
                       'loading': ['loaded', 'modifying', 'accessing'],
                       'modifying': ['modified', 'accessing'],
                       'saving': ['saved', 'modifying'],
                       'accessing': ['loading', 'saving', 'modifying', 'modified', 'saved']}
                   }
    default_column_sets = {'values': [('value', str), ('value_str', str), ('value_num', float)],
                           'info': [('name', str), ('description', str), ('symbol', str), ('unit', str)],
                           'type': [('float', bool), ('int', bool), ('bool', bool), ('str', bool)],
                           'io': [('fn_str', str), ('priority', float)],
                           'meta': [('setting', bool), ('function', str), ('runtime', bool), ('order', int)],
                           'repr': [('value', str), ('description', str), ('setting', bool)],
                           'comp': [('value', str), ('runtime', bool), ('setting', bool)]}  # plotting?
    # TODO: Add modified column in order to keep track of what's been modified since last save?
    # TODO: block reserved states being used
    reserved_item_names = ['all', 'ignore_state']

    def __init__(self, application, name):
        """Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. my_code
        - The name is a label for a particular set of settings for the application e.g. my_default_settings
        """
        assert isinstance(application, str)
        assert isinstance(name, str)
        assert Settings.get_instance(application, name) is None, ('Setting object {}:{} already exists.\n'
                'Use Settings.get(application, name) to ensure there is only one instance'.format(
                application, name))
        # TODO: Fix deepcopy of Settings objects
        self._reset_settings_attributes()
        self.call_table = {'modifying': {'enter': [self._block_protected]},
                           # 'modified': {'accessing': [self.save]}
                           }
        self.state = State(self, self.state_table, 'init', call_table=self.call_table)
        self.application = application  # property sets logfile
        self.name = name
        self._column_sets = Settings.default_column_sets   # Groups of columns with similar purposes
        self._column_sets_names = self.column_sets_names  # TODO: properly implement saving and loading of column sets

        if self.file_exists:
            self.load()
        else:
            logger.info('Initialising new empty settings ({}, {})'.format(application, name))
            self.init()

        self.add_instance()

    def _reset_settings_attributes(self):
        self.state = None
        self.call_table = None
        self._application = None
        self._name = None
        self._df = None
        self.log_file = None
        self._column_sets = None
        self._column_sets_names = None
        self._hash_id = None
        self._composite_settings = []  # list of composite settings instances that these settings currently belong to
        self._template = None

    @classmethod
    def get(cls, application=None, name=None, add_missing_template_items=False, update_template=False,
            auto_create_template=False, default_repeat=False):
        if update_template:
            auto_create_template = True
        if application is None:
            raise ValueError('No application supplied to Settings.get().\nExisting applications settings: {}'.format(
                    cls.existing_applications()))
        if name is None:
            # ToDo: lookup most recently accessed setting name from log file
            raise ValueError('No settings set name supplied to Settings,get().\n'
                             'Existing settings for application "{}": {}'.format(
                    application, cls.existing_settings(application)))
        existing_instance = Settings.get_instance(application, name)
        if existing_instance is not None:
            logger.debug('Returning existing instance of {}'.format(repr(existing_instance)))
            return existing_instance
        settings = Settings(application, name)
        settings._set_template(create_if_missing=auto_create_template)
        existing_settings = settings.log_file.names

        if add_missing_template_items and ('template' in existing_settings):
            # - Compare to template, if missing fields add them
            all_equal, df_out = settings.compare_to(application, 'template')
            if not all_equal:
                settings.update_from_other(application, 'template', add_missing_items=True, only_if_newer=False,
                                           update_values=False, update_columns=False)
            pass
        if update_template:
            # If loaded settings contain items not present in 'template', add them to template
            # TODO: Implement
            raise NotImplementedError


        return settings
        
    @classmethod
    def collect(cls, application=None, name=None, core_values={}, blacklist=[], whitelist=[], exclude_if_col_true=(),
                **kwargs):
        """Collect together settings from multiple settings files into one large settings collection.

        The settings file given by 'applicatoin' and 'name' will be loaded. Its values will be updated acording to the
        'values' keyword dictionary. Then each item in that settings file that corresponds to another settings file will
        be read in and combined into the same overaching settings collection. A CompositeSettings object is returned."""
        from .composite_settings import CompositeSettings
        exclude_if_col_true = make_iterable(exclude_if_col_true)
        settings = Settings.get(application, name)
        for item, value in copy(kwargs).items():
            if item in settings.items:
                core_values[item] = kwargs.pop(item)
            if (settings.list_item_indices(settings, item) is not False):
                core_values[item] = kwargs.pop(item)
        for item, value in core_values.items():
            if isinstance(value, Setting):
                value = value.value
            if (item not in settings.items) and (settings.list_item_indices(settings, item) is False):
                raise ValueError('Supplied core settings does not exist: {}, {}'.format(item, settings))
            settings.set(item, value, ignore=[None])
        composite_settings = CompositeSettings(application, name, blacklist=blacklist, whitelist=whitelist,
                                               exclude_if_col_true=exclude_if_col_true, update_values=kwargs)
        # composite_settings.set_value(**kwargs)
        # composite_settings.save()
        return composite_settings

    @classmethod
    def from_dict(cls, application, name, dictionary, _silent=False, **kwargs):
        if _silent:
            old_log_level = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)
        s = Settings.get(application, name)
        s.delete_items(s.items)
        for key, value in dictionary.items():
            s(key, value, _silent=_silent, **kwargs)
        if _silent:
            logger.setLevel(old_log_level)
        return s

    @in_state('init', 'modified')
    def init(self):
        columns = self.ordered_column_names
        types = self.column_sets_types
        self._df = pd.DataFrame({key: [] for key in columns})#, dtype=types)  # Initialise empty dataframe
        # Set the types of each column
        type_dict = [dict(v) for k,v in self.default_column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        self._df.loc[:, :] = self._df.astype(type_dict)
        # self.df.loc[:, [self.column_sets_names['type']]] = False

    def search(self, pattern):
        r = re.compile(pattern, re.IGNORECASE)
        newlist = list(filter(r.search, self.items))
        return newlist

    def view(self, items='all', cols='repr', order=None, ascending=True):
        """Return dataframe containing a subst of columns, with items ordered as requried"""
        if cols == 'all':
            col_set = self.ordered_column_names
        else:
            col_set = []
            cols = make_iterable(cols)
            for col in cols:
                if col in self.column_sets_names.keys():
                    col_set += self.column_sets_names[col]
                else:
                    assert col in self.columns, 'Column not found: {}'.format(col)
                    col_set += [col]
        if items == 'all':
            items = self.items
        elif isinstance(items, str):
            items = self.search(items)
        if order is None:
            df = self._df
        elif order == 'alphabetical':
            df = self._df.sort_index(ascending=ascending)
        elif order == 'custom':
            df = self._df.sort_values('order', ascending=ascending)
        else:
            raise ValueError('Order "{}" not recognised'.format(order))
        out = df.loc[items, col_set]
        return out

    def view_str(self, cols='repr', order=None, ascending=True):
        """Return string containing a subst of columns, with items ordered as requried and index replaced with name"""
        df = self.view(cols=cols, order=order, ascending=ascending)
        if 'name' in df.columns:
            columns = list(df.columns)
            out = df.reset_index().loc[:, columns].to_string(index=False, justify='left')
        else:
            out = df.to_string(index=True, justify='left')
        return out

    @in_state('modifying', 'modified')
    def __call__(self, item, value=None, _create_columns=False, _save=False, _silent=False, _verbose=True, **kwargs):
        """Set value of setting
        :item: name of setting to change
        :value: value to set the item to
        :kwargs: Use to set non-value columns
        TODO: Replace list setting modification with recursive __call__ for sub setting items
        TODO: Enable setting existing item's value to None without '*None*'
        """
        modified = False
        assert isinstance(item, str), 'Settings must be indexed with string, not "{}" ({})'.format(item, type(item))
        item, category = self.get_item_index(self, item, _raise=False)
        if (category not in (None, 'list')) and (self[item] == value) and (len(kwargs) == 0):
            # No changes
            # tmp - update legacy files
            self._df.loc[item, 'value'] = str(value)
            return
        # Item doesn't exist or updating value or columns
        if isinstance(value, (list, tuple)):
            # Create setting item/Update settings value
            # Store a list as an enumerated set of items
            if item in self.items:
                # scalar item being replaced by list item
                order = self._df.loc[item, 'order']
                kws = {k: self._df.loc[item, k] for k in ['description', 'runtime']}
                kws.update(kwargs)
                kwargs = kws
                self.delete_items(item)
            if len(value) == 0:
                value = ['[<empty>]']
            for i, v in enumerate(value):
                item_i = '{}:{}'.format(item, i)
                if i == 0:
                    if item_i in self.items:
                        # Existing list item being updated
                        order = self._df.loc[item_i, 'order']
                        if self._df.loc[item_i, 'value'] == str(v):
                            # No change
                            # TODO: deal with columns changing value
                            continue
                        kws = {k: self._df.loc[item_i, k] for k in ['description', 'runtime']}
                        kws.update(kwargs)
                        kwargs = kws
                        self.delete_items(item)
                    else:
                        order = len(self)
                else:
                    order += 1
                self(item_i, v, _create_columns=_create_columns, **kwargs)
                self.reorder_item(item_i, order, save=False)
                for com_setting in self._composite_settings:
                    # Update composite settings objects referencing these settings
                    com_setting._df.loc[item_i] = self._df.loc[item_i]
                    com_setting._subsetting_mod_times[self] = t_now_str('natural')
                    com_setting._hash_id = None
            while '{}:{}'.format(item, i+1) in self.items:
                # If new list is shorter, delete old additional items
                self.delete_items('{}:{}'.format(item, i+1))
                i += 1
            return
        else:
            # Scalar value
            df = self._df
            item_from_name = self.name_to_item(item)
            if item_from_name is not False:
                item = item_from_name
            # new = item not in list(self.items)
            if item not in list(self.items):
                # add item setting type columns appropriately
                self.add_item(item, value, _silent=_silent)
            elif value is not None:
                # set value given we already know type
                col = self.get_value_column(self._df, item)
                df.loc[item, col] = value
                df.loc[item, 'value'] = str(value)
                if _verbose:
                    logger.info('Existing item of {} set: "{}" = {}'.format(repr(self), col, value))
            for k, v in kwargs.items():
                # TODO: Deal with setting columns for list item
                if (k in self.columns) and (df.loc[item, k] == v):
                    # No change
                    continue
                elif (k in self.columns):
                    # Modify existing value
                    df.loc[item, k] = v
                    modified = True
                elif _create_columns:
                    # New column
                    df.loc[item, k] = v
                    logger.info('Added column {} to {}'.format(k, repr(self)))
                    modified = True
                    # TODO: Made sure state modified appropriately for save when columns changed
                else:
                    raise IndexError('{} is not a valid Settings column. Possible values: {}'.format(k, self.columns))
            cols = self.column_sets_names['type']
            df.loc[:, cols] = df.loc[:, cols].fillna(False)
            for com_setting in self._composite_settings:
                # Update composite settings objects to referencing these settings
                # TODO: Fix pandas warning: A value is trying to be set on a copy of a slice from a DataFrame
                com_setting._df.loc[item] = self._df.loc[item]
                com_setting._subsetting_mod_times[self] = t_now_str('natural')
                com_setting._hash_id = None
        if modified:
            if _save:
                self.save()
        else:
            # Don't register modification in state
            self.state.reverse(_ignore_state=True)

    @in_state('accessing')
    def __getitem__(self, item):
        # Check input is valid
        item, category = self.get_item_index(self, item)
        if category == 'list':
            # If item is a list, unpack it's values into a list
            return SettingList(self, item)
        if category == 'function':
            return self.get_func_name_args(item)

        # Isolate settings values
        df = self._df
        setting = df.loc[item]
        np_true = np.bool_(True)  # np bool array does not use python True/False instances
        # Return appropriate type of settings object
        if setting['float'] is np_true:
            out = SettingFloat(self, item)
        elif setting['int'] is np_true:
            out = SettingInt(self, item)
        elif setting['bool'] is np_true:
            out = SettingBool(self, item)
        else:
            out = SettingStr(self, item)
        return out

    def __contains__(self, item):
        # Includes list items
        h = re.compile(r'^{}:?\d*$'.format(item))
        for ind in self._df.index:
            m = h.match(ind)
            if m:
                return True
        return False

    @in_state('modifying', 'modified')
    def __setitem__(self, item, value):
        self(item, value)

    def __len__(self):
        if self._df is not None:
            return len(self._df)
        else:
            return 0

    def __iter__(self):
        for item in self._df.index:
            yield item

    @in_state('modifying', 'modified')
    def add_item(self, item, value, _silent=False):
        assert item not in list(self.items), 'Item {} already exists'.format(item)
        type_to_col = {'bool': 'value_num', 'int': 'value_num', 'float': 'value_num', 'str': 'value_str'}
        if isinstance(value, bool):
            dtype = 'bool'
        elif isinstance(value, numbers.Integral):
            dtype = 'int'
        elif isinstance(value, numbers.Real):
            dtype = 'float'
        elif isinstance(value, str):
            dtype = 'str'
        elif value is None:
            value = '*None*'
            dtype = 'str'
        else:
            raise TypeError('Setting {}={} of type {} is not supported.'.format(item, value, type(value)))
        df = self._df
        df.loc[item, [type_to_col[dtype], dtype]] = value, True
        other_types = list(set([k for k in type_to_col if k != dtype]))
        df.loc[item, other_types] = False
        df.loc[item, 'order'] = len(df) - 1
        df.loc[item, 'runtime'] = False  # default to False
        df.loc[item, 'setting'] = False  # default to False
        df.loc[item, 'value'] = str(value)
        df.loc[item, 'name'] = item.split(':')[0].replace('_', ' ')  # Default name to item key without underscores

        if not _silent:
            logger.info('Added new item to settings: {} = {}'.format(item, value))

    @in_state('modifying', 'modified')
    def add_items(self, items):
        for item, value in items.items():
            self.add_item(item, value)

    def set_column(self, col, value, items='all', regex=False):
        if regex:
            raise NotImplementedError
        elif items == 'all':
            items = self.items

        assert isinstance(value, self.column_types[col]), 'Trying to set {}={}, {}!={}'.format(
                                                col, value, self.column_types[col], type(value))
        kwargs = {col: value}
        for item in items:
            self(item, **kwargs)


    @in_state('modifying', 'modified')
    def rename_item(self, old_name, new_name):
        """Rename setting key"""
        assert old_name in self._df.index
        self._df = self._df.rename({old_name: new_name}, axis='index')

    def rename_items_with_pattern(self, pattern, replacement_string, force=False):
        """Replace all occurences of regex pattern in indices with 'replacement_string'"""
        p = re.compile(pattern)
        for item in self.items:
            if p.search(item):
                new_name = p.sub(replacement_string, item)
                if not force:
                    out = input('Rename item "{}" -> "{}"? (Y)/n: '.format(item, new_name))
                    if out.lower() not in ('', 'y'):
                        continue
                self._df = self._df.rename({item: new_name}, axis='index')
                # Default name to item key without underscores
                self._df.loc[new_name, 'name'] = new_name.split(':')[0].replace('_', ' ')

                self.state('modifying')
                logger.info('Renamed item {} -> {}'.format(item, new_name))
        if self.state == 'modifying':
            self.state('modified')

    @in_state('modifying', 'modified')
    def delete_items(self, items, _siltent=False):
        """Remove item(s) from settings"""
        items = make_iterable(items)
        for item in copy(items):
            if item not in self.items:
                # Check if item is a list item, in which case delete all parts
                i = 0
                while '{}:{}'.format(item, i) in self.items:
                    items.append('{}:{}'.format(item, i))
                    i += 1
                if i == 0:
                    raise ValueError('Item "{}" not in {}'.format(item, repr(self)))
                items.pop(items.index(item))
        if len(items) > 0:
            self._df = self._df.drop(items)
            if not _siltent:
                logger.info('Deleted items {} from settings: {}'.format(items, repr(self)))

    @in_state('modifying', 'modified')
    def clear(self):
        self.delete_items(self.items)
        return self

    @in_state('modifying', 'modified')
    def append_item(self, name, values={'value': []}, categories=[], create_cols=True):
        """Add item with an already existing name to settings.

        The item key will have a number appended to the end in format <name>::2 etc"""
        # TODO: implement settings append
        raise NotImplementedError

    @in_state('loading', 'loaded')
    def load(self):
        assert self.file_exists
        # TODO: Reimplement reading meta data...
        def read_settings_file(fn_path):
            # with Dataset(fn_path) as root:
            #     # self.__dict__.update(netcdf_to_dict(root, 'meta'))  # redundant info as stored in logfile
            #     self._column_sets_names = netcdf_to_dict(root, 'column_sets_names')
            self._df = xr.open_dataset(fn_path, group='df').to_dataframe()
            # self._column_sets_names = xr.open_dataset(fn_path, group='column_sets_names').to_dataframe()

        # Make multiple attempts reading file - sometimes inexplicably fails on first attempt
        n_attempts = 5
        error_message = 'Failed to read settings file after {{n_attempts}} attempts: {fn}'.format(fn=self.fn_path)
        attempt_n_times(read_settings_file, args=[self.fn_path], n_attempts=n_attempts, exceptions=(OSError,),
                        sleep_interval=3, error_message=error_message)

        self.log_file.loaded(self.name)
        self.check_consistency()
 
    def from_config(self, fn):
        assert os.path.isfile(os.path.expanduser(fn)), 'Config file does not exist: {}'.format(fn)
        config = configparser.ConfigParser()
        config.read(fn)
        modified = False
        for section, item in config.items():
            if len(item) == 0:
                continue
            for key, value in item.items():
                name = '{}::{}'.format(section, key)
                # If no change required, skip
                if name in self and self[name] == value:
                    continue
                self[name] = value
                self.state('modifying')
                modified = True
        if modified:
            self.state('modified')
        pass
    
    @in_state('saving', 'saved')
    def save(self, state_transition=None, update_template=True):
        self.check_consistency()
        if not self.file_exists:
            self.create_file()
            return
        # Don't resave if already saved
        if self.state.previous_state == 'saved' or self.state.previous_state == 'modifying':
            self.state.reverse()
            return
        gc.collect()
        meta = pd.Series({'t_created': self.t_created, 't_modified': self.t_modified, 't_accessed': self.t_accessed})
        meta = meta.to_xarray()
        ds = self._df.to_xarray()
        # col_sets = pd.Series(self.column_sets_names).to_xarray()
        n_attempts = 3
        attempt_n_times(ds.to_netcdf, self.fn_path, {'group': 'df', 'mode': 'w'}, n_attempts=n_attempts,
                        exceptions=(IOError,), sleep_interval=2,
                        error_message='Call to {}.save() df failed after {{n_attempts}} attempts'.format(repr(self)))
        attempt_n_times(meta.to_netcdf, self.fn_path, {'group': 'meta', 'mode': 'a'}, n_attempts=n_attempts,
                        exceptions=(IOError,), sleep_interval=2,
                        error_message='Call to {}.save() meta failed after {{n_attempts}} attempts'.format(repr(self)))
        attempt_n_times(dict_to_netcdf, (self.fn_path, 'column_sets_names', self.column_sets_names),
                        n_attempts=n_attempts, exceptions=(IOError,), sleep_interval=2,
                        error_message='Call to {}.save() col_groups failed after {{n_attempts}} attempts'.format(repr(self)))
        logger.debug('Saved settings: {}'.format(self.__repr__()))
        # for attempt in range(n_attempts):
        #     try:
        #         #TODO: delete/overwrite previous values? Use with?
        #         delete_file(self.fn_path, ignore_exceptions=())
        #         with Dataset(self.fn_path, "w", format="NETCDF4") as root:
        #             dict_to_netcdf(root, 'meta', meta)
        #             dict_to_netcdf(root, 'column_sets_names', self.column_sets_names)
        #         self._df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')
        #
        #     except PermissionError as e:
        #         if attempt == n_attempts-1:
        #             logger.exception('Unable to write to settings log file "{app}({name})": {path}, {e}'.format(
        #                 app=self.application, name=self.name, path=self.fn_path, e=e))
        #             raise e
        #     except RuntimeError as e:
        #         if attempt == n_attempts-1:
        #             logger.exception('Failed to update Settings File for application "{app}({name})": {path}, {e}'.format(
        #                 app=self.application, name=self.name, path=self.fn_path, e=e))
        #             raise e
        #     except OSError as e:
        #         if attempt == n_attempts - 1:
        #             logger.exception('Failed to update Settings File for application "{app}({name})": {path}, {e}'.format(
        #                 app=self.application, name=self.name, path=self.fn_path, e=e))
        #             raise e
        #     except Exception as e:
        #         logger.exception('Unanticipated error updating SettingsFile "{app}({name})": {path}, {e}'.format(
        #             app=self.application, name=self.name, path=self.fn_path, e=e))
        #         raise e
        #     else:
        #         logger.info('Updated/saved SettingsFile values for application "{app}({name})": {path}'.format(
        #                 app=self.application, name=self.name, path=self.fn_path))
        #         self.log_file.updated(self.name)
        #         break
        #     time.sleep(3)

    def delete_file(self, force=False):
        """Delete settings file for this settings instance"""
        if not force:
            answer = ask_input_yes_no('Are you sure you want to delete settings file "{}"')
            if not answer:
                return
        delete_file(self.fn_path)

    def refresh_file_from_memory(self):
        """Delete existing settings file and save new file with values in memory

        Useful for fixing corrupted netcdf files"""
        self.delete_file(force=True)
        self.save()

    def backup(self):
        """Backup current settings to backup folder"""
        # TODO: Clear recent backups
        # self.trim_backups()
        if self.state == 'modified':
            self.save()
        backup_path = os.path.join(self.path, 'backups')
        mkdir(backup_path)
        # TODO: check new backup isn't identical to previous backup
        backup_fn = '{fn}-{time}.nc'.format(fn=str(Path(self.fn).stem), time=t_now_str())
        fn_path = os.path.join(backup_path, backup_fn)
        shutil.copyfile(self.fn_path, fn_path)
        logger.info('Created backup of SettingsLogFile for "{app}, {name}": {path}'.format(
                app=self.application, name=self.name, path=fn_path))

    def trim_backups(self):
        """Reduce number of backups to save space"""
        raise NotImplementedError

    def restore_backup(self, recent=0):
        raise NotImplementedError

    def create_file(self):
        if self.file_exists:
            logger.debug('Settings file already exists: {}'.format(self.fn_path))
            return
        # meta = {key: self.__dict__[key] for key in ['_t_created', '_t_modified', '_t_accessed']}
        self.log_file.created(self.name)
        meta = {'t_created': self.t_created, 't_modified': self.t_modified, 't_accessed': self.t_accessed}
        if not os.path.isdir(self.path):
            mkdir(self.path, depth=1)
        try:
            with Dataset(self.fn_path, "w", format="NETCDF4") as root:
                dict_to_netcdf(root, 'meta', meta)
                dict_to_netcdf(root, 'column_sets_names', self.column_sets_names)
                pass
            self._df.to_xarray().to_netcdf(self.fn_path, mode='a', group='df')

        except PermissionError as e:
            logger.exception('Unable to write to file')
        except:
            if self.file_exists:
                self.delete_file(force=True)
            logger.exception('Failed to create Settings File for application "{app}": {path}'.format(
                app=self.application, path=self.fn_path))
        else:
            logger.info('Created SettingsFile for application "{app}", name {name}: {path}'.format(
                        app=self.application, name=self.name, path=self.fn_path))
        # TODO: Create settings_log file if new application
        self.state('saved')

    def delete_file(self, force=False):
        """Delete the configurations file for the current application"""
        if not force:
            print('Are you sure you want to delete the settings file:\n{}'.format(self.fn_path))
            out = input('(Y)/n')
            if not (out.lower() in ('y', '')):
                return
        os.remove(self.fn_path)
        logger.warning('Deleted settings file: {}'.format(self.fn_path))

    @classmethod
    def get_copy(cls, application=None, original=None, new_name=None, force_overwrite=False):
        """Get new settings object based on an existing 'original' template"""
        old_settings = cls.get(application, original)
        assert len(old_settings) > 0, 'Attempting to copy empty settings object'
        assert new_name is not None, 'Name required for new settings set'
        new_settings = old_settings.copy(new_name, force_overwrite=force_overwrite)
        del old_settings
        return new_settings

    def copy(self, new_name, force_overwrite=False):
        """Copy internal values to new settings set name"""
        if new_name in self.log_file.names and (not force_overwrite):
            out = input('New name "{}" already exists. Overwrite it (Y)/n? '.format(new_name))
            if out.lower() not in ('y', ''):
                return
        new_settings = Settings.get(self.application, new_name)
        new_settings.state('modifying')
        new_settings._df = copy(self._df.loc[:, :])
        new_settings._column_sets_names = copy(self.column_sets_names)
        new_settings.state('modified')
        new_settings.save()
        return new_settings

    @classmethod
    def is_item_in(cls, item, application, name):
        """Return True if setting 'item' is also present in the settings (application, name).

        If list of names supplied, return list of bools"""
        names = make_iterable(name)
        out = []
        for name in names:
            s = Settings.get(application=application, name=name)
            if item in s.items:
                out.append(True)
            else:
                out.append(False)
        if len(out) == 1:
            out = out[0]
        return out

    def is_item_equal_in(self, item, application, name):
        """Return (True, info) if setting item is equal in settings (appliction, name), otherwise (False, info).

        info is tuple of (descriptive_string, difference_data)
        """
        if item not in self:
            raise ValueError(item)
        names = make_iterable(name)
        out = []
        for name in names:
            s = Settings.get(application=application, name=name)
            if not Settings.is_item_in(item, application, name):
                # Item missing from other settings
                out.append((False, ('missing other', 'missing from other ({}, {})'.format(application, name))))
                continue
            elif (not Settings.is_item_in(item, self.application, self.name)):
                # Item missing from self
                out.append((False, ('missing self', 'missing from self ({}, {})'.format(self.application, self.name))))
                continue

            if self[item] != s[item]:
                # Values different
                out.append((False, ('not equal', (self[item].value, s[item].value))))
            else:
                col_diffs = np.array([self._df.loc[item, col] == s._df.loc[item, col] or
                                     (safe_isnan(self._df.loc[item, col]) and safe_isnan(s._df.loc[item, col]))
                                     for col in self._df.columns if col in s._df.columns])
                if np.all(col_diffs):
                    out.append((True, ('identical', )))
                else:
                    col_diff_values = (self._df.loc[item, ~col_diffs], s._df.loc[item, ~col_diffs])
                    out.append((True, ('columns differ', col_diff_values)))
        if len(out) == 1:
            out = out[0]
        return out

    def compare_to(self, application, name):
        """Compare settings to settings (appliction, name). Return Bool and DataFrame of differences."""
        s = Settings.get(application, name)
        all_items = set(self.items).union(s.items)
        cols = ['({app},{name})'.format(app=ss.application, name=ss.name) for ss in (self, s)]
        df_out = pd.DataFrame(columns=['equal']+cols+['columns'])
        for item in all_items:
            equal, info = self.is_item_equal_in(item, application, name)
            df_out.loc[item, 'equal'] = equal
            df_out.loc[item, cols[0]] = self[item].value
            df_out.loc[item, cols[1]] = s[item].value
            df_out.loc[item, 'columns'] = info[0]
        all_equal = np.all(df_out['equal'])
        return all_equal, df_out

    def compare_log_times(self, application, name, time_type='modified'):
        """Check if log time is more recent than other"""
        s = Settings.get(application, name)
        df_self = self.log_file._df.loc[self.name]
        df_other = s.log_file._df.loc[name]
        assert time_type in df_self.index
        t_self = str2datetime(df_self[time_type], self.t_formats['natural'])
        t_other = str2datetime(df_other[time_type], s.t_formats['natural'])
        if t_self > t_other:
            out = 'self is newer'
        elif t_self < t_other:
            out = 'self is older'
        else:
            out = 'same age'
        return out

    def copy_item_from(self, item, application, name, update_value=True, update_columns=True, preserve_order=True):
        s = Settings.get(application, name)
        assert item in s.items
        if item not in self.items:
            # TODO: handle list items...
            self._df.loc[item, :] = s._df.loc[item, :]
        else:
            if update_value:
                raise NotImplementedError

        if preserve_order:
            self.reorder_item(item, s[item]['order'])

    def update_from_other(self, application, name, only_if_newer=True, add_missing_items=True,
                          update_values=False, update_columns=False, save=True):
        """Update settings in self according to values in (application, name)"""
        if (not add_missing_items) and (not update_values):
            return
        # Get settings object to update from
        s = Settings.get(application, name)
        modified = False
        if only_if_newer:
            # Return if comparisson settings are older
            dt = self.compare_log_times(application, name)
            if dt != 'self is older':
                logger.debug('Not updating {} with {} as {}'.format(self, s, dt))
                return
        value_cols = [v[0] for v in self.default_column_sets['values']+self.default_column_sets['type']]
        for item in s.items:
            if add_missing_items and (item not in self.items):
                # Get column values for item (excluding value columns)
                kws = {col: s._df.loc[item, col] for col in s.columns if col not in value_cols}
                # Add missing items to self
                self(item, s[item].value, **kws)
                # Make sure the new item is inserted in the same position in self as it was in s
                self.reorder_item(item, s[item]['order'], save=False)
            else:
                if update_values:
                    # Update value of existing setting
                    if self[item] != s[item]:
                        # Only update if different
                        # This should handle list items correctly
                        self[item] = s[item].value
                        modified = True
                if update_columns:
                    # TODO: make sure column heading match
                    # Copy column values from s
                    non_value_cols = [c for c in self._df.columns if c not in value_cols]
                    self._df.loc[item, non_value_cols] = s._df.loc[item, non_value_cols]
                    modified = True
        if modified:
            self.state = 'modified'
        if save and modified:
            self.save()
        logger.info('Updated {} according to {}'.format(self, s))

    def new_time(self, format='natural'):
        # TODO: update with config file
        # Find times of exisiting settings for application
        format_str = self.t_formats[format]
        if self.file_exists:
            while t_now_str(format=format_str) in self.log_file:
                time.sleep(1.0)
        return t_now_str(format=format_str)

    def set(self, item, value, ignore=[None]):
        """Set existing setting provided it is not an ignore value"""
        if (item not in self.items) and (Settings.list_item_indices(self, item) is False):
            raise ValueError('Item "{}" is not in {}'.format(item, repr(self)))
        if value in ignore:
            return
        # No change, so do nothing
        if item in self and self[item] == value:
            return
        self[item] = value

    def check_consistency(self):
        """Checks on consistency of dataframe"""
        # TODO: Check format of columns
        # If columns in the class column_sets are missing from the dataframe, add them
        if not is_subset(self.ordered_column_names, self.columns):
            self._add_missing_columns()
        if True:
            self._reset_column_types()
        self._check_values()
        # Make sure each item only has one type
        unique_type = self.view(cols='type').astype(int).sum(axis=1) == 1
        if not all(unique_type):
            raise ValueError('Inconsistent types:\n{}'.format(self.view(cols='type').loc[~unique_type]))

    def _set_template(self, create_if_missing=True):
        """Load template settings for this application"""
        if self.name == 'template':
            self._template = self
        else:
            self._template = Settings.get(application=self.application, name='template')
            if len(self._template) > 0:
                if 'template' not in self.log_file.names:
                    # If settings file was not created locally it may not be present in the logfile
                    self._template.save(ignore_state=True)
            else:
                existing_settings = self.log_file.names
                if create_if_missing and (len(existing_settings) > 0) and ('template' not in existing_settings):
                    # If settings have been created for this application, but no 'template' settings have been created
                    # create template settings out of the most recently modified settings for this application
                    most_recent = self.log_file.times['date_time_mod'].argmax()
                    self._template = Settings.get_copy(self.application, most_recent, 'template')
                    logger.info('Created "template" settings for "{}" from "{}"'.format(self.application, most_recent))
                    answer = ask_input_yes_no('Save new template file')  # tmp
                    if not answer:
                        return
                    self._template.save()
        if len(self._template) == 0:
            self._template = None
        return self._template

    @in_state('modifying', 'modified')
    def _add_missing_columns(self):
        """Add empty collumns where missing from class defined column sets"""
        null_types = {int: 0, str: '', float: 0.0, bool: False}
        if is_subset(self.columns, self.ordered_column_names):
            for tup in self.default_column_sets.values():
                # TODO: Use DataFrame.assign to remove loop
                for col, type in tup:
                    if col not in self.columns:
                        self._df.loc[:, col] = null_types[type]
                        logger.warning('Added missing column "{}" to {}'.format(col, repr(self)))

    def _reset_column_types(self):
        """Set datatypes of dataframe columns"""
        # TODO: remove nans, etc
        type_dict = [dict(v) for k, v in self.default_column_sets.items()]
        type_dict = {k: v for d in type_dict for k, v in d.items()}
        try:
            self._df.loc[:, :] = self._df.astype(type_dict)
        except ValueError as e:
            # Values cannot be converted to new type safely, so need to reset their values
            init_values = {str: '', bool: False, int: 0, float: 0.0}
            for col, type in type_dict.items():
                try:
                    self._df.loc[:, col] = self._df[col].astype(type)
                except:
                    logger.warning('Column {} has wrong type ({}). Resetting all values to {}.'.format(
                            col, self._df[col].dtype, init_values[type]))
                    self._df.loc[:, col] = init_values[type]
        pass
    
    def _check_values(self):
        """Check values conform to standards"""
        # Make sure value column is not empty. 'value_str' col will be nan if not string type
        mask = (self._df['value'] == '') & (self._df['value_str'] != '')
        for item in self._df.loc[mask, 'value'].index.values:
            col = self.get_value_column(self._df, item)
            value = self._df.loc[item, col]
            self(item, value=value)
            logger.debug('Filled missing "value" entry for {}={}'.format(item, value))

    # def add_column(self, value):
    #     if value in self.columns:
    #         return
    #     self.df[value] = np.nan  # self.df.index
    #
    # def add_columns(self, values):
    #     for value in values:
    #         self.add_column(value)

    @classmethod
    def get_logfile(cls, application):
        """Get saved settings names for given application

        :param application: application settings are for
        :return:
        """
        log = SettingsLogFile.get(application)
        if log.file_exists:
            return log
        else:
            logger.warning('No saved settings for {}'.format(application))
            return None
    
    @classmethod
    def existing_applications(cls):
        """Get list of all applications with saved settings"""
        return SettingsLogFile.get_applications()

    @classmethod
    def existing_settings(cls, application):
        """Get list of all saved settings names for given application"""
        fn_pattern = 'settings-{app}-(.*).nc'.format(app=application)
        path = os.path.join(settings_dir, 'values', application)
        if os.path.isdir(path):
            files = filter_files_in_dir(path, fn_pattern, group_keys=['name'])
            names = list(files.keys())
        else:
            names = []
        return names
        # return SettingsLogFile(application).names

    @staticmethod
    def is_list_item(settings, item):
        r = re.compile(r'^{}:\d'.format(item))
        newlist = list(filter(r.match, settings.items))
        if len(newlist) > 0:
            return True
        else:
            return False

    @staticmethod
    def list_item_values(settings, item):
        """Return df items that are part of the item list. Return False if not a list."""
        r = re.compile(r'^{}:\d'.format(item))
        list_items = list(filter(r.match, settings.items))
        if (len(list_items) == 1) and (settings[list_items[0]].value == '[<empty>]'):
            return []
        elif len(list_items) > 0:
            return [settings[i].value for i in list_items]
        else:
            return False

    @staticmethod
    def list_item_indices(settings, item):
        """Return df items that are part of the item list. Return False if not a list."""
        r = re.compile(r'^{}:\d'.format(item))
        list_items = list(filter(r.match, settings.items))
        if len(list_items) > 0:
            return sorted(list_items)
        else:
            return False

    def item_is_func_arg(self, item):
        raise NotImplementedError

    def set_func_args(self, func, func_name=None, kwargs=None):
        raise NotImplementedError

    def get_func_name_args(self, func_name):
        """Return df items that are arguments for function with func_name. Return False if not a list."""
        mask = self._df.loc[:, 'function'] == func_name
        items = self._df.loc[mask, :].index.values
        out = {key: self(key) for key in items}
        return out

    def get_func_args(self, funcs, func_names=None, whitelist=(), blacklist=()):
        """Get arguments for function from settings object
        :param: funcs - function instances or strings describing the function name"""
        funcs = to_list(funcs)
        if func_names is not None:
            func_names = make_iterable(func_names)
        else:
            func_names = [None]*len(funcs)
        args, kws = [], {}
        for func, func_name in zip(funcs, func_names):
            if func_name is not None:
                # Get name of function/method/class
                func_name = func.__name__
                if func_name == '__init__':
                    # If func is an __init__ method, get the name of the corresponding class
                    func_name = get_methods_class(func).__name__
            sig = inspect.signature(func)
            for i, kw in enumerate(sig.parameters.values()):
                name = kw.name
                if name in blacklist:
                    continue
                if (name in self) or (name in whitelist):# and (self._df.loc[name, 'function'] == func_name):
                    kws[name] = self[name].value
                # if setting in kwargs:
                #     kws[name] = kwargs[name]
        return kws

    @classmethod
    def get_value_column(cls, df, item):
        """Return name of column item value is in"""
        if any(df.loc[item, ['bool', 'int', 'float']]):
            col = 'value_num'
        elif df.loc[item, 'str']:
            col = 'value_str'
        else:
            raise ValueError('Setting {} has no type!'.format(item))
        return col

    def reorder_item(self, item, new_order_index, save=True):
        """Reorder item in settings, given new order index value for column 'order'"""
        df = self._df
        # Set the new order value for this item
        self(item, order=new_order_index)
        # Increase the index of each item with a higher index than the new index by 1
        higher_indices = df.index[(df['order'] >= new_order_index).values & (df.index != item)]
        for index in higher_indices:
            df.loc[index, 'order'] = df.loc[index, 'order'] + 1
        # Make sure the dataframe is sorted by the order column
        df = df.sort_values('order', axis='index')
        # Remove any gaps in the oder values
        df['order'] = np.arange(len(df))
        self._df = df
        if save:
            self.state('modified')
            self.save()

    def update_from_dataframe(self, df):
        for item, values in df.iterrows():
            if 'value' in values:
                value = values['value']
            else:
                col = self.get_value_column(df, item)
                value = values[col]

            kwargs = {}
            for col, col_value in values.items():
                if (col in self.columns) and (col not in ('value_num', 'value_str')):
                    kwargs[col] = col_value
            kwargs.pop('value')
            self(item, value, **kwargs)

    def update_from_dict(self, dictionary, _silent=False, **kwargs):
        for item, value in dictionary.items():
            if isinstance(value, Setting):
                value = value.value
            self(item, value, _silent=_silent, **kwargs)
        return self
        # logger.debug('Updated {} with values: {}, cols: {}'.format(self, dictionary, kwargs))

    @staticmethod
    def get_item_index(settings, item, _raise=True):
        """Lookup item key given name"""
        df = settings._df
        if item in settings.items:
            # item is already an item key
            category = 'index'
        elif Settings.is_list_item(settings, item):
            # Item is a list item name with item format <name>:0, <name>:1, <name>:2
            category = 'list'
        elif item in df['name'].values:
            # Item is name of item
            mask = df['name'] == item  # boolian mask where name == 'item'
            n_match = np.sum(mask)
            if n_match == 1:
                item = df.index[mask].values[0]
                category = 'name'
            elif n_match > 1:
                raise ValueError('Setting item name {} is not unique so cannot be used to index {}'.format(
                        item, repr(settings)))
        elif item in df['function'].values:
            # TODO: look is split list of functions
            # item is the name of function to return args for
            category = 'function'
        else:
            if _raise:
                raise ValueError('Cannot locate item/name "{}" in {}'.format(item, repr(settings)))
            else:
                return item, None
        return item, category

    def name_to_item(self, name):
        """Lookup item key given name"""
        df = self._df
        if (name in list(self.items)):
            return name
        if Settings.is_list_item(self, name):
            return name
        elif name in df['name'].values:
            mask = df['name'] == name  # boolian mask where name == 'item'
            if np.sum(mask) == 1:
                item = df.index[mask].values[0]
            else:
                raise ValueError('Setting item name {} is not unique so cannot be used to index {}'.format(
                        name, repr(self)))
            return item
        else:
            # Name not already in df so return as is to create new item
            return name
    
    def to_dict(self):
        out = OrderedDict()
        for item in self.items:
            if re.match(r'^.*:\d+', item):
                key = item.split(':')[-1]
            else:
                key = item
            value = self[item]
            out[key] = value
        return out

    def add_instance(self):
        if self.application not in self.instances:
            self.instances[self.application] = {}
        self.instances[self.application][self.name] = self

    @classmethod
    def get_instance(self, application, name):
        if application in self.instances:
            if name in self.instances[application]:
                return self.instances[application][name]
        return None

    def compare_settings(self, other_settings, include_missing=True, raise_on_difference=True, log_difference=True):
        """Compare self to other settings object or dataframe"""
        if isinstance(other_settings, Settings):
            df = other_settings._df
        elif isinstance(other_settings, pd.DataFrame):
            df = other_settings
        else:
            raise ValueError('Input format not recognised: {}'.format(type(other_settings)))
        summary = {'same': [], 'different': [], 'missing': [], 'added': [], 'identical': False}
        for item in df.index:
            if item not in self:
                summary['added'].append(item)
            else:
                if df.loc[item, 'value'] == self._df.loc[item, 'value']:
                    summary['same'].append(item)
                else:
                    summary['different'].append(item)
        # Get items in self not present in comparison settings
        for item in self._df.index:
            if item not in df.index:
                summary['missing'].append(item)
        if len(summary['same']) != len(df):
            different_items = summary['different']+summary['added']
            if include_missing:
                different_items += summary['missing']
            df_diffs = copy(df.loc[different_items, ['value']]).rename(columns={'value': 'comparison'})
            df_diffs['self'] = self._df.loc[different_items, 'value']
            message = 'Settings comparison; Same: {}, Different: {}, Missing: {}\n{}'.format(
                    len(summary['same']), summary['different'], summary['missing'], df_diffs)
            # print(df_diffs)
            if raise_on_difference:
                raise ValueError(message)
            if log_difference:
                logger.warning(message)
        else:
            df_diffs = None
            summary['identical'] = True

        return summary, df_diffs
    
    def _block_protected(self):
        """Block modificaton of a protected file"""
        if (self.name in self.log_file) and (self.log_file[self.name]['protected']):
            raise RuntimeError('Cannot modify protected settings file {}!'.format(repr(self)))
    
    def __str__(self):
        # TODO: set ordering
        df = self.view() if self._df is not None else None
        return '{}:\n{}'.format(repr(self)[1:-1], str(df))

    def __repr__(self):
        return '<Settings: {app};{name}[{len}], {state}>'.format(app=self._application, name=self.name, 
                                                                 len=len(self.items), state=self.state)

    def __del__(self):
        try:
            self.instances[self.application].pop(self.name)
        except KeyError as e:
            logger.error('{}: {}'.format(self, e))
        # super().__del__()

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        assert isinstance(value, str)
        self._application = value
        self.log_file = SettingsLogFile.get(self.application)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        elif self.state == 'init':
            return None
        else:
            return self.t_modified

    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        self._name = value

    @property
    def t_created(self):
        return self.log_file._df.loc[self.name, 'created']

    @property
    def t_modified(self):
        if self.log_file is not None:
            return self.log_file._df.loc[self.name, 'modified']
        else:
            return None

    @property
    def t_accessed(self):
        return self.log_file._df.loc[self.name, 'loaded']

    @property
    def items(self):
        # TODO: Inlclude base list indices in returned values
        if self._df is not None:
            return self._df.index.values
        else:
            return []

    @property
    def columns(self):
        return list(self._df.columns.values)

    @property
    def column_sets_names(self):
        names = {key: [v[0] for v in value] for key, value in self._column_sets.items()}
        return names

    @property
    def ordered_column_names(self):
        names = [value for value in self.column_sets_names.values()]
        names = list((itertools.chain.from_iterable(names)))
        names = [name for name in remove_duplicates_from_list(names)]
        return names

        # columns = list(itertools.chain.from_iterable(self._column_sets.values()))
        # types = [value[1] for value in columns]
        # columns = [value[0] for value in columns]

    @property
    def column_sets_types(self):
        types = {key: [v[1] for v in value] for key, value in self._column_sets.items()}
        return types

    @property
    def column_types(self):
        types = {k: v for key, value in self._column_sets.items() for k, v in value}
        return types

    def hash_id(self):
        # TODO: Move method here = swap location
        from ccfepyutils.classes.composite_settings import CompositeSettings
        return CompositeSettings.hash_id(self)

    @property
    def path(self):
        """Path to settings files"""
        ## TODO: Load from config file
        return os.path.join(settings_dir, 'values', self.application)


    @property
    def fn(self):
        """Filename of current settings file"""
        assert self.name is not None
        return 'settings-{app}-{name}.nc'.format(app=self.application, name=self.name)

    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)


    @property
    def file_exists(self):
        """Return True if a settings file with the current application and name already exists else False"""
        return os.path.isfile(self.fn_path)

    # @property
    # def t_modified(self):
    #     return convert_str_datetime_format(self._t_modified, format1=self.t_compressed_format)


class SettingsLogFile(object):
    t_formats = {'compressed': "%y{dl}%m{dl}%d{dl}%H{dl}%M{dl}%S".format(dl=''),
                 'natural': "%H:%M:%S %d/%m/%y".format(dl='')}
    instances = {}
    def __init__(self, application):
        """ """
        assert isinstance(application, str)
        self.application = application
        self.instances[application] = self

    def __getitem__(self, name):
        """Get log information for setting set"""
        assert name in self.names
        return self._df.loc[name, :]

    def __call__(self, name, cols=None):
        """Get log information for setting set, specifying columns"""
        if name not in self._df.index:
            raise ValueError('SettingsLogFile: No "{}" settings for application "{}". Options: {}'.format(
                name, self.application, self._df.index))
        if cols is None:
            return self[name]
        else:
            cols = make_iterable(cols)
            return self._df.loc[name, cols]

    @classmethod
    def get(cls, application):
        """Treat SettingsLogFile as singleton to signle instance is always in line with netcdf file"""
        if application in cls.instances:
            # Return existing instance
            logfile = cls.instances[application]
        else:
            # Create new instance
            logfile = SettingsLogFile(application)
        return logfile

    def __del__(self):
        # Remove instance from instances when deleted
        try:
            self.instances.pop(self.application)
        except KeyError as e:
            logger.error('{}: {}'.format(self, e))

    def init(self):
        # TODO: add safety checks
        self._df = pd.DataFrame({'modified': [], 'loaded': [], 'created': [],
                                'mod_count': [], 'load_count': [], 'load_count_total': [],
                                'protected': []})  # Initialise empty dataframe
        self._df.index.name = 'name'
        self.save()

    def save(self):
        gc.collect()  # Fix problems with netcdf file handles persisting
        exists = self.file_exists
        # Don't save the log file if it is empty
        if not exists and len(self.names) == 0:
            return
        mkdir(self.path, depth=1)
        try:
            self._df.to_xarray().to_netcdf(self.fn_path, mode='w')
            if not exists:
                logger.info('Created SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
            else:
                logger.debug('Updated SettingsLogFile for application "{app}": {path}'.format(
                        app=self.application, path=self.fn_path))
        except PermissionError as e:
            logger.warning('Trouble updating {}. Deleting and refreshing file'.format(repr(self)))
            try:
                self.delete_file(True)
                self._df.to_xarray().to_netcdf(self.fn_path, mode='w')
                logger.warning('Successfully updated {} after refresh'.format(repr(self)))
            except PermissionError as e:
                logger.exception('Unable to write to file!')

    def load(self):
        """Load settings logfile for current application"""
        if not self.file_exists:
            self.init()
            return False
        n_attempts = 4
        for attempt in range(n_attempts):
            try:
                self._df = xr.open_dataset(self.fn_path).to_dataframe()
            except Exception as e:
                if attempt == (n_attempts-1):
                    logger.error('Failed to open settingslogfile: {}'.format(self.fn_path))
                    # TODO: copy backedup file?
                    raise e
                time.sleep(2)
        return True
    
    def created(self, name, time=None, overwrite=True):
        """Update time settings file for settings name was created"""
        import getpass

        df = self._df
        if name in self.names and (not overwrite or df.loc[name, 'protected']):
            raise RuntimeError('Cannot overwrite values for setting {}'.format(name))
        if time is None:
            time = t_now_str('natural')
        df.loc[name, ['created', 'modified', 'loaded']] = time
        df.loc[name, ['mod_count', 'load_count', 'load_count_total']] = 0
        df.loc[name, 'protected'] = False
        df.loc[name, 'creator'] = getpass.getuser()  # User creating file
        pass
    
    def loaded(self, name, time=None):
        """Update time settings was last accessed"""
        if name not in self._df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self._df.loc[name, 'loaded'] = time
        self._df.loc[name, 'load_count'] += 1  # since modification
        self._df.loc[name, 'load_count_total'] += 1  # since creation

    def updated(self, name, time=None):
        """Update the time string of a named settings configuration"""
        if name not in self._df.index:
            self.created(name)
        if time is None:
            time = t_now_str('natural')
        self._df.loc[name, 'modified'] = time
        self._df.loc[name, 'mod_count'] += 1
        self._df.loc[name, 'load_count'] = 0
        self.save()

    def rename(self, old, new):
        """Rename a settings configuration and update existing instances"""
        self._df.rename(index={old: new})
        self.save()

    def backup(self):
        """Backup current application settings configurations file to backup folder"""
        self.save()
        # TODO: Load path setttings from config file
        backup_path = os.path.join(settings_dir, 'backups')
        backup_fn = 'settings_backup-{app}-{time}.nc'.format(app=self.application, time=t_now_str())
        fn_path = os.path.join(backup_path, backup_fn)
        shutil.copyfile(self.fn_path, fn_path)
        logger.info('Created backup of SettingsLogFile for application "{app}": {path}'.format(
                app=self.application, path=fn_path))

    def delete_file(self, force=False):
        """Delete the configurations file for the current application"""
        if not force:
            print('Are you sure you want to delete the settings log file:\n{}'.format(self.fn_path))
            out = input('Y/n')
            if not (out.lower() in ('y', '')):
                return
        os.remove(self.fn_path)
        logger.warning('Deleted settings log file: {}'.format(self.fn_path))

    def refresh_logfile(self):
        """Ensure all settings files in the settings/appliction directory are represented in the log file"""
        # TODO: Implement
        raise NotImplementedError

    def print(self):
        print(self)

    @ classmethod
    def get_applications(cls):
        """Get list of all applications with saved settings"""
        applications = next(os.walk(os.path.join(settings_dir, 'values')))[1]
        return sorted(applications)

    def __repr__(self):
        return '<SettingsLogFile: {app}({l})>'.format(app=self.application, l=len(self._df))

    def toggle_protected(self, name):
        assert name in self.names
        self._df.loc[name, 'protected'] = ~self._df.loc[name, 'protected']
        logger.info('Set protected state of {} to {}'.format(name, self._df.loc[name, 'protected']))

    def __str__(self):
        # tmp = copy(self.df)
        # tmp.loc[:, 'modified'] = [convert_str_datetime_format(s, format2=self.t_display_format) for s in tmp['modified']]
        columns = ['modified', 'loaded']
        string = u'{}\n{}'.format(repr(self)[1:-1], str(self._df[columns]))
        return string

    def __contains__(self, item):
        if (item in self.times) or (item in self.names):
            return True
        else:
            return False

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value
        self.load()

    @property
    def path(self):
        """Path to settings log files"""
        out = Path(settings_dir) / 'log_files'
        return str(out)

    @property
    def fn(self):
        """Filename of current settings file"""
        return 'settings_log-{app}.nc'.format(app=self.application)

    @property
    def fn_path(self):
        return os.path.join(self.path, self.fn)

    @property
    def names(self):
        return list(self._df.index.values)

    @property
    def times(self):
        t_modified = self._df.loc[:, ['modified']]
        for item, t in t_modified.iterrows():
            t_modified.loc[item, 'date_time_mod'] = str2datetime(t['modified'], format=self.t_formats['natural'])
                # [str2datetime(t, format=self.t_formats['natural']) for t in t_modified['modified']]
        return t_modified

    @property
    def file_exists(self):
        """Return True if this log file already exists"""
        return os.path.isfile(self.fn_path)
    
    @property
    def dict(self):
        """Return dict of item: value pairs"""
        out = {}
        for item in self.items:
            out[item] = self[item]
        return out

def compare_settings_hash(application, name, settings_obj, n_output=1, skip_identical=False):
    """ Compare settings to saved settings hashes"""
    if isinstance(settings_obj, dict):
        settings_obj = Settings.from_dict(application, name, settings_obj, runtime=False)
    hash_path = os.path.expanduser(r'~/.ccfetools/settings/hash_records/{app}/{name}/'.format(app=application,
                                                                                              name=name))
    fn_pattern = r'settings_hash_record-(\w+).nc'
    files = filter_files_in_dir(hash_path, fn_pattern, ['hash_id'])
    differences = {}
    diff_table = pd.DataFrame(columns=['n_same', 'n_changes', 'n_missing', 'n_different'])
    for hash_id, fn in files.items():
        try:
            with xr.open_dataset(os.path.join(hash_path, fn), group='df') as ds:
                hash_settings = ds.to_dataframe()
        except Exception as e:
            raise e
        summary, df_diffs = settings_obj.compare_settings(hash_settings, raise_on_difference=False, log_difference=False)
        differences[hash_id] = df_diffs
        diff_table.loc[hash_id, ['n_same', 'n_changes', 'n_missing', 'n_different']] = (len(summary['same']),
                            len(settings_obj)-len(summary['same']), len(summary['missing']), len(summary['different']))
        diff_table.sort_values(['n_changes', 'n_missing'], ascending=True)
    for i in np.arange(np.min((n_output, len(diff_table)))):
        hash_id = diff_table.index[i]
        if skip_identical and (diff_table.loc[hash_id, 'n_changes'] == 0):
            continue
        logger.info('Rank {} closest match: {}\n{}'.format(i+1, hash_id, differences[hash_id]))
    return diff_table, differences

if __name__ == '__main__':
    s = Settings.get('Invertor')

    s = Settings.get('Invertor_resolution', '29840_eps_1')

    s = Settings.get('test_tmp', 'default')
    print(s)
    s.add_columns(['Description', 'I/O', 'Precedence', 'Representation'])
    print(s)
    s.add_item('path', '~')

    # print(s.columns)
    # print(s.items)
    print(s)
    pass