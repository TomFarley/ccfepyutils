import os
from collections import OrderedDict
from copy import copy, deepcopy
import logging
import inspect

import numpy as np
import pandas as pd
import xarray as xr
import re

from netCDF4 import Dataset

import ccfepyutils
from ccfepyutils.utils import make_iterable, t_now_str, get_methods_class
from ccfepyutils.io_tools import mkdir, attempt_n_times
from ccfepyutils.netcdf_tools import dict_to_netcdf
from ccfepyutils.classes.settings import Settings
from ccfepyutils.classes.state import State

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

settings_dir = ccfepyutils.settings_dir

class CompositeSettings(object):
    instances = {}
    def __init__(self, application, name, blacklist=[], whitelist=[], include_children=True, exclude_if_col_true=(),
                 update_values=None):
        """Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. my_code
        - The name is a label for a particular set of settings for the application e.g. my_default_settings
        """
        # TODO: implement state?
        self._reset_settings_attributes()
        # assert not (blacklist != [] and whilelist != []), 'Supply blacklist OR whitelist - NOT both'
        # self.state = State('init')
        self._application = application
        self._name = name
        self._blacklist = blacklist
        self._whitelist = whitelist
        self._exclude_if_col_true = exclude_if_col_true
        if (len(whitelist) > 0) and (application not in whitelist):
            # Make sure core settings are in whitelist else CompositeSettings will be empty
            self._whitelist = [application] + self._whitelist
        self.core = Settings.get(application, name)
        assert len(self.core._df) > 0, 'Empty core parent settings for {}, {}'.format(application, name)
        self.build_composite_df(include_children=include_children, exclude_if_col_true=exclude_if_col_true,
                                update_values=update_values)

    def _reset_settings_attributes(self):
        self.state = None
        self.core = None
        self._settings = None
        self._hash_id = None
        self._t_created = None
        self._t_modified = None
        self._subsetting_mod_times = {}

    def build_composite_df(self, include_children=True, exclude_if_col_true=(), update_values=None):
        """Expand settings items into full Settings objects"""
        self._settings = OrderedDict()
        self._items = OrderedDict()

        self.core._df.loc[:, 'parent'] = 'None'
        self._df = self.core._df[0:0]  # Get emtpy dataframe with same column structure
        self._df = self.append_settings_file(self._application, self._name, self._df, self._settings, self._items,
                                             include_children=include_children, exclude_if_col_true=exclude_if_col_true,
                                             update_values=update_values)
        self._hash_id = None
        pass

    def append_settings_file(self, application, name, df, settings, items,
                             add_to_whitelist=False, include_children=True, exclude_if_col_true=(), update_values=None):
        """ Add contents of settings file to CompositeSettings instance's dataframe
        :param: add_to_whitelist - treat application as if it in whitelist
        :param: include_children - include child settings even if they are not in the whitelist"""
        if (len(self._blacklist) > 0) and (application in self._blacklist):
            logger.debug('Skipping blacklist settings "{}:{}" from {}'.format(application, name, repr(self)))
            return df
        if (len(self._whitelist) > 0) and (application not in self._whitelist) and (not add_to_whitelist):
            logger.debug('Skipping non whitelist settings "{}:{}" from {}'.format(application, name, repr(self)))
            return df
        if update_values is None:
            update_values = {}
        exclude_if_col_true = make_iterable(exclude_if_col_true)
        logger.debug('Adding "{}:{}" settings to {}'.format(application, name, repr(self)))
        s = Settings.get(application, name)
        # Add parent column to note which Settings file each setting originates from
        if len(s) > 0:
            s._df.loc[:, 'parent'] = '{}:{}'.format(application, name)
        assert application not in settings.keys(), 'Application names must be unique: {}'.format(application)
        settings[application] = s
        if self not in s._composite_settings:
            s._composite_settings.append(self)
        # Record time settings file was last modified and update modification time for this composite settings
        t = t_now_str('natural')
        self._t_created = t
        self._t_modified = t
        self._subsetting_mod_times[application] = s.log_file(s.name, 'modified')

        list_items_updated = []
        for item in deepcopy(s.items):
            item_name_components = item.split(':')
            if item in update_values:
                s[item] = update_values.pop(item)
            elif (len(item_name_components) == 2):
                item_list_name = item_name_components[0]
                if (item_name_components[0] in update_values):
                    # NOTE: Under this implementation, assignement will be repeated oritginal length of list times
                    s[item_list_name] = update_values.pop(item_list_name)
                    list_items_updated.append(item_list_name)

                    # Bug fix for only subset of list being added to compostite settings
                    # Add all items for list in one go, then continue
                    for i, subitem_value in enumerate(s[item_list_name]):
                        subitem = '{}:{}'.format(item_list_name, i)
                        df = df.append(s._df.loc[subitem, :])
                    items[item] = [s]
                    continue
                elif item_list_name in list_items_updated:
                    # Item has just been set in previous loop iteration
                    continue
            elif (len(item_name_components) == 3) and (item_name_components[2] in update_values):
                # Function settings not implemented here
                pass
            df_nest = s._df
            exclude = False
            for excl_col in exclude_if_col_true:
                if df_nest.loc[item, excl_col] is np.True_:
                    exclude = True
            if not exclude:
                df = df.append(df_nest.loc[item, :])
                # Record which settings each item belongs to
                if item not in items.keys():
                    items[item] = [s]
                else:
                    items[item].append(s)
            if df_nest.loc[item, 'setting']:
                name = df_nest.loc[item, 'value']
                try:
                    # If include_children pass to subsequent calls, but set False for calls from core Settings set
                    df = self.append_settings_file(item, name, df, settings, items,
                                               add_to_whitelist=(include_children and (application != self._application)),
                                               include_children=include_children, exclude_if_col_true=exclude_if_col_true,
                                                   update_values=update_values)
                except Exception as e:
                    logger.warning('Failed to append settings file: {} {}'.format(item, name))
                    raise e
        return df

    def save(self, force=False):
        for settings in self._settings.values():
            if settings.state == 'modified' or force:
                settings.save()

    def backup(self):
        for settings in self._settings.values():
            settings.backup()

    def refresh(self):
        """Save any changes to file and refresh the internal df in case setting items have changed i.e. will load
        different settings files"""
        self.save()
        self.build_composite_df()
        logger.info('Rebuilt composite settings {}'.format(repr(self)))

    def view(self, items='all', cols=('comp', 'parent'), order=None, ascending=True, _expand_settings=True,
             _contains=True, _ignore_case=True):
        """Return dataframe containing a subst of columns, with items ordered as requried"""
        if cols == 'all':
            raise NotImplementedError
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
            # if cols == ['repr']:
            #     col_set.pop(col_set.index('name'))
            #     col_set += ['parent']
        if items == 'all':
            items = self.items
        elif isinstance(items, str):
            items = self.search_items(items, ignorecase=_ignore_case, contains=_contains)
        if order is None:
            df = self._df
        elif order == 'alphabetical':
            df = self._df.sort_index(ascending=ascending)
        elif order == 'custom':
            df = self._df.sort_values('order', ascending=ascending)
        df = df.loc[items]
        out = df
        if _expand_settings:
            # For items that correspond to a settings file, include the contents of that settings file in output
            #TODO: Integrate with ordering
            df2 = pd.DataFrame(columns=df.columns)
            for item in df.index:
                df2.loc[item] = df.loc[item]
                if df.loc[item, 'setting']:
                    df2 = df2.append(self._df.loc[self._df['parent'] == '{}:{}'.format(item, self[item].value), col_set])
            out = df2
        out = out.loc[:, col_set]
        return out

    def search_items(self, pattern, contains=True, ignorecase=True):
        args = [re.IGNORECASE] if ignorecase else []
        r = re.compile(pattern, *args)
        if contains:
            newlist = list(filter(r.search, self.items))
        else:
            newlist = list(filter(r.match, self.items))
        return newlist

    def add_item(self, application, item, value, **kwargs):
        """Add item to settings file contained within CompositeSettings instance"""
        assert application in self._settings, 'Application {} not recognised. Posibilities: {}'.format(
                application, self._settings.keys())
        settings = self._settings[application]
        settings(item, value, **kwargs)

    def __str__(self):
        # TODO: set ordering
        df = self.view()
        return '{}:\n{}'.format(repr(self)[1:-1], str(df))

    def __repr__(self):
        return '<CompositeSettings: {app};{name}[{len}], {state}>'.format(app=self._application, name=self._name,
                                                                 len=len(self.items), state=None)

    def __getitem__(self, item):
        """Call getitem from appropriate settings object"""
        settings = self.get_settings_for_item(item)
        return settings[item]

    def __setitem__(self, item, value):
        self(item, value)
    
    def __call__(self, item, value=None, _create_columns=False, _save=False, **kwargs):
        """Call __call__ from appropriate settings object"""
        item = Settings.name_to_item(self, item)
        settings = self.get_settings_for_item(item)
        out = settings(item, value=value, _create_columns=_create_columns, _save=_save, **kwargs)

        # Update combined settings instance to reflect change
        # if Settings.is_list_item(settings, item):
        #     for item0 in Settings.list_item_indices(settings, item):
        #         self._df.loc[item0, :] = settings._df.loc[item0, :]
        # else:
        #     self._df.loc[item, :] = settings._df.loc[item, :]
        # self._hash_id = None
        return out
    
    def __contains__(self, item):
        h = re.compile(r'{}:?\d*'.format(item))
        for ind in self._df.index:
            m = h.match(ind)
            if m:
                return True
        return False

    def __len__(self):
        if self._df is not None:
            return len(self._df)
        else:
            return 0

    def __iter__(self):
        for item in self._df.index:
            yield item

    def get_func_args(self, funcs, func_names=None, ignore_func_name=True, blacklist=()):
        """Get arguments for function from settings object
        :param: funcs - function instances or strings describing the function name"""
        if ignore_func_name is False:
            raise NotImplementedError
        funcs = make_iterable(funcs)
        blacklist = make_iterable(blacklist)
        if func_names is not None:
            func_names = make_iterable(func_names)
        else:
            func_names = [None]*len(funcs)
        args, kws = [], {}
        for func, func_name in zip(funcs, func_names):
            if func_name is None:
                # Get name of function/method/class
                func_name = func.__name__
                if func_name == '__init__':
                    # If func is an __init__ method, get the name of the corresponding class
                    func_name = get_methods_class(func).__name__
            sig = inspect.signature(func)
            for i, kw in enumerate(sig.parameters.values()):
                name = kw.name
                if ((name not in self.items) and
                    (not any(re.match(r'{}(:\d+)?$'.format(name), item) for item in self.items))):
                    continue
                if name in blacklist:
                    continue
                # compatible_functions = self._df.loc[name, 'function'].strip().split(',')
                # if (name in self) and ((func_name in compatible_functions) or ignore_func_name):
                kws[name] = self[name].value
                # if setting in kwargs:
                #     kws[name] = kwargs[name]
        return kws
    
    def call_with_args(self, func):
        """ Call supplied function with arguments from settings
        :param func: function to call with arguments
        :return: output of func.__call__(**kwargs)
        """
        kwargs = self.get_func_args(func)
        out = func(**kwargs)
        return out
        
    
    def set_value(self, **kwargs):
        """Set values of multiple existing items using keyword arguments. The set kwargs are removed from the dict."""
        updated = False
        for item, value, in copy(kwargs).items():
            if (item in self.items):
                if (value is None):
                    # Ignore None values
                    pass
                elif self[item] != value:
                    # Remove item from original dictionary
                    self(item, kwargs.pop(item))
                    updated = True
                    logger.debug('Set {}={} from kwargs'.format(item, value))
            else:
                raise KeyError('Item {} does not exist in {} to update value to {}'.format(item, self, value))
        if updated:
            self._hash_id = None

    def set_column(self, col, value, items='all', apply_to_groups=False):
        """Set value of column for a group of items"""
        assert col in self.columns, 'Column "{}" invalid. Options: {}'.format(col, self.columns)
        if items == 'all':
            items = self.items
        items = make_iterable(items)
        if apply_to_groups:
            # If item is the name of a settings file application modify its contents
            for item in copy(items):
                if item in self._settings.keys():
                    items = np.concatenate((items, self._settings[item].items))
                    items = np.delete(items, np.where(items==item))
        kws = {col: value}
        for item in items:
            self(item, **kws)

    def get_settings_for_item(self, item):
        """Return Settings object instance that item belongs to"""
        # Check input is valid
        item, category = Settings.get_item_index(self, item)
        if category == 'list':
            item = '{}:0'.format(item)
        if category == 'function':
            raise NotImplementedError
            return self.get_func_name_args(item)

        if len(self._items[item]) == 1:
            # Item is only in one collection of settings
            settings = self._items[item][0]
        elif len(self._items[item]) > 1:
            raise ValueError('Item "{}" is not unique. Occurs in: {}'.format(item, self._items[item]))
        else:
            raise Exception('Unexpected error. Item {} not in items dict'.format(item))
        return settings

    def compare_settings(self, other_settings, raise_on_difference=True):
        return Settings.compare_settings(self, other_settings, raise_on_difference=raise_on_difference)

    def save_to_hdf5(self, fn, group='settings'):
        """Save composite settings dataframe to hdf5 file group"""
        self._df.to_xarray().to_netcdf(fn, mode='a', group=group)
        logger.debug('Saved composite settings {} to file: {}'.format(repr(self), fn))

    def rename_items_with_pattern(self, pattern, replacement_string, force=False):
        """Replace all occurences of regex pattern in indices with 'replacement_string'"""
        for settings in self._settings.values():
            settings.rename_items_with_pattern(pattern, replacement_string, force=force)
        # TODO: Implement properly
        # self._df.loc[settings._df.index] = settings._df
        self._hash_id = None
        self.refresh()
        # logger.warning('Changes will not be reflected in composite settings df')

    def to_dict(self):
        """Return settings values as dict"""
        out = OrderedDict()
        for item in self.items:
            if re.match(r'^.*:\d+', item):
                item = item.split(':')[0]
            else:
                item = item
            value = self[item]
            out[item] = value
        return out

    def hash_id(self):
        from ccfepyutils.io_tools import gen_hash_id
        if self._hash_id is not None:
            return self._hash_id
        application, name = self._application, self._name
        mask = ~self._df['runtime'].astype(bool)
        df_hash = self._df.loc[mask]
        df_runtime = self._df.loc[~mask]
        hash_id = gen_hash_id(df_hash['value'])  # Only use value column for hash_id generation, not fine meta data
        path = os.path.join(settings_dir, 'hash_records', application)
        if not os.path.isdir(path):
            mkdir(path, depth=3)
        fn = 'settings_hash_record-{}-{}.nc'.format(application, hash_id)
        fn_path = os.path.join(path, fn)
        t0 = t_now_str(format='natural')
        if not os.path.isfile(fn_path):
        # if True:
            meta = {'application': self._application, 'name': self._name, 'first_used': t0, 'last_used': t0,
                    'protected': False, 'hash_id': hash_id}
            for i, col in enumerate(df_hash.columns):
                if (df_hash[col].dtype == bool) or (type(df_hash.iloc[0, i]) == bool):
                    df_hash[col] = df_hash[col].astype(int) # int?
            df_hash.to_xarray().to_netcdf(fn_path, mode='w', group='df')
            df_runtime.to_xarray().to_netcdf(fn_path, mode='a', group='runtime')
            with Dataset(fn_path, "a", format="NETCDF4") as root:
                dict_to_netcdf(root, 'meta', meta)
            logger.info('Created new settings hash file record: {}'.format(fn))
        else:
            logger.debug('Updating hash_id file: {}'.format(fn_path))
            n_attempts = 3
            for attempt in np.arange(n_attempts):
                try:
                    with Dataset(fn_path, "r+", format="NETCDF4") as root:
                        root['meta']['last_used'][0] = t0
                    break
                except IndexError as e:
                    logger.warning('Settings_hash_record file does not contain "meta/last_used" variable: {}'.format(fn))
                    exc = e
                except Exception as e:
                    logger.warning('Unanticipated error {}: {}'.format(e, fn))
                    exc = e
            else:
                logger.warning('Failed to update hash_id file after {} attempts'.format(n_attempts))
                raise exc

        self._hash_id = hash_id
        return hash_id

    @classmethod
    def load_from_hash_id(cls, application, hash_id):
        path = os.path.join(settings_dir, 'hash_records', application)
        fn = 'settings_hash_record-{}-{}.nc'.format(application, hash_id)
        fn_path = os.path.join(path, fn)
        s = None
        if os.path.isfile(fn_path):
            t0 = t_now_str(format='natural')
            df, success = attempt_n_times(xr.open_dataset, (fn_path,), kwargs={'group': 'df'})
            df = df.to_dataframe()
            s = Settings.get(application, hash_id)
            s._df = df
        else:
            logger.warning('Hash_id file does not exist: {}'.format(fn_path))
        return s

    @property
    def name(self):
        return self.core.name

    @property
    def column_sets_names(self):
        names = {key: [v[0] for v in value] for key, value in Settings.default_column_sets.items()}
        return names

    @property
    def items(self):
        return list(self._df.index)

    @property
    def columns(self):
        return list(self._df.columns.values)

    def __del__(self):
        # Clear up references to self in constituent settings objects
        for setting in self._settings.values():
            try:
                i = setting._composite_settings.index(self)
                setting._composite_settings.pop(i)
            except Exception as e:
                raise e

