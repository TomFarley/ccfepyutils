from collections import OrderedDict
from copy import copy
import logging
import inspect

import numpy as np
import re
from nested_dict import nested_dict
from ..utils import make_itterable

from .settings import Settings
from .state import State

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CompositeSettings(object):
    instances = nested_dict()
    def __init__(self, application, name, blacklist=[], whitelist=[], include_children=True, exclude_if_col_true=()):
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
        self.build_composite_df(include_children=include_children, exclude_if_col_true=exclude_if_col_true)

    def _reset_settings_attributes(self):
        self.state = None
        self.core = None

    def build_composite_df(self, include_children=True, exclude_if_col_true=()):
        """Expand settings items into full Settings objects"""
        self._settings = OrderedDict()
        self._items = OrderedDict()

        self.core._df.loc[:, 'parent'] = 'None'
        self._df = self.core._df[0:0]  # Get emtpy dataframe with same column structure
        self._df = self.append_settings_file(self._application, self._name, self._df, self._settings, self._items,
                                             include_children=include_children, exclude_if_col_true=exclude_if_col_true)

        pass

    def append_settings_file(self, application, name, df, settings, items,
                             add_to_whitelist=False, include_children=True, exclude_if_col_true=()):
        """ Add contents of settings file to CompositeSettings instance's dataframe
        :param: add_to_whitelist - treat application as if it in whitelist
        :param: include_children - include child settings even if they are not in the whitelist"""
        if (len(self._blacklist) > 0) and (application in self._blacklist):
            logger.debug('Skipping blacklist settings "{}:{}" from {}'.format(application, name, repr(self)))
            return df
        if (len(self._whitelist) > 0) and (application not in self._whitelist) and (not add_to_whitelist):
            logger.debug('Skipping non whitelist settings "{}:{}" from {}'.format(application, name, repr(self)))
            return df
        logger.debug('Adding "{}:{}" settings to {}'.format(application, name, repr(self)))
        s = Settings.get(application, name)
        # Add parent column to note which Settings file each setting originates from
        if len(s) > 0:
            s._df.loc[:, 'parent'] = '{}:{}'.format(application, name)
        assert application not in settings.keys(), 'Application names must be unique: {}'.format(application)
        settings[application] = s
        df_nest = s._df
        for item in s.items:
            exclude = False
            for excl_col in exclude_if_col_true:
                if df_nest.loc[item, excl_col] is np.True_:
                    exclude = True
            if not exclude:
                df = df.append(df_nest.loc[item, :])
                if item not in items.keys():
                    items[item] = [s]
                else:
                    items[item].append(s)
            if df_nest.loc[item, 'setting']:
                name = df_nest.loc[item, 'value']
                # If include_children pass to subsequent calls, but set False for calls from core Settings set
                df = self.append_settings_file(item, name, df, settings, items,
                                           add_to_whitelist=(include_children and (application != self._application)),
                                           include_children=include_children)
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

    def view(self, items='all', cols='repr', order=None, ascending=True):
        """Return dataframe containing a subst of columns, with items ordered as requried"""
        if cols == 'all':
            raise NotImplementedError
            col_set = self.ordered_column_names
        else:
            col_set = []
            cols = make_itterable(cols)
            for col in cols:
                if col in self.column_sets_names.keys():
                    col_set += self.column_sets_names[col]
                else:
                    assert col in self.columns, 'Column not found: {}'.format(col)
                    col_set += [col]
            if cols == ['repr']:
                col_set.pop(col_set.index('name'))
                col_set += ['parent']
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
        out = df.loc[items, col_set]
        return out

    def search(self, pattern):
        r = re.compile(pattern, re.IGNORECASE)
        newlist = list(filter(r.search, self.items))
        return newlist

    def add_item(self, application, item, value, **kwargs):
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
    
    def __call__(self, item, value=None, create_columns=False, _save=False, **kwargs):
        """Call __call__ from appropriate settings object"""
        item = self.name_to_item(item)
        settings = self.get_settings_for_item(item)
        out = settings(item, value=value, create_columns=create_columns, _save=_save, **kwargs)
        # Update combined settings instance to reflect change
        self._df.loc[item, :] = settings._df.loc[item, :]
        return out
    
    def __contains__(self, item):
        if item in self._df.index:
            return True
        else:
            return False

    def __len__(self):
        if self._df is not None:
            return len(self._df)
        else:
            return 0

    def get_func_args(self, funcs, func_names=None, ignore_func_name=False):
        """Get arguments for function from settings object
        :param: funcs - function instances or strings describing the function name"""
        funcs = make_itterable(funcs)
        if func_names is not None:
            func_names = make_itterable(func_names)
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
                if name not in self.items:
                    continue
                compatible_functions = self._df.loc[name, 'function'].strip().split(',')
                if (name in self) and ((func_name in compatible_functions) or ignore_func_name):
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
        """Set values of multiple items using keyword arguments"""

        for item, value, in copy(kwargs).items():
            if (item in self.items) and (value is not None):
                self(item, kwargs.pop(item))
                logger.debug('Set {}={} from kwargs'.format(item, value))

    def set_column(self, col, value, items='all', apply_to_groups=False):
        """Set value of column for a group of items"""
        assert col in self.columns, 'Column "{}" invalid. Options: {}'.format(col, self.columns)
        if items == 'all':
            items = self.items
        items = make_itterable(items)
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

    def save_to_ddf5(self, fn, group='settings'):
        """Save composite settings dataframe to hdf5 file group"""
        self._df.to_xarray().to_netcdf(fn, mode='a', group=group)
        logger.debug('Saved composite settings {} to file: {}'.format(repr(self), fn))

    def name_to_item(self, name):
        """Lookup item key given name"""
        df = self._df
        if name in df['name'].values:
            mask = df['name'] == name  # boolian mask where name == 'item'
            if np.sum(mask) == 1:
                item = df.index[mask].values[0]
            else:
                raise ValueError('Setting item name {} is not unique so cannot be used to index {}'.format(
                        name, repr(self)))
            return item
        elif (name in self.items):
            return name
        else:
            return name

    def rename_items_with_pattern(self, pattern, replacement_string, force=False):
        """Replace all occurences of regex pattern in indices with 'replacement_string'"""
        for settings in self._settings.values():
            settings.rename_items_with_pattern(pattern, replacement_string, force=force)
            self.df.loc[:,:] = settings._df

    def to_dict(self):
        """Return settings values as dict"""
        out = OrderedDict()
        for item in self.items:
            if re.match(r'^.*:\d+', item):
                key = item.split(':')[-1]
            else:
                key = item
            value = self[item]
            out[key] = value
        return out

    @property
    def hash_id(self):
        from ccfepyutils.io_tools import gen_hash_id
        mask = ~self._df['runtime']
        df = self._df.loc[mask, 'value']
        hash_id = gen_hash_id(df)
        return hash_id

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
