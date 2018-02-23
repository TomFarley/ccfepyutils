from collections import OrderedDict
from copy import copy
import logging

import numpy as np
import re
from nested_dict import nested_dict
from ..utils import make_itterable

from .settings import Settings
from .state import State

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CompositeSettings(object):
    instances = nested_dict()
    def __init__(self, application, name):
        """Settings must have an 'application' and a 'name'
        - The application is the context in which the settings are used e.g. my_code
        - The name is a label for a particular set of settings for the application e.g. my_default_settings
        """
        # TODO: implement state?
        self._reset_settings_attributes()
        # self.state = State('init')
        self._application = application
        self._name = name
        self.core = Settings.get(application, name)
        self.build_composite_df()


    def _reset_settings_attributes(self):
        self.state = None
        self.core = None

    def build_composite_df(self):
        """Expand settings items into full Settings objects"""
        self._settings = OrderedDict()
        self._items = OrderedDict()

        self._df = self.core._df[0:0]  # Get emtpy dataframe with same column structure
        self._df = self.append_settings_file(self._application, self._name, self._df, self._settings, self._items)

        pass

    def append_settings_file(self, application, name, df, settings, items):
        logger.debug('Adding "{}:{}" settings to {}'.format(application, name, repr(self)))
        s = Settings.get(application, name)
        assert application not in settings.keys(), 'Application names must be unique: {}'.format(application)
        settings[application] = s
        df_nest = s._df
        for item in s.items:
            df = df.append(df_nest.loc[item, :])
            if item not in items.keys():
                items[item] = [s]
            else:
                items[item].append(s)
            if df_nest.loc[item, 'setting']:
                name = df_nest.loc[item, 'value']
                df = self.append_settings_file(item, name, df, settings, items)
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

    def view(self, cols='repr', order=None, ascending=True):
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
        if order is None:
            df = self._df
        elif order == 'alphabetical':
            df = self._df.sort_index(ascending=ascending)
        elif order == 'custom':
            df = self._df.sort_values('order', ascending=ascending)
        out = df.loc[:, col_set]
        return out

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
    
    def get_settings_for_item(self, item):
        if item not in self.items:
            raise ValueError('Item "{}" not in {}'.format(item, repr(self)))
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
            return False

    def rename_items_with_pattern(self, pattern, replacement_string, force=False):
        """Replace all occurences of regex pattern in indices with 'replacement_string'"""
        for settings in self._settings.values():
            settings.rename_items_with_pattern(pattern, replacement_string, force=force)
            self.df.loc[:,:] = settings._df

    @property
    def column_sets_names(self):
        names = {key: [v[0] for v in value] for key, value in Settings.default_column_sets.items()}
        return names

    @property
    def items(self):
        return list(self._df.index)

