#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict
from datetime import datetime
from time import time

from copy import deepcopy

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)


class State(object):
    def __init__(self, owner, table, initial_state):
        self._owner = owner
        self._table = table
        self._current = initial_state
        self._history = [initial_state]
        assert initial_state in self.possible_states
        self.check_table_consistency()

    def check_table_consistency(self):
        # Check all transition states are valid starting states
        for key, values in self._table:
            assert all([v in self.possible_states for v in values])

    @property
    def possible_states(self):
        return self._table.key()

    @property
    def accessible_states(self):
        return self._table[self._current]

    @property
    def current_state(self):
        return self._current

    @property
    def history(self):
        return ' -> '.join(self._history)

    def __call__(self, new_state, *args, **kwargs):
        assert new_state in self.possible_states
        if (new_state != self.current_state):  # No change
            pass
        elif (new_state in self.accessible_states):  # Update state
            self._current = new_state
            self._history.append(new_state)
        else:
            raise RuntimeError('{owner} cannot perform state switch {old} -> {new}. Accessible states: {avail}'.format(
                    owner=repr(self._owner), old=self._current, new=new_state, avail=self.accessible_states))
        return self

    def __str__(self):
        return '{owner} in state {current}'.format(owner=repr(self._owner), current=self.current_state)

    def __repr__(self):
        return '<State: {owner};{current}>'.format(owner=repr(self._owner), current=self.current_state)

    def __eq__(self, other):
        """Return True if other is current state name or state object with same self.__dict__"""
        if other == self.current_state:
            return True
        if isinstance(other, State) and other.__dict__ == self.__dict__:
            return True
        else:
            return False