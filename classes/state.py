#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict
from datetime import datetime
from time import time

from copy import deepcopy

from ccfepyutils.utils import args_for

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def in_state(state):
    """Decorator to change state of object method belongs to for duration of method call"""
    def in_state_wrapper(func):
        def func_wrapper(self, *args, **kwargs):
            self.state(state)
            func(*args, **kwargs)
            self.state.reverse()
        return func_wrapper
    return in_state_wrapper

class State(object):
    call_patterns = ('enter', 'exit')  # Patterns for calls in state transitions
    def __init__(self, owner, table, initial_state, call_table=None):
        self._owner = owner
        self._table = table
        self._current = initial_state
        self._history = [initial_state]
        self._history_all = [initial_state]
        self._call_table = call_table
        assert initial_state in self.possible_states
        self.check_tables()

    def check_tables(self):
        """Check formatting and consistency of tables"""

        # Check all transition states are valid starting states
        for key, values in self._table.items():
            assert all([v in self.possible_states for v in values])

        # Check structure of call table
        if self._call_table is not None:
            for key, values in self._call_table:
                assert key in self.possible_states  # check states
                assert all([(v in self.call_patterns) for v in values])  # check call patterns
                for pattern in self.call_patterns:
                    callables = values[pattern]
                    if pattern not in values:  # Complete call table with empty lists
                        self._call_table[key][pattern] = []
                    elif not isinstance(callables, (list, tuple)):  # If not collection of callables nest in list
                        values[pattern] = [values[pattern]]
                    assert all([callable(v) for v in callables])  # check callables are callable
            for state in self.possible_states:  # Complete call table with empty lists
                if state not in self._call_table:
                    for pattern in self.call_patterns:
                        self._call_table[state][pattern] = []

    @property
    def possible_states(self):
        return self._table.keys()

    @property
    def accessible_states(self):
        return self._table[self._current]

    @property
    def current_state(self):
        return self._current

    @current_state.setter
    def current_state(self, value):
        assert value in self.possible_states
        self._current = value

    @property
    def history(self):
        return ' -> '.join(self._history)

    def call_transition(self, old, new, *args, **kwargs):
        if self._call_table is None:
            return
        kwargs.update(dict(
                ('state_transition', dict( (('old', old), ('new', new)),),)
                ))  # Add information out state change
        for func in self._call_table[old]['exit']:
            # TODO: Use args
            kws = args_for(func, kwargs)
            func(*args, **kws)
        for func in self._call_table[new]['enter']:
            # TODO: Use args
            kws = args_for(func, kwargs)
            func(*args, **kws)

    def __call__(self, new_state, ignore=False, *args, **kwargs):
        """Change state to new_state. Return True if call lead to a state change, False if already in new_state.
        :param new_state: name of state to change to"""
        old_state = self.current_state
        assert new_state in self.possible_states
        if (new_state == self.current_state):  # No change
            pass
        elif (new_state in self.accessible_states):  # Update state
            self.current_state = new_state
            self._history.append(new_state)
            self.call_transition(old_state, new_state, *args, **kwargs)
        else:
            raise RuntimeError('{owner} cannot perform state switch {old} -> {new}. Accessible states: {avail}'.format(
                    owner=repr(self._owner), old=self._current, new=new_state, avail=self.accessible_states))
        logger.debug('{owner} state changed {old} -> {new}'.format(
                    owner=repr(self._owner), old=self._current, new=new_state))
        self._history_all.append(new_state)  # Update irrespective of change
        return new_state != old_state

    def __getitem__(self, item):
        """Get state name from history
        eg state[-1] returns previous state"""
        assert item <= 0
        if item < -len(self._history):
            raise IndexError('State history only has length {}'.format(len(self._history)))
        return self._history[item]  # TODO: return Sate instance using undo

    def __str__(self):
        return self.current_state
        # return '{owner} in state {current}'.format(owner=repr(self._owner), current=self.current_state)

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

    def reverse(self, steps=1):
        """Reverse to past state if past state is different to current state
        :param steps: number of steps in history_all to reverse"""
        assert steps > 0
        new_state = self._history_all[-steps]
        if new_state != self:
            self(self._history_all[-steps])

    def undo(self, n):
        """Undo n state changes"""
        raise NotImplementedError