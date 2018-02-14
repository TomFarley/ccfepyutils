#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
from nested_dict import nested_dict
from datetime import datetime
from time import time

from copy import deepcopy

from ccfepyutils.utils import args_for, caller_details

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
TODO:
- add sig2fwhm arg to plot ellipses _/
- scale match radius
- reduce fil size
- mod fil posns
"""

def in_state(state, end_state=None):
    """Decorator to change state of object method belongs to for duration of method call"""
    def in_state_wrapper(func):
        def func_wrapper(*args, **kwargs):
            self = args[0]
            self.state(state, call=(func, args, kwargs))
            # TODO: log function call and args in state object
            func(*args, **kwargs)
            # Return to state before decorated method was called
            if end_state is None:
                self.state.reverse()
            # Don't make transition if internally state has already been changed
            elif self.state != end_state:
                self.state(end_state)
        return func_wrapper
    return in_state_wrapper

class State(object):
    """Easy to implement state class
    
    Supplying a state_table dictionary specifies what state transitions are valid.  
    Supplying a call_table dictionary specifies what functions should be called during particular state transitions.  
    
    """
    call_patterns = ('enter', 'exit')  # Patterns for calls in state transitions
    def __init__(self, owner, table, initial_state, call_table=None, init_call=None, record_calls=True):
        self._owner = owner
        self._table = table
        self._current = initial_state
        self._record_calls = record_calls
        self._history = [initial_state]
        self._history_all = [{'state': initial_state, 'call': init_call}]  # caller_details()}]  # TODO: Implement caller_details
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
            # Check input values in call table are callables and keys are valid
            for key, values in self._call_table.items():
                assert key in self.possible_states  # check states
                assert all([(v in self.call_patterns) for v in values])  # check call patterns
                for pattern in self.call_patterns:
                    if pattern not in values:  # Complete call table with empty lists
                        self._call_table[key][pattern] = []
                    callables = values[pattern]
                    if not isinstance(callables, (list, tuple)):  # If not collection of callables nest in list
                        values[pattern] = [callables]
                        callables = values[pattern]
                    assert all([callable(v) for v in callables])  # check callables are callable
            # Complete call table with dicts of empty lists
            for state in self.possible_states:
                if state not in self._call_table:
                    self._call_table[state] = {}
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
    
    @property
    def previous_state(self):
        if len(self._history_all) > 1:
            return self._history_all[-2]['state']
        else:
            return None

    def call_transition(self, old, new, *args, **kwargs):
        """Call function from call_table during state transition"""
        if self._call_table is None:
            return
        kwargs.update(dict((
                ('state_transition', dict((
                    ('old', old), ('new', new))
                )),
                ),))  # Add information out state change
        for func in self._call_table[old]['exit']:
            # TODO: Use args
            kws = args_for(func, kwargs)
            func(*args, **kws)
        for func in self._call_table[new]['enter']:
            # TODO: Use args
            kws = args_for(func, kwargs)
            func(*args, **kws)
        # Execute functions in call table specificall for this transition 
        if new in self._call_table[old]:
            for func in self._call_table[old][new]:
                # TODO: Use args
                kws = args_for(func, kwargs)
                func(*args, **kws)

    def __call__(self, new_state, call=None, ignore=False, *args, **kwargs):
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
                    owner=repr(self._owner), old=old_state, new=new_state))
        self._history_all.append({'state': new_state})
        if self._record_calls:
            assert call is None or (isinstance(call, tuple) and len(call) == 3)  # Needs format (func, args, kwargs)
            self._history_all[-1]['call'] = call
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
        return '<State: ({owner});{current}>'.format(owner=repr(self._owner)[1:-1], current=self.current_state)

    def __eq__(self, other):
        """Return True if other is current state name or state object with same self.__dict__"""
        if other == self.current_state:
            return True
        if isinstance(other, State) and other.__dict__ == self.__dict__:
            return True
        else:
            return False
        
    def repeated(self):
        """Return True if last state change call was to the same state"""
        return self == self.previous_state

    def reverse(self, steps=1):
        """Reverse to past state if past state is different to current state
        :param steps: number of steps in history_all to reverse"""
        assert steps > 0 and steps < len(self._history_all)
        new_state = self._history_all[-(steps+1)]['state']
        self(new_state)

    def undo(self, n):
        """Undo n state changes"""
        raise NotImplementedError