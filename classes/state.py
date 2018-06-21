#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import numbers
# from nested_dict import nested_dict
from datetime import datetime
from time import time
import itertools    

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

def in_state(transition_state, end_state=None):
    """Decorator to change state of object method belongs to for duration of method call"""
    def in_state_wrapper(func):
        def func_wrapper(*args, **kwargs):
            if 'ignore_state' in kwargs:
                # Don't perform state transition
                kwargs.pop('ignore_state')
                return func(*args, **kwargs)
                
            self = args[0]
            self.state(transition_state, call=(func, args, kwargs))
            # TODO: log function call and args in state object
            out = func(*args, **kwargs)
            # Set state to end state unless already set within func call
            if (end_state is not None) and (self.state == transition_state):
                self.state(end_state)
            # Return to state before decorated method was called, provided state not changed within func
            if self.state.current_group == 'transient':
                self.state.reverse()
            return out
        return func_wrapper
    return in_state_wrapper

class State(object):
    """Easy to implement state class
    
    Supplying a state_table dictionary specifies what state transitions are valid.  
    Supplying a call_table dictionary specifies what functions should be called during particular state transitions.  
    
    """
    call_patterns = ['enter', 'exit']  # Patterns for calls in state transitions
    state_groups = ['core', 'transient']
    def __init__(self, owner, table, initial_state, call_table=None, init_call=None, record_calls=True):
        self._owner = owner
        self._table = table
        self._call_table = call_table
        self._record_calls = record_calls
        self._current = {group: None for group in self.state_groups}
        self._history = {group: [None] for group in self.state_groups}
        self._history_all = []  # {'state': initial_state, 'call': init_call}]  # caller_details()}]  # TODO: Implement caller_details
        self.check_tables()
        self.set_state(initial_state)

    def check_tables(self):
        """Check formatting and consistency of tables"""

        # Check states groups are valid state groups
        assert all([group in self.state_groups for group in self._table]), 'State group not recognised: {}'.format(
                self._table.keys())
        # Fill in missing state groups
        for group in self.state_groups:
            if group not in self._table:
                self._table[group] = {}
        # Check all transition states are valid starting states
        for group in self.state_groups:
            for key, values in self._table[group].items():
                assert all([v in self.possible_states for v in values])

        # Check structure of call table
        if self._call_table is not None:
            # Check input values in call table are callables and keys are valid
            for key, values in self._call_table.items():
                assert key in self.possible_states  # check states
                assert all([(v in self.call_patterns+self.possible_states) for v in values])  # check call patterns
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
    def current_states(self):
        return self._current

    @property
    def current_states_str(self):
        return '; '.join([state for state in self.current_states.values() if state is not None])

    @property
    def current_state(self):
        """Return lowest level state ie transient state if not None"""
        for group in reversed(self.state_groups):
            if self._current[group] is not None:
                return self._current[group]

    @property
    def current_group(self):
        """Return lowest level group whos state is not None eg 'transient'"""
        for group in reversed(self.state_groups):
            if self._current[group] is not None:
                return group

    @property
    def core_states(self):
        return list(self._table['core'].keys())

    @property
    def transient_states(self):
        return list(self._table['transient'].keys())

    @property
    def possible_states(self):
        out = list(itertools.chain.from_iterable([self._table[group].keys() for group in self.state_groups])) + [None]
        return out

    @property
    def accessible_states(self):
        return self._table[self.current_group][self.current_state] + [None]

    @property
    def core_state(self):
        return self._current['core']

    @core_state.setter
    def core_state(self, value):
        assert value in self.possible_states
        self._current['core'] = value

    def get_state_group(self, state):
        for group in self.state_groups:
            if state in self._table[group]:
                return group
        return False

    @current_group.setter
    def current_group(self, value):
        assert value in self.state_groups
        self._current_group = value

    @property
    def history(self):
        return ': '.join(' -> '.join(value for value in self._history[group].values()) for group in self.state_groups)
    
    @property
    def previous_state(self):
        if len(self._history_all) > 1:
            return self._history_all[-2]['state']
        else:
            return None

    def check_accessible(self, new_state, raise_ex=True):
        if self.core_state is None:
            return True
        if new_state in self.accessible_states:
            return True
        else:
            if raise_ex:
                raise RuntimeError('{owner} cannot perform state switch "{old}" -> "{new}".\n'
                                   'Accessible states: {avail}'.format(
                        owner=repr(self._owner), old=self.current_state, new=new_state, avail=self.accessible_states))
            else:
                return False

    def call_transition(self, old, new, call=None, *args, **kwargs):
        """Call function from call_table during state transition"""
        if self._call_table is None:
            return
        if old is None:
            # Don't perform transition when initialising
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

        if self._record_calls:
            assert call is None or (isinstance(call, tuple) and len(call) == 3)  # Needs format (func, args, kwargs)
            self._history_all[-1]['call'] = call

    def state_group(self, group):
        assert group in self.state_groups
        return list(self._table[group].keys())

    def state_in_group(self, state, group):
        return state in self.state_group(group)

    def set_state(self, new_state, call=None, ignore=False, *args, **kwargs):
        old_state = self.current_state
        if new_state == old_state:
            return False
        # Check new state is accessible
        self.check_accessible(new_state, raise_ex=True)

        if new_state is not None:
            # Set the value of the appropriate state group and set lower level state values to None
            located = False
            for group in self.state_groups:
                if located:
                    self._current[group] = None
                    self._history[group] = [None]
                elif self.state_in_group(new_state, group):
                    self.call_transition(old_state, new_state, call=call, *args, **kwargs)
                    self._current[group] = new_state
                    self._history[group].append(new_state)
                    self._history_all.append({'state': new_state})
                    located = True
            if not located:
                raise ValueError('"{}" is not a valid state. \nOptions: {}'.format(new_state, self.possible_states))
        else:
            reset = False
            for group in self.state_groups:
                if group == self.current_group:
                    reset = True
                if reset:
                    self._current[group] = None
                    self._history[group] = [None]
        logger.debug('{owner} state changed {old} -> {new}'.format(
                owner=repr(self._owner), old=old_state, new=self.current_state))
        return new_state != old_state

    def __call__(self, new_state, call=None, ignore=False, *args, **kwargs):
        """Change state to new_state. Return True if call lead to a state change, False if already in new_state.
        :param new_state: name of state to change to"""
        return self.set_state(new_state, call=None, ignore=False, *args, **kwargs)

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
        return '<State: ({owner}): {current}>'.format(owner=repr(self._owner)[1:-1], current=self.current_states_str)

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
        # new_state = self._history_all[-(steps+1)]['state']
        new_state = self._history[self.current_group][-(steps+1)]
        self(new_state)

    def undo(self, n):
        """Undo n state changes"""
        raise NotImplementedError