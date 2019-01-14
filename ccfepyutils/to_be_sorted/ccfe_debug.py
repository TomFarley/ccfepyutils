#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_debug.py: Object used to assist debugging, giving information about line numbers and file locations etc

Detailed description:

Notes:
    self.debug_ON is the value for the instance, Debug.debug_ON
    @bug:

Todo:
    @todo:

Info:
    @since: 18/09/2015
"""

import numpy as np
import matplotlib.pyplot as plt     # Plotting library
from pprint import pprint  # Pretty printing
import os
import shutil
import inspect              # Line numbers etc
import sys

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"


class Debug:
    """ Debugging features """
    debug_ON_class = False
    plot_ON_class = False
    lines_ON_class = False
    init = False
    ndebug_ON = 0
    nOFF = 0
    lines = {}
    functidebug_ONs = {'debug_ON':[], 'off': []}
    # List of all files read so far (in order)

    def __init__(self, debug_ON=False, lines_ON = False, plot_ON=True):
        self.line = inspect.currentframe().f_back.f_lineno
        ## First time Debug class is initiated print location if Debug ON
        if (not Debug.init) and debug_ON:
            print(whereami(level=1)+'tf_libs <Debug> instance created')#.format(self.line)
            Debug.init = True
        self.debug_ON = debug_ON
        self.lines_ON = lines_ON
        self.plot_ON = plot_ON
        if plot_ON:
            self.plot_ON = True

        debug_print(0, init_debug_ON=self.debug_ON)


    def __call__(self, *args, **kwargs):
        """ Perform a Debug print (or plot if requested) """
        ## Extract specific keywords
        plot = kwargs.pop('PLOT', False)
        force = kwargs.pop('FORCE', False) ## Force this call to print even if instance turned off

        self.line = inspect.currentframe().f_back.f_lineno

        if not plot: # Normal Debug print opperation
            # debug_print(1, debug_ON=self.debug_ON)
            if self.debug_ON or force or Debug.debug_ON_class:
                debug_print(self.debug_ON, lines_on=(self.lines_ON or force), *args, **kwargs)
                Debug.ndebug_ON += 1
                Debug.lines[self.line] = True
            else:
                Debug.nOFF += 1
                Debug.lines[self.line] = False
        else: # debug_plot
            if Debug.plot_ON:
                debug_plot(*args, **kwargs) 

    def on(self):
        Debug.debug_ON = True
        print('Debugging (tf_libs): \tdebug_ON')  

    def off(self):
        Debug.debug_ON = False
        print('Debugging (tf_libs): \tOFF')

    def l(self):
        line = inspect.currentframe().f_back.f_back.f_lineno
        print('(line: {}) {}'.format(line, text))

    def count(self):
        print("line {}: {} <Debug> debug_ON, {} Debug OFF".format(self.line, Debug.ndebug_ON, Debug.nOFF))

    def whereami(self, level=1):
        print(whereami(level=level))

    def trace_back(self, level=1):
        print(traceback(level=level))

    def force_all(self, on=True):
        if on:
            Debug.debug_ON_class = True
        else:
            Debug.debug_ON_class = False

    def info(self):
        print("line {}: {} <Debug> ON, {} <Debug>: OFF".format(self.line, Debug.ndebug_ON, Debug.nOFF))
        for line, debug_ON in self.lines.items():
            print(' ', line, debug_ON, ((debug_ON and ' <--') or ''))
        
def get_verbose_prefix():
    """Returns an informative prefix for verbose Debug output messages"""
    s = inspect.stack()
    module_name = inspect.getmodulename(s[1][1])
    func_name = s[1][3]
    return '%s->%s' % (module_name, func_name)

def func_name(level=0):
    return inspect.stack()[level+1][3]

def module_name(level=0):
    """ Return name of the module level levels from where this 
    function was called. level = 1 goes 1 level further up the stack """
    # return inspect.getmodulename(inspect.stack()[level+1][1])
    # print 'tf_debug.py, 85:', os.path.basename(inspect.stack()[level+1][1])
    try:
        name = os.path.basename(inspect.stack()[level+1][1])
    except IndexError:
        print('tf_debug: Failed to return module name. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        name = '*UNKNOWN*'
    return name

def line_no(level=0):
    """ Return line number level levels from where this 
    function was called. level = 1 goes 1 level further up the stack """
    try:
        line = str(inspect.stack()[level+1][2])
    except IndexError:
        print('tf_debug: Failed to return line number. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        line = '*UNKNOWN*'
    return line

def whereami(level=0):
    """ Return a string detailing the line number, function name and filename from level relative to where this
    function was called """

    # string = module_name(level=level+1)+', '+func_name(level=level+1)+', '+line_no(level=level+1)+': '
    string = line_no(level=level+1)+', '+func_name(level=level+1)+', '+module_name(level=level+1)+':\t'
    return string

def file_line(level=1):
    """ Return string containing filename and line number at level """
    return module_name(level=level+1)+', '+line_no(level=level+1)+': '


def traceback(level=0):
    """ Return string listing the full fraceback at the level relative to where this function was called """
    string = 'Traceback:\n'
    while not (func_name(level=level) == '<module>'):
        string += line_no(level=level+1)+', '+func_name(level=level+1)+', '+module_name(level=level+1)+'\n'
        level += 1
    return string.rstrip()

def debug_print( debug, *args, **kwargs ):
    """
    Purpose:
      Functidebug_ON to print informatidebug_ON about variables that can be easily
     turned debug_ON and off for debugging purposes. 
      If normal parameters are supplied, multiple variables are printed 
     to the same line, separated by tabs.
      If keyword arguments are supplied then the variables are printed 
     aldebug_ONgside the keyword names.
    
    Inputs:
     Debug      bool	toggle Debug mode debug_ON and off
     *args      any     variables to print
     **kwargs   any     variables to print with names
     
    Outputs: 
     (Ndebug_ONE)

    Call examples: 
     debug_print(1, var1, var2, var3)
     debug_print(Debug, var1 = var1, var2 = var2, var3 = var3)
    """
    ## TODO: Use pprint.pformat

    lines_on = kwargs.pop('lines_on', False)

    ## If lines_on prefix printed info with file name and line number
    if lines_on:
        prefix = file_line(level=2)
        print(prefix, end='')
    else: prefix = ''

    if debug:
        if args: # If there is anything in args
            for i, arg in enumerate(args):
                if i != len(args)-1:
                    ## print arguements on one line separated by commas (,)
                    print(' '+str(arg)+',\t', end=' ')
                else:
                    ## print final arguement with new line
                    print(arg)
        if kwargs: # If there is anything in args
            i = 0
            for key, value in kwargs.items():
                ## If the variable is a list that won't fit on one line start it on a new line
                ## NOTE: does not check for numpy arrays

                if i>0:
                    print(end=' '*len(prefix))
                i += 1

                if (type(value) is list) and (len(value) > 6):
                    print('{} \t=\n{}'.format( key, value ))
                else:
                    print('{} \t= {}'.format( key, value ))

def debug_plot(*args, **kwargs):
    "Quickly plot the supplied variables"
    level = kwargs.pop('level', 0)
    xlabel = kwargs.pop('xlabel', 'x')
    ylabel = kwargs.pop('ylabel', 'y')

    # print 'Debug plot on line {}'.format(line_no(level))

    if len(args) == 1: # and len(kwargs) == 0:
        x = np.arange(len(args[0]))
        y = args[0]
    elif len(args) == 2: # and len(kwargs) == 0:
        x = args[0]
        y = args[1]
    else:
        print('WARNING: Incorrect arguements to debug_plot')
        return

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(1,1,1)
    plt.plot( x, y, 'k-x', **kwargs)
    plt.grid(True)
    plt.title(r"Debugging plot")#.format(iv.fn))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend(loc='best')


def plineno(text=False):
    """Return the current line number in the calling program.
    TODO: Fix problem with inspect.currentframe().f_back.f_back returning None
    """
    line = inspect.currentframe().f_back.f_back.f_lineno
    if text:
        print('(line: {}) {}'.format(line, text))
    return line

def demo_print(command, info=""):
    """ Doesn't work if func not defined in this namespace
    """
    if info:
        info = ' - '+info
        print('\n'+command+info)
        # print(exec(command))
    else:
        print('\n'+command+' = ', end='')
        # print(exec(command))
    # return exec(command)

def debug_demo():
    print('*** tf_debug.py demo ***')
    db = Debug(1,1,1)

    x = np.linspace(0,10,100)
    y = np.linspace(10,30,100)
    a = [1,2,3]
    print()
    print('Line number (~206):', line_no(level=0))
    print('Function name (debug_demo):', func_name(level=0))
    print('Module name (tf_debug.py):', module_name(level=0))
    print(get_verbose_prefix(), '<Message>')
    print(whereami())
    print()
    print('debug_print tests:')
    a = 4
    arr = [4,7,3]
    str = 'hello!'
    print(a, arr, str)
    debug_print( 1, a, arr, str )
    debug_print( 1, a=a, arr=arr, str=str )

if __name__ == "__main__":
    debug_demo()
    pass	


