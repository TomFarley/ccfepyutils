#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_classes.py:

Detailed description: Data manipulation classes

Notes:
    @bug:

Todo:
    @todo: update tf.axis_range to use axis DATA limits
    @todo: Use param module
    @todo: Use scipy units


Info:
    @since: 20/09/2015
"""

import numpy as np  # Maths library
import scipy as sp  # Data analysis library
import matplotlib.pyplot as plt  # Plotting library

from scipy.optimize import curve_fit  # Curve fitting
from scipy.signal import find_peaks_cwt, argrelmax  # Peak finding
from scipy.interpolate import interp1d  # Interpolation

import os  # System directory/file operations
import shutil  # High-level file operations
import re  # Regular expressions
import copy

from pprint import pprint  # Pretty printing

## CAN import:    all tf_* files
## CANNOT import: None
import tf_libs.tf_simple as tf_simple
import tf_libs.tf_array as tf_array
import tf_libs.tf_plot as tfp
import tf_libs.tf_debug as tf_debug

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"

db = tf_debug.Debug(1,1,0)

class PhysQuant(object):
    def __init__(self, name, symbol, unit):
        self.name  = name
        self.symbol = symbol
        self.unit  = unit


class ParamFloat(np.ndarray):
    """ Experimental parameter of float type """

    def __new__(cls, input_array, phys_quant, error=None, fn=None, description=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        # obj.info = info

        if type(phys_quant) == str:
            obj.phys_quant = obj.get_dflt_phys_quant(phys_quant)
            obj.name = obj.phys_quant.name
            obj.symbol = obj.phys_quant.symbol
            obj.unit = obj.phys_quant.unit
        elif len(phys_quant) == 3:
            obj.phys_quant = PhysQuant(phys_quant[0],phys_quant[1],phys_quant[2])
            obj.name = phys_quant[0]
            obj.symbol = phys_quant[1]
            obj.unit = phys_quant[2]
        else:
            assert (type(phys_quant) == str) or (len(phys_quant) == 3), 'Incorrect type/format for phys_quant'

        ## All atributes added here must also be added in array_finalise
        obj.values = input_array
        obj.error = error
        obj.fn    = fn
        obj.description = description

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default values for atributes, because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.phys_quant = getattr(obj, 'phys_quant', 'arb')  # Default if not found is 3rd argument
        self.values = getattr(obj, 'values', None)  # Default if not found is 3rd argument
        self.error = getattr(obj, 'error', None)  # Default if not found is 3rd argument
        self.fn = getattr(obj, 'fn', None)  # Default if not found is 3rd argument
        self.description = getattr(obj, 'description', None)  # Default if not found is 3rd argument

        if type(self.phys_quant) == str:
            self.phys_quant = self.get_dflt_phys_quant(self.phys_quant)
            self.name = self.phys_quant.name
            self.symbol = self.phys_quant.symbol
            self.unit = self.phys_quant.unit
        elif ((type(self.phys_quant) == tuple) and (len(self.phys_quant) == 3)):
            self.phys_quant = PhysQuant(self.phys_quant[0],self.phys_quant[1],self.phys_quant[2])
            self.name = self.phys_quant[0]
            self.symbol = self.phys_quant[1]
            self.unit = self.phys_quant[2]        
        if type(self.phys_quant) == PhysQuant:
            self.name = self.phys_quant.name
            self.symbol = self.phys_quant.symbol
            self.unit = self.phys_quant.unit
        else:
            raise('Incorrect type/format for phys_quant')

        # print(self[:])

        # We do not need to return anything

    def __init__(self, value, phys_quant, error=None, fn=None, description=None):
        pass
        # self[:] = value
        #
        # if type(phys_quant) == str:
        #     self.phys_quant = self.get_dflt_phys_quant(phys_quant)
        #     self.name = self.phys_quant.name
        #     self.symbol = self.phys_quant.symbol
        #     self.unit = self.phys_quant.unit
        # elif len(phys_quant) == 3:
        #     self.phys_quant = PhysQuant(phys_quant[0],phys_quant[1],phys_quant[2])
        #     self.name = phys_quant[0]
        #     self.symbol = phys_quant[1]
        #     self.unit = phys_quant[2]
        # else:
        #     assert (type(phys_quant) == str) or (len(phys_quant) == 3), 'Incorrect type/format for phys_quant'
        #
        # self.error = error
        # self.fn    = fn
        # self.description = description




        # self.lname = name.lower()
        # self.uname = name.capitalize()
        # self.label = (name+' ['+unit+']').encode('string-escape') if (not unit == '') else name
        # self.equals = (name+' = '+repr(value)+' '+unit).encode('string-escape')
        # self.fn    = params

    def __repr__(self):
        return 'ParamFloat<'+self.name+' ['+self.unit+']>'

    def __str__(self):
        #print help(self[:])
        # return self.name+' = '+repr(self[:])+' '+self.unit
        return self.name+': '+np.ndarray.__str__(self) +' '+self.unit

    def __eq__(self, other):
        """ two seperate instances with identical atributes will evaluate as the same. Handles inhertiance correctly.
        """
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __call__(self):
    #     " Value of float parameter "
    #     return self[:]
    #
    # def __getitem__(self,index):
    #     if tf_array.is_scalar(self[:]):
    #         raise IndexError('This <float parameter> is scalar and cannot be indexed')
    #     return self[:][index]
    #
    # def __len__(self):
    #     """ Length of parameter value array """
    #     return tf_array.safe_len(self[:])

    # def __copy__(self):
    #     ## Shallow copy
    #     return type(self)(self.name, self.unit, self[:], fn=self.fn)
    #
    # def __deepcopy__(self):
    #     ## Broken?
    #     return copy.deepcopy(type(self)(self.name, self.unit, self[:], fn=self.fn))
    #
    # def __del__(self):
    #     """ Unfinished """
    #     class_name = self.__class__.__name__

    def fit(self, func='poly8', x0=[]):
        """ Values from fitting func to self[:]s vs regularly spaced x variable using sp.curvefit """
        pass

    def valueSI(self):
        if self.unit == 'mTorr':
            return self[:] * 133.3224e-3
        elif self.unit == 'eV':
            return self[:] * 1.60217657e-19 / 1.3806488e-23

    def label(self):
        return self.name+' ['+self.unit+']'

    def legend(self):
        """ Return description of parameter appropriate for a plot legend """
        if self.description:
            return self.description
        else:
            return self.name

    def val_unit(self):
        return repr(self[:])+' '+self.unit

    def update(self, value):
        self[:] = value

    def info(self, quiet=False):
        if not quiet:
            pprint (vars(self))
        return vars(self)

    def get_dflt_phys_quant(self, symbol):
        dflts = {
            'arb':('Arbitrary quantity', 'No units'),
            'I':('Current','A'),
            't':('Time', 's'),
            'P':('Probability', None),
            'x':('X coordinate', 'm'),
            'y':('Y coordinate', 'm'),
            'z':('Z coordinate', 'm')
                 }
        assert symbol in dflts.keys(), 'Symbol for physical quantity not recognised'

        return PhysQuant(dflts[symbol][0],symbol,dflts[symbol][1])





class Param_string:
    """ Experimental parameter of string type """

    def __init__(self, name, value, fn=None):
        self.name  = name
        self.value = value
        self.fn    = fn

        self.lname = name.lower()
        self.uname = name.capitalize()
        self.label = name
        self.equals = name+' = '+repr(value)
        # self.fn    = params

    def __repr__(self):
        return 'ParamFloat<'+self.name+'> object'

    def __str__(self):
        return 'ParamFloat<'+self.Name+'>'

    def __del__(self):
        class_name = self.__class__.__name__

    def info(self):
        pprint (vars(self))


class Data(object):
    """ Ideas to include:
    - Key atributes: x, y, z, x_err, y_err, z_err, y_fit, z_fit
    - Auto data fit atribute
    - methods: average, differentiate, subtract given columns etc
    """

class Plot(object): # Inherit plt.figure ?
    nfig = 0 # Counter for number of Plot objects in existence
    def __init__(self, x=None, y=None, title = None, subplot=111, defaults=1, text=None, block=True, dir_fig = './Figures/', fn='Figure_tmp', cm='jet'):
        ## Check arguments are as expected
        db(tx=type(x))
        if not x==None: assert type(x) == ParamFloat, 'x plotting variable must be a <ParamFloat> (tuple not accepted)'
        if not y==None: assert type(y) == ParamFloat or (type(y) == tuple and type(y[0])==ParamFloat), 'y plotting variable must be a <ParamFloat> or tuple of <ParamFloat>s'
        if text: assert (type(text)==tuple and len(text)>=3), 'text keyword must have format (x,y,str,{fontsize})'

        PlotLines.nfig += 1  # Increase counter for number of plot instances
        self.fig_no = PlotLines.nfig  # Set the figure number to the number of this instance
        self.shown = False

        ## Store arguments in self
        self.fig = None
        self.title = title
        self.x = x
        self.y = tf_simple.make_iter(y) # Always make y a tuple even if it only contains one set of data
        self.text = [] # text is a list of tuples containing (x,y,str,{fontsize})
        if text: self.text.append(text) # Text to annotate plot
        self.subplot = subplot
        self.block = block
        self.dir_fig = dir_fig
        self.fn = fn
        self.cm = cm

        ## Set default mpl properties (font sizes etc)
        tfp.set_mpl_defaults(defaults=defaults)
        plt.ion() # Interactive plotting (draw implicit)

        ## Create the figure and axes
        self.ax, self.fig = tfp.new_axis(subplot, fig_no=self.fig_no)

        ## Set axis title and labels
        if self.title:
            self.set_title(title)
        if not self.x == None:
            self.set_xlabel(self.x.label())
        if not self.y == None:
            self.set_ylabel(self.y[0].label()) # set y axis using first y parameter in tuple

        if text:
            self.set_text()

            # self.fig.show()
            # plt.draw()
            # plt.show()

    def set_title(self,title):
        assert type(title) == str, 'title must be a string'
        self.ax.set_title(title)

    def set_xlabel(self,xlabel):
        assert type(xlabel) == str, 'xlabel must be a string'
        self.ax.set_xlabel(xlabel)

    def set_ylabel(self,ylabel):
        assert type(ylabel) == str, 'ylabel must be a string'
        self.ax.set_ylabel(ylabel)

    def set_text(self, text=None, center = False, fontsize = 16):
        """ Add text to plot at given plot coordinates """
        if text:
            self.text.append(text) # If called individually store supplied text tuple
        # print(self.text)
        if self.text:
            for t in self.text:
                if len(t)==4:
                    fontsize=t[3]
                else: fontsize=None
                tfp.text_poss(t[0],t[1],t[2], self.ax, center = False, fontsize=fontsize)

    def add_legend(self, force_legend = False):
        """ Add legend to plot with default setting (best position, dragable, transparrent etc) """
        self.force_legend = force_legend
        if len(self.lines) > 1 or force_legend: # Only show the legend if there is more than one plot
            tfp.legend_dflt(self.ax)

    def set_strings(self, force_legend=False):
        self.set_title(self.title)
        self.set_xlabel(self.x.label())
        self.set_ylabel(self.y[0].label())
        # self.add_legend(force_legend=force_legend) # don't want to add legend before lines

    def save_fig(self, dir_fig='./Figures/', fn='Figure_tmp', ext='.png', dpi=300, silent=False, create_dir=False):
        """ Save figure as an image """
        tfp.save_fig(self.fig, dir_fig=dir_fig, fn=fn, ext=ext, dpi=dpi, silent=silent, create_dir=create_dir)

    def show(self):
        if ((not self._internal) and (self.block)):
            # self.fig.show() # Why does this make figure window non-blocking?
            plt.show()
            self.shown=True  # Once the fig has been shown it needs to be redrawn, so record this

            if (self.fn):
                self.save_fig(create_dir=True, dpi=300)

class PlotLines(Plot):
    """ Wrapper for matplotlib 2D line plots with handy default behaviour """

    def __init__(self, cm = 'jet', padx = [0, 0], pady = [0, 0], pass_zero=(False, False), force_legend=False,
                 **kwargs):

        self.lines = [] # List of all lines plotted

        super().__init__(**kwargs) # Run __init__ for Plot base class

        self._internal = True # Prevent functions performing blocking actions and saving files after internal chain calls

        self.plot2D(self.x, self.y) # Plot the data
        self.update_colours(cm=cm) # Set line colours acording to colour map for clarity
        self.update_ranges(padx = padx, pady = pady, pass_zero=pass_zero) # Extend the axes ranges
        self.add_legend(force_legend=force_legend) # Add legend with good default settings

        self._internal = False # Allow functions called separately to perform blocking actions and save files ie call show()

        self.show() # Show the plotting window and save file

    def plot2D(self, x, y, block=True):
        """ Plot the y parameters (stored in a tuple) vs the x parameter """
        db(_internal=self._internal)
        if not self._internal: # If called individually, redraw the plot as was
            if self.shown: self.draw() # If the plot has already been shown and closed, redraw it
            self.block = block
        y = tf_simple.make_lt(y)

        for l in y: # Loop over y parameters and plot a line for each (a single y param is nested in a tuple)
            line = self.ax.plot(x[:], l[:], label=l.legend())
            self.lines.append(line)
        if not self._internal:
            self.update_colours(cm=self.cm) # Is this needed here?
            self.add_legend() # Update legend with new lines when called individually
            self.update_ranges(padx=self.padx, pady = self.pady, pass_zero=self.pass_zero)
            db('updated colours, legend and ranges')
        db(internal=self._internal)
        self.show()

        db(a=1,b=2,c=3)


    def refresh_lines(self):
        """ Redraw existing lines stored in self.lines, without appending any new elements
        Redundant now using _internal flag
        """

    def update_colours(self, cm='jet', min_lines=6):
        """ Update line colours to span supplied colour map """
        self.cm = cm
        tfp.update_colors(self.ax, cm=cm, min_lines=min_lines)

    def update_ranges(self, padx = [5, 5], pady = [5, 5], pass_zero=False):
        """ Update axis ranges """
        self.padx = padx
        self.pady = pady
        self.pass_zero = pass_zero
        ## Need to update axis_range to use axis DATA limits
        tfp.axis_range(self.ax, padx = padx.copy(), pady = pady.copy(), pass_zero=pass_zero) # send a copy so values in self are not modified (/100)

    def set_plot(self):
        """ Plot lines, tweek display and show legend etc based on previously stored values"""
        self.lines = [] # Empty list of all lines plotted so they are not duplicated when replotted
        self.plot2D(self.x, self.y) # Plot the data
        self.update_colours(cm=self.cm) # Set line colours acording to colour map for clarity
        self.update_ranges(padx = self.padx, pady = self.pady, pass_zero = self.pass_zero) # Extend the axes ranges
        self.add_legend(force_legend=self.force_legend) # Add legend with good default settings
        self.set_text()

    def draw(self):
        """ Clear and redraw the figure so it can be shown again """
        self._internal = True
        # if self.fig: self.fig.clear() # if the fig is being redrawn, clear it first
        self.ax, self.fig = tfp.new_axis(self.subplot, fig_no=self.fig_no)
        self.set_strings()
        self.set_plot()
        self.shown = False
        self._internal = False


class Plot3D(Plot):
    def __init__(self, z=None):
        pass

if __name__ == "__main__":
    pass