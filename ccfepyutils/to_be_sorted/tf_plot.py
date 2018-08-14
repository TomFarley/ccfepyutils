#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_plot.py: Frequently used plotting operations and wrappers.

Detailed description:

Notes:
    @bug:

Todo:
    @todo: update tf.axis_range to use axis DATA limits
    @todo:

Info:
    @since: 18/09/2015
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl                  # For transforms etc ?
from matplotlib import transforms   # For transforms etc

import tf_libs.tf_debug as tf_debug
import tf_libs.tf_array  as tf_array
import tf_libs.tf_simple as tf_simple
import tf_libs.tf_string as tf_string

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"

db = tf_debug.Debug(1,1,0)

def set_mpl_defaults(defaults=0):
	""" Sets defaults for future mpl plots this session
	defaults = 0: Normal mpl defaults
			 = 1: More readable
			 = 2: Publication setting """
	if defaults == 0: # matplotlib defaults
		mpl.rcdefaults()
	elif defaults == 1:
		ax_labelsize = 20
		ax_titlesize = 22
		tick_labelsize = 'large'
		major_tick = dict(size=6, width=1.5, pad=4)
		minor_tick = dict(size=3, width=1, pad=4)
		lines = dict(linewidth=2.0, markersize=8)

		mpl.rc('axes', labelsize=ax_labelsize, titlesize = ax_titlesize)

		mpl.rc('xtick', labelsize=tick_labelsize)
		mpl.rc('ytick', labelsize=tick_labelsize)

		mpl.rc('xtick.major', **major_tick)
		mpl.rc('xtick.minor', **minor_tick)

		mpl.rc('ytick.major', **major_tick)
		mpl.rc('ytick.minor', **minor_tick)

		mpl.rc('lines', **lines)

	else:
		raise ValueError('mpl defaults defaults \'%d\' not recognised' % defaults)

	return

def new_axis(subplot=111, fig_no=None):
	"""
	Make new plot and reuturn its axes - used for finding axes ranges etc:

	CODE:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	"""
	if fig_no:
		fig = plt.figure(fig_no)
	else:
		fig = plt.figure()
	ax = fig.add_subplot(subplot)
	return ax, fig

def vline_label(x, label, ypos=0.8, xoffset=0.01, color = 'k'):
    """ Plot labeled vline, x in data coordinates, y in axis coordinates """
    fig = plt.gcf()
    ax = plt.gca()
    xlim = plt.xlim() # could use ax.get_xlim()
    xran = xlim[1] - xlim[0]
    # The x coords of this transformation are data, and the y coord are axes
    transy = transforms.blended_transform_factory(
                        ax.transData, ax.transAxes)
    plt.axvline( x, linestyle='--', color = color)
    plt.text(x+xoffset*xran, ypos, label, transform=transy, color = color )

def text_poss( x, y, string, ax, center = False, fontsize = 18 ):
	""" Given a string, an axis, and fractional axis coordinates, plot text 
	at those coordiantes (eg 20% from left, 20% from bottom)  
	ie will plot text string at:
	x_data = x_min + x_range * x_frac
	y_data = y_min + y_range * y_frac

	Inputs:
        string:   	string to add to plot
        ax:			axis of the current plot (in order to get limits)
        x: 			fractional x coordiante at which to plot string
        y: 			fractional y coordiante at which to plot string
    Outputs:
        NONE. (plots text) 
	"""

	if fontsize==None: fontsize=18

	if center == True:
		plt.text( x, y, string, 
				horizontalalignment='center', verticalalignment='center',
				transform=ax.transAxes, fontsize=fontsize)
	else:
		text = plt.text( x, y, string,
				transform=ax.transAxes, fontsize=fontsize)
		# text.draggable()

	## Get axis limits
	# ax_limits = ax.axis()
	# x_data = tf_num.frac_range( ax_limits[0:2], x )
	# y_data = tf_num.frac_range( ax_limits[2:4], y )
	# plt.text( string, x_data, y_data )

def extend_range(lims, pad=[5,5], absolute=False, pass_zero = False):
	""" Extend range of two element array
	"""
	assert (len(lims) == 2), 'Not two element array'
	if ~absolute:
		## Pad percentage values
		pad[0] /= 100.0 # Convert to percentage
		pad[1] /= 100.0

	lims_min = min(lims)
	lims_max = max(lims)
	range = lims_max - lims_min

	## Stop extended range passing zero (the origin)
	if (not pass_zero) and (lims_min-pad[0]*range)*lims_min < 0: # if extension of min changes sign of min stop at 0
		lims_min = 0
	else: # extend range as usual
		lims_min = lims_min - pad[0]*range

	lims_max = lims_max + pad[1]*range

	return np.array([lims_min, lims_max])

def axis_range(ax, padx = [5,5], pady = [5,5], pass_zero = (False,False), **kwargs):
    """ Extend axis ranges by given percentages
    Default to extend x and y ranges by 5%
    Set pass_zero = True, to allow range to cross origin
    """

    if tf_simple.is_scalar(pass_zero):
        pass_zero = (pass_zero, pass_zero)

    ## Update the axis view limits to first match the data limits
    ax.autoscale(enable=True, axis=u'both', tight=True)
    # ax.relim(visible_only=False) # matplotlib.axes.Axes.relim() prior to calling autoscale_view.
    # ax.autoscale_view(tight=None, scalex=True, scaley=True)

    ## Get the axis limits
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    # ax.set_autoscalex_on(False)
    # ax.set_autoscaley_on(False)

    ## Extend the axis ranges by required amounts
    ax.set_xlim(extend_range(x_range, pad = padx, pass_zero = pass_zero[0], **kwargs))
    ax.set_ylim(extend_range(y_range, pad = pady, pass_zero = pass_zero[1], **kwargs))

    return

def legend_dflt(ax, handles=None, labels=None, **kwargs):
    """ Add a legend to an axis with nice default behaviour	"""

    no_lines = len(ax.lines) # number of currently plotted lines associated with this legend

    ## If sufficiently many lines plotted, decrease the legend font size
    mod_fontsize = (no_lines>5) * (no_lines-4)

    options = { 'loc' : 'best',
                'ncol' : 1,
                'title' : None,
                'framealpha' : 0.5,
                'fontsize' : 18 - mod_fontsize,
                'fancybox' : True }

    ## update the default values if any of these kwargs have been passed
    options.update(kwargs)

    ## If separate handles and labels supplied, use them
    if handles and labels:
        legend = ax.legend(handles, labels, **kwargs)
    else:
        legend = ax.legend(**options)
    legend.draggable(state=True)



    return legend

def update_colors(ax, cm = 'jet', update_legend=True, min_lines=6):
	""" Update colour of existing lines to span colour table range
	Function from: http://stackoverflow.com/questions/20040597/matplotlib-change-colormap-after-the-fact """
	if cm and len(ax.lines) >= min_lines:
		lines = ax.lines
		cm = plt.get_cmap(cm) # Get colour map: eg OrRd, jet
		colors = cm(np.linspace(0, 1, len(lines)))
		for line, c in zip(lines, colors):
			line.set_color(c)
		if update_legend: legend_dflt(ax) # Update legend so its colours match the lines

def arr_hist(arr, nbins='auto', av_per_bin=40):
    """ Plot histogram of contents of arr
    setting mbins overrides av_per_bin """

    ## If nbins set to auto choose it so that there are on average av_per_bin
    if nbins == 'auto':
        if type(arr) == tuple:
            nbins = int(len(arr[0])/av_per_bin)
        else:
            nbins = int(len(arr)/av_per_bin)
        ## Make sure there are at least 10 bins
        if nbins < 10:
            nbins = 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## the histogram of the data with histtype='step'
    n, bins, patches = plt.hist(arr, nbins, normed=0, histtype='stepfilled', color=['g','b','c'][0:len(arr)])

    # plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

    x = (0.04, 0.8)
    for i, subarr in enumerate(tf_simple.make_iter(arr)):
        tf_string.str_moment(subarr, ax = ax, xy=(x[i],0.7))

    plt.show()
    return

def save_fig(fig, dir_fig='./Figures/', fn='Figure_tmp', ext='.png', dpi=300, silent=False, create_dir=False):
	""" Save figure as image """
	## If directory does not exist, create it if required
	if not create_dir:
		assert (os.path.isdir(dir_fig)), 'Path %s does not exist.'
	elif not os.path.isdir(dir_fig):
		os.makedirs(dir_fig)

	path = dir_fig+fn+ext
	try:
		fig.savefig( dir_fig+fn+ext, dpi=dpi )
		if not silent:
			print('Saved:  '+repr(fn+ext)+'\t\t to '+repr(dir_fig)) # Use repr for inverted commas
	except FileNotFoundError as e: #
		print('WARNING: Driectory: %s does not exist. %s' % (dir_fig, repr(e)))
		print('Failed to save file: %s' % fn+ext)

	return

def replot():
	""" Clear replot the figure with all it saved settings """
	pass

if __name__ == "__main__":
	print("axis_range([0,1,2,3,10])")
	print(axis_range([0,1,2,3,10]))

	mu, sigma = 200, 25
	x = mu + sigma*np.random.randn(10000)
	# x = np.linspace(0,100)
	arr_hist((x,x*.5))







