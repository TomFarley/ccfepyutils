#!/usr/bin/env python

import os, numbers
from collections import OrderedDict
from copy import copy, deepcopy

import numpy as np

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batch_mode = os.getenv('LOADL_ACTIVE', None)
if batch_mode == 'yes':
    import matplotlib
    try:
        matplotlib.use('Agg')
        print('In batch mode')
    except Exception:
        logger.warning('Failed to switch matplotlib backend in batch mode')
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from ccfepyutils.utils import make_iterable, make_iterables, args_for, to_array, is_scalar, none_filter
from ccfepyutils.io_tools import pos_path
from ccfepyutils.classes.state import State, in_state
from ccfepyutils.classes.fitter import Fitter
from ccfepyutils.data_processing import pdf
try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

class Plot(object):
    """Convenience plotting class

    TODO:
    - Save output using settings object
    - Change default save filetype easily (png -> eps etc)
    - Share axis
    - Multiple axes
    - Seg line plot
    - Arrow plot
    """
    # Plotting modes compatible with different input data dimensions. The first mode for each dimension is the default
    dim_modes = dict((((0, 0, None), ('scatter',)),  # single point
                      ((1, None, None), ('line', 'scatter', 'pdf')),
                      ((1, 1 , None), ('line', 'scatter')),
                      # ((1, 1 , 1), ('scatter3D', 'line3D', 'segline')),
                      ((2, 2 , 2), ('contourf', 'surface3D',)),
                      ((1, 1, 2), ('contourf', 'contour', 'image', 'surface_3d')),
                      ((2, None, None), ('contourf', 'contour', 'image')),
                      ((1, 1, 1), ('scatter_3d')),
                      # ((None, None , 2), ('image', 'contour', 'contourf'))
                      ))
    instances = []  # List of all Plot instances
    state_table = {'core': {
                        'init': ['ready', 'plotting'],
                        'ready': ['plotting']},
                    'transient': {
                        'plotting': ['ready']},
                   }
    plot_args = ['ls', 'lw', 'c', 'color', 'marker', 'label']  # TODO: extend plot_args
    scatter_args = ['s', 'c', 'color', 'marker', 'label']  # TODO: extend plot_args
    other_args = ['show', 'xlabel', 'ylabel', 'legend', 'ax']
    args = plot_args + scatter_args + other_args # List of all possible args for use with external kwargs

    def __init__(self, x=None, y=None, z=None, num=None, axes=(1, 1), default_ax=(0, 0), ax=None, mode=None,
                 legend='each axis', save=False, show=False, fig_args={}, **kwargs):
        """

        :param x:
        :param y:
        :param z:
        :param num:
        :param axes:
        :param default_ax: default axis for future calls if ax argument not supplied (starts at 1)
        :param mode:
        :param legend:
        :param save:
        :param show:
        :param fig_args:
        :param kwargs:
        """
        logger.debug('In Plot.__init__')

        self.call_table = {'ready': {'enter': self.set_ready}}
        self._reset_attributes()

        self.state = State(self, self.state_table, 'init', call_table=self.call_table)

        self._default_ax = default_ax  # Default axis for actions when no axis is specified
        self._current_ax = ax
        self._data = OrderedDict()  # Data used in plots stored for provenance
        self._log = []  # Log of method calls for provenance
        self._legend = legend

        self.instances.append(self)
        # TODO: find cleaner way of using existing plot instance!
        self = self.set_figure_variables(ax=ax, num=num, axes=axes, **fig_args)  # TODO: Add axis naming? .set_axis_names - replace defaults?
        # kws = args_for(self.plot, kwargs, include=self.plot_args)
        self.plot(x, y, z, mode=mode, **kwargs)
        self.show(show, **args_for(self.show, kwargs))
        self.save(save, **args_for(self.save, kwargs))

    def _reset_attributes(self):
        self._num = None  # Name of figure window
        self._ax_shape = ()  # Shape of axes grid
        self._default_ax = None # Default axis for actions when no axis is specified
        self._current_ax = None # Current axis for when actions are chained together. None as soon as out of Plot scope.
        self._data = OrderedDict()  # Data used in plots stored for provenance
        self._log = []  # Log of method calls for provenance
        self._legend = None

        self.fig = None  # figure object
        self.axes = None  # flat array of axes?
        self.gs = None  # grid spec
        self._gs_slices = None  # grid spec slices of existing axes
        self._axes_dict = None  # dict linking grid spec slices to existing axes
        self._axes_names = None  # dict linking string names for axes to their grid spec slices
        self.return_values = None  # values returned by internal function calls

    def set_figure_variables(self, ax=None, num=None, axes=None, **kwargs):
        """Set figure attributes"""
        if ax is None:
            self.make_figure(num, axes, **kwargs)
        elif hasattr(ax, 'ccfe_plot'):
            # Axis is from an exisiting ccfe_plot instance
            self = ax.ccfe_plot
        elif isinstance(ax, matplotlib.axes.Axes):
            self.fig = ax.figure
            self.axes = np.array(self.fig.axes)
            self._ax_shape = np.array(np.array(self.fig.axes).shape)  # TODO fix so 2 element shape, not len
            self._default_ax = np.where(self.axes == ax)[0][0]
            if len(self._ax_shape) == 1:
                self._ax_shape = np.insert(self._ax_shape, 0, 1)
                self.axes = np.array([self.axes])
        else:
            raise ValueError('Unexpected axis object! {}'.format(ax))
        self._num = self.fig.canvas.get_window_title()
        return self

    def make_figure(self, num, axes, **kwargs):
        """Create new figure if no axis instance passed"""
        assert isinstance(num, (string_types, str, int, type(None)))
        assert isinstance(axes, (tuple, list, np.ndarray))
        # if plt.fignum_exists(num):
        #     plt.figure(num).clear()  # unnessesary?
        # self.fig, self.axes = plt.subplots(num=num, *axes, **kwargs)
        kws = {'tight_layout': False}
        kws.update(kwargs)
        self.fig = plt.figure(num=num, **kws)
        self.gs = gridspec.GridSpec(axes[0], axes[1])
        self._gs_slices = OrderedDict()
        for i in range(axes[0]):
            for j in range(axes[1]):
                ax = self.fig.add_subplot(self.gs[i, j])
                ax.ccfe_plot = self
                self._gs_slices[(i, j)] = ax
        self.axes = to_array(self.fig.axes)
        self._axes_names = OrderedDict()
        self._ax_shape = np.array(axes)  # Shape of axes grid
        if len(self._ax_shape) == 1:
            self._ax_shape = np.insert(self._ax_shape, 0, 1)
        pass

    def set_ready(self):
        """Set plot object in 'ready' state by finalising/tidying up plot actions"""
        self._current_ax = None

    def __repr__(self):
        """Representation: Plot(<axes_shape>;<default_axis>:<name>)"""
        out = '<Plot{};{}:"{}">'.format(tuple(self._ax_shape), self._default_ax, self._num)
        return out

    def _get_modes(self, x, y, z, data_requried=False):
        """Inspect supplied data to see whether further actions should be taken with it"""
        data = {k: np.array(v) for k, v in zip(['x', 'y', 'z'], [x, y, z]) if v is not None}  # data that is not None
        shapes = {k: v.shape for k, v in data.items()}
        dims = {k: v.ndim for k, v in data.items()}
        dim = tuple((np.array(v).ndim if v is not None else None for v in (x, y, z)))
        if len(data) == 0:
            if data_requried:
                raise ValueError('No data supplied to Plot object')
            else:
                return [None]
        if dim in self.dim_modes:
            return self.dim_modes[dim]
        else:
            raise ValueError('Dimensions of input data {} are not amongst those supported: {}'.format(dim, self.dim_modes))

    def _name2ax(self, ax, name=None):
        """Return axis object given axis index, grid spec slice or string name"""
        # Convert to gridspec slice
        if isinstance(ax, matplotlib.axes.Axes):  # TODO: improve this!
            return ax  # already an axis instance
        ax_names = self._axes_names
        if isinstance(ax, numbers.Integral):
            shape = self._ax_shape
            assert ax <= np.prod(shape), 'axes {} is outside of axes shape {}'.format(ax, shape)
            index = (ax // shape[1], ax % shape[1])
        elif isinstance(ax, string_types):
            assert ax in ax_names, 'Axis name "{}" not recognised. Options: {}'.format(ax, ax_names)
            index = ax_names[ax]
            # raise NotImplementedError
        elif isinstance(ax, (tuple, list)):
            assert np.array(ax).shape == self._ax_shape, 'Axis tuple selection of wrong shape: {} not {}'.format(
                    np.array(ax).shape, self._ax_shape)
        else:
            raise TypeError('ax argument has unexpected type "{}": {}'.format(type(ax), ax))

        if self.gs is None:
            ax = self.axes[index]
        # Convert grid spec slice to axis instance
        elif index in self._gs_slices:
            # Axis instance already exists
            ax = self._gs_slices[index]
            ax.ccfe_plot = self
            if name is not None:
                # Rename axis
                if index in ax_names.values():
                    old_name = list(ax_names.keys())[list(ax_names.values()).index(index)]
                    del ax_names[old_name]
                ax_names[name] = index
        else:
            index = ax
            logger.debug('index {} not in self._gs_slices {}'.format(index, self._gs_slices))
            logger.debug('Adding axis at index {} to fig {} in {} with axes {}'.format(index, self.fig, self,
                                                                                       self.fig.axes))
            ax = self.fig.add_subplot(self.gs[index])
            ax.ccfe_plot = self
            self._gs_slices[index] = ax
            ax.ccfe_plot = self
            self._gs_slices[index] = ax
            if name is None:
                name = str(index)
            ax_names[name] = index
            self.axes = to_array(self.fig.axes)
        return ax

    def _check_mode(self, x, y, z, mode):
        assert isinstance(mode, string_types)
        pos_modes = self._get_modes(x, y, z)
        if pos_modes is None:
            raise ValueError('No data')
        if mode in pos_modes:
            return True
        else:
            return False

    def ax(self, ax=None, name=None):
        """Return axis instance."""
        if ax is None:
            # If no information supplied use current/default axis
            if self._current_ax is not None:
                ax = self._current_ax
            else:
                ax = self._default_ax
        elif isinstance(ax, matplotlib.axes.Axes):
            ax = ax
        elif isinstance(ax, string_types+(int, tuple, list)):
            ax = ax
        # Convert axis index/name to axis instance
        ax = self._name2ax(ax, name=name)
        self._current_ax = ax
        try:
            # Set as current axis for plt. calls
            plt.sca(ax)
        except ValueError as e:
            logger.error('Failed to set current plotting axis! {}'.format(e))
        return ax

    def _use_data(self, x, y, z, mode):
        """Inspect supplied data to see whether further actions should be taken with it"""
        raise NotImplementedError

    def call_if_args(self, ax, kwargs, raise_on_exception=True):
        kwargs['ax'] = ax
        for func in (self.set_axis_labels, self.set_axis_limits, self.show):
            kws = args_for(func, kwargs, remove=True)
            if len(kws) > 0:
                func(**kws)
        if len(kwargs) > 0:
            raise TypeError('Invalid keyword argument(s): {}'.format(kwargs))
        self.state('ready')
                
    @in_state('plotting')
    def plot(self, x=None, y=None, z=None, ax=None, mode=None, fit=None, smooth=None, fit_kwargs=None, **kwargs):
        """Common interface for plotting data, whether 1d, 2d or 3d. 
        
        Type of plotting is controlled by mode."""
        if (x is None) and (y is None) and (z is None):
            self.call_if_args(None, kwargs)
            return  # No data to plot
        ax = self.ax(ax)
      
        if smooth is not None:
            raise NotImplementedError

        if mode is None:
            mode = self._get_modes(x, y, z)[0]  # take first compatible mode as default
        self._check_mode(x, y, z, mode)  # Check mode is compatible with supplied data
        if mode == 'pdf':
            kws = args_for(plot_pdf, kwargs, include=self.plot_args, remove=True)
            bin_edges, bin_centres, counts = plot_pdf(x, ax, **kws)
            self.return_values = bin_edges, bin_centres, counts
        elif mode == 'line':
            kws = args_for((plot_1d, matplotlib.lines.Line2D), kwargs, include=self.plot_args, remove=True)
            plot_1d(x, y, ax, **kws)
        elif mode == 'scatter':
            kws = args_for((scatter_1d, plt.scatter), kwargs, include=self.scatter_args, remove=True)
            scatter_1d(x, y, ax, **kws)
        elif mode == 'contourf':
            kws = args_for((contourf, plt.contourf), kwargs, remove=True)
            contourf(x, y, z, ax, **kws)
        elif mode in ('image', 'imshow'):
            kws = args_for((imshow, plt.imshow), kwargs, remove=True)
            imshow(ax, x, y, z, **kws)
        elif mode == 'scatter_3d':
            ax = self.convert_ax_to_3d(ax)
            kws = args_for((plot_scatter_3d, plt.scatter), kwargs, remove=True)
            plot_scatter_3d(ax, x, y, z, **kws)
        elif mode == 'surface_3d':
            ax = self.convert_ax_to_3d(ax)
            kws = args_for(plot_surface(), kwargs, remove=True)
            plot_surface(ax, x, y, z, **kws)
        else:
            raise NotImplementedError('Mode={}'.format(mode))

        if fit is not None:
            kws = {'envelope': False}
            kws.update(none_filter({}, fit_kwargs))
            f = Fitter(x, y).plot(ax=ax, data=False, show=False, **kws)

        self.call_if_args(ax, kwargs)
        return self

    def set_axis_labels(self, xlabel=None, ylabel=None, zlabel=None, label_fontsize=None, tick_fontsize=None,
                        ax=None, tight_layout=False):
        assert isinstance(xlabel, (string_types, type(None)))
        assert isinstance(ylabel, (string_types, type(None)))
        if ax == 'all':
            for ax in self.axes:
                self.set_axis_labels(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, label_fontsize=label_fontsize,
                                     tick_fontsize=tick_fontsize, tight_layout=tight_layout, ax=ax)
            return
        ax = self.ax(ax)
        if (xlabel is not None) or (label_fontsize is not None):
            ax.set_xlabel(xlabel if xlabel is not None else ax.get_xlabel(), fontsize=label_fontsize)
        if (ylabel is not None) or (label_fontsize is not None):
            ax.set_ylabel(ylabel if ylabel is not None else ax.get_ylabel(), fontsize=label_fontsize)
        if (zlabel is not None) or (label_fontsize is not None) and isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D):
            ax.set_zlabel(zlabel if zlabel is not None else ax.get_zlabel(), fontsize=label_fontsize)
        if tick_fontsize is not None:
            if not isinstance(tick_fontsize, (tuple, list)):
                tick_fontsize = (tick_fontsize, tick_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize[0])
            ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize[1])
        if tight_layout:
            self.fig.tight_layout()

    def set_axis_limits(self, xlim=None, ylim=None, ax=None):
        if ax == 'all':
            for ax in self.axes:
                self.set_axis_limits(xlim=xlim, ylim=ylim, ax=ax)
            return
        ax = self.ax(ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    def legend(self, ax=None, legend=True, legend_fontsize=14):
        """Finalise legends of each axes"""
        ax = none_filter(self._legend, ax)
        if ax == 'each axis':
            axes = self.axes.flatten()
        else:
            axes = [self.ax(ax)]
        for ax in axes:
            # TODO: check if more than one legend handels exist
            try:
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 1:  # Only produce legend if more than one artist has a label
                    leg = ax.legend(fontsize=legend_fontsize)
                    leg.draggable()
            except ValueError as e:
                #  https: // github.com / matplotlib / matplotlib / issues / 10053
                logger.error('Not sure how to avoid this error: {}'.format(e))
            if not legend:
                leg = ax.legend()
                leg.remove()

    @in_state('plotting')
    def plot_ellipses(self, ax=None, obj=None, convention='ellipse_axes', **kwargs):
        """Plot ellipses either by providing keyword arguments 'major, minor and angle' or by providing an Ellipses
        onject"""
        from ccfepyutils.geometry import Ellipses
        ax = self.ax(ax)
        if isinstance(obj, Ellipses):
            args = {}
            args['major'], args['minor'], args['angle'] = obj.get(convention, nested=True)
            args['x'], args['y'] = obj.position
            kwargs.update(args)
        kws = args_for(plot_ellipses, kwargs)
        plot_ellipses(ax, **kws)
        self.call_if_args(kwargs)

    @in_state('plotting')
    def plot_rectangle(self, x_lims, y_lims, fill=False, alpha=None, facecolor=None,
                    edgecolor=None, linewidth=None, linestyle='solid', ax=None, **kwargs):
        ax = self.ax(ax)
        plot_rectangles(ax, x_lims, y_lims, fill=fill, alpha=alpha, facecolor=facecolor,
                    edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, **kwargs)

    def convert_ax_to_3d(self, ax):
        ax = self.ax(ax)
        if isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D):
            return ax
        elif not isinstance(ax, matplotlib.axes._subplots.Axes):
            raise TypeError('ax must be axis instance')
        nx, ny = self._ax_shape[0], self._ax_shape[1]
        for i in np.arange(nx):
            axes_subset = self.axes[i] if nx > 1 else self.axes  # Extract row of axes
            for j, ax0 in enumerate(axes_subset):
                if ax0 is ax:
                    ax.remove()
                    # self.fig.axes.pop(self.fig.axes.index(ax))
                    ax = self.fig.add_subplot(nx, ny, i*ny+j+1, projection='3d')
                    axes_subset[j] = ax
                    self._gs_slices[(i, j)] = ax
                    if nx > 1:
                        # TODO: Fix!
                        self.axes[i] = axes_subset
                        # self.fig.axes = self.axes
                    else:
                        self.axes = to_array(self.fig.axes)
                    return axes_subset[j]
        raise RuntimeError("2D axis wasn't replaced with 3D one")

    def save(self, save=False, settings=None, prefix='', description=None,
             bbox_inches='tight', transparent=True, dpi=90):
        if save is False:  # Don't save!
            return
        elif pos_path(save):  # string path
            path_fn = save
        else:
            assert settings is not None
            raise NotImplementedError
        self.fig.savefig(path_fn, bbox_inches=bbox_inches, transparent=transparent, dpi=dpi)

    def save_image(self, z, fn, bit_depth=12):
        """Save image to file preserving resolution"""
        import scipy.misc
        scipy.misc.toimage(z, cmin=0.0, cmax=2 ** bit_depth).save(fn)  # Ensure preserve resolution and bit depth
        logger  # TODO logger

    def show(self, show=True, tight_layout=True, legend=True):
        if (not show) or (self.fig is None):
            return
        if legend:
            self.legend()
        if tight_layout:
            plt.tight_layout()
        plt.show()

    def to_plotly(self):
        """Convert figure to plotly figure"""
        import plotly
        import plotly.plotly as py
        import plotly.tools as tls
        # Converting to Plotly's Figure object..
        try:
            plotly_fig = tls.mpl_to_plotly(self.fig)
            py.iplot(plotly_fig, filename='Plot_obj_tmp')
            # raise NotImplementedError
        except plotly.exceptions.PlotlyEmptyDataError as e:
            logger.error('Could not convert mpl plot to plotly format')


# TODO: Move functions to separate file
def plot_1d(x, y, ax, ls=None, lw=None, alpha=None, **kwargs):
    # TODO: Add error bar plot functionality
    #
    if y is not None:
        ax.plot(x, y, ls=ls, lw=lw, alpha=alpha, **kwargs)
    else:
        ax.plot(x, ls=ls, lw=lw, alpha=alpha, **kwargs)

def plot_pdf(x, ax, label=None, nbins=None, bin_edges=None, min_data_per_bin=10, nbins_max=40, nbins_min=3,
        max_resolution=None, density=False, max_1=False, **kwargs):
    bin_edges, bin_centres, counts = pdf(x, nbins=nbins, bin_edges=bin_edges, min_data_per_bin=min_data_per_bin,
                                         nbins_max=nbins_max, nbins_min=nbins_min, max_resolution=max_resolution,
                                         density=density, max_1=max_1)
    ax.plot(bin_centres, counts, label=label, **kwargs)
    if ax.get_xlabel() == '':
        ax.set_xlabel(label)
    if ax.get_ylabel() == '':
        if density:
            y_label = 'Frequency density [N/A]'
        elif max_1:
            y_label = 'Normalised frequency [N/A]'
        else:
            y_label = 'Frequency [N/A]'
        ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0)
    return bin_edges, bin_centres, counts

def scatter_1d(x, y, ax, **kwargs):
    ax.scatter(x, y, **kwargs)

def contourf(x, y, z, ax, colorbar=True, cbar_label=None, levels=200, cmap='viridis', transpose=False, **kwargs):
    """ """
    assert not np.all([i is None for i in (x, y, z)])

    if (z is None) and (y is None) and (np.array(x).ndim == 2):  # if x supplied as z coord, swap them around
        x, z = z, x

    if transpose:
        z = z.T

    if np.ptp(z) == 0:  # If 
        v = z.flatten()[0]
        levels = [v, 0.1]

    # if (x is not None) and (y is not None):
    #     if (np.array(x).ndim == 1) and (np.array(y).ndim == 1):
    #         x, y = np.meshgrid(y, x)

    try:
        if not any(v is None  for v in (x, y, z)):
            im = ax.contour(x, y, z, levels, cmap=cmap, **kwargs)  # prevent white lines between contour fils
            im = ax.contourf(x, y, z, levels, cmap=cmap, **kwargs)
        else:
            im = ax.contour(z, levels, cmap=cmap, **kwargs)  # prevent white lines between contour fils
            im = ax.contourf(z, levels, cmap=cmap, **kwargs)
    except ValueError as e:
        logger.exception('Failed to plot contour. min(z)={}, max(z)={}'.format(np.min(z), np.max(z)))
        im = None

    # TODO: move to Plot class
    if colorbar and im:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fmt = '%.2g' if colorbar != '%' else '%.2%'
        if cbar_label is None:
            cbar_label = ''
        cbar = plt.colorbar(im, cax=cax, format=fmt)
        cbar.set_label(cbar_label)

def imshow(ax, x=None, y=None, z=None, origin='lower', interpolation='none', cmap='viridis', set_axis_limits=False,
           show_axes=False, fil_canvas=False, transpose=False, **kwargs):
    """Plot data as 2d image

    Note:
    - aspect = 'auto' useful for extreeme aspect ratio data
    - transpose = True useful since imshow and contourf expect arrays ordered (y, x)
"""
    if (x is not None) and (y is None) and (z is None):  # if 2d data passed to x treat x as z data
        x, y, z = y, z, x
    if x is None and y is None:
        ax.set_axis_off()  # Don't display axes and labels
        if len(ax.figure.axes) == 1:
            ax.figure.subplots_adjust(0, 0, 1, 1)  # maximise figure margins so image fills full canvas
    else:
        kwargs.update({'extent': (np.min(x), np.max(x), np.min(y), np.max(y))})
    if transpose:
        z = np.array(z).T
    if fil_canvas:
        ax.figure.subplots_adjust(0, 0, 1, 1)  # maximise figure margins so image fills full canvas
    if set_axis_limits:
        if x is None and y is None:
            ax.set_xlim(-0.5, z.shape[1] - 0.5)
            ax.set_ylim(z.shape[0] - 0.5, -0.5)
        else:
            ax.set_xlim(np.min(x), np.max(x))  # TODO: Handle half pixel offset properly...
            ax.set_ylim(np.min(y), np.max(y))
    if show_axes:
        ax.set_axes_on()
    else:
        ax.set_axis_off()
    # if aspect:
    #     ax.set_aspect(aspect)
    img = ax.imshow(z, cmap=cmap, interpolation=interpolation, origin=origin, **kwargs)
    return img

def plot_2d(self, z, x, y, ax, raw=False, show=True, save=False, annotate=True,
         path='~/elzar/images/frames/', extension='.png',
         contour=False, colorbar=True,
         transpose=False, facecolor=(242/255,241/255,239/255), **kwargs):
    if ax is None:
        fig = plt.figure(repr(self), facecolor=facecolor, edgecolor=facecolor)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if raw:  # raw frame data
        data = self.raw
    else:  # enhanced frame data
        data = self.frame

    if not contour:
        img = ax.imshow(data, cmap='gray', interpolation='none', **kwargs)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)  # maximise figure margins so image fills full canvas
        ax.set_xlim(-0.5, data.shape[1] - 0.5)
        ax.set_ylim(data.shape[0] - 0.5, -0.5)
    else:
        if not transpose:
            im = ax.contourf(np.array(self.df.columns), np.array(self.df.index), data, 200, **kwargs)
        else:
            # Doesn't work....?
            z = data.T
            # labels = {'R': 'a', 'tor': 'b', 'dR': 'c', 'dtor': 'd'}
            # for key in labels.keys():
            #     if isinstance(save, basestring) and ('-'+key in save):
            #         ax.text(0.05, 0.95, labels[key], transform=ax.transAxes, ha='left', va='top',
            #         fontsize=60, color='k')
            if contour == 'cm':  # m -> cm
                z = z*1e2
            cmap = 'viridis'
            if np.all(z<0):
                cmap += '_r'
            im = ax.contour(np.array(self.df.index), np.array(self.df.columns), z, 200, cmap=cmap, **kwargs) # prevent white lines between contour fils
            im = ax.contourf(np.array(self.df.index), np.array(self.df.columns), z, 200, cmap=cmap, **kwargs)
            ax.set_xlabel('$R$ [m]')
            ax.set_ylabel('$\phi R$ [m]')
        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fmt =  '%.2g' if colorbar != '%' else '%.2%'
            if contour == 'cm':
                label = 'Error [cm]'
            elif contour == '%':
                label = 'Fractional error'
            else:
                label = ''
            cbar = plt.colorbar(im, cax=cax, format=fmt)
            cbar.set_label(label)

    if annotate:
        n_str = 'Frame: {:d}'.format(self.n) if self.n is not None else ''
        t_str = '  Time: {:0.5f} [s]'.format(self.t) if self.t is not None else ''
        text = n_str + t_str
        frametxt = ax.annotate(text, xy=(0.05,0.95), xycoords='axes fraction', color='white', fontsize=8)
        frametxt.set_bbox(dict(color='k', alpha=0.5, edgecolor=None))
    # plt.tight_layout()
    if save:
        fn = self.frame_history.out_name(n=self.n, prefix='raw_'*raw+'frame',
                                                   dtype=True, resolution=False, ident=False, enhancements=True)
        fn = out_path(save, path, fn, extension)

        # if not isinstance(save, basestring):
        #     if self.frame_history is not None:  # TODO: remove Elzar path hardwiring...
        #         try:
        #             save = self.frame_history.out_name(n=self.n, prefix='frame', extension=extension,
        #                                            dtype=True, resolution=False, ident=False, enhancements=True)
        #         except AttributeError as e:
        #             save = 'frame_tmp'
        #     assert isinstance(save, basestring)
        #     # fn = 'frame-'+self.name_out+'.png'
        #     # save = os.path.join(self.image_path, 'frames', fn)
        # save = os.path.join(os.path.expanduser('~/elzar'), 'images', 'error_maps', save)
        if annotate or contour:
            fig.savefig(fn, bbox_inches='tight', transparent=True, dpi=90)
        else:
            import scipy.misc
            scipy.misc.toimage(data, cmin=0.0, cmax=2**12).save(fn)  # Ensure preserve resolution and bit depth
            # plt.imsave(fn, data, cmap='gray')
        # print('Saved enhanced frame data (same resolution) to: '+fn)

        # consider plt.saveim()
        logger.info('Frame image saved to: '+fn)
    if show:
        plt.show()
    return ax

def plot_scatter_3d(ax, x, y, z, **kwargs):
    points = ax.scatter(x, y, z, **kwargs)

def plot_surface(ax, x, y, z, cmap='viridis', colorbar=True, **kwargs):
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z, cmap=cmap, **kwargs)
    if colorbar:  # TODO: Move to Plot object?
        ax.figure.colorbar(surf, shrink=0.5, aspect=5, pad=0.015)


def plot_ellipses(ax, major, minor, angle, x=None, y=None, a=None, a_lims=None, color='k', lw=1, ls='-', alpha=0.7,
                  label=False, half_widths=False, scale_factor=1, **kwargs):
    """Plot ellipses (2d outlines) with centres marked
    :param ax: axis to plot to
    :param major: ellipse major diameter
    :param minor: ellipse minor diameter
    :param angle: ellipse tilt angle anticlockwise between major axis and x axis
    :param x: ellipse centre x coords
    :param y: ellipse centre y coords
    :param a: amplitude to scale size of ellipse centre dot with
    :param a_lims: amplitude limits
    :param color: shape color
    :param lw: shape line widths
    :param ls: shape line style
    :param alpha: transparency
    :param label: add numerical labels to ellipse centres (bool)
    :param half_widths: if true double input major and minor to get diameters
    :param scale_factor: value to scale ellipse dimensions by (eg 2 will double widths) - can be a list
    :param kwargs: keyword arguments to pass to Ellipse patch object
    :return:
    """
    from matplotlib.patches import Ellipse

    if x is None:
        x = np.zeros_like(major)
    if y is None:
        y = np.zeros_like(major)

    # If half widths passed to function multiply by two for full widths wanted by matplotlib
    if half_widths:
        major = 2 * copy(major)
        minor = 2 * copy(minor)

    x, y, major, minor, angle = make_iterables(x, y, major, minor, angle)

    for i, (x0, y0, major0, minor0, angle0) in enumerate(list(zip(x, y, major, minor, angle))):
        # Matplotlib ellipse takes:
        #  width:  total length (diameter) of horizontal axis
        #  height: total length (diameter) of vertical axis
        #  angle:  rotation in degrees (anti-clockwise)

        # Scale ellipse size
        scale_factor = to_array(scale_factor)
        for scale, alpha0 in zip(scale_factor, np.linspace(1, 0.3, len(scale_factor))):
            major0i = scale * major0
            minor0i = scale * minor0
            alpha0i = alpha * alpha0

            ax.add_patch(Ellipse((x0, y0), major0i, minor0i, angle=angle0,
                                  facecolor='none', edgecolor=color, lw=lw, alpha=alpha0, ls=ls, zorder=2, **kwargs))

        if label:
            plt.text(x0, y0, '{}'.format(i + 1), ha="center", family='sans-serif', size=10, color=color)
        # Mark centres of ellipses with dots
        if a is None:
            ax.plot(x0, y0, '.', color=color, ms=3, lw=0.3, alpha=0.6)
        else:  # Scale point size by intensity
            a_lims = [np.min(a), np.max(a)] if a_lims is None else a_lims
            a -= a_lims[0]
            a /= (a_lims[1] - a_lims[0])  # Normalise amplitudes
            min_size = 4
            max_size = 25
            ax.scatter(x, y, s=np.array(min_size + a * (max_size - min_size)), c=color, lw=0, alpha=0.6)

def plot_rectangles(ax, x_lims, y_lims, fill=False, alpha=None, facecolor=None,
                    edgecolor=None, linewidth=None, linestyle='solid',
                    **kwargs):
    """Plot rectangle on axis.
    :param x_lims: x limits of rectangle to plot or array from which min and max limits will be taken
    :param y_lims: y limits of rectangle to plot or array from which min and max limits will be taken
    """
    import matplotlib.patches as patches
    x1, x2 = np.min(x_lims), np.max(x_lims)
    y1, y2 = np.min(y_lims), np.max(y_lims)
    rectangle = patches.Rectangle((x1, y1),  # (x,y)
                                  x2-x1,  # width
                                  y1-y1,  # height
                                  fill=fill, alpha=alpha, facecolor=facecolor,
                                  edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle,
                                  **kwargs)
    ax.add_patch(rectangle)

if __name__ == '__main__':
    x = np.linspace(-10, 10, 41)
    y = np.sin(x)
    z = np.ones((41, 41)) * y

    # plot = Plot(x, y, label='test_tmp 1', axes=(2,1))
    # plot.plot(x, -y, ax=1, label='test_tmp 2')

    plot = Plot(x, y, z)

    plot.show()