#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from copy import copy, deepcopy
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

from ccfepyutils.utils import make_itterable, args_for, to_array
from ccfepyutils.classes.state import State, in_state
from ccfepyutils.classes.fitter import Fitter
try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

class Plot(object):
    """Convenience plotting class"""
    # Plotting modes compatible with different input data dimensions. The first mode for each dimension is the default
    dim_modes = dict((((1, None, None), ('line', 'scatter', 'pdf')),
                      ((1, 1 , None), ('line', 'scatter')),
                      # ((1, 1 , 1), ('scatter3D', 'line3D', 'segline')),
                      ((2, 2 , 2), ('surface3D')),
                      ((1, 1, 2), ('contourf', 'contour', 'image', 'surface3D')),
                      ((2, None, None), ('contourf', 'contour', 'image')),
                      # ((None, None , 2), ('image', 'contour', 'contourf'))
                      ))
    instances = []  # List of all Plot instances
    state_table = {'init': ['ready', 'plotting'],
                   'ready': ['plotting'],
                   'plotting': ['ready'],
                   }
    plot_args = ['ls', 'lw', 'c', 'color', 'marker', 'label']  # TODO: extend plot_args
    scatter_args = ['s', 'c', 'color', 'marker', 'label']  # TODO: extend plot_args

    def __init__(self, x=None, y=None, z=None, num=None, axes=(1,1), default_ax=1, ax=None, mode=None,
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
        self.call_table = {'ready': {'enter': self.set_ready}
                      }
        self.state = State(self, self.state_table, 'init', call_table=self.call_table)

        self._num = num  # Name of figure window
        self._ax_shape = np.array(axes)  # Shape of axes grid
        self._default_ax = default_ax  # Default axis for actions when no axis is specified
        self._current_ax = ax
        self._data = OrderedDict()  # Data used in plots stored for provenance
        self._log = []  # Log of method calls for provenance
        self._legend = legend

        self.fig = None
        self.axes = None
        self.instances.append(self)
        self.make_figure(num, axes, **fig_args)  # TODO: Add axis naming? .set_axis_names - replace defaults?
        # kws = args_for(self.plot, kwargs, include=self.plot_args)
        self.plot(x, y, z, mode=mode, **kwargs)
        self.show(show)
        self.save(save)

    def make_figure(self, num, axes, **kwargs):
        assert isinstance(num, (string_types, str, int, type(None)))
        assert isinstance(axes, (tuple, list))
        self.fig, self.axes = plt.subplots(num=num, *axes, **kwargs)
        self.axes = to_array(self.axes)

    def set_ready(self):
        """Set plot object in 'ready' state by finalising/tidying up plot actions"""
        self._current_ax = None

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
                return None
        if dim in self.dim_modes:
            return self.dim_modes[dim]
        else:
            raise ValueError('Dimensions of input data {} are not amongst those supported: {}'.format(dim, self.dim_modes))

    def _name2ax(self, ax):
        if isinstance(ax, int):
            assert ax <= self.axes.size
            ax = self.axes.flatten()[ax-1]  # TODO: Make compatible with 2d grid of axes
        elif isinstance(ax, string_types):
            raise NotImplementedError
        elif isinstance(ax, (tuple, list)):
            ax = self.axes[self._ax_shape-1]
        elif isinstance(ax, matplotlib.axes.Axes):  # TODO: improve this!
            ax = ax  # already an axis instance
        else:
            raise TypeError
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

    def ax(self, ax=None):
        """Return axis instance."""
        if ax is None:
            if self._current_ax is not None:
                ax = self._current_ax
            else:
                ax = self._default_ax
        elif isinstance(ax, matplotlib.axes.Axes):
            ax = ax
        elif isinstance(ax, string_types+(int, tuple, list)):
            ax = ax
        ax = self._name2ax(ax)  # Convert axis name to axis instance
        self._current_ax = ax
        try:
            plt.sca(ax)  # Set as current axis for plt. calls
        except ValueError:
            logger.exception('Axis problem!')
        return ax

    def _use_data(self, x, y, z, mode):
        """Inspect supplied data to see whether further actions should be taken with it"""
        raise NotImplementedError

    def call_if_args(self, kwargs, raise_on_exception=True):
        for func in (self.set_axis_labels, self.show):
            kws = args_for(func, kwargs, remove=True)
            if len(kws) > 0:
                func(**kws)
        if len(kwargs) > 0:
            raise TypeError('Invalid keyword argument(s): {}'.format(kwargs))
        self.state('ready')
                
    @in_state('plotting')
    def plot(self, x=None, y=None, z=None, ax=None, mode=None, fit=None, smooth=None, **kwargs):
        """Common interface for plotting data, whether 1d, 2d or 3d. 
        
        Type of plotting is controlled by mode."""
        ax = self.ax(ax)
      
        if smooth is not None:
            raise NotImplementedError

        if mode is None:
            mode = self._get_modes(x, y, z)[0]  # take first compatible mode as default
        self._check_mode(x, y, z, mode)  # Check mode is compatible with supplied data
        if mode == 'line':
            kws = args_for(plot_1d, kwargs, include=self.plot_args, remove=True)
            plot_1d(x, y, ax, **kws)
        elif mode == 'scatter':
            kws = args_for(scatter_1d, kwargs, include=self.scatter_args, remove=True)
            scatter_1d(x, y, ax, **kws)
        elif mode == 'contourf':
            kws = args_for(contourf, kwargs, remove=True)
            contourf(x, y, z, ax, **kws)
        elif mode == 'surface3D':
            ax = self.convert_ax_to_3d(ax)
            plot3d_surface(ax, x, y, z)
        else:
            raise NotImplementedError('Mode={}'.format(mode))

        if fit is not None:
            kws = args_for(Fitter.plot, kwargs, remove=True)
            f = Fitter(x, y).plot(ax=ax, data=False, envelope=False, show=False, **kws)

        self.call_if_args(kwargs)

    def set_axis_labels(self, xlabel, ylabel, ax=None):
        assert isinstance(xlabel, string_types)
        assert isinstance(ylabel, string_types)
        if ax == 'all':
            for ax in self.axes:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        else:
            ax = self.ax(ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)


    def legend(self):
        """Finalise legends of each axes"""
        if self._legend == 'each axis':
            for ax in self.axes.flatten():
                # TODO: check if more than one legend handels exist
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 1:  # Only produce legend if more than one artist has a label
                    leg = ax.legend()
                    leg.draggable()

    def plot_ellipses(self, ax=None, obj=None, convention='ellipse_axes', **kwargs):
        """Plot ellipses either by providing keyword arguments 'major, minor and angle' or by providing an Ellipses
        onject"""
        self.state('plotting')
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
        self.state('ready')

    def convert_ax_to_3d(self, ax):
        ax = self.ax(ax)
        nx, ny = self._ax_shape[0], self._ax_shape[1]
        for i in np.arange(nx):
            axes_subset = self.axes[i] if nx > 1 else self.axes  # Extract row of axes
            for j, ax0 in enumerate(axes_subset):
                if ax0 is ax:
                    ax.remove()
                    axes_subset[j] = self.fig.add_subplot(nx, ny, i*ny+j+1, projection='3d')
                    return axes_subset[j]

    def save(self, save=False):
        if save:
            raise NotImplementedError

    def save_image(self, z, fn, bit_depth=12):
        """Save image to file preserving resolution"""
        import scipy.misc
        scipy.misc.toimage(z, cmin=0.0, cmax=2 ** bit_depth).save(fn)  # Ensure preserve resolution and bit depth
        logger  # TODO logger

    def show(self, show=True, tight_layout=True):
        if self.fig is None:
            return
        self.legend()
        if tight_layout:
            plt.tight_layout()
        if show:
            plt.show()


# TODO: Move functions to separate file
def plot_1d(x, y, ax, **kwargs):
    # TODO: Add error bar plot functionality
    #
    ax.plot(x, y, **kwargs)

def scatter_1d(x, y, ax, **kwargs):
    ax.scatter(x, y, **kwargs)

def contourf(x, y, z, ax, colorbar=True, cbar_label=None, levels=200, cmap='viridis', transpose=False, **kwargs):
    """ """
    assert not np.all([i is None for i in (x, y, z)])

    if (z is None) and (y is None) and (np.array(x).ndim == 2):  # if x supplied as z coord, swap them around
        x, z = z, x

    if transpose:
        z = z.T

    if not None in (x, y, z):
        im = ax.contour(x, y, z, levels, cmap=cmap, **kwargs)  # prevent white lines between contour fils
        im = ax.contourf(x, y, z, levels, cmap=cmap, **kwargs)
    else:
        im = ax.contour(z, levels, cmap=cmap, **kwargs)  # prevent white lines between contour fils
        im = ax.contourf(z, levels, cmap=cmap, **kwargs)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fmt = '%.2g' if colorbar != '%' else '%.2%'
        if cbar_label is None:
            cbar_label = ''
        cbar = plt.colorbar(im, cax=cax, format=fmt)
        cbar.set_label(cbar_label)

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

def plot3d_surface(ax, x, y, z, cmap='viridis', colorbar=True, **kwargs):
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z, cmap=cmap, **kwargs)
    if colorbar:  # TODO: Move to Plot object?
        ax.figure.colorbar(surf, shrink=0.5, aspect=5, pad=0.015)


def plot_ellipses(ax, major, minor, angle, x=None, y=None, a=None, a_lims=None, color='k', lw=1, ls='-', alpha=0.7,
                  label=False, half_widths=False, **kwargs):
    from matplotlib.patches import Ellipse

    if x is None:
        x = np.zeros_like(major)
    if y is None:
        y = np.zeros_like(major)

    # If half widths passed to function multiply by two for full widths wanted by matplotlib
    if half_widths:
        major = 2 * copy(major)
        minor = 2 * copy(minor)

    for i, (x0, y0, major0, minor0, angle0) in enumerate(list(zip(x, y, major, minor, angle))):
        # Matplotlib ellipse takes:
        #  width:  total length (diameter) of horizontal axis
        #  height: total length (diameter) of vertical axis
        #  angle:  rotation in degrees (anti-clockwise)
        ax.add_patch(Ellipse((x0, y0), major0, minor0, angle=angle0,
                              facecolor='none', edgecolor=color, lw=lw, alpha=alpha, ls=ls, zorder=2, **kwargs))
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

if __name__ == '__main__':
    x = np.linspace(-10, 10, 41)
    y = np.sin(x)
    z = np.ones((41, 41)) * y

    # plot = Plot(x, y, label='test 1', axes=(2,1))
    # plot.plot(x, -y, ax=1, label='test 2')

    plot = Plot(x, y, z)

    plot.show()