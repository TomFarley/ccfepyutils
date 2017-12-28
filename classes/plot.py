#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
from logging.config import fileConfig, dictConfig
fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)

from ccfepyutils.utils import make_itterable


class Plot(object):
    """Convenience plotting class"""
    # Plotting modes compatible with different input data dimensions. The first mode for each dimension is the default
    dim_modes = dict((((1, None, None), ('line', 'scatter', 'pdf')),
                      ((1, 1 , None), ('line', 'scatter')),
                      # ((1, 1 , 1), ('scatter3D', 'line3D', 'segline')),
                      ((1, 1 , 2), ('contourf', 'contour', 'image')),
                      # ((None, None , 2), ('image', 'contour', 'contourf'))
                      ))

    def __init__(self, x=None, y=None, z=None, num=None, axes=(1,1), current_ax=0, mode=None, legend='each axis',
                 save=False, show=False, fig_args={}, **kwargs):
        self._num = num  # Name of figure window
        self._ax_shape = axes  # Shape of axes grid
        self._current_ax = current_ax  # Default axis for actions when no axis is specified
        self._data = OrderedDict()  # Data used in plots stored for provenance
        self._log = []  # Log of method calls for provenance
        self._legend = legend

        self.fig = None
        self.axes = None
        self.make_figure(num, axes, **fig_args)
        self.plot(x, y, z, mode=mode, **kwargs)
        self.show(show)

    def make_figure(self, num, axes, **kwargs):
        assert isinstance(num, (str, int, type(None)))
        assert isinstance(axes, (tuple, list))
        self.fig, self.axes = plt.subplots(num=num, *axes, **kwargs)
        self.axes = make_itterable(self.axes)


    def _get_modes(self, x, y, z):
        """Inspect supplied data to see whether further actions should be taken with it"""
        data = {k: np.array(v) for k, v in zip(['x', 'y', 'z'], [x, y, z]) if v is not None}  # data that is not None
        shapes = {k: v.shape for k, v in data.items()}
        dims = {k: v.ndim for k, v in data.items()}
        dim = tuple((np.array(v).ndim if v is not None else None for v in (x, y, z)))
        if len(data) == 0:
            return None
        if dim in self.dim_modes:
            return self.dim_modes[dim]
        else:
            raise ValueError('Dimensions of input data {} are not amongst those supported: {}'.format(dim, self.dim_modes))

    def _name2ax(self, ax):
        if isinstance(ax, int):
            return self.axes[ax]
        elif isinstance(ax, str):
            raise NotImplementedError

    def _check_mode(self, x, y, z, mode):
        assert isinstance(mode, str)
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
            ax = self._current_ax
        elif isinstance(ax, type(self.axes[0])):  # tmp TODO: fix this!
            return ax
        return self._name2ax(ax)

    def _use_data(self, x, y, z, mode):
        """Inspect supplied data to see whether further actions should be taken with it"""
        raise NotImplementedError


    def plot(self, x=None, y=None, z=None, ax=None, mode=None, **kwargs):
        if mode is None:
            mode = self._get_modes(x, y, z)[0]  # take first compatible mode as default
        self._check_mode(x, y, z, mode)
        ax = self.ax(ax)
        if mode == 'line':
            plot_1d(x, y, ax, **kwargs)
        elif mode == 'contourf':
            contourf(x, y, z, ax, **kwargs)
        # raise NotImplementedError

    def set_axis_labels(self, xlabel, ylabel, ax=None):
        assert isinstance(xlabel, str)
        assert isinstance(ylabel, str)
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
            for ax in self.axes:
                # TODO: check if more than one legend handels exist
                leg = ax.legend()
                leg.draggable()

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


def plot_1d(x, y, ax, **kwargs):
    ax.plot(x, y, **kwargs)

def contourf(x, y, z, ax, colorbar=True, cbar_label=None, levels=200, cmap='viridis', transpose=False, **kwargs):
    """ """
    if transpose:
        z = z.T

    im = ax.contour(x, y, z, levels, cmap=cmap, **kwargs)  # prevent white lines between contour fils
    im = ax.contourf(x, y, z, levels, cmap=cmap, **kwargs)

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

if __name__ == '__main__':
    x = np.linspace(-10, 10, 41)
    y = np.sin(x)
    z = np.ones((41, 41)) * y

    # plot = Plot(x, y, label='test 1', axes=(2,1))
    # plot.plot(x, -y, ax=1, label='test 2')

    plot = Plot(x, y, z)

    plot.show()