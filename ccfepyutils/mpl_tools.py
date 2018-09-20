## Function from http://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib

import logging, warnings, os, itertools

import numpy as np
from matplotlib.collections import LineCollection

from ccfepyutils.io_tools import mkdir
from ccfepyutils.utils import make_iterable, safe_len

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_matplotlib_backend(use_non_visual_backend, non_visual_backend='Agg', visual_backend='Qt5Agg'):
    import matplotlib
    # TODO: Supress backend warnings
    if use_non_visual_backend:
        matplotlib.use(non_visual_backend)
        logger.info('Using non-visual backend')
    else:
        matplotlib.use(visual_backend)
        # matplotlib.use('Qt4Agg')
        pass

from ccfepyutils import batch_mode
set_matplotlib_backend(batch_mode, non_visual_backend='Agg', visual_backend='Qt5Agg')
import matplotlib
import matplotlib.pyplot as plt
colormap_names = sorted(m for m in plt.cm.datad if not m.endswith("_r"))

def get_fig_ax(ax=None, num=None, subplot=111, **fig_kwargs):
    """Return new/existing figure and axis instances"""
    if ax is None:
        fig = plt.figure(num=num, **fig_kwargs)
        ax = fig.add_subplot(subplot)
    else:
        fig = ax.figure
    return fig, ax

def seg_line_plot(x, y, z, ax=None, z_out='color', in_range=None, out_range=None, fig=None, color='b', lw = 2, ls='-',
                  cmap='Spectral_r', label=None, alpha=1):
    """ Colour line plot. Plot 2D line with z value represented by colour of points on line
    """
    if ax is None:
        ax = plt.gca()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Scale z values to [0,1] over z_range
    if in_range is None:  # scale z values linearly between 0 and 1 over full range of z values
        if np.min(z) < 0:
            z += np.abs(np.min(z))
        z_scaled = (z.astype(float) - z.min()) / (np.max(z) - np.min(z))
    else:  # scale z linearly so that inp_range[0] becomes 0 and inp_range[1] becomes 1
        z_scaled = (z.astype(float) - in_range[0]) / (in_range[1] - in_range[0])

    if out_range is not None:
        out_range = [np.min(out_range), np.max(out_range)]
        # Scale values to range [z_range]
        z_scaled *= (out_range[1] - out_range[0])
        z_scaled += out_range[0]

    if 'color' in z_out:
        lc = LineCollection(segments, cmap=plt.get_cmap(cmap))#, norm=plt.Normalize(250, 1500))
        lc.set_array(z_scaled)
    else:
        lc = LineCollection(segments, cmap=None)#, norm=plt.Normalize(250, 1500))
        lc.set_color(color)

    if 'linewidth' in z_out:
        # Scale z values between min and max linewidths
        lw = z_scaled
    if 'alpha' in z_out:
        from matplotlib import colors as mcolors  # Need uptodate matplotlib
        color = mcolors.to_rgba_array(color)
        color = np.repeat(color, len(z), axis=0)
        color[:,3] = z_scaled  # set alpha to z
        lc.set_color(color)
    else:
        lc.set_alpha(alpha)

    lc.set_linewidth(lw)
    lc.set_linestyle(ls)
    # elif z_out=='color+linewidth':
    #     lc = LineCollection(segments, cmap=plt.get_cmap(cmap))#, norm=plt.Normalize(250, 1500))
    #     lc.set_array(z_scaled)
    #     # Scale z values between min and max linewidths
    #     linewidth = lw_range[0] + lw_range[1] * z_scaled
    #     lc.set_linewidth(linewidth)
    if 'color' not in z_out and 'linewidth' not in z_out and 'alpha' not in z_out:
        print('WARNING: {} not recognised. Aborting seg_line_plot.'.format(z_out))

    ax.add_collection(lc)

    ## Colorbar for line
    if fig and 'color' in z_out:
        axcb = fig.colorbar(lc, shrink=1.0, pad=0.02)#, format='%.0f')
        if label: axcb.set_label('cota (m)')
        plt.tight_layout()
    if label is not None:
        lc.set_label(label)
    return lc

def arrowplot(axes, x, y, narrs=30, dspace=0.5, direc='pos', \
                          hl=0.3, hw=6, c='black', alpha=1):
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace)
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]:
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2. + ax
        dy0 = np.cos(theta)*hl/2. + ay
        dx1 = -1.*np.sin(theta)*hl/2. + ax
        dy1 = -1.*np.cos(theta)*hl/2. + ay

        if direc is 'neg' :
          ax0 = dx0
          ay0 = dy0
          ax1 = dx1
          ay1 = dy1
        else:
          ax0 = dx1
          ay0 = dy1
          ax1 = dx0
          ay1 = dy0

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                        xytext=(ax1, ay1), textcoords='data',
                        arrowprops=dict( headwidth=hw, frac=1., ec=c, fc=c,alpha=alpha),
                        alpha=alpha)

    axes.plot(x, y, pix_color= c, alpha=alpha)
    # axes.set_xlim(x.min()*.9,x.max()*1.1)
    # axes.set_ylim(y.min()*.9,y.max()*1.1)

def scatter_fit(x, y, fit='linear', label=None, color='blue', ax=None, x_label=None, y_label=None, fit_only=False,
                remove_outliers=False, outlier_thresh=0.7, y_eq_x=False, **kwargs):
    """ Plot scatter with fit curve
    """
    if remove_outliers:
        try:
            import statsmodels.api as smapi
            from statsmodels.formula.api import ols
            import statsmodels.graphics as smgraphics
        except:
            remove_outliers = False
    if len(x) < 2 or len(y) < 2:
        logger.warning('Not enough data for scatter_fit. Lengths: x:{}, y:{}'.format(len(x), len(y)))
        return
    if ax is None:
        ax = plt.gca()

    color_pairs = {'blue': 'darkblue', 'green': 'darkgreen', 'cornflowerblue': 'royalblue'}
    if not fit_only:
        ax.scatter(x, y, label=label, alpha=0.7, s=6, color=color)

    color_fit = color_pairs[color] if color in color_pairs.keys() else color
    if fit == 'linear':
        from elzar.tools.utils import correlation
        if y_eq_x:
            x_sort = sorted(x)
            ax.plot(x_sort, x_sort, ls='--', color='k', alpha=0.6, label='$y = x$')
        if not remove_outliers:  # normal least squares fit
            pear_r, (m, c) = correlation(x, y)
        else:
            regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
            test = regression.outlier_test()
            mask = (test.iloc[:, 2] < outlier_thresh).values
            outliers = (x[mask], y[mask])
            inliers = (x[~mask], y[~mask])
            pear_r, (m, c) = correlation(inliers[0], inliers[1])
            ax.scatter(outliers[0], outliers[1], label='Outliers', alpha=0.8, s=8, color='r', marker='x')
        ax.plot(x, x * m + c, color=color_fit, label="y = {:0.3g}x + {:0.3g}, r = {:0.2f}".format(m, c, pear_r),
                **kwargs)

    elif fit == 'moving_av':
        from elzar.tools.utils import moving_average
        iorder = np.argsort(x)
        x = x[iorder]
        y = y[iorder]
        width = np.max([int(0.15 * len(y)), 1])  # average over 15% of range
        y2 = moving_average(y, width)
        x2 = x[int(width/2): int(width/2) + len(y2)]
        plt.plot(x2, y2, color=color_fit, label="Moving average ({}/{})".format(width, len(y)), alpha=0.8, **kwargs)
    plt.autoscale(enable=True)
    if x_label is not None and y_label is not None:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

def plot_density_contour(xy, xy0=False, nxny=(10,10), levels='range', points=True, contours=None, cbar_sf=2,
                         cbar_range=[None, None], hlines=None, leg_title=None, normed=True,
                         ax=None, show=False, save=False):
    """Produce 2d density contour plot normalised to second data set.
    Inputs:
    xy      -  (x, y) values of points
    xy0     -  (x, y) values of points to normalise to
    nxny    -  number of bins in each dimension
    levels  -  number of contour levels
    points  -  flag for whether to plot points as well as contours
    """
    from elzar.tools.utils import moving_average, to_array
    x, y = xy
    x, y = to_array(x), to_array(y)
    x0, y0 = xy0 if xy0 is not False else [[], []]
    x0, y0 = to_array(x0), to_array(y0)
    nx, ny = nxny
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate common bins for both datasets
    x_range = (np.min(x), np.max(x))
    y_range = (np.min(y), np.max(y))
    x_bins = np.linspace(x_range[0], x_range[1], nx)
    y_bins = np.linspace(y_range[0], y_range[1], ny)
    bins = (x_bins, y_bins)
    try:
        counts, xbins, ybins = np.histogram2d(x, y, bins=bins)
    except ValueError as e:
        logger.warning('Unable to produce 2d histogram of points. Returning.')
        return

    if xy0 is not False:  # normalise to second dataset
        counts0, xbins, ybins = np.histogram2d(x0, y0, bins=(xbins, ybins))
        if normed:  # Normalise to xy0
            counts /= np.max(counts)
            counts0 /= np.max(counts0)

        with warnings.catch_warnings():   # silence RuntimeWarning: divide by zero encountered in divide
            warnings.simplefilter('ignore')
            counts /= counts0  # 0 / 0 = nan
        counts[np.isnan(counts)] = 0  # 2 * np.nanmax(counts[np.isfinite(counts)])
        counts[np.isinf(counts)] = np.nan
        # counts[np.isinf(counts)] = 1.1 * np.nanmax(counts[np.isfinite(counts)])
        levels = 200 if levels == 'range' else levels
        if cbar_range != [None, None]:
            levels = np.linspace(cbar_range[0], cbar_range[1], levels)
        else:
            levels = np.linspace(np.min(counts), np.max(counts), levels)
            # print(levels)
            # print(np.min(counts), np.max(counts), levels)
    else:
        levels = int(np.max(counts) - np.min(counts) +2) if levels == 'range' else levels
        levels = np.linspace(np.min(counts)-1, np.max(counts), levels)
    x_centre, y_centre = moving_average(xbins, 2), moving_average(ybins, 2)
    cax = ax.contourf(x_centre, y_centre, counts.transpose(), levels)#
                # extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                # linewidths=3, colors='black', linestyles='solid')
    cbar = plt.colorbar(cax, format='%.{}g'.format(cbar_sf))#, vmin=cbar_range[0], vmax=cbar_range[1])
    if contours:
        # cont = ax.contour(x_centre, y_centre, counts.transpose(), contours, colors='k')
        cont = ax.contour(cax, levels=contours, colors='k')
        plt.clabel(cont, fontsize=9, inline=1, fmt='%1.2f')  # numeric labels for contours
        cbar.add_lines(cont)
    if hlines:
        ax.axvline(x=(hlines), ls='--', linewidth=1, color='k', alpha=0.8)
    if points:
        # Plot indicidual points contributing to distribution
        if xy0:
            ax.plot(x0, y0, 'o', color='red', ms=1, alpha=0.6, label='Normalising points ({})'.format(len(x0)))
        ax.plot(x, y, 'o', color='green', ms=1.5, alpha=0.6, label='Points ({})'.format(len(x)))
        if xy0:
            leg = ax.legend(loc='best', title=leg_title, fancybox=True, fontsize=8)
            leg.get_frame().set_alpha(0.6)
    if save:
        plt.savefig(save, bbox_inches='tight', transparent=True, dpi=90)
    if show:
        plt.show()

def set_cycler(properties, ax='current'):
    """Set property cycler for given axis
    Combine properties that change together in a signle dictionary and put properties that should be returned as a
    product in separate dictionaries. Color can be passed as a tuple of (colormap_name, n_colors) which will split the
    colormap range into n_colors and cycle over them.

    Examples:
        properties =
            [{'color': ('r', 'g', 'b'), 'linestyle': ['-', '--', '-.']},]
        -->
        list(cycler_new) =
            {'color': 'r', 'linestyle': '-'}
            {'color': 'g', 'linestyle': '--'}
            {'color': 'b', 'linestyle': '-.'}

        properties =
            [{'color': ('r', 'g', 'b')}, {'linestyle': ['-', '--', '-.']}]
        -->
        list(cycler_new) =
            {'color': 'r', 'linestyle': '-'}
            {'color': 'r', 'linestyle': '--'}
            {'color': 'r', 'linestyle': '-.'}
            {'color': 'g', 'linestyle': '-'}
            {'color': 'g', 'linestyle': '--'}
            {'color': 'g', 'linestyle': '-.'}
            {'color': 'b', 'linestyle': '-'}
            {'color': 'b', 'linestyle': '--'}
            {'color': 'b', 'linestyle': '-.'}

        properties =
            [{'color': ('jet', 7)},]
        -->
            list(cycler_new) =
            [{'color': array([ 0.        ,  0.        ,  0.53565062,  1.        ])},
             {'color': array([ 0.        ,  0.19019608,  1.        ,  1.        ])},
             {'color': array([ 0.        ,  0.84901961,  1.        ,  1.        ])},
             {'color': array([ 0.49019608,  1.        ,  0.47754586,  1.        ])},
             {'color': array([ 1.       ,  0.9157589,  0.       ,  1.       ])},
             {'color': array([ 1.        ,  0.30573711,  0.        ,  1.        ])},
             {'color': array([ 0.53565062,  0.        ,  0.        ,  1.        ])}]

    """
    from cycler import cycler
    if ax == 'current':
        ax = plt.gca()
    properties = make_iterable(properties, nest_types=(dict,))

    cycler_new = None
    for prop_dict in properties:
        prop_dict = {key: make_iterable(values) for key, values in prop_dict.items()}
        length = max(map(len, prop_dict.values()))
        for key, values in prop_dict.items():
            if len(values) == 1:
                prop_dict[key] = list(values) * length
            if (key == 'color') and (len(values) == 2) and (values[0] in colormap_names):
                cmap = plt.get_cmap(values[0])
                # Convert array of colors to lists for == comparison
                prop_dict[key] = [list(v) for v in cmap(np.linspace(0.01, 0.99, values[1]))]
            elif len(values) != length:
                raise ValueError('cycler properties must be all be same length or length 1: {}'.format(prop_dict))
        # Multiple kwargs same as adding cyclers
        cycler_tmp = cycler(**prop_dict)
        if cycler_new is None:
            cycler_new = cycler_tmp
        else:
            # Product of cyclers
            cycler_new = cycler_new * cycler_tmp

    if ax is None:
        pass
    elif ax == 'all':
        plt.rc('axes', prop_cycle=cycler_new)
    else:
        cycler_old = ax._get_lines.prop_cycler
        cycler_old_tmp, cycler_old = itertools.tee(cycler_old)
        # compare next 3 items
        next_items = [next(cycler_old_tmp) for i in np.arange(4)]
        if np.all(np.array([item in cycler_new for item in next_items])):
            # cycler already set - don't change
            # ax.set_prop_cycle(cycler_old)
            pass
        else:
            # New cycler is different
            ax.set_prop_cycle(cycler_new)
    return cycler_new

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    return


def to_percent(y, position, dp=0):
    """ Need to write function wrapper to use dp with FuncFormatter
    """
    # From: http://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    import matplotlib
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    y = 100 * y
    if dp == 0:
        y = int(y)
    s = str(y)
    # s = ['{1:0.{0}f}'.format(dp).format(x) for x in s]
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def set_yaxis_percent(ax, dp=0):
    from matplotlib.ticker import FuncFormatter
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    return ax

def save_fig(fn, fig=None, directory=None, transparent=True, bbox_inches='tight', dpi=90, save=True, mkdir_depth=None,
             mkdir_start=None, description=''):
    if not save:
        return
    if fig is None:
        fig = plt.gcf()
    if directory is not None:
        fn = os.path.join(directory, fn)
    fn = os.path.expanduser(fn)
    if (mkdir_depth is not None) or (mkdir_start is not None):
        mkdir(os.path.dirname(fn), depth=mkdir_depth, start_dir=mkdir_start)
    fig.savefig(fn, bbox_inches=bbox_inches, transparent=transparent, dpi=dpi)
    logger.info('Saved {} plot to: {}'.format(description, fn))

def color_shade(color, percentage):
    """Crude implementation to make color darker or lighter until matplotlib function is available:
    https://github.com/matplotlib/matplotlib/pull/8895"""
    from matplotlib import colors as mcolors
    c = mcolors.ColorConverter().to_rgb(color)
    c = np.clip(np.array(c)+percentage/100., 0, 1)
    return c

def get_previous_line_color(ax=None):
    """Return color of previous line plotted to axis"""
    if ax is None:
        ax = plt.gca()
    if len(ax.lines) == 0:
        logger.warning("Can't repeat line color - no previous lines on axis")
        return 'k'
    color = ax.lines[-1].get_color()
    return color

def repeat_color(string, ax=None):
    if ax is None:
        ax = plt.gca()
    color = get_previous_line_color(ax)
    if '+' in string or '-' in string:
        percentage = float(string.split('+')[-1].split('-')[-1])
        c = color_shade(color, percentage)
    return c

def close_all_mpl_plots(close_all=True, verbose=True):
    """Close all existing figure windows"""
    if close_all:
        nums = plt.get_fignums()
        for n in nums[:-1]:
            plt.close(n)
    if verbose:
        logger.info('Closed all mpl plot windows')

def show_if(show, close_all=False):
    """If show is true show plot. If clear all is true clear all plot windows before showing."""
    close_all_mpl_plots(close_all)
    if show:
        plt.show()

if __name__ == '__main__':
    fig = plt.figure()
    axes = fig.add_subplot(111)

    cycler = set_cycler([{'color': ('r', 'g', 'b')}, {'linestyle': ['-', '--', '-.']}])
    cycler = set_cycler([{'color': ('r', 'g', 'b'), 'linestyle': ['-', '--', '-.']},])
    cycler = set_cycler([{'color': ('jet', 7)},])

    # my random data
    scale = 10
    np.random.seed(101)
    x = np.random.random(10)*scale
    y = np.random.random(10)*scale
    # arrowplot(axes, x, y )


    x = np.linspace(0,10,41)
    y = np.sin(x)
    z = x*y

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    print(points)

    seg_line_plot(axes,x,y,z, fig=fig, z_out='color+linewidth')
    axes.set_xlim([0,10])
    axes.set_ylim([-1,1])

    plt.show()