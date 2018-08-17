import numpy as np

from ccfepyutils.mast_data.get_data import get_data, signal_abbreviations, signal_ylabels
from ccfepyutils.classes.plot import Plot
from ccfepyutils.mpl_tools import get_previous_line_color

def parameter_space_plot(time_windows, x_signal='ne2', y_signal='Ip', legend_resolution={'Pnbi': 0.1},
                         error_bars='range', dividers=None, machine='MAST'):
    if machine != 'MAST':
        raise NotImplementedError
    if x_signal in signal_abbreviations:
        x_signal = signal_abbreviations[x_signal]
    if y_signal in signal_abbreviations:
        y_signal = signal_abbreviations[y_signal]
    plot = Plot(num='parameter_space_plot: {}'.format(list(time_windows.keys())), xlabel=signal_ylabels[x_signal], ylabel=signal_ylabels[y_signal])
    ax = plot.ax()

    if dividers is not None:
        if 'x' in dividers:
            for xi in dividers['x']:
                ax.axvline(xi, ls='--', color='k', alpha=0.5)
        if 'y' in dividers:
            for yi in dividers['y']:
                ax.axhline(yi, ls='--', color='k', alpha=0.5)

    legend_params = {}

    for pulse, twins in time_windows.items():
        for twin in twins:
            x_data = get_data(x_signal, pulse, twin=twin)['data']
            y_data = get_data(y_signal, pulse, twin=twin)['data']

            x = np.mean(x_data)
            y = np.mean(y_data)

            for z_signal, resolution in legend_resolution.items():
                z_data = get_data(z_signal, pulse, twin=twin)
                if z_data['erc'] == 0:
                    z = np.mean(z_data['data'])
                else:
                    z = 0

            if error_bars is None:
                xerr = None
                yerr = None
            elif error_bars == 'range':
                xerr = np.abs(np.array([np.min(x_data), np.max(x_data)]-x))
                yerr = np.abs(np.array([np.min(y_data), np.max(y_data)]-y))
            elif error_bars == 'std':
                xerr = np.std(x_data)
                yerr = np.std(y_data)
            else:
                raise ValueError('Error bar typle "{}" not recognised'.format(error_bars))

            kws = {}
            if (len(legend_params) > 0):
                # Not first line
                if (np.min(np.abs(np.array(list(legend_params.keys()))-z)) < resolution):
                    # Within resolution of existing point
                    keys = np.array(list(legend_params.keys()))
                    key = keys[np.argmin(np.abs(keys - z))]
                    kws['color'] = legend_params[key]
                    z = key
                    label = None
                else:
                    # New point type
                    label = '{} = {:0.1f}'.format(z_signal, z)
            else:
                # legend_params[z] = 'k'
                label = '{} = {:0.1f}'.format(z_signal, z)

            plot.plot(x, y, xerr=xerr, yerr=yerr, mode='errorbar', point_annotations=str(pulse), label=label,
                      show=False, point_annotation_offset=(0.01e19, 2), point_annotation_offset_mode='data',
                      annotate_points_kwargs={'color': 'dimgrey'}, **kws)

            if label is not None:
                legend_params[z] = get_previous_line_color(ax)

    plot.show()

if __name__ == '__main__':
    # time_windows = {29840: [[0.135, 0.160], [0.201, 0.2125], [0.262, 0.272]],
    time_windows = {29991: [[0.1289, 0.1469], [0.182, 0.195], [0.255, 0.272]],
                    29840: [[0.1289, 0.1469]],
                    # 29879: [[0.255, 0.272]],  # 1.3 MW
                    # 29908: [[0.255, 0.272]],  # 1.55 MW
                    # 29958: [[0.152, 0.20]],
                    29852: [[0.163, 0.1748], [0.255, 0.272]], #, [0.2545, 0.265]],   # [0.1664, 0.1740], [0.29, 0.314], [0.295, 0.354]
                    # 29902: [[0.28, 0.32]],
                    # 29764: [[0.23, 0.30]],
                    # 29994: [[0.23, 0.30]],
                    }
    parameter_space_plot(time_windows, x_signal='ne', y_signal='Ip', dividers={'x': [2e19, 3e19, 3.9e19], 'y': [500, 700]},
                         legend_resolution={'Pnbi': 0.4})