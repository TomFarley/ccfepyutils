
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from ccfepyutils.utils import make_iterable
from ccfepyutils.classes.plot import Plot
from ccfepyutils.mpl_tools import get_previous_line_color
from ccfepyutils.mast_data.get_data import signal_abbreviations, get_data, signal_ylabels

class PulseOverivew:

    def __init__(self):
        # self._pulses = pulses
        pass

    def plot(self, pulses, signals, xlim=None, twins=None, show=False, save=False):

        pulses = make_iterable(pulses)
        signals = make_iterable(signals)
        plot = Plot(axes=(len(signals), 1), num='Overview plot: {}, {}'.format(pulses, signals))#, cycler=[{'color': ('jet', len(pulses))}])
        pulse_colors = {}
        for i, signal in enumerate(signals):
            ax = plot.ax(i)
            if signal in signal_abbreviations.keys():
                signal = signal_abbreviations[signal]
            if twins is not None:
                for twin in twins:
                    ax.axvspan(twin[0], twin[1], facecolor='gray', alpha=0.3)  # '#2ca02c'
            plot.set_axis_cycler([{'color': ('Vega20', len(pulses))}], ax=i)
            for pulse in pulses:
                data = get_data(signal, pulse)
                try:
                    ylabel = '{} [{}]'.format(data['dlabel'], data['dunits']) if signal not in signal_ylabels else signal_ylabels[signal]
                    x, y = data['time'], data['data']
                    if signal == 'ESM_NE_BAR':
                        y /= 1e19
                    color = pulse_colors[pulse] if pulse in pulse_colors else None
                    plot.plot(x, y, ax=i, ylabel=ylabel, xlim=xlim, label=str(pulse), alpha=0.75, grid=True, color=color)
                    if pulse not in pulse_colors:
                        pulse_colors[pulse] = get_previous_line_color(ax)
                except:
                    print('Failed to plot {}:{}'.format(pulse, signal))
                if signal == 'ESM_NE_BAR':
                    plot.set_axis_limits(ylim=[0, 5], ax=i)

        plot.set_axis_labels(xlabel='$t$ [s]', ax=len(signals)-1)
        plot.set_axis_labels(label_fontsize=18, tick_fontsize=14, tight_layout=True, ax='all')
        plot.set_axis_appearance(grid=True, grid_which='both', ax='all', sharex='all')
        plt.setp(plot.ax(0).get_xticklabels(), visible=False)
        plot.fig.subplots_adjust(hspace=0.015)

        # plot.legend(ax=-1, legend_fontsize=5)  # 14
        plot.legend(ax='all', legend_fontsize=14)  # 5, 14
        if save:
            plot.save(save=save)
        plot.show(show=show, tight_layout=False, legend=False)


if __name__ == '__main__':
    po = PulseOverivew()
    # po.plot([29840, 29023], ['Ip', 'ne'], xlim=[0, 0.6])
    # po.plot([29840, 29023], ['Ip', 'ne', 'Da'], xlim=[0, 0.6])
    # po.plot([28996, 29835, 29825, 29767], ['Ip', 'ne', 'Da'], xlim=[0, 0.6], save='/home/tfarley/tmp/EPS_pulse_low_ne.png')
    # po.plot([28996, 29767], ['Ip', 'ne', 'Pnbi', 'Da'], xlim=[0, 0.6], save='/home/tfarley/tmp/EPS_pulse_low_ne.png')
    # po.plot([29811, 29815, 29827, 29852, 29834, 29808, 29823], ['Ip', 'ne2', 'Pnbi', 'sXray'], xlim=[0, 0.6], show=True)
    # po.plot([29761], ['Ip', 'ne', 'ne2', 'ne3', 'Pnbi', 'sXray'], xlim=[0, 0.6], show=True)
    # po.plot([29754, 29761, 29763, 29764, 29785, 29848, 29849, 29852, 29902, 29958, 29960, 29972, 29993, 29994], ['Ip', 'ne3', 'Pnbi', 'q0'], xlim=[0, 0.6], show=True)
    # po.plot([29840, 29991, 29852, 29764, 29958, 29908, 29879], ['Ip', 'ne', 'Pnbi', 'q0'], xlim=[0, 0.6], show=True,
    #         twins=[[0.1280, 0.1460], [0.182, 0.195], [0.23, 0.30], [0.255, 0.272], [0.28, 0.32]])
    po.plot([29840, 29991, 29852], ['Ip', 'ne', 'Pnbi', 'q0'], xlim=[0, 0.45], show=True,
            twins=[[0.1289, 0.1469], [0.182, 0.195], [0.255, 0.272], [0.163, 0.1748]])


