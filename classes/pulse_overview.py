
import matplotlib.pyplot as plt

from ccfepyutils.utils import make_iterable
from ccfepyutils.classes.plot import Plot
from ccfepyutils.mast_data.get_data import get_data


signal_keys = {
    'Ip': "amc_plasma current",
    'ne': "ESM_NE_BAR",  # (electron line averaged density)
    'ne2': "ane_density",  # Gives lower density - what is this?
    'Pnbi': "anb_tot_sum_power",  # Total NBI power
    'Pohm': "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
    'Ploss': "esm_p_loss",  # Total power crossing the separatrix
    'q95': "efm_q_95",  # q95
    'q0': "efm_q_axis",  # q0
    'q952': "EFM_Q95",  # (q95)
    'Da': "ada_dalpha integrated",
    # 'Da-mp': 'ph/s/cm2/sr',
    'sXray': 'xsx/tcam/1',
    'Bphi': 'efm_bphi_rmag',
    'zmag': "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
    'dn_edge': "ADG_density_gradient",
    'Bvac': "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
    'LPr': "arp_rp radius"  # (radial position of the reciprocating probe)
}

ylabels = {'ESM_NE_BAR': r'$n_e$ [$\times 10^{19}$ m$^{-3}$]',
           'amc_plasma current': '$I_p$ [kA]',
           "ada_dalpha integrated": r'$D_{\alpha}$ [ph.$s^{-1}cm^{-2}sr^{-1}$]',
           'Pnbi': '$P_{NBI}$ [MW]'}

class PulseOverivew:

    def __init__(self):
        # self._pulses = pulses
        pass

    def plot(self, pulses, signals, xlim=None, show=False, save=False):

        pulses = make_iterable(pulses)
        signals = make_iterable(signals)
        plot = Plot(axes=(len(signals), 1))

        for pulse in pulses:
            for i, signal in enumerate(signals):
                if signal in signal_keys.keys():
                    signal = signal_keys[signal]
                data = get_data(signal, pulse)
                try:
                    ylabel = '{} [{}]'.format(data['dlabel'], data['dunits']) if signal not in ylabels else ylabels[signal]
                    x, y = data['time'], data['data']
                    if signal == 'ESM_NE_BAR':
                        y /= 1e19
                    plot.plot(x, y, ax=i, ylabel=ylabel, xlim=xlim, label=str(pulse))
                except:
                    print('Failed to plot {}:{}'.format(pulse, signal))

        plot.set_axis_labels(xlabel='$t$ [s]', ax=len(signals)-1)
        plot.set_axis_labels(label_fontsize=18, tick_fontsize=14, tight_layout=True, ax='all')
        plt.setp(plot.ax(0).get_xticklabels(), visible=False)
        plot.fig.subplots_adjust(hspace=0.015)

        plot.legend(ax=0)
        if save:
            plot.save(save=save)
        plot.show(show=show, tight_layout=False, legend=False)


if __name__ == '__main__':
    po = PulseOverivew()
    # po.plot([29840, 29023], ['Ip', 'ne'], xlim=[0, 0.6])
    # po.plot([29840, 29023], ['Ip', 'ne', 'Da'], xlim=[0, 0.6])
    # po.plot([28996, 29835, 29825, 29767], ['Ip', 'ne', 'Da'], xlim=[0, 0.6], save='/home/tfarley/tmp/EPS_pulse_low_ne.png')
    po.plot([28996, 29767], ['Ip', 'ne', 'Pnbi', 'Da'], xlim=[0, 0.6], save='/home/tfarley/tmp/EPS_pulse_low_ne.png')