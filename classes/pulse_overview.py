
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

ylabels = {'ESM_NE_BAR': '$n_e$ [m$^{-3}$]',
           'amc_plasma current': '$I_p$ [kA]',
           "ada_dalpha integrated": r'$D_{\alpha}$ [ph.$s^{-1}cm^{-2}sr^{-1}$]'}

class PulseOverivew:

    def __init__(self):
        # self._pulses = pulses
        pass

    def plot(self, pulses, signals, xlim=None):

        pulses = make_iterable(pulses)
        signals = make_iterable(signals)
        plot = Plot(axes=(len(signals), 1))

        for pulse in pulses:
            for i, signal in enumerate(signals):
                if signal in signal_keys.keys():
                    signal = signal_keys[signal]
                data = get_data(signal, pulse)
                ylabel = '{} [{}]'.format(data['dlabel'], data['dunits']) if signal not in ylabels else ylabels[signal]
                plot.plot(data['time'], data['data'], ax=i, xlabel='$t$ [s]', ylabel=ylabel, xlim=xlim, label=str(pulse))
        plot.legend(ax=0)
        plt.tight_layout()
        plot.save(save='/home/tfarley/tmp/EPS_pulse.png')
        plot.show()


if __name__ == '__main__':
    po = PulseOverivew()
    po.plot([29840, 29023], ['Ip', 'ne'], xlim=[0, 0.6])
    # po.plot([29840, 29023], ['Ip', 'ne', 'Da'], xlim=[0, 0.6])