#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger(__name__)

signal_abbreviations = {
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
signal_sets = {
    'set1': [
        "amc_plasma current",
        "ESM_NE_BAR",  # (electron line averaged density)
        "ane_density",  # Gives lower density - what is this?
        "anb_tot_sum_power",  # Total NBI power
        "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
        "esm_p_loss",  # Total power crossing the separatrix
        "efm_q_95",  # q95
        "efm_q_axis",  # q0
        "EFM_Q95",  # (q95)
        "ada_dalpha integrated",
        'xsx/tcam/1',  # soft xray 1
        'efm_bphi_rmag',
        "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
        "ADG_density_gradient",
        "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
        "arp_rp radius"]   # (radial position of the reciprocating probe)
    }

def get_data(signal, pulse, save_path='~/data/MAST_signals/', save=True, *args, **kwargs):
    """Get data with IDL_bridge getdata if available, else load from pickle store."""
    pulse = int(pulse)
    if signal in signal_abbreviations:
        signal = signal_abbreviations[signal]
    save_path = os.path.expanduser(save_path)
    if save:
        pulse_str = '{pulse:d}'.format(pulse=pulse)
        fn = signal.replace('/', '_').replace(' ', '_')+'.p'
        mkdir(os.path.join(save_path, pulse_str), start_dir=save_path)
    try:
        import idlbridge as idl
        getdata = idl.export_function("getdata")
        d = getdata(signal, pulse, *args, **kwargs)
        if d['erc'] != 0:
            logger.warning('Failed to load data for {}: {}'.format(pulse_str, signal))
        elif save:
            pickle_dump(d, os.path.join(save_path, pulse_str, fn), protocol=2)
            logger.info('Saved data for {}; {} to {}'.format(pulse_str, signal, os.path.join(save_path, pulse_str, fn)))
        return d
    except ImportError:
        try:
            d = pickle_load(os.path.join(save_path, pulse_str, fn))
            return d
        except IOError:
            logger.warning('Cannot locate data for {}:{} in {}'.format(pulse_str, signal, save_path))

def store_mast_signals(signals, pulses, save_path='~/data/MAST_signals/', *args, **kwargs):
    if isinstance(signals, (str, basestring)) and signals in signal_sets:
        signals = signal_sets[signals]
    pulses = make_iterable(pulses)
    signals = make_iterable(signals)
    save_path = os.path.expanduser(save_path)
    assert os.path.isdir(save_path), 'Save path does not exist'
    for pulse in pulses:
        for signal in signals:
            get_data(signal, pulse, save_path=save_path, noecho=1, *args, **kwargs)