""" Filter a set of pulses based on various plasma parameter conditions
"""
import numpy as np
import os
from ccfepyutils.classes.pulse_filter import PulseFilter

## Range of pulses to analyse
# pulse_range = [29737, 30021]
# pulse_range = [21712, 21712]
# pulse_range = [29852, 29852]
# pulse_range = [29767, 29767]
pulse_range = None

## Files contianing list of pulses to analyse (overloads pulse_range). Intersection of pulse lists will be analysed.
pulse_lists = ['../../misc/SA1_HDD_midplane_pulse_list-all.txt']
# pulse_lists = None
# pulse_lists = np.array([29023, 29003, 28996, 28998, 29026, 29007])

## Constrians to apply to pulses in filtering process over chosen time window (typically Ip flat top)
# constraints = {
# "Ip": {'range': [300, 1000], 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "ne": {'range': [0.5e19, 10e19], 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "Bphi": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "zmag": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "q0": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "q95": {'range': [2,15], 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "dn_edge": {'range': None, 'mean': [0.3e21, 1e23], 'percent_fluct': None, 'smoothness': None}
# }
constraints = {
# "Ip": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
"Ip": {'range': None, 'mean': [360, 440], 'percent_fluct': None, 'smoothness': None},
"ne": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None}, #'mean': [0.4e19, 2.0e19]
# "ne": {'range': None, 'mean': [0.8e19, 1.6e19], 'percent_fluct': None, 'smoothness': None}, #'mean': [0.4e19, 2.0e19]
# "Bphi": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
"Pnbi": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None, 'missing': True},
# "zmag": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "q0": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "q95": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
# "dn_edge": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None}
}

# ft_sig = 'ne'  # 'Ip-4'
ft_sig = 'Ip-2'  # 'Ip-4'

## Apply filter useing constraints dictionary and use flat top time window found from ft_sig='Ip-2'
pf = PulseFilter(constraints, pulse_lists=pulse_lists, pulse_range=pulse_range, ft_sig=ft_sig)

# Plot output path
# path = 'plots/similar_Militello2013/'
# path = 'plots/insteresting_pulses/'
# path = 'plots/Militello2016_LP_pulses/'
# plots_path = '~/figures/pulse_filter/SA1_similar_to_29023/'
plots_path = '~/figures/pulse_filter/SA1_similar_to_29029/'

plots_path = os.path.expanduser(plots_path)


## Plot the set of pulses that have 'pass'ed the filtering process and save them to path
pf.plot(['ne', 'Ip', 'q95'], info=['Ip', 'ne', 'q95', 'q0', 'Pnbi', 'Bphi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
        ft_sig=ft_sig, set='pass', save=True, show=False, path=plots_path)
# pf.plot(['Ip', 'ne', 'Da', 'q95'], info=['Ip', 'ne', 'q95', 'q0', 'Bphi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
#         ft_sig='Ip-2', set='pass', save=True, show=False, path=path)

## Print a summary of the accepted pulses to the terminal
pf.print_summary()