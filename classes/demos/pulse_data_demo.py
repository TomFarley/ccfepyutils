""" Analyse and plot data for individual pulse
"""

from ccfepyutils.classes.pulse_data import PulseData

# p = PulseData(29840, ft_sig='Ip-2')
# p = PulseData(28996, ft_sig='Ip-2')
# p = PulseData(29835, ft_sig='Ip-2')
p = PulseData(29767, ft_sig='Ip-2')

p.plot(['Ip', 'ne', 'dn_edge'], info=['Ip', 'ne', 'q95', 'q0', 'Bphi', 'Pnbi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
       save=False, show=True)
