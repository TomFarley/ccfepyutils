#!/usr/bin/env python
"""
Pulse filtering by signal chriteria
Tom Farley, May 2016
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import inspect
import collections
from collections import OrderedDict
# import idlbridge as idl
# getdata = idl.export_function("getdata")
from ccfepyutils.mast_data.get_data import get_data
# import idam
# idam.setProperty("verbose")
# idam.setProperty("verbose", False)
# idam.setProperty("verbose", False)
from copy import deepcopy
# import idlbridge as idl
import logging
logger = logging.getLogger(__name__)

from ccfepyutils.io_tools import mkdir
from ccfepyutils.utils import make_iterable, is_number
from ccfepyutils.classes.pulse_data import PulseData

class PulseFilter(object):  # collections.MutableMapping
    """ Class for filtering MAST pulses by a set of constraints from IDAM data signals
    """

    def __init__(self, constraints, pulse_range=None, pulse_lists=None, ft_sig='Ip', name=None):
        """

        :param constraints: Dict of plasma parameter constrains
        :param pulse_range: List [min, max] of pulse number range to analyse (as opposed to pulse_list)
        :param pulse_lists: Filename of pulse list to consider (as opposed to pulse_range)
        :param ft_sig: Signal to use to detect flattop region (eg flattop in current or density)
        :param name: Name of pulse filter instance
        """
        # TODO: Add find similar pulse feature / similar with particular change
        self.contraints = constraints
        self.pulse_range = pulse_range
        self.pulse_lists = pulse_lists
        self.ft_sig = ft_sig
        self.name = name

        assert (type(constraints) is dict)

        if pulse_lists is not None:
            if isinstance(pulse_lists, np.ndarray):
                self.pulses = pulse_lists
            else:
                # Load files containing lists of pulses to consider
                self.pulses = self.combine_pulse_files(self.pulse_lists)
        elif pulse_range is not None:
            # Use list to generate pulse range
            assert (len(pulse_range) == 2)
            self.pulses = np.arange(pulse_range[0], pulse_range[1] + 1).astype(int)
        else:
            self.pulses = None
            print('WARNING: No pulse_list files for pulse range has been provided for filtering.')

        self.results = None
        self.summary = None

        if self.pulses is not None:
            self.results = {key: [] for key in ['pulse', 'flattop', 'pass', 'reject'] +
                            [PulseData.sig_short(sig) for sig in self.contraints.keys()]}

            self.run_filter(ft_sig=ft_sig)
            self.print_summary()
            self.write_summary()

    def run_filter(self, verbose=True, ft_sig='Ip'):
        """ Find pulses that satisfy all supplied constraints
        """
        for pulse in self.pulses:
            # Start pff accepting the pulse and reject it only once it has failed one of the tests
            accept = True
            result = {}  # dict to record which constraints are passed and failed

            p = PulseData(pulse)
            self.results['flattop'].append(p.find_flattop(signal=ft_sig, test=False))
            if p.twin_ft is None:
                continue

            for sig, constraint in self.contraints.items():
                sig = PulseData.sig_short(sig)
                result[sig] = p.check_constraint(sig, constraint)
                self.results[sig].append(result[sig])
                if result[sig] is not True:
                    accept = False

            if accept:
                print('PASS: {}'.format(pulse))
                self.results['pass'].append(int(pulse))
            else:
                print('FAIL: {}, {}'.format(pulse, result))
                self.results['reject'].append(pulse)
            self.results['pulse'].append(pulse)

            # p.plot('Bphi', smooth=None, window_len=1000)

        if verbose:
            print('\n\n\n\n\nFiltered pulse list:\n{}\n{}/{} ({:0.1%}) pulses passed\n\n\n\n\n'.format(
                self.results['pass'],
                len(self.results['pass']), len(self.pulses), float(len(self.results['pass'])) / len(self.pulses)))

    def generate_summary(self, refresh=False):
        """ Generate a dictionary containing the constraint parameter values for the accepted pulses
        """
        # if summary has already been generated, use previous result
        if self.summary is not None and not refresh:
            return self.summary

        data = {key: [] for key in self.contraints.keys()}
        for pulse in self.results['pass']:
            p = PulseData(pulse, ft_sig=self.ft_sig)
            values = p.av_params(list(self.contraints.keys()), twin='flattop')
            for key in self.contraints.keys():
                data[key].append(values[key])
            p.find_flattop(test=False)
            data['t_ft'] = p.twin_ft[1] - p.twin_ft[0] if p.twin_ft is not None else np.nan

        self.summary = data
        return data

    def print_summary(self, stdout=True):
        """ Print summary output to terminal
        """
        self.generate_summary()
        print('\n\n\n\n\nFiltered pulse list:\n{}\n{}/{} ({:0.1%}) pulses passed\n'.format(self.results['pass'],
                                                                                           len(self.results['pass']),
                                                                                           len(self.pulses), float(
                len(self.results['pass'])) / len(self.pulses)))

        # print(p.av_params(self.contraints.keys(), header=True))

        # for i, pulse in enumerate(self.results['pulse']):
        #     format = '{{1}[{0}]}\t'*len(self.contraints.keys())
        #     print(format.format(*self.contraints.values()))

        # print(data.keys())

        out = [(key, list) for key, list in self.summary.items()]
        out = [('Pulse', self.results['pass'])] + out
        df = pd.DataFrame.from_dict(dict(out))
        if stdout:
            print(df)

        # print('Ip (Filtered):\n{}'.format(self.summary['Ip']))
        return df

    def write_summary(self, path='./results/'):
        """ Write summary output to file
        """
        mkdir(path, start_dir='.')

        if self.name is not None:
            fn = 'pulse_filter_summary_{}.txt'.format(self.name)
        else:
            fn = 'pulse_filter_summary.txt'

        with open(path + fn, 'w') as f:
            # f.writelines('\n'.join(['Constraints', json.dumps(self.contraints, sort_keys=True,separators=(', ', ':\n  '))]))#indent=2)]))
            f.writelines('\n'.join(['Constraints', json.dumps(self.contraints, sort_keys=True, indent=3)]))
            f.writelines('\n'.join(['\n{} pulses accepted'.format(len(self.results['pass'])),
                                    json.dumps(self.results['pass'])]))


        out = [(key, list) for key, list in self.summary.items()]
        out = [('Pulse', self.results['pass'])] + out
        df = pd.DataFrame.from_items(out)
        if self.name is not None:
            df.to_csv(path + 'pulse-filter_data-summary_{}.csv'.format(self.name))
        else:
            df.to_csv(path + 'pulse-filter_data-summary.csv')
        logger.info('Summary written to: {}'.format(os.path.abspath(fn)))

        # df.to_excel()

    # for pulse in pulses:
    #     for signal in constraints.keys():
    #         sig = idam.Data(signal, pulse)  # Read a data object for MAST shot
    #
    #         print(sig)
    #
    #         plt.plot(sig.time, sig.data)
    #         plt.show()

    def plot(self, signals=['Ip', 'ne', 'q95'], info=None, ft_sig='Ip', show=True, save=True, path='plots/', set='pass',
             **kwargs):
        """ Produce plots of all pulses that satisfy constraints
        """
        print('Plotting traces for {} signals for all pulses that satisfy constraints'.format(' '.join(signals)))
        for pulse in self.results[set]:
            p = PulseData(pulse)
            try:
                p.av_params(["amc_plasma current", "ane_density", "efm_magnetic_axis_z", "efm_q_95"])
                p.find_flattop(test=False, signal=ft_sig)
                p.plot(signals=signals, info=info, show=show, save=save, path=path, **kwargs)
            except Exception as e:
                logger.exception('Failed to plot signals for pulse {}'.format(pulse))

    def read_pulse_file(self, fn):
        """ Read a file composed one pulse number per line and return output in an array.
        Skips non-numeric lines (eg header)
        """
        if not os.path.isfile(fn):
            print("WARNING: File: '{}' does not exist".format(fn))
            raise FileNotFoundError('File {} does not exist'.format(fn))
        pulses = []
        with open(fn, 'r') as f:
            for line in f:
                line.rstrip()
                if is_number(line):
                    pulses.append(int(line))

        return np.array(pulses)

    def combine_pulse_files(self, fns, fn_out='combined_pulse_list'):
        """ Take text files, each with a pulse number on each line and produce a file with the list of pulses common
        to all files
        """
        if type(self.pulse_lists) is str:
            self.pulse_lists = [self.pulse_lists]  # nest in list
        assert isinstance(fns, (list, tuple))

        pulses = []
        for fn in fns:
            pulses.append(self.read_pulse_file(fn))

        ar1 = pulses[0]
        for ar2 in pulses[1:]:
            ar1 = np.intersect1d(ar1, ar2)

        print('Reduced file list from files {}:\n{}'.format(', '.join(fns), ar1))

        with open(fn_out, 'w') as f:
            f.write('Pulses common to files: {}\n'.format(', '.join(fns)))
            f.write('{}'.format('\n'.join([str(p) for p in ar1])))

        return ar1

    def signal_has_data(self, signal, data=None):
        if data is None:
            data = self[signal]
        if (data is not None) and ('time' in data):
            exists = True
        elif (data is None) or (data['erc'] == -1):
            exists = False
        else:
            raise ValueError('Unexpected value for data dict: {}'.format(data))
            exists = False
        return exists


# def str2tex(string):
#     """Convert common mathematical strings to latex friendly strings for plotting"""
#
#     def insert(string, char, i):
#         """ insert character after index """
#         if i == -1:  # effecively last character: -1-1
#             return string + char
#         # elif i == 0:  # effecively first character: 0-1
#         #     return char + string
#         else:
#             return string[0:i + 1] + char + string[i + 1:]
#
#     first = 1000
#     last = -1
#     close_chars = [' ', '=', '^', '_']
#
#     for char_enclose in ['_', '^']:
#         start = 0
#         while string.find(char_enclose, start + 1, len(string)) != -1:
#             if string.find(char_enclose, start + 1, len(string)) == string.find(char_enclose + '{', start + 1,
#                                                                                 len(string)):
#                 continue
#             start = string.find(char_enclose, start, len(string))
#             if start < first:
#                 first = np.max(start - 1, 0)
#             string = insert(string, '{', start)
#             found = False
#             end_chars = deepcopy(close_chars)
#             end_chars.remove(char_enclose)
#             for char in end_chars:
#                 if string.find(char, start, len(string)) != -1:
#                     end = string.find(char, start, -1)
#                     if end > last:
#                         last = end - 1
#                     string = insert(string, '}', end - 1)
#                     found = True
#                     break
#             if found is False:
#                 string += '}'
#                 last = 1000
#
#     for char_escape in ['%', 'Psi', 'phi', 'theta', 'sigma']:
#         start = 0
#         while string.find(char_escape, start, len(string)) != -1:
#             if string.find(char_escape, start + 1, len(string)) == string.find('\\' + char_escape, start + 1,
#                                                                                len(string)) - 1:
#                 continue
#             start = string.find(char_escape, start, len(string))
#             if start < first:
#                 first = np.max(start - 1, 0)
#             string = insert(string, '\\', start - 1)
#             start += 2
#             if start + len(char_escape) > last:
#                 last = start + len(char_escape)
#
#     if first != 1000:
#         first = string.rfind(' ', 0, first + 1)
#         if first == -1:
#             first = 0
#         if last == 1000:
#             last = len(string)
#         string = insert(string, '$', first)
#         string = insert(string, '$', last)
#
#     string = re.sub('nounits', 'N/A', string)
#
#     return string
#
#
# def findNearest(arr, point, index=True, normalise=False):
#     """ Find closest point to supplied point in either 1d array, 2d grid or 2xn array of coordinate pairs
#     """
#     inparr = np.asarray(arr)
#     # print('point', point)
#     # print('inparr.shape', inparr.shape)
#     # print('inparr', inparr)
#     if isinstance(point, numbers.Number) or len(inparr.shape) == 1:  # if point is a single number take array to be 1d
#         if index:
#             return np.abs(inparr - point).argmin()
#         else:
#             return inparr[np.abs(inparr - point).argmin()]
#     else:
#         # Make sure array in two column format
#         shape = inparr.shape
#         if shape[1] == 2 and shape[0] > 0:
#             inparr = inparr.T
#             shape = inparr.shape
#
#         if shape[0] == 2 and shape[1] > 0:
#             (valx, valy) = point
#             normarr = deepcopy(inparr)
#             # Treat x and y coordinates as having the same fractional accuracy ie as if dx=dy
#             if normalise:
#                 normarr[0] = (normarr[0] - np.min(normarr[0])) / (np.max(normarr[0]) - np.min(normarr[0]))
#                 normarr[1] = (normarr[1] - np.min(normarr[1])) / (np.max(normarr[1]) - np.min(normarr[1]))
#                 valx = (valx - np.min(inparr[0])) / (np.max(inparr[0]) - np.min(inparr[0]))
#                 valy = (valy - np.min(inparr[1])) / (np.max(inparr[1]) - np.min(inparr[1]))
#             ixy = (((normarr[0, :] - valx) ** 2.0 + (normarr[1, :] - valy) ** 2.0) ** 0.5).argmin()
#             if index:
#                 return ixy
#             else:
#                 return inparr[:, ixy]
#
#         elif len(shape) == 3:
#             # incomplete!
#             (valx, valy, valz) = point
#             return (((inparr[:, 0] - valx) ** 2.0 + (inparr[:, 1] - valy) ** 2.0 + (
#                         inparr[:, 2] - valz) ** 2.0) ** 0.5).argmin()
#
#         else:
#             raise RuntimeError('findNearest: Input array did not match any anticipated format')


def pulse_files():
    pf = PulseFilter(None, None)
    print(pf.read_pulse_file('SA1_midplane-pulse-list_description-filtered.txt'))

    pf.combine_pulse_files(['SA1_midplane-pulse-list_description-filtered.txt', 'MP_M9_pulse_list.txt'])


def analyse_pulse():
    # p = PulseData(29737)
    p = PulseData(29852, ft_sig='Ip-2')
    print('nx:', p['nx'])
    # p.av_params(["amc_plasma current","ane_density","efm_magnetic_axis_z","efm_q_95"])
    p.find_flattop(signal='Ip-2', test=False)
    p.plot(['Ip', 'ne', 'dn_edge', 'Da'], info=['Ip', 'ne', 'q95', 'q0', 'Bphi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
           save=False, show=True)
    # p.plot('Bphi')
    p.plot("/xrp/isat/p2")
    print('series output:\n{}'.format(p.print()))

    print('"amc_plasma current" ->', PulseData.sig_short("amc_plasma current"))


def filter():
    pulse_range = [29737, 30021]

    # pulse_lists=['SA1_midplane-pulse-list_description-filtered.txt']
    pulse_lists = ['SA1_HDD_midplane_pulse_list-all.txt']

    constraints = {
        # "ada_dalpha integrated",
        "Ip": {'range': [300, 1000], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "ne": {'range': [0.5e19, 10e19], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "Bphi": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "zmag": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "q0": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "q95": {'range': [2, 15], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "dn_edge": {'range': None, 'mean': [0, 0.3e21], 'percent_fluct': None, 'smoothness': None}
        # "q0": {'range': [1,20], 'mean': None, 'percent_fluct': None, 'smoothness': None}
        # "efm_magnetic_axis_z": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None}
        # "/xrp/isat/p2": {'range': None},
        # "/xrp/isat/p5": {'range': None},
        # "/xrp/isat/p8": {'range': None}
        # arp_rp radius (radial position of the reciprocating probe)
    }

    # name = None
    name = 'Lmode-mean'

    pf = PulseFilter(constraints, pulse_lists=pulse_lists, pulse_range=None, ft_sig='Ip-2', name=name)
    pf.plot(['ne', 'Da', 'dn_edge'], info=['Ip', 'ne', 'q95', 'q0', 'Bphi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
            ft_sig='Ip-2', save=True, show=False, path='./plots/{}/'.format(name))
    pf.print_summary()


def filter2():
    pulse_lists = ['SA1_HDD_midplane_pulse_list-all.txt']

    constraints = {
        "Ip": {'range': [300, 1000], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "ne": {'range': [0.5e19, 10e19], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "Bphi": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "zmag": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "q0": {'range': None, 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "q95": {'range': [2, 15], 'mean': None, 'percent_fluct': None, 'smoothness': None},
        "dn_edge": {'range': [0.3e21, 1e23], 'mean': None, 'percent_fluct': None, 'smoothness': None}
    }

    # name = None
    name = 'Lmode-mean'

    pf = PulseFilter(constraints, pulse_lists=pulse_lists, pulse_range=None, ft_sig='Ip-2', name=name)
    pf.plot(['ne', 'Da', 'dn_edge'], info=['Ip', 'ne', 'q95', 'q0', 'Bphi', 'zmag', 'Da', 'nx', 'mode', 'dn_edge'],
            ft_sig='Ip-2', save=True, show=False, path='./plots/{}/'.format(name))
    pf.print_summary()


if __name__ == '__main__':
    # pulse_files()
    # analyse_pulse()
    filter()
    # filter2()
