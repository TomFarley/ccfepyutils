#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
"""
Pulse filtering by signal chriteria
Tom Farley, May 2016
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
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

from ccfepyutils.utils import make_iterable
from ccfepyutils.data_processing import smooth, data_split, savitzky_golay, conv_diff
from ccfepyutils.io_tools import mkdir
from ccfepyutils.string_formatting import str2tex
from ccfepyutils.utils import is_scalar, is_in, is_numeric

logger = logging.getLogger(__name__)

class PulseData(collections.MutableMapping):
    ## Short signal names (shorter and easier to remember!)
    signals = {
        'Ip': "amc_plasma current",
        'ne': 'ayc_ne',  # Core Thomson scattering data - Electron Density
        'ne3': "ESM_NE_BAR",  # (electron line averaged density)
        'ne2': "ane_density",  # Gives lower density - what is this? - CO2 Interferometry
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

    ft_sig = {
                'Ip': {'signal': 'Ip', 'min_length': 0.10, 'threshold': 0.023, 'background': 0.4, 'window': 'hanning',
                       'window_len': 100, 'top_mul': 3},  # Good flattop
                'Ip-2': {'signal': 'Ip', 'min_length': 0.02, 'threshold': 0.040, 'background': 0.4, 'window': 'hanning',
                         'window_len': 100, 'top_mul': 2},  # Less stringent flattop
                'Ip-3': {'signal': 'Ip', 'min_length': 0.02, 'threshold': 0.080, 'background': 0.4, 'window': 'hanning',
                        'window_len': 100, 'top_mul': 2},  # Less stringent flattop
                'Ip-4': {'signal': 'Ip', 'min_length': 0.02, 'threshold': 0.150, 'background': 0.4, 'window': 'hanning',
                         'window_len': 100, 'top_mul': 2},  # Less stringent flattop
                'ne': {'signal': 'ne', 'min_length': 0.10, 'threshold': 0.25, 'background': 0.3, 'window': 'hanning',
                       'window_len': 100, 'top_mul': 3}
            }
    #"anb_ss_sum_power"

    def __init__(self, pulse=None, ft_sig='Ip'):
        self.ft_sig = ft_sig  # signal settings in PulseData.ft_sig used to identify flattop region of pulse
        self.twin_ft = None  # flat top time window limits
        self.i_ft = None  # indices of current data over flat top
        self.i_top = None  # indices higher tollerance flat top current data
        self.load(pulse)
        self.processed_signals = {
            'nx': self.xpoint_config,
            'mode': self.confinement_mode
            }

    def __getitem__(self, key):
        if key in self.store.keys():  # has already been loaded
            return self.store[key]
        if key in self.signals.values() and self.signals[key] in self.store.keys():  # has already been loaded
            # TODO: fix
            raise NotImplementedError
            return self.store[self.signals[key]]
        else:
            return self.sig(key)  # load data from idam

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        return str(self.store)

    @classmethod
    def sig_short(self, sig):
        """ Convert long signal name to short signal name """
        if sig in self.signals.values():
            sig = list(self.signals.keys())[list(self.signals.values()).index(sig)]
        return sig

    @classmethod
    def sig_long(self, sig):
        """ Convert short signal name to long signal name """
        if sig in self.signals.keys():
            sig = self.signals[sig]
        return sig

    def load(self, pulse):
        self.pulse = pulse
        self.store = {}

    def sig(self, signal, verbose=True):
        if signal in self.processed_signals.keys():
            return self.processed_signals[signal]()

        sig = self.signals[signal] if signal in self.signals.keys() else signal
        try:
            # d = idam.Data(sig, self.pulse)
            # d = getdata(sig, int(self.pulse), noecho=1)
            d = get_data(sig, int(self.pulse), noecho=1)
            self.store[signal] = d
        except RuntimeError as e:
            if verbose: print('Could not load signal "{}": {}'.format(signal, e))
            return
        return d

    def it_win(self, sig, twin):
        """ Return indices of data within time window.
        If time resolution is too small for a point to fall within the window, return the index of the nearest
        data point
        """
        t = self[sig]['time']
        i = np.where(np.array(t >= twin[0]) & np.array(t <= twin[1]))
        if len(i) == 0:
            if twin[0] >= t[0] and twin[-1] <= t[-1]:
                i = findNearest(t, np.average(twin))
        return i

    def t_win(self, sig, twin='flattop', output='data'):
        """ Return signal data within time window
        """
        d = self[sig]
        if is_scalar(d['data']):
            data = d['data']
            if output == 'data':
                return data
            elif output == 'itd':
                return None, None, data
        if twin == 'flattop':
            if self.twin_ft is None:
                self.find_flattop(signal=self.ft_sig)
            twin = self.twin_ft
        if twin == 'top':
            if self.twin_ft is None:
                self.find_flattop(signal=self.ft_sig)
            twin = self.twin_top
        elif twin == 'all':
            if d is not None:
                twin = [d.time[0], d.time[-1]]
            else:
                twin = None

        if twin is None:  # eg failed to find flattop
            if output=='data':
                return []
            elif output=='itd':
                return [], [], []

        i = self.it_win(sig, twin) if d is not None else []  # If no data return None
        # t = d.time[i] if d is not None else []  # If no data return None
        # d = d.data[i] if d is not None else []  # If no data return None
        t = d['time'][i] if d is not None else []  # If no data return None
        d = d['data'][i] if d is not None else []  # If no data return None
        if len(d) == 0:
            i, t, d = [], [], []

        if output == 'data':
            return d
        elif output == 'itd':
            return i, t, d

    def av_params(self, signals, twin='flattop', header=False):
        if header:
            header = '/t'.join(list(signals))
            return header
        sigs = deepcopy(signals)
        proc_sigs = []
        for sig in sigs:
            if sig in self.processed_signals.keys():
                proc_sigs.append(sig)
                sigs.remove(sig)

        params = OrderedDict()
        for sig in sigs:
            sig = PulseData.sig_short(sig)
            if not self.signal_has_data(sig):
                params[sig] = np.nan
                continue
            value = self.t_win(sig, twin)
            if is_numeric(value):
                value = np.nanmean(value) if value is not None else None
            params[sig] = value

        for sig in proc_sigs:
            params[sig] = self.processed_signals[sig]()
        return params

    def find_flattop(self, signal='Ip', min_length=0.10, threshold=0.023, background=0.4, window='hanning',
                     window_len=125, top_mul=3, test=False, verbose=True):
        """ Find flat top in current signal
        background: Fraction of lower range of signal to be excluded
        """
        if self.pulse is None: return
        if self.twin_ft is not None: return

        ## Get settings for flat top detection for this signal
        if signal in PulseData.ft_sig.keys():
            set = PulseData.ft_sig[signal]
        else:
            set = {'signal': signal, 'min_length': min_length, 'threshold': threshold, 'background': background,
                   'window': window, 'window_len': window_len, 'top_mul': top_mul}
        self.ft_settings = set
        signal = self.sig_long(signal)

        # if signal == 'Ip':
        #     min_length, threshold, background, window, window_len = 0.2, 0.025, 0.4, 'hanning', 125
        # elif signal == 'ne':
        #     min_length, threshold, background, window, window_len = 0.2, 0.25, 0.3, 'hanning', 1000
        # elif signal == 'custom':
        #     pass  # use keyword argument values
        # else:
        #     print('Flat top signal value not recognised ')

        data = self[set['signal']]
        if not self.signal_has_data(set['signal'], data=data):
            logger.warning('Pulse: {}, signal: {}, error: {}'.format(self.pulse, signal, data['errmsg']))
            return None
        else:
            d = data['data']
            t = data['time']
            # d = data.data
            # t = data.time
            d_raw = deepcopy(d)
            t_raw = deepcopy(t)
        ## Remove extreem spikes
        p1 = np.percentile(d, 0.3)
        p99 = np.percentile(d, 99.7)
        if np.max(d) > 2*p99: d[np.where(d>p99)[0]] = p99
        if np.min(d) < 2*p1: d[np.where(d<p1)[0]] = p1
        if np.abs(np.max(d)) > np.abs(np.min(d)):
            ipos = np.where(d > 0.0 + set['background']*(np.max(d)-0.0))[0]
        else:  # reverse current
            ipos = np.where(d < 0.0 - set['background']*(np.min(d)-0.0))[0]

        # Exclude data around zero (using minimum encounters problems when there is a large -ve spike)
        if len(ipos) == 0:
            if verbose: print('No significant current detected for pulse {}'.format(self.pulse))
            return
        dpos = d[ipos]
        # Take median of the non-zero data to establish flattop height
        median = np.median(dpos)

        # diff = 3*np.std(np.diff(dpos))
        # i = np.where(np.logical_and(dpos>median-diff, dpos<median+diff))[0]
        # print(ipos, i)

        # Smooth and then differentiate the set['signal'] # 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        self.dsmooth = smooth(d, window_len=set['window_len'], window=set['window'])
        self.dxdy = conv_diff(self.dsmooth)
        # Smooth the differentiated set['signal']
        self.dxdy = smooth(self.dxdy, window_len=50, window=set['window'])
        # Rescale differential so it can be plotted on the same axes as the data
        self.dxdy *= np.max(d)/np.max(self.dxdy)*3
        # Also get unsmoothed differential
        self.dxdy_raw = conv_diff(d)
        self.dxdy_raw *= np.max(d)/np.max(self.dxdy_raw)*3

        # Find number of peices of data equal to time set['window'] length
        data_length = int(set['min_length'] / (t[1]-t[0]))

        def threshold_section(thresh, type='ft'):
            # set['threshold'] for peak gradient permissable in flat top
            self.thresh = thresh * np.max(np.abs(self.dxdy[ipos]))
            # Get all indices where differential is within set['threshold']
            i = np.where(np.abs(self.dxdy[ipos]) < self.thresh)[0]
            i = ipos[i]
            # Break indices into sets of contiguous indices
            i_, t_, d_ = data_split(t[i], self.dsmooth[i],gap_length=2, data_length=data_length, verbose=False)
            if (np.sum(i_.shape) == 0):
                message = 'No "{}" current section longer than {}s indentified for pulse {}'.format(
                    type, set['min_length'], self.pulse)
                print(message)
                self.__dict__['t_'+type] = None
                self.__dict__['I_'+type] = None
                return None
            else:
                try:
                    ind = np.array([len(x) for x in t_]).argmax()
                except ValueError as e:
                    raise e
                self.__dict__['t_'+type] = t_[ind]
                self.__dict__['I_'+type] = d_[ind]
                return i[i_[ind]]

        self.i_top = threshold_section(set['top_mul']*set['threshold'], type='top')
        self.i_ft = threshold_section(set['threshold'], type='ft')
        self.twin_ft = [t[self.i_ft[0]], t[self.i_ft[-1]]] if self.i_ft is not None else None
        self.twin_top = [t[self.i_top[0]], t[self.i_top[-1]]] if self.i_top is not None else None
        self.twin_pos = [t[ipos][0], t[ipos][-1]]

        if test:
            fig, ax = plt.subplots(1, sharex=True)
            # Original data
            plt.plot(t, d, c='k', lw=1, label='Raw data')
            plt.xlabel(r't $[s]$')
            plt.ylabel(r'{} $[{}]$'.format(self['Ip'].label.decode("utf-8"), self['Ip'].units.decode("utf-8")))
            plt.ylim([-10,np.max(d)*1.1])

            plt.axhline([median], ls='--', c='k')
            plt.axhline([median+self.thresh], ls='--', c='r')
            plt.axhline([median-self.thresh], ls='--', c='r')

            plt.axhline([median+set['top_mul']*self.thresh], ls='--', c='c')
            plt.axhline([median-set['top_mul']*self.thresh], ls='--', c='c')

            print('thresholds: {}, {}'.format(median+self.thresh, median-self.thresh))
            print('Median: {}, self.thresh: {}'.format(median, self.thresh))

            plt.xlim([np.min(t[ipos])-0.2*(np.max(t[ipos])-np.min(t[ipos])),
                      np.max(t[ipos])+0.2*(np.max(t[ipos])-np.min(t[ipos]))])

            plt.plot(t, self.dxdy_raw+median, c='c', label='dy/dx', lw=0.5, alpha=0.4)
            plt.plot(t, self.dsmooth, c='b', label='Smoothed data')
            plt.plot(t, self.dxdy+median, c='r', label='dy/dx (smoothed)', lw=1)

            # plt.plot(d.time[self.i], self.dsmooth[self.i]-412, c='g', ls='-', lw=3, label='Flat top')
            if self.i_top is not None:
                plt.plot(self.t_top, self.I_top, c='c', ls='-', lw=2, label='Top')
                length = '{:0.3f}'.format(t[self.i_top[-1]]-t[self.i_top[0]])
            if self.i_ft is not None:
                plt.plot(self.t_ft, self.I_ft, c='g', ls='-', lw=3, label='Flat top')
                length = '{:0.3f}'.format(t[self.i_ft[-1]]-t[self.i_ft[0]])
            else:
                message = 'No flat top section longer than {}s indentified for pulse {}'.format(
                    set['min_length'], self.pulse)
                plt.annotate(message, xy=(0, 1), xycoords='axes fraction', fontsize=14,
                horizontalalignment='left', verticalalignment='bottom')
                length = '-'
            leg = plt.legend(loc='best', fancybox=True, title='Pulse: {}\nFlat top: {}s'.format(self.pulse, length))
            leg.draggable()
            leg.get_frame().set_alpha(0.8)

            plt.savefig('/home/tfarley/tmp/pulse_filter/flattop_p{}_{}.png'.format(self.pulse, set['signal']), tight=True)
            print('Plotting test plot of flattop detection')
            plt.show()

        return True

    def check_constraint(self, sig, constraint, twin='flattop'):
        accept = True
        if not isinstance(constraint, dict):
            raise ValueError('Unexpected {} constraint format not dict: {}'.format(sig, constraint))
        if all((c is None for c in constraint.values())):
            # Return true if no constrains on this signal
            return True
        # get data over flat top region
        d = self[sig]
        if d['erc'] != 0:
            if ('missing' in constraint) and constraint['missing'] is True:
                return True
            else:
                logger.warning('No {} data for pulse {}'.format(sig, self.pulse))
                return np.nan
        if is_scalar(d['data']):
            data = d['data']
        else:
            if twin == 'flattop':
                data = self.t_win(sig, self.twin_ft)
            else:
                assert(type(twin) is list or type(twin) is np.ndarray)
                data = self.t_win(sig, twin)
        if data == []:
            logger.warning('No {} data in time window: {}'.format(sig, twin))
            return False
        if 'missing' in constraint.keys() and constraint['missing'] is not None:
            accept = False
        if 'equal' in constraint.keys():
            if np.all(data != constraint['equal']):
                accept = False
        if 'not_equal' in constraint.keys():
            if np.all(data == constraint['not_equal']):
                accept = False
        if 'is_in' in constraint.keys():
            if data not in constraint['is_in']:
                accept = False
        if 'contains' in constraint.keys():
            if not is_in(constraint['is_in'], data):
                accept = False
        if 'range' in constraint.keys() and constraint['range'] is not None:
            try:
                if not (np.min(data) >= constraint['range'][0] and np.max(data) <= constraint['range'][1]):
                    accept = False
            except:
                pass
        if 'mean' in constraint.keys() and constraint['mean'] is not None:
            if not (np.nanmean(data) >= constraint['mean'][0] and np.nanmean(data) <= constraint['mean'][1]):
                accept = False
        if 'max' in constraint.keys() and constraint['max'] is not None:
            if not (np.nanmax(data) >= constraint['max'][0] and np.nanmax(data) <= constraint['max'][1]):
                accept = False
        ## Accept if range of values is less than given threshold percentage of mean value in the time window
        if 'percent_fluct' in constraint.keys() and constraint['percent_fluct'] is not None:
            if np.abs(np.nanmax(data)-np.nanmin(data)) >= (constraint['percent_fluct']/100. * np.nanmean(data)):
                accept = False
        ## Accept if deviation from smoothed curve is less than the supplied smoothness value
        if 'smoothness' in constraint.keys() and constraint['smoothness'] is not None:
            data_norm = data / np.max(np.abs(data.min), np.abs(data.max))  # set peak value to 1
            smoothness = np.sum(np.abs(smooth(data_norm, window='hanning', window_len=50)-data_norm))
            if not smoothness < constraint['smoothness']:
                accept = False
        return accept

    def signal_has_data(self, signal, data=None):
        if data is None:
            data = self[signal]
        if data is None:
            return False
        elif ('time' in data) and ('data' in data) and (len(data['time']) > 10):
            exists = True
        elif ('time' not in data) and ('data' in data) and is_scalar(data['data']):
            exists = True
        elif (data['erc'] == -1):
            exists = False
        else:
            logger.warning('Unexpected values for "{}" data dict: {}'.format(signal, data))
            exists = False
        return exists

    def plot(self, signals=['Ip', 'ne', 'q95'], info=None, twin='pos', smooth=None, window_len=50, show=True,
             save=False, path='plots/'):
        if info == 'default':
                info = ['Ip', 'ne', 'q95', 'Bphi', 'Da', 'nx']

        if type(signals) == str:
            signals = [signals]

        fig, axs = plt.subplots(len(signals), sharex=True, figsize=(12, 7))
        if isinstance(axs, matplotlib.axes.Axes):
            axs = np.array([axs])


        for ax, sig in zip(axs, signals):
            data = self[sig]
            if (data is None) or (data['erc'] == -1):
                continue
                return
            try:
                x, y = data['time'], data['data']
            except Exception as e:
                pass

            ax.plot(x, y, c='k', lw=1.5, label='Raw data')

            if smooth:
                d_smooth = smooth(y, window_len=window_len, window=smooth)
                ax.plot(x, y, c='b', lw=1, label='Smoothed data')

            i_top, t_top, d_top = self.t_win(sig, twin='top', output='itd')
            if not i_top == []:  # Check there is a 'top'
                ax.plot(t_top, d_top, c='c', lw=1.5, label='Top')
            i_ft, t_ft, d_ft = self.t_win(sig, twin='flattop', output='itd')
            if not i_ft == []:  # Check there is a flatop
                ax.plot(t_ft, d_ft, c='b', lw=1.5, label='Flat top')

            ax.tick_params(axis='both', which='both', bottom='on', left='on')

            ax.set_xlabel(r't $[s]$')
            ylabel = r'{} [{}]'.format(str2tex(data['dlabel']), str2tex(data['dunits']))
            ylabel = ylabel.replace('Integrated Dalpha', r'$D_\alpha$')
            ylabel = ylabel.replace('Plasma Current', r'$I_p$')
            if sig == 'dn_edge':
                ylabel = r'$dn_{edge} [m^{-4}]$'
            ax.set_ylabel(ylabel)  # .decode("utf-8")

            if sig == 'ne':
                ## Make sure axis limits are sensible - cut out spikes etc
                lims = list(ax.axis())
                if lims[2] < 0: lims[2] = 0  # ymin axis limit
                if lims[3] > 5e19: lims[3] = np.percentile(y, 99)*1.1  # ymax
                ax.axis(lims)

        if twin == 'pos':
            t_lim = self.twin_pos
        elif twin == 'ft' or 'flattop':
            t_lim = t_ft if t_ft != [] else t_top
        elif twin == 'top':
            t_lim = t_top
        else:
            t_lim = x

        i_top, t_top, d_top = self.t_win('Ip', twin='top', output='itd')
        # t_lim = t_top
        if t_lim !=[]: ax.set_xlim([np.max([0, t_lim[0]-0.3*(t_lim[-1]-t_lim[0])]),
                                               t_lim[-1]+0.2*(t_lim[-1]-t_lim[0])])
        if self.twin_ft is not None:
            title = 'Pulse: {}\nFlat top: {:0.3f}s'.format(self.pulse, self.twin_ft[1]-self.twin_ft[0])
        else:
            title = 'Pulse: {}\nFlat top: None detected'.format(self.pulse)
        xy = (0.34, 0.735) if len(signals)==3 and info is not None else \
                            (0.5-0.16*(info is not None), 1-1/(len(signals)+1))
        ax.annotate(title, xy=xy, xycoords='figure fraction',
                    bbox=dict(boxstyle="round, pad=0.5", fc="w", alpha=0.7))
        try:
            leg = ax.legend(loc='best', fancybox=True, title=None)
            leg.draggable()
            leg.get_frame().set_alpha(0.8)
        except AttributeError as e:
            logger.warning('Failed to create legend')

        if info is not None:
            title = self.print(signals=info, stdout=False)
            figleg = fig.legend([], [] , loc=2, title=title,
                                bbox_to_anchor=(0.76, 0.965), bbox_transform=plt.gcf().transFigure)
            plt.subplots_adjust(left=0.15, bottom=0.12, right=0.76, top=0.95)
        else:
            plt.tight_layout()

        # plot_margin = 1
        # x0, x1, y0, y1 = plt.axis()
        # plt.axis((x0, x1+plot_margin, y0, y1))

        if show:
            print('Plotting {}'.format(signals))
            plt.show()
        if save:
            fn = '{}_{}.png'.format(self.pulse, '-'.join(signals))
            mkdir(path, depth=2)
            plt.savefig(path+fn, transparent=True, dpi=100)
            print('Saved plot of signals {} for pulse {} to {}'.format(', '.join(signals), self.pulse, path+fn))
        return ax

    def plot_multi(self, signals=['Ip', 'ne', 'q95']):

        for sig in signals:
            ax = self.plot(sig)

    def print(self, signals=['nx', 'Ip', 'ne'], twin='flattop', stdout=True):
        out = [('Signal', 'Value')]

        if (twin == 'flattop' or twin == 'ft') and self.twin_ft == None:
            twin = 'top'

        sigs = deepcopy(signals)
        if 'nx' in sigs:  # Number of x-points
            sigs.remove('nx')
        for sig in sigs:
            av = self.av_params(sigs, twin=twin)
        if 'nx' in signals:  # Estimate number of x-points from position of magnetic axis
            zav = self.av_params(['zmag'], twin=twin)
            print(zav)
            if zav['zmag'] < -0.01:
                av['nx'] = 'LSN'
            else:
                av['nx'] = 'DND'

        out = out+[(key, list) for key, list in av.items()]
        out = zip(list(av.keys()), list(av.values()))
        out = [(key, av[key]) for key in av.keys()]
        # print(out)
        # df = pd.DataFrame(av)
        ser=pd.Series(av)
        # pd.set_option('precision', 3)
        pd.options.display.float_format = '{:,.3g}'.format
        pd.set_option('colheader_justify', 'left')
        pd.set_option('column_space', 6)
        if stdout: print(ser)

        out = str(ser)
        if '\ndtype: object' in out:
            out = out.replace('\ndtype: object', '')
        if '\ndtype: float64' in out:
            out = out.replace('\ndtype: float64', '')

        title = 'Flat top averages\n' if (twin == 'flattop' or twin == 'ft') else 'Top averages\n'
        return title+out

    def format_str(self, string):
        # print(type(string))
        # if string[0:2] == "b'" and string[-1] == "'":
        #     string = string[2:-2]
        return string.decode("utf-8")

    def xpoint_config(self, twin='top'):
        """ Use position of magnetic axis to infer if the plasma is in a LSND or DND configuration
        """
        if 'nx' in self.store.keys():
            return self.store['nx']

        if twin == 'top':
            self.find_flattop('Ip')
            if self.twin_top is None:
                twin = 'all'

        zav = self.av_params(['zmag'], twin=twin)
        if zav['zmag'] < -0.01:
            self.store['nx'] = 'LSN'
        else:
            self.store['nx'] = 'DND'

        return self.store['nx']

    def confinement_mode(self, twin='top'):
        """ Use edge density gradient to infer if the plasma is in L-mode or H-mode
        """
        ## TODO: Could also add check to see if Dalpha drops bellow eg 50% of overal maximum during flat top
        ## TODO: Could also add check to 'smoothness' of Dalpha to get indication of ELMS
        if 'mode' in self.store.keys():
            return self.store['mode']

        if twin == 'top':
            self.find_flattop('Ip')
            if self.twin_top is None:
                twin = 'all'

        grad = self.av_params(['dn_edge'], twin=twin)
        if grad['dn_edge'] > 0.3e21:
            self.store['mode'] = 'H-mode'
        else:
            self.store['mode'] = 'L-mode'

        return self.store['mode']
