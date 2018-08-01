try:
    import cpickle as pickle
except ImportError:
    import pickle
import logging, os
import pandas as pd
from ccfepyutils.io_tools import pickle_dump, pickle_load, mkdir
from ccfepyutils.utils import make_iterable
logger = logging.getLogger(__name__)

signal_abbreviations = {
    'Ip': "amc_plasma current",
    'ne': 'ayc_ne_core',  # Core Thomson scattering data - (peak?) Electron Density
    'ne2': "ESM_NE_BAR",  # (electron line averaged density) - Gives better plots - same as session logs
    'ne3': "ane_density",  # Gives lower density - what is this? - CO2 Interferometry
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
# TODO: Remove duplication of abbreviations in pulse_data.py

signal_ylabels = {'ESM_NE_BAR': r'$n_e$ [$\times 10^{19}$ m$^{-3}$]',
                  'ayc_ne_core': r'$n_e$ [$\times 10^{19}$ m$^{-3}$]',
                   'amc_plasma current': '$I_p$ [kA]',
                   "ada_dalpha integrated": r'$D_{\alpha}$ [ph.$s^{-1}cm^{-2}sr^{-1}$]',
                   'anb_tot_sum_power': '$P_{NBI}$ [MW]',
                   'xsx/tcam/1': '$I_{sXray}$ [V]',
                   'efm_q_axis': '$q_0$',
                  }

session_log_columns = ['useful', 'shot_type', 'preshot', 'postshot', 'plasma_shape', 'ip_range', 'heating',
                       'divertor_config', 'rmptype', 'scenario', 'pellets']

pycpf_columns = [('exp_number', 'MAST Experiment Pulse Number'),
                 ('exp_date', 'MAST Experiment Pulse Date'),
                 ('exp_time', 'MAST Experiment Pulse Time'),
                 ('preshot', 'Session Leaders Pre-Shot Comment'),
                 ('postshot', 'Session Leaders Post-Shot Comment'),
                 ('creation', 'Data Dictionary Entry Creation Date'),
                 ('gdc_duration', 'Glow Discharge Duration'),
                 ('ibgas_pressure', 'Inboard Gas Pressure prior to Pulse'),
                 ('sl', 'Session Leader'),
                 ('pic', 'Physicist in Charge'),
                 ('gdc_time', 'GDC Time'),
                 ('abort', 'Shot Aborted'),
                 ('useful', 'Useful Shot'),
                 ('reference', 'Reference Shot'),
                 ('program', 'Scientific Program'),
                 ('objective', 'Scientific Objective'),
                 ('summary', 'Session Summary'),
                 ('term_code', 'Plasma Termination Reason Code'),
                 ('tstart', 'Plasma Current Start Time'),
                 ('tend', 'Plasma Current End Time'),
                 ('tipmax', 'Time of Maximum Plasma Current'),
                 ('truby', 'Ruby Thomson Scattering Time'),
                 ('tftstart', 'Plasma Current Flat-Top Start Time'),
                 ('tftend', 'Plasma Current Flat-Top End Time'),
                 ('tstart_ibgas', 'Inboard Gas Puff Start Time'),
                 ('tend_ibgas', 'Inboard Gas Puff End Time'),
                 ('tte0_max', 'Time of Maximum Core Electron Temperature (from Nd.YAG TS)'),
                 ('tne0_max', 'Time of Maximum Core Electron Density (from Nd.YAG TS)'),
                 ('tpe0_max', 'Time of Maximum Core Electron Pressure (from Nd.YAG TS)'),
                 ('tpnbi_max_ss', 'Time of Maximum SS NBI Power'),
                 ('tpnbi_max_sw', 'Time of Maximum SW NBI Power'),
                 ('tstart_nbi_ss', 'SS NBI Start Time'),
                 ('tend_nbi_ss', 'SS NBI End Time'),
                 ('tstart_nbi_sw', 'SW NBI Start Time'),
                 ('tend_nbi_sw', 'SW NBI End Time'),
                 ('tpohm_max', 'Time of Maximum Ohmic Heating'),
                 ('tndl_co2_max', 'Time of Maximum Electron Density Line Integral'),
                 ('tamin_max', 'Time of Maximum Plasma Minor Radius'),
                 ('tarea_max', 'Time of Maximum Poloidal Cross-Sectional Area'),
                 ('tbepmhd_max', 'Time of Maximum Beta Poloidal MHD'),
                 ('tbetmhd_max', 'Time of Maximum Beta MHD'),
                 ('tbt_max', 'Time of Maximum Toroidal Magnetic Field Strength'),
                 ('tdwmhd_max', 'Time of Maximum Rate of Change of MHD Stored Energy'),
                 ('tkappa_max', 'Time of Maximum Plasma Elongation'),
                 ('tprad_max', 'Time of Maximum Plasma Radiated Power Loss'),
                 ('tq95_min', 'Time of Minimum q 95'),
                 ('trgeo_max', 'Time of Maximum Geometrical Mid-plane Center Major Radius'),
                 ('tsarea_max', 'Time of Maximum Plasma Surface Area'),
                 ('ttautot_max', 'Time of Maximum Total Energy Confinement Time'),
                 ('tvol_max', 'Time of Maximum Plasma Volume'),
                 ('twmhd_max', 'Time of Maximum MHD Stored Energy'),
                 ('tzeff_max', 'Time of Maximum Plasma Z-Effective'),
                 ('pohm_ipmax', 'Ohmic Heating Rate at time of Peak Current'),
                 ('johm_ipmax', 'Ohmic Heating Energy Input at time of Peak Current'),
                 ('johm_total', 'Total Ohmic Heating Energy Input'),
                 ('pnbi_max_ss', 'Peak NBI Power from SS Beam'),
                 ('pnbi_max_sw', 'Peak NBI Power from SW Beam'),
                 ('enbi_max_ss', 'NBI Injection Energy from SS Beam at time of Peak power'),
                 ('enbi_max_sw', 'NBI Injection Energy from SW Beam at time of Peak power'),
                 ('pnbi_ipmax_ss', 'NBI Power from SS Beam at time of Peak Current'),
                 ('pnbi_ipmax_sw', 'NBI Power from SW Beam at time of Peak Current'),
                 ('jnbi_ipmax_ss', 'NBI Injected Energy from SS Beam at time of Peak Current'),
                 ('jnbi_ipmax_sw', 'NBI Injected Energy from SW Beam at time of Peak Current'),
                 ('jnbi_total_ss', 'Total NBI Injected Energy from SS Beam '),
                 ('jnbi_total_sw', 'Total NBI Injected Energy from SW Beam '),
                 ('pnbi_truby_ss', 'NBI Power from SS Beam at time of Ruby TS'),
                 ('pnbi_truby_sw', 'NBI Power from SW Beam at time of Ruby TS'),
                 ('pohm_max', 'Maximum Ohmic Heating Rate'),
                 ('pohm_truby', 'Ohmic Heating Rate at time of Ruby TS'),
                 ('johm_max', 'Ohmic Heating Energy Input at time of Maximum Ohmic Heating'),
                 ('johm_truby', 'Ohmic Heat Energy Input at time of Ruby TS'),
                 ('te0_max', 'Peak Core Electron Temperature (Nd.YAG)'),
                 ('ne0_max', 'Peak Core Electron Density (Nd.YAG)'),
                 ('ndl_co2_max',
                  'Maximum Electron Density Line Integral observed by CO2 Interferometer'),
                 ('dwmhd_max', 'Maximum Rate of Change of Total Stored Energy'),
                 ('prad_max', 'Maximum Total Radiated Power Loss'),
                 ('tautot_max', 'Maximum value of Energy Confinement Time'),
                 ('wmhd_max', 'Maximum Stored Energy'),
                 ('zeff_max', 'Maximum Plasma Z-Effective'),
                 ('te0_ipmax', 'Core Electron Temperature (Nd.YAG) at time of Peak Current'),
                 ('ne0_ipmax', 'Core Electron Density (Nd.YAG) at time of Peak Current'),
                 ('ndl_co2_ipmax',
                  'Electron Density Line Integral at time of Peak Plasma Current'),
                 ('dwmhd_ipmax',
                  'Rate of Change of Total Stored Energy at time of Peak Plasma Current'),
                 ('prad_ipmax', 'Total Radiated Power Loss at time of Peak Plasma Current'),
                 ('tautot_ipmax', 'Energy Confinement Time at time of Peak Plasma Current'),
                 ('wmhd_ipmax', 'Stored Energy at time of Peak Plasma Current'),
                 ('zeff_ipmax', 'Plasma Z-Effective at time of Peak Plasma Current'),
                 ('te0ruby', 'Ruby TS Core Electron Temperature'),
                 ('te0_truby', 'Core Electron Temperature (Nd.YAG) at time of Ruby TS'),
                 ('ne0ruby', 'Ruby Core Electron Density'),
                 ('ne0_truby', 'Core Electron Density (Nd.YAG) at time of Ruby TS'),
                 ('ndl_co2_truby', 'Electron Density Line Integral at time of Ruby TS'),
                 ('dwmhd_truby',
                  'Rate of Change of Total Stored MHD Energy at time of Ruby TS'),
                 ('prad_truby', 'Total Radiated Power Loss at time of Ruby TS'),
                 ('tautot_truby', 'Energy Confinement Time at time of Ruby TS'),
                 ('wmhd_truby', 'Stored Energy at time of Ruby TS'),
                 ('zeff_truby', 'Plasma Z-Effective at time of Ruby TS'),
                 ('pe0ruby', 'Ruby TS Core Electron Pressure'),
                 ('pe0_truby', 'Core Electron Pressure (Nd.YAG) at time of Ruby TS'),
                 ('pe0_max', 'Maximum value of Core Electron Pressure (Nd.YAG)'),
                 ('pe0_ipmax',
                  'Core Electron Pressure (Nd.YAG) at time of Peak Plasma Current'),
                 ('rinner_da', 'Rinner major Radius from Visible D_Alpha Light'),
                 ('router_da', 'Router major Radius from Visible D_Alpha Light'),
                 ('rinner_efit', 'Rinner major Radius from EFIT Equilibrium'),
                 ('router_efit', 'Router major Radius from EFIT Equilibrium'),
                 ('rmag_efit', 'Magnetic Axis major Radius from EFIT Equilibrium'),
                 ('zmag_efit', 'Magnetic Axis height above Mid-Plane from EFIT Equilibrium'),
                 ('amin_max', 'Maximum value of Minor Radius'),
                 ('area_max', 'Maximum Poloidal Cross-Sectional Area'),
                 ('kappa_max', 'Maximum value of Plasma Elongation'),
                 ('rgeo_max', 'Maximum value of Geometrical Center Major Radius'),
                 ('sarea_max', 'Maximum Total Surface Area'),
                 ('vol_max', 'Maximum Plasma Volume'),
                 ('amin_ipmax', 'Minor Radius at time of Peak Plasma Current'),
                 ('area_ipmax',
                  'Poloidal Cross-Sectional Area at time of Peak Plasma Current'),
                 ('kappa_ipmax', 'Plasma Elongation at time of Peak Plasma Current'),
                 ('rgeo_ipmax',
                  'Geometrical Center Major Radius at time of Peak Plasma Current'),
                 ('sarea_ipmax', 'Total Surface Area at time of Peak Plasma Current'),
                 ('vol_ipmax', 'Plasma Volume at time of Peak Plasma Current'),
                 ('amin_truby', 'Minor Radius at time of Ruby TS'),
                 ('area_truby', 'Poloidal Cross-Sectional Area at time of Ruby TS'),
                 ('kappa_truby', 'Plasma Elongation at time of Ruby TS'),
                 ('rgeo_truby', 'Geometrical Center Major Radius at time of Ruby TS'),
                 ('sarea_truby', 'Total Surface Area at time of Ruby TS'),
                 ('vol_truby', 'Plasma Volume at time of Ruby TS'),
                 ('li_3_ipmax', 'li(3) at time of Peak Current'),
                 ('bepmhd_max', 'Maximum Beta poloidal'),
                 ('betmhd_max', 'Maximum value of Beta'),
                 ('bt_max', 'Maximum Toroidal Field Strength'),
                 ('q95_min', 'Minimum value of q-95'),
                 ('bepmhd_ipmax', 'Beta poloidal at time of Peak Plasma Current'),
                 ('betmhd_ipmax', 'Beta at time of Peak Plasma Current'),
                 ('bt_ipmax', 'Toroidal Field Strength at time of Peak Plasma Current'),
                 ('q95_ipmax', 'q-95 at time of Peak Plasma Current'),
                 ('bepmhd_truby', 'Beta poloidal at time of Ruby TS'),
                 ('betmhd_truby', 'Beta at time of Ruby TS'),
                 ('bt_truby', 'Toroidal Field Strength at time of Ruby TS'),
                 ('q95_truby', 'q-95 at time of Ruby TS'),
                 ('li_2_truby', 'li(2) at time of Ruby TS'),
                 ('jnbi_truby_sw', 'NBI Injected Energy from SW Beam at time of Ruby TS'),
                 ('jnbi_truby_ss', 'NBI Injected Energy from SS Beam at time of Ruby TS'),
                 ('jnbi_max_sw', 'NBI Injected Energy from SW Beam at time of Peak power'),
                 ('jnbi_max_ss', 'NBI Injected Energy from SS Beam at time of Peak power'),
                 ('jnbi_max', 'NBI Injected Energy from SS+SW Beams at time of Peak power'),
                 ('jnbi_ipmax',
                  'NBI Injected Energy from SS+SW Beams at time of Peak Current'),
                 ('jnbi_truby', 'NBI Injected Energy from SS+SW Beams at time of Ruby TS'),
                 ('jnbi_total', 'Total NBI Injected Energy from SS+SW Beams'),
                 ('pnbi_ipmax', 'NBI Power from SS+SW Beams at time of Peak Current'),
                 ('pnbi_truby', 'NBI Power from SS+SW Beams at time of Ruby TS'),
                 ('pnbi_max', 'NBI Power from SS+SW Beams at time of Peak power'),
                 ('tstart_nbi', 'NBI Start Time'),
                 ('tend_nbi', 'NBI End Time'),
                 ('tpnbi_max', 'Time of Maximum NB Power'),
                 ('column_temp_in', 'Centre Column Inlet Temperature'),
                 ('column_temp_out', 'Centre Column Outlet Temperature'),
                 ('log_base_pressure', 'TG1 Base Pressure prior to Shot'),
                 ('li_2_max', 'li(2) Maximum value'),
                 ('li_2_ipmax', 'li(2) at time of Peak Current'),
                 ('li_3_max', 'li(3) Maximum value'),
                 ('li_3_truby', 'li(3) at time of Ruby TS'),
                 ('tli_2_max', 'Time of Maximum li(2)'),
                 ('tli_3_max', 'Time of Maximum li(3)'),
                 ('pradne2',
                  'Flat Top Ratio of Power Radiated to Line Density squared averaged over Ip flat top from 150ms'),
                 ('ne0ratio_ipmax', 'Ratio of ne0 to line average ne'),
                 ('c2ratio', 'CII Line Ratio averaged over Ip flat top from 150ms'),
                 ('o2ratio', 'OII Line Ratio averaged over Ip flat top from 150ms'),
                 ('te0ratio_ipmax', 'ratio of Te0 to line average Te'),
                 ('ip_av', 'Time Averaged Plasma Current'),
                 ('ip_max', 'Maximum Value of Plasma Current'),
                 ('ne_bar_ipmax',
                  'Mid-plane line average electron density from CO2 interferometer'),
                 ('ngreenwaldratio_ipmax',
                  'Ratio of Greenwald density limit to mid-plane line averaged electron density'),
                 ('te_yag_bar_ipmax',
                  'Mid-plane line average electron temperature from YAG Thomson scattering'),
                 ('ne_yag_bar_ipmax',
                  'Mid-plane line average electron density from YAG Thomson scattering'),
                 ('ngreenwald_ipmax', 'Greenwald Density Limit')]

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

def get_data(signal, pulse, save_path='~/data/MAST_signals/', save=True, load_saved=True, *args, **kwargs):
    """Get data with IDL_bridge getdata if available, else load from pickle store."""
    pulse = int(pulse)
    if signal in signal_abbreviations:
        signal = signal_abbreviations[signal]
    save_path = os.path.expanduser(save_path)
    if save or load_saved:
        pulse_str = '{pulse:d}'.format(pulse=pulse)
        fn = signal.replace('/', '_').replace(' ', '_')+'.p'
        mkdir(os.path.join(save_path, pulse_str), start_dir=save_path)
        fn_path = os.path.join(save_path, pulse_str, fn)
    success = False
    if load_saved:
        # First try to load from pickle file
        try:
            d = pickle_load(fn_path)
            logger.info('Loaded signal "{}" for pulse {} from pickle file'.format(signal, pulse))
            success = True
        except IOError as e:
            logger.debug('Could not load signal "{}" for pulse {} in "{}", {}'.format(signal, pulse, save_path, e))
    if not success:
        try:
            import idlbridge as idl
            getdata = idl.export_function("getdata")
            d = getdata(signal, pulse, *args, **kwargs)

            if d['erc'] != 0:
                d2 = get_session_log_data(signal, pulse)
                if d2['erc'] == 0:
                    d = d2
            if d['erc'] != 0:
                logger.warning('Failed to load data for {}: {}'.format(pulse_str, signal))
            elif save:
                pickle_dump(d, fn_path, protocol=2)
                logger.info('Saved data for {}; "{}" to: {}'.format(pulse_str, signal, os.path.join(save_path, pulse_str, fn)))
        except ImportError as e:
            logger.error(e)
    if success:
        return d
    else:
        return d


path = os.path.expanduser('~/repos/ccfepyutils/misc/')
fn = 'session_log_info_29737_30023.csv'
fn_path = os.path.join(path, fn)
df_session_log = pd.read_csv(fn_path, header=34, index_col=0)
logger.info('Loaded session logs csv: {}'.format(fn_path))
def get_session_log_data(signal, pulse):
    """Get information from session logs

    Temporary fix reading from local csv file until have proper access to CPF data"""
    # from ccfepyutils.io_tools import
    out = {'name': signal, 'pulse': pulse, 'data': None, 'erc': -1}
    try:
        out['data'] = df_session_log.loc[pulse, signal]
        out['erc'] = 0
    except KeyError as e:
        pass
    except Exception as e:
        raise e
    if out['erc'] != 0:
        try:
            import pycpf
            out['data'] = pycpf.query([signal], filters=['exp_number = {}'.format(pulse)])[signal][0]
            out['erc'] = 0
        except ImportError as e:
            logger.warning('{}'.format(e))
        except Exception as e:
            logger.debug(e)
    return out


def store_mast_signals(signals, pulses, save_path='~/data/MAST_signals/', *args, **kwargs):
    if isinstance(signals, (str)) and signals in signal_sets:
        signals = signal_sets[signals]
    pulses = make_iterable(pulses)
    signals = make_iterable(signals)
    save_path = os.path.expanduser(save_path)
    assert os.path.isdir(save_path), 'Save path does not exist'
    for pulse in pulses:
        for signal in signals:
            get_data(signal, pulse, save_path=save_path, noecho=1, *args, **kwargs)
