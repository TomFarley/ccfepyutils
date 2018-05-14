import configparser
import os
import logging
import pickle
import re
from pathlib import Path
from tkinter import Tk, filedialog as askopenfilename

import numpy as np
import xarray as xr
from nested_dict import nested_dict
from past.types import basestring

from ccfepyutils.utils import string_types, signal_abbreviations, logger, signal_sets, make_itterable, compare_dict, \
    is_number, is_subset

try:
    from natsort import natsorted
    sorted = natsorted
except ImportError as e:
    logger.debug('Please install natsort for improved sorting')

logger = logging.getLogger(__name__)

def create_config_file(fn, dic):
    """Create ini config file structured around supplied dictionary"""
    config = configparser.ConfigParser()
    for key, value in dic.items():
        if isinstance(value, dict):
            config[key] = {}
            for k, v in value.items():
                config[key][k] = v
        else:
            config[key] = value
    with open(fn, 'w') as configfile:
        config.write(configfile)
    logging.info('Wrote config file to {}. Sections: {}'.format(fn, config.sections()))


def get_from_ini(config, setting, value):
    """Return value for setting from config ini file if value is None"""
    raise NotImplementedError

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
    pulses = make_itterable(pulses)
    signals = make_itterable(signals)
    save_path = os.path.expanduser(save_path)
    assert os.path.isdir(save_path), 'Save path does not exist'
    for pulse in pulses:
        for signal in signals:
            get_data(signal, pulse, save_path=save_path, noecho=1, *args, **kwargs)


def rm_files(path, pattern, verbose=True):
    path = str(path)
    if verbose:
        print('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
    for fn in os.listdir(path):
        if re.search(pattern, fn):
            os.remove(os.path.join(path, fn))
            if verbose:
                print('Deleted file: {}'.format(fn))


def getUserFile(type=""):
    Tk().withdraw()
    filename = askopenfilename(message="Please select "+type+" file:")
    return filename


def filter_files_in_dir(path, fn_pattern, group_keys=(), raise_on_incomplete_match=False, **kwargs):
    """Return dict of filenames in given directory that match supplied regex pattern

    The keys of the returned dict are the matched groups for each file from the fn_pattern.
    :param path: path in which to search for/filter filenames
    :param fn_pattern: regex pattern to match against files. kwargs will be substituted into the pattern (see example)
    :param group_keys: list that links the ordering of the regex groups to the kwargs keys. Warnings are raised
                         if not all of the kwarg values are mmatched to files.
    :param raise_on_incomplete_match: raise an exception if not all kwarg values are located
    :param kwargs: values are substituted into the fn_pattern (provided the pattern contains a format key matching that
                    of the kwarg) with lists/arrays of values converted to the appropriate regex pattern.
    e.g. to get the files with values of n between 10 and 50 use. The returned dict will be keyed by the number n and
    the last group in the filename (<n>, <.*>). You will be warned if any of the requested n's are not found.
    fn_pattern = 'myfile-n({n})-(.*).nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(20,51))

    """
    from ccfepyutils.utils import PartialFormatter
    fmt = PartialFormatter()

    path = Path(path)
    path = path.expanduser()
    if not path.is_dir():
        raise IOError('Search directory "{}" does not exist'.format(path))
    # If kwargs are supplied convert them to re patterns
    re_patterns = {}
    for key, value in kwargs.items():
        if isinstance(value, (np.ndarray, list, tuple)):  # and is_number(value[0]):
            # List/array of numbers, so match any number in list
            re_patterns[key] = '[{}]'.format('|'.join([str(v) for v in value]))
    # fn_pattern = fn_pattern.format(**re_patterns)
    fn_pattern = fmt.format(fn_pattern, **re_patterns)
    filenames_all = sorted(os.listdir(str(path)))
    out = {}
    i = 0
    for fn in filenames_all:
        m = re.search(fn_pattern, fn)
        if m is None:
            continue
        ngroups = len(m.groups())
        if ngroups == 0:
            key = i
        elif ngroups == 1:
            key = m.groups()[0]
        else:
            key = m.groups()
        n = m.groups()  # Get frame number of match
        out[key] = fn
        i += 1

    if len(out) == 0:
        raise IOError('Failed to locate any files with pattern "{}" in {}'.format(fn_pattern, path))
    for i, group_key in enumerate(group_keys):
        if (group_key not in kwargs) or (kwargs[group_key] is None):
            continue
        # List of located values for group cast to same type
        located_values = [type(kwargs[group_key][0])(key[i]) for key in out.keys()]
        if not is_subset(kwargs[group_key], list(located_values)):
            message = 'Could not locate files with {} = {}'.format(group_key,
                                                                  set(kwargs[group_key]) - set(located_values))
            if raise_on_incomplete_match:
                raise RuntimeError(message)
            else:
                logger.warning(message)
    return out

# def filter_files_in_dir(path, extension='.p', contain=None, not_contain=None):
#
#     for (dirpath, dirnames, filenames) in os.walk(path_in):
#         break  # only look at files in path_in
#     if extension is not None:
#         filenames = [f for f in filenames if f[-len(extension):] == extension]  # eg only pickle files
#     if contain is not None:
#         if isinstance(contain, basestring):
#             contain = [contain]
#         for pat in contain:
#             filenames = [f for f in filenames if pat in f] # only files with fixed variable
#     if not_contain is not None:
#         if isinstance(not_contain, basestring):
#             not_contain = [not_contain]
#         for pat in not_contain:
#             filenames = [f for f in filenames if pat not in f] # only files with fixed variable
#
#     fn = filenames[0]
#
#     return filenames


def mkdir(dirs, start_dir=None, depth=None, info=None, verbose=False):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary. Alternatively, if a depth
    is supplied only the last <depth> levels of directories will be created i.e. the path <depth> levels above must
    pre-exist.
    Inputs:
        dirs 			- Directory path
        start_dir       - Path from which all new directories must branch
        depth           - Maximum levels of directories what will be created for each path in <dirs>
        info            - String to write to DIR_INFO.txt file detailing purpose of directory etc
        verbatim = 0	- True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    from pathlib import Path
    # raise NotImplementedError('Broken!')
    if start_dir is not None:
        start_dir = os.path.expanduser(str(start_dir))
        if isinstance(start_dir, Path):
            start_dir = str(start_dir)
        start_dir = os.path.abspath(start_dir)
        if not os.path.isdir(start_dir):
            print('Directories {} were not created as start directory {} does not exist.'.format(dirs, start_dir))
            return 1

    if isinstance(dirs, Path):
        dirs = str(dirs)
    if isinstance(dirs, basestring):  # Nest single string in list for loop
        dirs = [dirs]
    # import pdb; pdb.set_trace()
    for d in dirs:
        if isinstance(d, Path):
            d = str(d)
        d = os.path.abspath(os.path.expanduser(d))
        if depth is not None:
            depth = np.abs(depth)
            d_up = d
            for i in np.arange(depth):  # walk up directory by given depth
                d_up = os.path.dirname(d_up)
            if not os.path.isdir(d_up):
                print('Directory {} was not created as start directory {} (depth={}) does not exist.'.format(
                    d, d_up, depth))
                continue
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                print('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            os.makedirs(d)
            print('Created directory: ' + d)
            if info:  # Write file describing purpose of directory etc
                with open(os.path.join(d, 'DIR_INFO.txt'), 'w') as f:
                    f.write(info)
        else:
            if verbose:
                print('Directory "' + d + '" already exists')
    return 0


def test_pickle(obj):
    """Test if an object can successfully be pickled and loaded again
    Returns True if succeeds
            False if fails
    """
    import pickle
    # sys.setrecursionlimit(10000)
    path = 'test_tmp.p.tmp'
    if os.path.isfile(path):
        os.remove(path)  # remove temp file
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print('Pickled object')
        with open(path, 'rb') as f:
            out = pickle.load(f)
        print('Loaded object')
    except Exception as e:
        print('{}'.format(e))
        return False
    if os.path.isfile(path):
        print('Pickled file size: {:g} Bytes'.format(os.path.getsize(path)))
        os.remove(path)  # remove temp file
    import pdb; pdb.set_trace()
    print('In:\n{}\nOut:\n{}'.format(out, obj))
    if not isinstance(obj, dict):
        out = out.__dict__
        obj = obj.__dict__
    if compare_dict(out, obj):
        return True
    else:
        return False


def pickle_dump(obj, path, **kwargs):
    """Wrapper for pickle.dump, accepting multiple path formats (file, string, pathlib.Path).
    - Automatically appends .p if not pressent.
    - Uses cpickle when possible.
    - Automatically closes file objects."""
    if isinstance(path, Path):
        path = str(path)

    if isinstance(path, basestring):
        if path[-2:] != '.p':
            path += '.p'
        with open(path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif isinstance(path, file):
        pickle.dump(obj, path, **kwargs)
        path.close()
    else:
        raise ValueError('Unexpected path format')


def pickle_load(path, base=None, **kwargs):
    if isinstance(path, Path):
        path = str(path)

    if base is not None:
        path = os.path.join(base, path)

    if isinstance(path, basestring):
        if path[-2:] != '.p':
            path += '.p'
        with open(path, 'rb') as f:
            try:
                out = pickle.load(f, **kwargs)
            except EOFError as e:
                logger.error('path "{}" is not a pickle file. {}'.format(path, e))
                raise e
            except UnicodeDecodeError as e:
                logger.error('Failed to read pickle file "{}". Wrong pickle protocol? {}'.format(path, e))
                raise e
    elif isinstance(path, file):
        out = pickle.load(path, **kwargs)
        path.close()
    else:
        raise ValueError('Unexpected path format')
    return out


def to_csv(x, ys, fn, xheading=None, yheadings=None, description='data'):
    """Quickly and easily save data to csv, with one dependent variable"""
    import pandas as pd
    ys = np.squeeze(ys)
    if ys.shape[0] != len(x) and ys.shape[1] == len(x):
        ys = ys.T
    df = pd.DataFrame(data=ys, index=make_itterable(x), columns=yheadings)
    if xheading:
        df.index.name = xheading
    # df.columns.name = 'tor'
    df.to_csv(str(fn))
    logger.info('Saved {} to: {}'.format(description, str(fn)))


def out_path(input, default_path, default_fn, default_extension, path_obj=False):
    """Generate output path given input and variable input.

    Any information missing in input with be replaced with defaults.
    Input can be eg:
    Full path including filename and extension
    Just path, no filename
    Just filename, no path
    Not a string/Path, in which case defaults will be used """
    default_path = os.path.expanduser(default_path)
    try:
        if isinstance(input, Path):
            input = str(input)
        input = os.path.expanduser(input)
        if os.path.isdir(input) and input[-1] != os.sep:  # Make sure a directory ends in slash
            input += os.sep
        # Split up input information
        path, fn = os.path.split(input)
        base, ext = os.path.splitext(fn)
        if len(ext) > 5:  # Avoid full stops in filenames being treated as extensions - 4 char ext len limit
            base += ext
            ext = ''

        # Replace missing information with defaults
        if not path:
            path = default_path
        if not fn:
            fn = os.path.splitext(default_fn)[0]
        if not ext:
            ext = default_extension
        fn = join_with_one('.', fn, ext)
        out = os.path.join(path, fn)

    except AttributeError as e:
        # TODO: allow no extension and extension included in default_fn
        fn = join_with_one('.', default_fn, default_extension)
        out = os.path.join(default_path, fn)
    if path_obj:
        out = Path(out)
    return out


def join_with_one(sep, *args):
    """Join strings with exactly one separator"""
    l = len(sep)
    out = ''
    for i, (arg1, arg2) in enumerate(zip(args, args[1:])):
        arg12 = [arg1, arg2]
        if i == 0:
            # If arg1 ends with separator, remove it
            while arg12[0].endswith(sep):
                arg12[0] = arg12[0][:len(arg12[0])-l]
            out += arg12[0]
        # check if arg2 begins or ends with separator (need to loop over mutable!)
        while arg12[1].startswith(sep):
            arg12[1] = arg12[1][l:]
        while arg12[1].endswith(sep):
            arg12[1] = arg12[1][:len(arg12[1])-l]
        # add to output
        out += sep+arg12[1]
    return out


def python_path(filter=None):
    import os
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        print('PYTHONPATH not set')
        user_paths = []
    if filter:
        filter = os.path.expanduser(filter)
        user_paths = [p for p in user_paths if filter in p]
    return user_paths


def locate_file(paths, fns, path_kws=None, fn_kws=None, return_raw_path=False, return_raw_fn=False, _raise=True,
                verbose=False):
    """Return path to file given number of possible paths"""
    # TODO: detect multiple occurences/possible paths
    if path_kws is None:
        path_kws = {}
    if fn_kws is None:
        fn_kws = {}

    for path_raw in paths:
        # Insert missing info in
        path_raw = str(path_raw)
        path = path_raw.format(**path_kws)
        try:
            path = Path(path).expanduser()
        except RuntimeError as e:
            if "Can't determine home directory" in str(e):
                continue
            else:
                raise e
        if not path.is_dir():
            continue
        path = path.resolve()
        for fn_raw in fns:
            fn = str(fn_raw).format(**fn_kws)
            fn_path = path / fn
            if fn_path.is_file():
                path_out = path_raw if return_raw_path else path
                fn_out = fn_raw if return_raw_fn else fn
                if verbose >= 2:
                    logging.info('Located "{}" in {}'.format(fn_out, path_out))
                return path_out, fn_out
    if _raise:
        raise IOError('Failed to locate file in paths "{}" with formats: {}'.format(paths, fns))
    else:
        if verbose:
            logger.warning('Failed to locate file in paths "{}" with formats: {}'.format(paths, fns))
        return None, None

def gen_hash_id(obj, mode='ripemd160'):
    import hashlib
    h = hashlib.new(mode)
    h.update(bytes(str(obj), 'utf-8'))
    hash_id = h.hexdigest()
    return hash_id


def fn_filter(path, fn_pattern, recursive=False, unique=False):
    """ Filenames in a given directory that match the search pattern
    TODO: add compatibility for non raw string file paths
    """
    fns = os.listdir(path)
    p = re.compile(fn_pattern)
    matches = []
    for fn in fns:
        if p.search(fn):
            matches.append(fn)
    if matches == []:
        print('No files match the supplied pattern: "%s"' % fn_pattern)
    if unique:  # expect unique match so return scalar string that matches
        if len(matches) == 1:
            return matches[0]
        else:
            raise ValueError('WARNING: fn_filter(unique=True): {} matches: {}'.format(len(matches), matches))
    else:
        return matches

def fn_filter_numeric_range(path_in, fn_pattern, numeric_range, sort_output=True):
    """Return sorted subset of filenames within a directory, within a numeric range

    The fn_pattern must contain {number} which must contain an integer in the numeric range"""
    assert '{number}' in fn_pattern, 'Include "{number}" in pattern when using file range'
    fn_pattern = fn_pattern.format(number=regexp_int_range(*numeric_range))

    filenames = fn_filter(path_in, fn_pattern)
    if sort_output:
        filenames = sorted(filenames)
    return filenames

def regexp_int_range(low, high, compile=False):
    fmt = '%%0%dd' % len(str(high))
    if compile:
        return re.compile('(%s)' % '|'.join(fmt % i for i in range(low, high + 1)))
    else:
        return '(%s)' % '|'.join('{:d}'.format(i) for i in range(low, high + 1))


def pos_path(value, require_dir_path=True):
    """Return True if value is a potential file path else False"""
    if not isinstance(value, string_types):
        return False
    value = os.path.expanduser(value)
    path, fn = os.path.split(value)
    if os.path.isdir(path):
        return True
    elif (not require_dir_path) and (path == ''):
        return True
    else:
        return False


def read_netcdf_group(fn_path, group):
    with xr.open_dataset(fn_path, group=group, autoclose=True) as match_data:
            match_data = match_data.copy(deep=True)
    return match_data

if __name__ == '__main__':
    path = '/home/tfarley/elzar2/checkpoints/MAST/SynthCam/single_filament_scan/Corrected_inversion_data/6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3/'
    fn_pattern = 'corr_inv-test1-n({n})-6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(2,8))

    fn = os.path.expanduser('~/repos/elzar2/elzar2/default_settings/elzar_defaults.ini')
    file = nested_dict()
    file['Paths']['elzar_path'] = '~/elzar/:'
    file['Paths']['data'] = ''

    file['Movie']['source'] = 'repeat'

    file['Invertor']['type'] = 'PsfInvertor'
    file['Invertor']['settings'] = 'repeat'
    file['Invertor']['resolution'] = 'repeat'

    file['Detector']['type'] = 'QuadMinEllipseDetector'
    file['Detector']['settings'] = 'repeat'

    file['Tracker']['type'] = 'NormedVariationTracker'
    file['Tracker']['settings'] = 'repeat'

    file['Benchmarker']['type'] = 'ProximityBenchmarker'
    file['Tracker']['settings'] = 'repeat'
    # file['elzar_path']['path'] = os.path.expanduser('~/elzar/')
    create_config_file(fn, file)