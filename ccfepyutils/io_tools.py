import configparser
import os, time
import logging
import pickle
import re
from pathlib import Path

import numpy as np
from past.types import basestring

from ccfepyutils.utils import make_iterable, compare_dict, is_number, is_subset, str_to_number, args_for
from ccfepyutils.debug import get_traceback_location

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    from natsort import natsorted
    sorted = natsorted
except ImportError as e:
    logger.debug('Please install natsort for improved sorting')

try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3

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

def delete_file(fn, path=None, ignore_exceptions=(), raise_on_fail=True, verbose=True):
    """Delete file with error handelling
    :param fn: filename
    :param path: optional path to prepend to filename
    :ignore_exceptions: Tuple of exceptions to pass over (but log) if raised eg (FileNotFoundError,) """
    fn = str(fn)
    if path is not None:
        fn_path = os.path.join(path, fn)
    else:
        fn_path = fn
    success = False
    try:
        os.remove(fn_path)
        success = True
        if verbose:
            logger.info('Deleted file: {}'.format(fn_path))
    except ignore_exceptions as e:
        logger.debug(e)
    except Exception as e:
        if raise_on_fail:
            raise e
        else:
            logger.warning('Failed to delete file: {}'.format(fn_path))
    return success

def rm_files(path, pattern, verbose=True, match=True, ignore_exceptions=()):
    path = str(path)
    if verbose:
        logger.info('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
    for fn in os.listdir(path):
        if match:
            m = re.search(pattern, fn)
        else:
            m = re.search(pattern, fn)
        if m:
            delete_file(fn, path, ignore_exceptions=ignore_exceptions)
            if verbose:
                logger.info('Deleted file: {}'.format(fn))


def getUserFile(type=""):
    from tkinter import Tk, filedialog as askopenfilename
    Tk().withdraw()
    filename = askopenfilename(message="Please select "+type+" file:")
    return filename

def filter_files_in_dir(path, fn_pattern, group_keys=(), modified_range=(None, None), n_matches_expected=None,
                        raise_on_incomplete_match=False,
                        raise_on_missing_dir=True, raise_on_no_matches=True, depth=0, include_empty_dirs=False,
                        **kwargs):
    """Return dict of filenames in 'path' directory that match supplied regex pattern

    The keys of the returned dict are the matched groups for each file from the fn_pattern.
    :param filenames: filenames to be filtered
    :param fn_pattern: regex pattern to match against files. kwargs will be substituted into the pattern (see example)
    :param path: path where files are located (only needed to querying files modification dates etc)
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
    path = Path(path).expanduser()
    if not path.is_dir():
        if raise_on_missing_dir:
            raise IOError('Search directory "{}" does not exist'.format(path))
        else:
            return {}
    path = path.resolve()
    # filenames_all = sorted(os.listdir(str(path)))
    out = {}
    n_matches = 0
    for i, (root, dirs, files) in enumerate(os.walk(str(path), topdown=True)):
        level = root.replace(str(path), '').count('/')
        if level > depth:
            break
        out_i = filter_files(files, fn_pattern, path=root, group_keys=group_keys,
                             raise_on_incomplete_match=raise_on_incomplete_match,
                             raise_on_no_matches=False, verbose=False, **kwargs)
        if (len(out_i) == 0) and not include_empty_dirs:
            continue
        else:
            n_matches += len(out_i)
            out[root] = out_i
        if (n_matches_expected is not None) and (n_matches >= n_matches_expected):
            # If all the required files have been found, ignore the remaining files
            break
    if (n_matches == 0):
        message = 'Failed to locate any files with pattern "{}" in {}, depth={}'.format(fn_pattern, path, depth)
        if raise_on_no_matches:
            raise IOError(message)
        else:
            logger.warning(message)
    if (depth ==0) and (len(out) == 1):
        out = list(out.values())[0]
    return out


def filter_files(filenames, fn_pattern, path=None, group_keys=(), modified_range=(None, None), raise_on_incomplete_match=False,
                        raise_on_no_matches=True, verbose=True, **kwargs):
    """Return dict of filenames from given set of filenames that match supplied regex pattern

    The keys of the returned dict are the matched groups for each file from the fn_pattern.
    :param filenames: filenames to be filtered
    :param fn_pattern: regex pattern to match against files. kwargs will be substituted into the pattern (see example)
    :param path: path where files are located (only needed to querying files modification dates etc)
    :param group_keys: list that links the ordering of the regex groups to the kwargs keys. Warnings are raised
                         if not all of the kwarg values are mmatched to files.
    :param raise_on_incomplete_match: raise an exception if not all kwarg values are located
    :param kwargs: values are substituted into the fn_pattern (provided the pattern contains a format key matching that
                    of the kwarg) with lists/arrays of values converted to the appropriate regex pattern.

    e.g. to get the files with values of n between 10 and 50 use. The returned dict will be keyed by the number n and
    the last group in the filename (<n>, <.*>). You will be warned if any of the requested n's are not found.
    fn_pattern = 'myfile-n({n})-(.*)\.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(20,51))

    e.g. to get all files in a directory sorted by some number in their filename i.e. catch all values of n:
    fn_pattern = 'myfile-n(\d+)-.*\.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'])
    """
    # TODO: Use glob for path selection
    from ccfepyutils.utils import PartialFormatter
    fmt = PartialFormatter()

    if (modified_range != (None, None)):
        assert path is not None, 'A path must be supplied to filter files by modified date'
        assert len(modified_range) == 2, 'Modifed range must have start and end'
        assert os.path.isdir(path)

    # If kwargs are supplied convert them to re patterns
    re_patterns = {}
    for key, value in kwargs.items():
        if isinstance(value, (np.ndarray, list, tuple)):  # and is_number(value[0]):
            # List/array of numbers, so match any number in list
            re_patterns[key] = '{}'.format('|'.join([str(v) for v in value]))
        elif isinstance(value, str):
            # Replace python format codes eg {time:0.3f} with regex pattern eg ([.\d]{3,4})
            fmt_pattern = '{{{key:}[^_]*}}'.format(key=key).replace('{', '\{').replace('}', '\}')
            fn_pattern = re.sub(fmt_pattern, value, fn_pattern)
    # fn_pattern = fn_pattern.format(**re_patterns)
    try:
        # fn_pattern = fmt.format(fn_pattern, **re_patterns)
        fn_pattern = fn_pattern.format(**re_patterns)
    except IndexError as e:
        pass
    out = {}
    i = 0
    for fn in filenames:
        # Check if each filename matches the pattern
        m = re.search(fn_pattern, fn)
        if m is None:
            continue
        if path is not None:
            fn_path = os.path.join(path, fn)
            t_now = time.time()
            t_day = 24*60*60
            t_age = t_now-os.path.getmtime(fn_path)
            if (modified_range[0] is not None) and (t_age < modified_range[0]*t_day):
                continue
            if (modified_range[1] is not None) and (t_age < modified_range[1]*t_day):
                continue

        ngroups = len(m.groups())
        if ngroups == 0:
            # Use index of element as output key
            key = i
        elif ngroups == 1:
            # Remove nesting tuple
            key = str_to_number(m.groups()[0])
        else:
            # Use tuple of elements from pattern matches as key
            key = tuple(str_to_number(v) for v in m.groups())
        out[key] = fn
        i += 1

    if len(out) == 0:
        message = 'Failed to locate any files with pattern "{}" in {}'.format(fn_pattern, filenames)
        if raise_on_no_matches:
            raise IOError(message)
        else:
            if verbose:
                logger.warning(message)
            return {}
    for i, group_key in enumerate(group_keys):
        if (group_key not in kwargs) or (isinstance(kwargs[group_key], (str, type(None)))):
            continue
        # List of located values for group cast to same type
        if ngroups == 0:
            raise ValueError('fn_pattern doesn not contain any regex groups "()"')
        if ngroups == 1:
            located_values = list(out.keys())
        else:
            located_values = [type(kwargs[group_key][0])(key[i]) for key in out.keys()]
        if not is_subset(kwargs[group_key], list(located_values)):
            message = 'Could not locate files with {} = {}'.format(group_key,
                                                                  set(kwargs[group_key]) - set(located_values))
            if raise_on_incomplete_match:
                raise RuntimeError(message)
            else:
                if verbose:
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
                logger.info('Directory {} was not created as start directory {} (depth={}) does not exist.'.format(
                    d, d_up, depth))
                continue
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                logger.info('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            try:
                os.makedirs(d)
                logger.info('Created directory: {}   ({})'.format(d, get_traceback_location(level=2)))
                if info:  # Write file describing purpose of directory etc
                    with open(os.path.join(d, 'DIR_INFO.txt'), 'w') as f:
                        f.write(info)
            except FileExistsError as e:
                logger.warning('Directory already created in parallel thread/process: {}'.format(e))
        else:
            if verbose:
                logger.info('Directory "' + d + '" already exists')
    return 0

def sub_dirs(path):
    """Return subdirectories contained within top level directory/path"""
    out = [p[0] for p in os.walk(path)]
    out.pop(out.index(path))
    return out

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
    df = pd.DataFrame(data=ys, index=make_iterable(x), columns=yheadings)
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

def attempt_n_times(func, args=None, kwargs=None, n_attempts=3, exceptions=(IOError,), sleep_invterval=0.5,
                    error_message='Call to {func} failed after {n_attempts} attempts',
                    call_on_fail=(), raise_on_fail=True, verbose=True):
    """Attempt I/O call multiple times with pauses in between to avoid read/write clashes etc."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    call_on_fail = make_iterable(call_on_fail)
    attempt = 1
    success = False
    while (success is False):
        try:
            # logger.debug('Attempt {} to call function "{}({})"'.format(
            #                 attempt, func.__name__, ', '.join([str(a) for a in args])))
            out = func(*args, **kwargs)
            success = True
            # logger.debug('Suceeded on attempt {} to call function "{}({})"'.format(
            #                 attempt, func.__name__, ', '.join([str(a) for a in args])))
        except exceptions as e:
            logger.warning('Attempt {} to call function "{}({})" failed'.format(
                            attempt, func.__name__, ', '.join([str(a) for a in args])))
            if attempt <= n_attempts:
                time.sleep(sleep_invterval)
                attempt += 1
            else:
                if error_message is not None:
                    logger.error(error_message.format(func=func.__name__, n_attempts=n_attempts))
                for func in call_on_fail:
                    raise NotImplementedError  # Need args_for to pass positional args
                    args, kwargs = args_for(func, kwargs)
                    func(*args0, **kws)
                if raise_on_fail:
                    raise e
                else:
                    out = e
                    break
    return out, success


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

def regexp_int_set(values, compile=False):
    fmt = '%%0%dd' % len(str(np.max(values)))
    if compile:
        return re.compile('(%s)' % '|'.join(fmt % i for i in values))
    else:
        return '(%s)' % '|'.join('{:d}'.format(i) for i in values)

def pos_path(value, allow_relative=True):
    """Return True if value is a potential file path else False"""
    if not isinstance(value, string_types):
        return False
    value = os.path.expanduser(value)
    if allow_relative:
        value = os.path.abspath(value)
    path, fn = os.path.split(value)
    if os.path.isdir(path):
        return True
    # elif (allow_relative) and (path == ''):
    #     return True
    else:
        return False


def read_netcdf_group(fn_path, group):
    import xarray as xr
    with xr.open_dataset(fn_path, group=group, autoclose=True) as match_data:
            match_data = match_data.copy(deep=True)
    return match_data

if __name__ == '__main__':
    # path = '/home/tfarley/elzar2/checkpoints/MAST/SynthCam/single_filament_scan/Corrected_inversion_data/6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3/'
    # fn_pattern = 'corr_inv-test1-n({n})-6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3.nc'
    # fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(4600,4650), depth=1)
    path = '/home/tfarley/elzar2/checkpoints/MAST/SA1.1/29991/Corrected_inversion_data/'
    fn_pattern = 'corr_inv-test1-n({n})-\w+.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(4600,4650), depth=1)

    fn = os.path.expanduser('~/repos/elzar2/elzar2/default_settings/elzar_defaults.ini')
    # from nested_dict import nested_dict
    # file = nested_dict()
    # file['Paths']['elzar_path'] = '~/elzar/:'
    # file['Paths']['data'] = ''
    #
    # file['Movie']['source'] = 'repeat'
    #
    # file['Invertor']['type'] = 'PsfInvertor'
    # file['Invertor']['settings'] = 'repeat'
    # file['Invertor']['resolution'] = 'repeat'
    #
    # file['Detector']['type'] = 'QuadMinEllipseDetector'
    # file['Detector']['settings'] = 'repeat'
    #
    # file['Tracker']['type'] = 'NormedVariationTracker'
    # file['Tracker']['settings'] = 'repeat'
    #
    # file['Benchmarker']['type'] = 'ProximityBenchmarker'
    # file['Tracker']['settings'] = 'repeat'
    # # file['elzar_path']['path'] = os.path.expanduser('~/elzar/')
    # create_config_file(fn, file)