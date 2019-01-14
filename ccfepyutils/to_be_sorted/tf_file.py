#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

""" tf_file.py: Frequently used file operations and wrappers.

Detailed description:

Notes:
    @bug:

Todo:
    @todo: find_latest_file function using sort_dates()
    @todo: add compatibility for non raw string file paths
    @todo: add recursive functionality to fn_filter



Info:
    @since: 17-06-14
"""

import numpy as np
from pprint import pprint  # Pretty printing
import os
import re
# import shutil

## CAN import:    tf_debug, tf_array
## CANNOT import: tf_class
# from . import tf_numeric
import tf_libs.tf_simple as tf_simple
import tf_libs.tf_string as tf_string
from tf_libs.tf_string import comb_str

__author__ = 'Tom Farley'
__copyright__ = "Copyright 2015, TF Library Project"
__credits__ = []
__email__ = "farleytpm@gmail.com"
__status__ = "Development"
__version__ = "1.0.1"


def mkdir(dirs, start_dir=None, verbose=True):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary.
    Inputs:
        dir 			- Directory path
        verbatim = 0	- True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    if start_dir is not None:
        start_dir = os.path.abspath(start_dir)
        if not os.path.isdir(start_dir):
            if verbose:
                print('Directories {} were not created as start directory {} does not exist.'.format(dirs, start_dir))
            return 1

    if isinstance(dirs, basestring):  # Nest single string in list for loop
        dirs = [dirs]
    # import pdb; pdb.set_trace()
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                if verbose:
                    print('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            os.makedirs(d)
            if (verbose == 0 or verbose):
                print('Created directory: ' + d)
        else:
            if verbose:
                print('Directory "' + d + '" already exists')
    return 0

def fn_filter(dir, pattern, recursive=False, unique=False):
    """ Filenames in a given directory that match the search pattern
    TODO: add compatibility for non raw string file paths
    """
    fns = os.listdir(dir)
    p = re.compile(pattern)
    matches = []
    for fn in fns:
        if p.search(fn):
            matches.append(fn)
    if matches == []:
        print('No files match the supplied pattern: "%s"' % pattern)
    if unique:  # expect unique match so return scalar string that matches
        if len(matches) == 1:
            return matches[0]
        else:
            raise ('WARNING: fn_filter(unique=True): {} matches: {}'.format(len(matches), matches))
    else:
        return matches


def batch_rename(directory, pattern, rep_str, recursive=False, user_confirm=False, dir_rename=False):
    """ For all files in directory containing re pattern, replace pattern with rep_str
    @todo: update to not rename directories!"""
    ## Get list of all files in directory containing pattern
    fn_matches = fn_filter(directory, pattern, recursive=recursive)
    if fn_matches == []:
        print('No files were renamed.')
        return
    p = re.compile(pattern)
    print(os.path.abspath(directory + '/file1.txt'))
    fn_new = []
    n = 0
    for fn in fn_matches:
        fn_new.append(p.sub(rep_str, fn))
        n += 1
    if user_confirm:
        print('The following %d files will be renamed in: "%s":' % (n, directory))
    if user_confirm:
        print(tf_string.str_cols(fn_matches, fn_new, sep=' --> '))
        choice = input('Rename %d file(s)? [y/n]: ' % n)
        if not choice == 'y':
            print('No files were renamed')
            return  ## return from function without renaming
    for fn_old, fn_repl in zip(fn_matches, fn_new):
        old = os.path.abspath(comb_str(directory, fn_old))
        new = os.path.abspath(comb_str(directory, fn_repl))
        os.rename(old, new)
    print("Renamed %d files in %s: '%s' --> '%s'" % (n, directory, pattern, rep_str))

    return fn_new


def ncol(fn_list, cols, dir='', filekey=False, np_arrays=False, header_len=0, debug=False):
    """
    set filekey = True to nest data inside dictionary keyed to filenames
    """
    # cols should be a dictionary of col#:'variable_name' pairs
    # The returned dictionary has the variable_names as keys associated
    # with the data in that column

    if isinstance(fn_list, str):  # make fn_list a list if it isn't already
        fn_list = [fn_list]
        if debug:
            print('tf_file: Filename string converted to list: ', fn_list)

            # data = {array([])} if np_arrays==True else {[]}

    # Initialise dictionary to hold data from all required files
    fn_data = {}

    for fn in fn_list:
        if debug:
            print('tf_file: Reading file: ' + dir + fn)
        # Create a dictionary of arrays to hold the data from each column of one file
        data = {}
        bad_lines = []  # Record of line number that were not read correctly
        for name in list(cols.values()):
            data[name] = []

        for (i, line) in enumerate(open(dir + fn, 'rb')):
            if debug == 10: print('tf_file: line {}: '.format(i + 1), line.rstrip('\n'))  # don't print newline char

            if i <= header_len - 1:  # Skip column headings etc
                continue

            values = line.split()  # .strip()

            for i, val in enumerate(values):  # Convert numeric values to floats
                values[i] = float(val) if tf_simple.is_number(val) else val

            try:
                for col_no, name in cols.items():
                    data[name].append(values[col_no - 1])
            except IndexError:
                print('Not enough columns on line {}? of file: {}'.format(i + 1,
                                                                          fn))  # Seems to always print line number as 1?
                bad_lines.append(i)
                if debug:
                    print('tf_file: line {}: '.format(i + 1), line.rstrip('\n'))
                    print('tf_file: values: ', values)
            continue

        # Associate data from file with the filename
        fn_data[fn] = data

    if bad_lines != []:
        print('tf_file: Line numbers not read correctly: ', bad_lines)
    # print data
    # print fn_data
    # print
    # pprint.pprint(fn_data)

    # | If only given one filename and filekey == False do not nest data in filenames dictionary
    if len(fn_list) == 1 and filekey == False:
        if debug: print('tf_file: returned data without filename nesting (filekey == 0)')
        return data
    else:
        if debug: print('tf_file: returned data with filename nesting')
        return fn_data


def find_latest_file(file_names):
    """ Return filename containing most recent date	"""

    return


def main():
    fn = ['chi_im_tot']
    data = ncol(fn, {2: 'q', 8: 'chi'})
    print(data)
    pass


if __name__ == "__main__":
    main()
