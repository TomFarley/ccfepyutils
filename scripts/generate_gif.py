#!/usr/bin/env python
"""Script for generating .gif files from sets of images
NOTE: requires imageio package"""

import imageio
import os
import numpy as np
from pprint import pprint

from io_tools import fn_filter, regexp_int_range

try:
    from natsort import natsorted as sorted
except ImportError:
    pass

def gen_gif(path_in, pattern=r'\S+?(?:jpg|jpeg|png)', fn_out='movie.gif', duration=0.5, file_range=None, repeat={}, path_out=None,
            user_confirm=True):
    """Generate a gif from a collection of images in a given directory.
    path_in:        path of input images
    pattern         regular expression matching file to include in gif
    fn_out          filename for output gif
    duration        duration between frames in seconds
    file_range      replaces "{number}" in pattern with re matching a range of numbers
    repeat          dict of frame numbers and the number of times those frames should be repeated
    path_out        directory to write gif to
    """
    assert os.path.isdir(path_in)
    if path_out is None:
        path_out = path_in
    assert os.path.isdir(path_out)

    if (file_range is not None) and ('{range}' in fn_out):
        fn_out.format(range='{}-{}'.format(file_range[0], file_range[1]))

    if file_range is not None:
        assert '{number}' in pattern, 'Include "{number}" in pattern when using file range'
        pattern = pattern.format(number=regexp_int_range(*file_range))

    filenames = fn_filter(path_in, pattern)
    filenames = sorted(filenames)

    nframes = len(filenames)
    assert nframes > 0, 'No frames to create gif from'

    if -1 in repeat.keys():  # If repeating final frame, replace '-1' with index
        repeat[nframes-1] = repeat[-1]
        repeat.pop(-1)

    if user_confirm:
        print('{} frames will be combined into gif in: {}'.format(nframes, os.path.join(path_in, fn_out)))
        if nframes < 60:
            pprint(filenames)
        choice = raw_input('Proceed? [y/n]: ')
        if not choice == 'y':
            print('gif was not produced')
            return  ## return from function without renaming

    with imageio.get_writer(os.path.join(path_out, fn_out), mode='I', duration=duration) as writer:  # duration = 0.4

        for i, filename in enumerate(filenames):
            image = imageio.imread(os.path.join(path_in, filename))
            writer.append_data(image)
            if repeat is not None and i in repeat.keys():
                for j in np.arange(repeat[i]):
                    writer.append_data(image)

    print('Wrote gif containing {} frames to: {}'.format(nframes, os.path.join(path_in, fn_out)))

if __name__ == '__main__':
    # Path of images to be compiled into a gif (also the output dir)
    path_in = '/home/tfarley/elzar/images/frames/elm_bgsub/'

    # Name of output gif file (produced in same directory as input images)
    fn_out = 'movie.gif'  # Output file name
    # fn_out = '{range}.gif'  # Output file name

    # Regex pattern describing files to include in gif. Use {number} to filter files by numbering.
    pattern = '.*.png'
    # pattern = '.*f{number}.png'

    # Range of numbers to be substituted into {number} in regex filename pattern
    file_number_range = None  # No number filter
    # file_number_range = [350, 469]  # range of numbers permitted in filename filter

    # Number of additional times to repeat each frame number. Useful for creating pause and beginning and end of gif
    repeat = {0: 3, -1: 0}

    # Frame duration in seconds
    duration = 0.3

    # Only path of input images is a required arg - by default will take all image files (jpeg, jpg, png) in directory
    gen_gif(path_in, pattern=pattern, duration=duration, fn_out=fn_out, file_range=file_number_range, repeat=repeat)