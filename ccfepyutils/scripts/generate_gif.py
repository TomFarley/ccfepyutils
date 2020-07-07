#!/usr/bin/env python
"""Script for generating .gif files from sets of images
NOTE: requires imageio package"""

import imageio
import os
import numpy as np
from pprint import pprint

from ccfepyutils.io_tools import fn_filter, regexp_int_set

try:
    from natsort import natsorted as sorted
except ImportError:
    pass

def gen_gif(path_in, pattern=r'\S+?(?:jpg|jpeg|png)', fn_out='movie.gif', duration=0.5, file_numbers=None, repeat={}, path_out=None,
            user_confirm=True, palettesize=256):
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

    if (file_numbers is not None) and ('{range}' in fn_out):
        fn_out.format(range='{}-{}'.format(np.min(file_numbers), np.max(file_numbers)))

    if file_numbers is not None:
        assert '{number}' in pattern, 'Include "{number}" in pattern when using file numbers'
        pattern = pattern.format(number=regexp_int_set(file_numbers))


    filenames = fn_filter(path_in, pattern)
    filenames = sorted(filenames)

    nframes = len(filenames)
    assert nframes > 0, 'No frames to create gif from'

    # Fill in information in filename
    fn_out = fn_out.format(n_images=nframes)

    if -1 in repeat.keys():  # If repeating final frame, replace '-1' with index
        repeat[nframes-1] = repeat[-1]
        repeat.pop(-1)

    if user_confirm:
        print('{} frames will be combined into gif in: {}'.format(nframes, os.path.join(path_in, fn_out)))
        if nframes < 60:
            pprint(filenames)
        choice = input('Proceed? [Y/n]: ')
        if choice.lower() in ('n', 'no'):
            print('gif was not produced')
            return  ## return from function without renaming

    with imageio.get_writer(os.path.join(path_out, fn_out), mode='I', duration=duration, palettesize=palettesize,
                            subrectangles=False) as writer:  # duration = 0.4

        for i, filename in enumerate(filenames):
            image = imageio.imread(os.path.join(path_in, filename))
            writer.append_data(image)
            if repeat is not None and i in repeat.keys():
                for j in np.arange(repeat[i]):
                    writer.append_data(image)

    print('Wrote gif containing {} frames to: {}'.format(nframes, os.path.join(path_in, fn_out)))

if __name__ == '__main__':
    # Path of images to be compiled into a gif (also the output dir)
    # path_in = '/home/tfarley/elzar/images/frames/elm_bgsub/'
    path_in = '/home/tfarley/elzar2/results/MAST/SynthCam/29840_storm_solps_neutrals/overview_plot/c1e484722fc7d1100721de2b2bbd00c0'
    # path_in = '/home/tfarley/elzar2/results/MAST/SynthCam/29840_storm_solps_neutrals/overview_plot/148ed4290c116def531c2099f0e6b82f/'
    # path_in = '/home/tfarley/elzar2/results/MAST/SynthCam/Nfil_40-Amp_exp-Width_log/overview_plot/3af0dda7d8c6d3570e5ad82aa066266c/'
    # path_in = '/home/tfarley/elzar2/results/MAST/SA1.1/29852/overview_plot/7e25f017a6eaf2f655e8de7aybde0faefc249b272/'
    # path_in = '/home/tfarley/elzar2/results/MAST/SA1.1/29852/overview_plot/7e25f017a6eaf2f655e8de7abde0faefc249b272/
    # path_in = '/home/tfarley/elzar2/results/MAST/CherabStorm/storm_21712_iaea_1/overview_plot/11b8717068c74a0ddff758d781899c512c7a8a14/'

    # Name of output gif file (produced in same directory as input images)
    fn_out = 'movie_{n_images}.gif'  # Output file name
    # fn_out = '{range}.gif'  # Output file name

    # Regex pattern describing files to include in gif. Use {number} to filter files by numbering.
    pattern = '.*.png'
    # pattern = '.*f{number}.png'
    # pattern = '.*-n_{number}-.*.png'

    # Range of numbers to be substituted into {number} in regex filename pattern
    file_numbers = None  # No number filter
    # file_numbers = np.arange(10, 50+1, 1)  # range of numbers permitted in filename filter

    # Number of additional times to repeat each frame number. Useful for creating pause and beginning and end of gif
    repeat = {0: 2, -1: 0}

    # Frame duration in seconds
    duration = 0.25    # Exp camera data 10us/frame, want 40us/s in gif -> 0.25s duration
    # duration = 0.1   # Cherab camera data 4us/frame, want 40us/s in gif -> 0.10s duration

    # Only path of input images is a required arg - by default will take all image files (jpeg, jpg, png) in directory
    gen_gif(path_in, pattern=pattern, duration=duration, fn_out=fn_out, file_numbers=file_numbers, repeat=repeat)