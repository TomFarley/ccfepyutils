
import os, re

from ccfepyutils.io_tools import mkdir

def copy_movie_files():
    print('Creating links to stereo movie files')
    start_dir = '/net/fuslsa/data/MAST_IMAGES/023/'

    out_dir = '/projects/SOL/Data/Cameras/StereoA/{pulse}/'
    pattern = re.compile('rba(\d{6}).ipx')
    # out_dir = '/projects/SOL/Data/Cameras/StereoB/{pulse}/'
    # pattern = re.compile('rbb(\d{6}).ipx')

    # pattern_b = re.compile('rbb(\d{6}).ipx')

    print('start_dir', start_dir)
    for root, dirs, files in os.walk(start_dir, topdown=False, followlinks=True):
        print(root, dirs, files)
        for name in files:
            m = pattern.match(name)
            if m:
                pulse = m.groups()[0][1:]
                mkdir(out_dir.format(pulse=pulse), depth=1)
                fn_in = os.path.join(root, name)
                fn_out = os.path.join(out_dir.format(pulse=pulse), name)
                print(pulse, name, fn_in)
                print('os.symlink({fn_in}, {fn_out})'.format(fn_in=fn_in, fn_out=fn_out))
                if not os.path.isfile(fn_out):
                    os.symlink(fn_in, fn_out)
                # return
        # for name in dirs:
        #     print(os.path.join(root, name))

if __name__ == '__main__':
    print('Creating links to stereo movie files')

    copy_movie_files()