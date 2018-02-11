#!/usr/bin/env python

import numbers
from collections import defaultdict
from pathlib import Path
from copy import copy, deepcopy
import logging
import inspect

import numpy as np
import pandas as pd
import xarray as xr

from pyIpx.movieReader import ipxReader,mrawReader,imstackReader

from ccfepyutils.classes.data_stack import Stack, Slice
from ccfepyutils.classes.settings import Settings
from ccfepyutils.utils import return_none, is_number, none_filter, to_array, make_itterable, args_for
from ccfepyutils.classes.plot import Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_mast_camera_data_path(camera, pulse):
    """Return path to movie file"""
    # TODO: get paths from config file
    if camera == 'SA1.1':
        path = '/home/tfarley/data/camera_data/SA1/'
        fn = 'C001H001S0001-{n:02d}.mraw'
    else:
        raise ValueError('Camera "{}" file lookup not yet supported'.format(camera))
    path = Path(path).expanduser().resolve()
    assert path.is_dir()
    if is_number(pulse):
        pulse = str(int(pulse))
    assert isinstance(pulse, str)
    path = path / pulse / 'C001H001S0001' / fn
    if not path.parent.is_dir():
        raise IOError('Path "{}" does not exist'.format(str(path)))
    if not Path(str(path).format(n=0)).is_file():
        raise IOError('Cannot locate file "{}" in path {}'.format(fn, str(path)))
    return str(path)

def get_mast_movie_transforms(camera, pulse):
    if camera == 'SA1.1':
        transforms = ['transpose', 'reverse_y']
    return transforms

class Frame(Slice):
    """Frame object returned by Movie class"""
    def __init__(self, movie, dim, value, roi=None):
        """"""
        super().__init__(movie, dim, value, roi=None)

    def __repr__(self):
        """Frame representation"""
        # TODO: add enhancement information
        if 't' in self.stack.meta.columns:
            t = ', t={}'.format(self.stack.time_format)
            t = t.format(self.stack.lookup('t', n=self.value))
        else:
            t = ''
        n = '{dim}={value}'.format(dim=self.dim, value=self.value)
        out = '<Frame("{name}") {res}[{n}{t}]>'.format(name=self.stack.name, res=self.stack.image_resolution, n=n, t=t)
        return out

    @property
    def raw(self):
        if self.stack._raw_movie is not None:
            raw = self.stack._raw_movie
            return raw(raw=True, **{self.dim: self.value})
            pass
        else:
            raise ValueError('Frame does not have a raw movie')

    def plot(self, ax=None, annotate=True, **kwargs):
        # TODO: Add automatic axis labeling once parameter class is complete
        kws = {'mode': 'image', 'cmap': 'gray', 'show': False}
        kws.update(kwargs)
        show = args_for(Plot.show, kws)
        plot = Slice.plot(self, ax=ax, show=False, **kws)
        if annotate:
            t = self.stack.lookup('t', **{self.stack.stack_dim: self.value}) if 't' in self.stack.meta.columns else None
            self._annotate_plot(plot.ax(), self.value, t=t)
        plot.show(**show)
        return plot

    @classmethod
    def _annotate_plot(cls, ax, n, t=None, xy=(0.05, 0.95)):
        """Add text annotation with frame number/time"""
        try:
            n_str = 'Frame: {:d}'.format(int(n))
            # TODO: implement time string in annotation
            t_str = '  Time: {:0.5f}s'.format(t) if t is not None else ''

            text = n_str + t_str
            frametxt = ax.annotate(text, xy=xy, xycoords='axes fraction', color='white', fontsize=8)
            frametxt.set_bbox(dict(color='k', alpha=0.5, edgecolor=None))
        except Exception as e:
            logger.exception('Cannot annotate frame plot. {}'.format(e))

class Movie(Stack):
    # TODO: Load compatibilities from config file
    compatibities = dict((
        ('MAST', dict((
            ('SA1.1', dict((
                ('get_path', get_mast_camera_data_path), ('transforms', get_mast_movie_transforms)
                ))),
            ),)),
        ))
    slice_class = Frame
    time_format = '{:0.5f}s'
    def __init__(self, pulse=None, machine=None, camera=None, fn=None, **kwargs):
        # TODO: load default machine and camera from config file
        assert (fn is not None) or all(value is not None for value in (pulse, machine, camera)), 'Insufficient inputs'
        self._pulse = None
        self._machine = None
        self._camera = None
        self._frame_range = None  # Dict of info detailing which frames to read from movie file
        self._transforms = None  # Transformations applied to frames when they are read eg rotate
        self._movie_meta = None  # meta data for movie format etc
        self._enhancements = None  # Flag for whether this data is enhanced or not
        self._enhanced_frames = [] #  Which frames have currently had enhancements applied
        self._enhancer = None
        self._enhanced_movie = None  # Movie instance containing enhanced data
        self._raw_movie = None  # Movie instance containing raw (unenhanced) data
        self.meta = None  # Dataframe of meta data related to each frame

        x = defaultdict(return_none, name='n')
        y = defaultdict(return_none, name='x_pix')
        z = defaultdict(return_none, name='y_pix')
        # quantity = defaultdict(return_none, name='pix_intensity')
        quantity = 'pix_intensity'

        kws = args_for(super(Movie, self).__init__, kwargs)
        super(Movie, self).__init__(x, y, z, quantity=quantity, stack_axis='x', **kws)
        kws = args_for(self.set_movie_file, kwargs)
        # TODO: change to set movie source
        self.set_movie_file(pulse, machine, camera, fn=fn, **kws)
        logger.debug('Initialised {}'.format(repr(self)))

    def __repr__(self):
        enhanced = self._enhanced_movie is not None
        out = '<Movie "{name}" {n}x{res}, enhanced={enh}>'.format(name=self.name, n=self.nframes,
                                                                res=self.image_resolution, enh=enhanced)
        return out

    def set_movie_file(self, pulse=None, machine=None, camera=None, fn=None, **kwargs):
        """Locate movie file to load data from"""
        self.pulse = pulse
        self.machine = machine
        self.camera = camera
        if fn is None:
            fn, self._transforms = self.locate_movie_file(self.pulse, self.machine, self.camera)
        fn = Path(fn)

        # Check file path exists
        if not fn.parent.is_dir():
            raise IOError('Path "{}" does not exist'.format(fn.parent))
        if not Path(str(fn).format(n=0)).is_file():
            raise IOError('Cannot locate file "{}" in path {}'.format(fn.name, fn.parent))

        self.fn_path = str(fn)
        self.path = str(fn.parent)
        self.fn = str(fn.name)
        self.movie_format = str(fn.suffix)

        self._movie_meta = {'format': self.movie_format}
        if self.movie_format == '.mraw':
            self._movie_meta.update(self.get_mraw_file_info(self.fn_path))
        else:
            raise NotImplementedError('set_movie_file for format "{}" not implemented'.format(self.movie_format))
        logger.debug('{} movie file set to: {}'.format(repr(self), self.fn_path))

    @classmethod
    def locate_movie_file(cls, pulse, machine, camera, **kwargs):
        """Locate movie file given movie info"""
        if machine not in cls.compatibities:
            raise ValueError('Movie class is not currently compatible with machine "{}". Compatibilties: {}'.format(
                    machine, cls.compatibities.values()))
        if camera not in cls.compatibities[machine]:
            raise ValueError('Movie class is not currently compatible with camera "{}". Compatibilties: {}'.format(
                    camera, cls.compatibities[machine].values()))
        fn = cls.compatibities[machine][camera]['get_path'](camera, pulse, **kwargs)
        transforms = cls.compatibities[machine][camera]['transforms'](camera, pulse)
        # cls._transforms = transforms
        return fn, transforms

    def set_frames(self, frames=None, start_frame=None, end_frame=None, start_time=None, end_time=None,
                   nframes=None, duration=None, stride=1, all=False, transforms=None):
        """Set frame range in file to read"""
        if self._movie_meta is None:
            assert RuntimeError('The movie file must be set before the frame range is set')
        self._transforms = none_filter(self._transforms, transforms)  # update transforms if passed

        if all:
            # Read whole movie
            raise NotImplementedError
        elif frames is None:
            # Complete information about frame range
            if any((start_time, end_time, duration)):
                raise NotImplementedError
            if start_frame is not None and end_frame is not None:
                assert start_frame < end_frame
                nframes = end_frame - start_frame + 1
            elif start_frame is not None and nframes is not None:
                end_frame = start_frame + nframes - 1
            else:
                raise ValueError('Insufficient input information to build frame range, {}-{} ({})'.format(
                        start_frame, end_frame, nframes))
            start_frame, end_frame = start_frame, end_frame
            assert isinstance(stride, (int, float))
            stride = int(stride)
            frames = np.arange(start_frame, end_frame+1, stride)
        else:
            frames = np.array(frames)
            start_frame, end_frame, nframes = frames.min(), frames.max(), len(frames)

        # tODO: change start stop to range list []
        frame_range = np.array([start_frame, end_frame])
        self._frame_range = {'frame_range': frame_range, 'n': nframes, 'stride': stride, 'frames': frames}

        self.check_frame_range()

        fps = self._movie_meta['fps']
        t_range = self._movie_meta['t_range'][0] + frame_range / fps
        t = np.arange(t_range[0], t_range[1]+1.0/fps, 1.0/fps)
        self._frame_range['t_range'] = t_range
        self._frame_range['t'] = t
        x_dim = {'name': 'x_pix', 'values': np.arange(self._movie_meta['frame_shape'][1])}
        y_dim = {'name': 'y_pix', 'values': np.arange(self._movie_meta['frame_shape'][0])}
        t_dim = {'name': 'n', 'values': frames}
        self.set_dimensions(x=t_dim, y=y_dim, z=x_dim)

        self.meta = pd.DataFrame({'t': t, 'n': frames, 'i': np.arange(0, nframes),
                                  'enhanced': np.zeros(nframes).astype(bool)})
        # self.meta.index.name = 'i'
        assert len(frames) == len(t) == nframes
        pass

    def check_frame_range(self):
        # TODO: break into sub functions for each file format
        movie_range = self._movie_meta['frame_range']
        frame_range = self._frame_range['frame_range']
        if (movie_range[0] > frame_range[0]) or (movie_range[1] < frame_range[1]):
            raise ValueError('Frame range {} set outside of movie file frame range {}'.format(frame_range, movie_range))

    @classmethod
    def get_mraw_file_info(cls, fn):
        movie_meta = {}
        mraw_files = pd.DataFrame({'StartFrame': []})
        # Get frame range in each mraw file
        n = 0
        while Path(fn.format(n=n)).is_file():
            vid = mrawReader(filename=fn.format(n=n))
            header = vid.file_header
            start_frame = int(header['StartFrame'].strip())
            mraw_files.loc[n, 'StartFrame'] = start_frame
            for key in ['TotalFrame', 'StartFrame']:  # 'OriginalTotalFrame', 'CorrectTriggerFrame', 'ZeroFrame']:
                mraw_files.loc[n, key] = int(header[key].strip())
            vid.release()
            n += 1
        assert n > 0, 'No mraw files read'
        # Mraw movie frames don't start at zero so record start frame and offset by it so start at 0
        movie_meta['frame0'] = int(mraw_files.loc[0, 'StartFrame'])
        mraw_files.loc[:, 'StartFrame'] -= movie_meta['frame0']
        mraw_files.loc[:, 'EndFrame'] = mraw_files.loc[:, 'StartFrame'] + mraw_files.loc[:, 'TotalFrame'] - 1
        # Get additional meta data eg frame shape
        movie_meta['mraw_files'] = mraw_files
        movie_meta['mraw_header'] = header
        movie_meta['frame_range'] = [int(mraw_files.loc[0, 'StartFrame']), int(mraw_files.loc[n-1, 'EndFrame'])]
        movie_meta['frame_shape'] = (int(header['ImageWidth'].strip()), int(header['ImageHeight'].strip()))
        movie_meta['fps'] = int(header['RecordRate(fps)'].strip())
        # Get time information from 1st and last frames
        vid = mrawReader(filename=fn.format(n=0))
        t_start = vid.set_frame_number(0).read()[2]['time_stamp']
        vid.release()
        end_index = int(mraw_files.loc[n - 1, 'TotalFrame']) - 1
        vid = mrawReader(filename=fn.format(n=n-1))
        t_end = vid.set_frame_number(end_index).read()[2]['time_stamp']
        vid.release()
        movie_meta['t_range'] = np.array([t_start, t_end])
        return movie_meta

    def _fill_values(self):
        """Called by Stack when data is accessed to ensure self._values is not empty"""
        if self._values is None:
            if self.fn_path is None:
                Stack._fill_values(self)
            else:
                self.load_movie_data()

    def load_movie_data(self):  # , pulse=None, machine=None, camera=None):
        """Load movie data into xarray given previously loaded movie file information"""
        if self.fn_path is None:
            raise ValueError('No movie file has been set for reading')
        if self._frame_range is None:
            raise ValueError('A frame range must be set before a movie can be read')
        # Initialise array for data to be read into
        data = np.zeros((self._frame_range['n'], *self._movie_meta['frame_shape']))
        if self.movie_format == '.mraw':
            i = 0
            n = self._frame_range['frame_range'][0]
            end = self._frame_range['frame_range'][1]
            frames = self._frame_range['frames']
            vid = mrawReader(filename=self.fn_path.format(n=0))
            vid.set_frame_number(n)
            while n <= end:
                ret, frame, header = vid.read(transforms=self._transforms)  # frames are read with 16 bit dynamic range, but values are 10 bit!
                if n in frames:
                    data[i] = frame
                i += 1
                n += 1
            vid.release()
        elif self.movie_format == '.ipx':
            raise NotImplementedError
        else:
            raise ValueError('Movie class does not currently support "{}" format movies'.format(self.movie_format))
        self.set_data(data, reset=True)
        logger.debug('{} loaded movie data from {}'.format(repr(self), self.fn_path))

    def lookup(self, output='frame_coord', **kwargs):
        """Return meta data value corresponding to supplied meta data input
        :param value: value of inp to look up corresponding out value for
        :param inp: coordinate type of 'value'
        :param out: output meta data type to look up
        :return: out value"""
        if len(kwargs) == 0 :
            raise ValueError('No input meta data passed to lookup')
        if output == 'frame_coord':
            output = self.stack_dim
        elif output not in self.meta.columns:
            raise ValueError('out={} is not a valid meta data type. Options: {}'.format(output, self.meta.columns))

        # TODO: Loop for multiple values per key
        mask = np.ones(len(self.meta)).astype(bool)
        for key, value, in kwargs.items():
            if key not in self.meta.columns:
                raise ValueError('inp={} is not a valid meta data type. Options: {}'.format(key, self.meta.columns))
            values = self.meta[key].values
            if value not in values.astype(type(value)):
                raise ValueError('value {} is not a valid "{}" value. Options: {}'.format(value, key, self.meta[key]))
            mask *= self.meta[key] == value
        new_value = self.meta.loc[mask, output].values
        if len(new_value) == 1:
            new_value = new_value[0]
        return new_value

    def __getitem__(self, item, raw=False):
        """Return data slices from the stack of data"""
        # If have enhaced data return that over raw data
        if self._enhanced_movie is not None and not raw:
            movie = self._enhanced_movie
        else:
            movie = self

        movie._init_xarray()
        item = movie.lookup_slice_index(item)
        return movie.get_slice(item)

    def __call__(self, raw=False, **kwargs):
        assert len(kwargs) > 0, 'Movie.__call__ requires keyword arg meta data to select frame'
        item = self.lookup(self.stack_dim, **kwargs)
        return self.__getitem__(item, raw=raw)

    @classmethod
    def get_frame_list(cls, current, n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1, skip_backwards=0,
                       skip_forwards=0, limits=None, unique=True, verbose=False):
        """ Return list of frame numbers (frame marks) given input
        """
        frame_list_settings = {'n_backwards': n_backwards, 'n_forwards': n_forwards, 'skip_backwards': skip_backwards,
                               'skip_forwards': skip_forwards,
                               'step_backwards': step_backwards, 'step_forwards': step_forwards}
        # import pdb; pdb.set_trace()

        ## Get list of frames equal to length of frame history that bracket the current frame and do not go outside
        ##  the range of frame numbers
        frameNumStart = current - frame_list_settings['skip_backwards'] - frame_list_settings['step_backwards'] * (
        frame_list_settings['n_backwards'] - 1) - 1
        frameNumEnd = current + frame_list_settings['skip_forwards'] + frame_list_settings['step_forwards'] * (
        frame_list_settings['n_forwards'] - 1) + 1

        frame_nos = (np.linspace(frameNumStart,
                                 current - frame_list_settings['skip_backwards'] - 1,
                                 num=frame_list_settings['n_backwards']),
                    # np.array([frameNum0]),
                    np.linspace(current + frame_list_settings['skip_forwards'] + 1,
                                frameNumEnd,
                                num=frame_list_settings['n_forwards']))
        frame_nos = np.round(np.hstack(frame_nos)).astype(int)
        logger.debug('Frames in frame_list:  {}'.format(str(frame_nos)))

        # Make sure frames are in frame range
        if limits is not None:
            frame_nos = frame_nos.clip(limits[0], limits[1])
        # frameMarks = frameNos + self.frameNumbers[0]
        if unique:  # remove duplicates
            frame_nos = list(set(frame_nos))
        return frame_nos

    def _enahnace_frame(self, enhancements, **kwargs):
        # TODO: Make Enhancements class
        # TODO: Enable storing of multiple enhancements at once? bg, fg, raw?
        assert len(kwargs) > 0, 'Enahnace_frame requires keyword arg meta data to select frame'
        i = self.lookup('i', **kwargs)
        if self._enhanced_movie is None:
            self._enhanced_movie = deepcopy(self)
            self._enhanced_movie._enhancements = enhancements
            self._enhanced_movie._raw_movie = self
            self._enhanced_movie.name = self.name+'_enhanced'
            self._enhancer = Enhancer(setting='default')  # TODO: specify enhancement settings
            self._enhanced_frames = []
        frame = self._enhanced_movie(i=i)
        frame[:] = self._enhancer(enhancements, frame[:])
        self._enhanced_frames.append(i)
        # raise NotImplementedError

    def enhance(self, enhancements, frames='all', keep_raw=False, **kwargs):
        # TODO: make t ect valid input
        if frames == 'all':
            frames = self.stack_axis_values
        frames = to_array(frames)
        # TODO: check all frame values in stack axis
        for n in frames:
            self._enahnace_frame(enhancements, n=n)
        if not keep_raw:
            self._data = None
        # raise NotImplementedError

    def to_hdf5(self, fn=None):
        raise NotImplementedError

    @property
    def data(self):
        """Return self._data if not enha
        Overloads method in Stack"""
        self._init_xarray()
        if self._enhanced_movie is not None:
            return self._enhanced_movie._data
        elif self._data is not None:
            return self._data
        else:
            raise ValueError('Both _data and _enhanced_movie are None: {}'.format(repr(self)))

    @data.setter
    def data(self, value):
        """Main data xarray
        Overloads method in Stack"""
        self._init_xarray()
        assert isinstance(value, xr.DataArray)
        if self._enhanced_movie is not None:
            self._enhanced_movie._data = value
        elif self._data is not None:
            self._data = value
        else:
            raise ValueError('Both _data and _enhanced_movie are None: {}'.format(repr(self)))

    @property
    def raw_data(self):
        """Return self._data
        Overloads method in Stack"""
        self._init_xarray()
        if self._data is not None:
            return self._data
        else:
            raise ValueError('Error returning movie data for {}'.format(repr(self)))

    @property
    def pulse(self):
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        if value is not None:
            assert isinstance(value, numbers.Number)
            self._pulse = value

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, value):
        if value is not None:
            if value not in self.compatibities:
                raise ValueError('Movie class is not compatible with machine "{}". Options: {}'.format(
                        value, self.compatibities.keys()))
            self._machine = value

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        if value is not None:
            assert self.machine is not None, 'Machine must be set before camera'
            if value not in self.compatibities[self.machine]:
                raise ValueError('Movie class is not compatible with camera "{}". Options: {}'.format(
                        value, self.compatibities[self.machine].keys()))
            self._camera = value

    @property
    def movie_frame_range(self):
        if self._movie_meta is None:
            return None
        else:
            return self._movie_meta['frame_range']

    @property
    def nframes(self):
        if self.meta is not None:
            return len(self.meta)
        else:
            return None

    @property
    def image_resolution(self):
        return self._movie_meta['frame_shape']

class Enhancer(object):
    """Class to apply image enhancements to arrays of data"""
    desciptions = {'bgsub': {'requires_window': True}}
    default_settings = {}
    from ccfepyutils.image import threshold, reduce_noise, sharpen
    functions = {'threshold': threshold, 'reduce_noise': reduce_noise, 'sharpen': sharpen}

    def __init__(self, setting='default', movie=None):
        self.settings = Settings('Enhancer', setting)
        self.movie = movie

    def __repr__(self):
        return '<Enhancer: {}>'.format(self.settings.name)

    def get_func_args(self, func):
        """Get arguments for enhancement function from settings object"""
        func_name = func.__name__
        args, kwargs = [], {}
        sig = inspect.signature(func)
        for i, kw in enumerate(sig.parameters.values()):
            name = kw.name
            if name == 'image':
                continue
            setting = '{func}_{arg}'.format(func=func_name, arg=name)
            if setting in self.settings:
                kwargs[name] = self.settings[setting].value
        return args, kwargs

    def __call__(self, enhancements, data):
        out = copy(data)
        funcs = self.functions
        enhancements = make_itterable(enhancements)
        for enhancement in enhancements:
            if enhancement not in funcs:
                raise ValueError('Enhancement {} not recognised'.format(enhancement))
            ## TODO:
            func = funcs[enhancement]
            args, kwargs = self.get_func_args(func)
            out = funcs[enhancement](out, *args, **kwargs)
        return out

    def apply(self, enhancements, x):
        """Apply enhancements to frame x"""
        raise NotImplementedError
        results = pool.map(func, inputs)
        intensity, phase, contrast = zip(*results)

    def chain_enhancements(self):
        raise NotImplementedError

    @classmethod
    def to_8bit(cls, data):
        raise NotImplementedError

    @classmethod
    def to_orig_format(cls, data):
        raise NotImplementedError

    @classmethod
    def set_brightness(cls, data, gamma):
        raise NotImplementedError

    @classmethod
    def set_contrast(cls, data, gamma):
        raise NotImplementedError

    @classmethod
    def set_gamma(cls, data, gamma):
        raise NotImplementedError

    @classmethod
    def threshold(cls, data, level):
        raise NotImplementedError

    def get_background(cls):
        raise NotImplementedError

    def get_foreground(cls):
        raise NotImplementedError

