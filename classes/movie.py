#!/usr/bin/env python

"""Classes for working with fusion camera data"""

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
from ccfepyutils.utils import return_none, is_number, none_filter, to_array, make_itterable, args_for, is_subset
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
        # transforms = ['transpose', 'reverse_y']
        transforms = ['transpose']#, 'reverse_y']
    return transforms

class Frame(Slice):
    """Frame object returned by Movie class"""
    def __init__(self, movie, dim, value, roi=None):
        """"""
        super().__init__(movie, dim, value, roi=None)

    def __repr__(self):
        """Frame representation"""
        # TODO: add enhancement information
        if 't' in self.stack._meta.columns:
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
            return self.stack(raw=True, **{self.dim: self.value})
            # raise ValueError('Frame does not have a raw movie')

    def plot(self, ax=None, annotate=True, **kwargs):
        # TODO: Add automatic axis labeling once parameter class is complete
        kws = {'mode': 'image', 'cmap': 'gray', 'show': False}
        kws.update(kwargs)
        show = args_for(Plot.show, kws)
        plot = Slice.plot(self, ax=ax, show=False, **kws)
        if annotate:
            t = self.stack.lookup('t', **{self.stack.stack_dim: self.value}) if 't' in self.stack._meta.columns else None
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
    def __init__(self, pulse=None, machine=None, camera=None, fn=None, settings='repeat', source=None, range=None, 
                 enhancer=None, **kwargs):
        # TODO: load default machine and camera from config file
        # assert (fn is not None) or all(value is not None for value in (pulse, machine, camera)), 'Insufficient inputs'
        self._reset_stack_attributes()  # Initialise attributes to None
        self._reset_movie_attributes()  # Initialise attributes to None
        self.settings = Settings.get('Movie', settings)
        self.settings.set('Movie_source', source, ignore=[None])
        self.settings.set('Movie_range', range, ignore=[None])
        self.settings.set('Enhancer_settings', enhancer, ignore=[None])
        self.source_settings = Settings.get('Movie_source', self.settings['Movie_source'])
        self.range_settings = Settings.get('Movie_range', self.settings['Movie_range'])
        s = self.source_settings
        s.set('pulse', pulse, ignore=[None])
        s.set('machine', machine, ignore=[None])
        s.set('camera', camera, ignore=[None])

        # Locate source of movie data
        kws = args_for(self.set_movie_file, kwargs)
        # TODO: change to set movie source
        self.set_movie_file(pulse, machine, camera, fn=fn, **kws)

        x = defaultdict(return_none, name='n')
        y = defaultdict(return_none, name='x_pix')
        z = defaultdict(return_none, name='y_pix')
        # quantity = defaultdict(return_none, name='pix_intensity')
        quantity = 'pix_intensity'
        
        # Initialise data xarray
        kws = args_for(super(Movie, self).__init__, kwargs)
        super(Movie, self).__init__(x, y, z, quantity=quantity, stack_axis='x', **kws)

        self.set_frames(**self.range_settings.get_func_args(self.set_frames))
        logger.debug('Initialised {}'.format(repr(self)))

    def _reset_movie_attributes(self):
        """Set all Stack class attributes to None"""
        self.source_settings = None
        self._frame_range = None  # Dict of info detailing which frames to read from movie file
        self._transforms = None  # Transformations applied to frames when they are read eg rotate
        self._movie_meta = None  # meta data for movie format etc
        self._enhancements = None  # Enhancements applied to _ENHANCED_movie data
        self._enhancer = None
        self._enhanced_movie = None  # Movie instance containing enhanced data
        self._raw_movie = None  # Movie instance containing raw (unenhanced) data

        self.fn_path = None
        self.path = None
        self.fn = None
        self.movie_format = None

        self._movie_meta = None
        self.movie_format = None

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
        
        frames, nframes, frame_range = self.get_frame_range(frames=frames, 
                                                            start_frame=start_frame, end_frame=end_frame, 
                                                            start_time=start_time, end_time=end_time,
                                                            nframes=nframes, duration=duration, stride=stride)
        
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

        self._meta = pd.DataFrame({'t': t, 'n': frames, 'i': np.arange(0, nframes),
                                  'enhanced': np.zeros(nframes).astype(bool)})
        # self._meta.index.name = 'i'
        assert len(frames) == len(t) == nframes
        pass
        
    
    def get_frame_range(self, frames=None, start_frame=None, end_frame=None, start_time=None, end_time=None,
                   nframes=None, duration=None, stride=1, all_=False):
        if all_:
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
        return frames, nframes, frame_range

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
        elif output not in self._meta.columns:
            raise ValueError('out={} is not a valid meta data type. Options: {}'.format(output, self._meta.columns))

        # TODO: Loop for multiple values per key
        mask = np.ones(len(self._meta)).astype(bool)
        for key, value, in kwargs.items():
            if key not in self._meta.columns:
                raise ValueError('inp={} is not a valid meta data type. Options: {}'.format(key, self._meta.columns))
            # Allow negative indexing with 'i'
            if key == 'i' and value < 0:
                value = self._meta[key].values[value]
            values = self._meta[key].values
            if value not in values.astype(type(value)):
                raise ValueError('value {} is not a valid "{}" value. Options: {}'.format(value, key, self._meta[key]))
            mask *= self._meta[key] == value
        new_value = self._meta.loc[mask, output].values
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
    def get_frame_list(cls, n, current=True, n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1, skip_backwards=0,
                       skip_forwards=0, limits=None, unique=True, verbose=False):
        """ Return list of frame numbers (frame marks) given input
        """
        frame_list_settings = {'n_backwards': n_backwards, 'n_forwards': n_forwards, 'skip_backwards': skip_backwards,
                               'skip_forwards': skip_forwards,
                               'step_backwards': step_backwards, 'step_forwards': step_forwards}
        # import pdb; pdb.set_trace()

        # Get list of frames equal to length of frame history that bracket the current frame and do not go outside
        #  the range of frame numbers
        start_frame = n - frame_list_settings['skip_backwards'] - frame_list_settings['step_backwards'] * (
        frame_list_settings['n_backwards'] - 1) - 1
        end_frame = n + frame_list_settings['skip_forwards'] + frame_list_settings['step_forwards'] * (
        frame_list_settings['n_forwards'] - 1) + 1

        # Frames preceding current frame
        frame_nos = [np.linspace(start_frame,
                                 n - frame_list_settings['skip_backwards'] - 1,
                                 num=frame_list_settings['n_backwards'])]
        # Include current frame
        if current:
            frame_nos.append(np.array([n]))
        # Frames after current frame
        frame_nos.append(np.linspace(n + frame_list_settings['skip_forwards'] + 1,
                                end_frame,
                                num=frame_list_settings['n_forwards']))
        frame_nos = np.round(np.hstack(frame_nos)).astype(int)

        # Make sure frames are in frame range
        if limits is not None:
            frame_nos = frame_nos.clip(limits[0], limits[1])
        # frameMarks = frameNos + self.frameNumbers[0]
        if unique:  # remove duplicates
            frame_nos = list(set(frame_nos))
        logger.debug('Frames in frame_list:  {}'.format(str(frame_nos)))
        return frame_nos

    def extract_frame_stack(self, frames):
        """Extract array of frame data for set of frames"""
        # TODO: move functionality to Stack class
        self._init_xarray()
        frame_stack = self._data.loc[{'n': frames}].values
        return frame_stack

    def extract_frame_stack_window(self, current, n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1,
                                   skip_backwards=0, skip_forwards=0, unique=True, verbose=False, raw=False):
        frames = self.get_frame_list(current, n_backwards=n_backwards, n_forwards=n_forwards,
                                    step_backwards=step_backwards, step_forwards=step_forwards,
                                    skip_backwards=skip_backwards, skip_forwards=skip_forwards,
                                    unique=unique, limits=self._frame_range['frame_range'])
        # If movie is enhanced, but some of the required frames are not enhanced, enhance those that have been missed
        #  -> return consistent enhanced output
        if self.is_enhanced and not raw:
            # Make sure all the required frames have been enhanced
            mask = (self._raw_movie._meta['enhanced'] == True).values
            not_enhanced = np.array(list(set(frames) - set(self._raw_movie._meta['n'][mask].values)))
            self.enhance(self._raw_movie._enhancements, not_enhanced)
            # Get frames from enhanced movie
            frame_stack = self._enhanced_movie.extract_frame_stack(frames)
        else:
            frame_stack = self.extract_frame_stack(frames)
        return frame_stack

    def _setup_enhanced_movie(self, enhancements):
        """Reset enhanced movie if enhancements are to change"""
        if not self.is_enhanced or self._enhancements != enhancements:
            # Update attributes of self
            self._enhancer = Enhancer(setting=self.settings['Enhancer_settings'])  # TODO: specify enhancement settings
            self._enhancements = []  # Enhancements applied to _ENHANCED_movie
            self._meta.loc[:, 'enhanced'] = False
            # Set fresh enhanced movie to copy of self
            # Settings objects don't like being deep coppied, so set all settings to None and reassign them afterwares
            # TODO: Fix deepcopy of Settings objects
            movie_copy = copy(self)
            movie_copy._enhanced_movie = None
            movie_copy._raw_movie = None
            movie_copy.settings = None
            movie_copy.source_settings = None
            movie_copy.range_settings = None
            movie_copy._enhancer = None

            self._enhanced_movie = deepcopy(movie_copy)
            # Reassign settings objects after deepcopy - temporary fix...
            movie_copy.settings = self.settings
            movie_copy.source_settings = self.source_settings
            movie_copy.range_settings = self.range_settings

            enhanced_movie = self._enhanced_movie
            enhanced_movie._enhancer = None
            enhanced_movie._enhancements = None
            enhanced_movie._raw_movie = self
            enhanced_movie.name = self.name + '_enhanced'
            return True
        else:
            return False

    def _apply_enhancement(self, frames, enhancement, **kwargs):
        # TODO: Make Enhancements class
        # TODO: Enable storing of multiple enhancements at once? bg, fg, raw?
        # assert len(kwargs) > 0, 'Enahnace_frame requires keyword arg meta data to select frame'
        # i = self.lookup('i', **kwargs)
        # n = self.lookup('n', **kwargs)
        # enhanced_movie = self._enhanced_movie
        # self._enhancements.append(enhancement)
        # from concurrent import futures
        # Executor = futures.ProcessPoolExecutor
        # with Executor(max_workers=max_workers) as executor:
        #     results = executor.map(self._enhancer, args[:, 0], args[:, 1], args[:, 2], args[:, 3])
        #     intensity, phase, contrast = zip(*results)
        args = []
        frame_arrays = []
        # TODO: Make parallel - threading?
        for n in frames:
            args.append(self._enhancer.prepare_arguments(enhancement, self, n))
            frame_arrays.append(self._enhanced_movie(n=n)[:])

        # TODO: Make parallel - processes
        for j, n in enumerate(frames):
            i = self.lookup('i', n=n)
            # Skip frame if already enhanced
            if self._meta.loc[i, 'enhanced'] == True:
                continue
            # NOTE: movie and n args only needed for fg and bg extraction
            self._enhanced_movie(n=n)[:] = self._enhancer(enhancement, frame_arrays[j], **args[j])

    def enhance(self, enhancements, frames='all', keep_raw=False, **kwargs):
        """Apply mutiple enhancements to a set of frames in order"""
        # TODO: make t ect valid input
        self._init_xarray()
        if frames == 'all':
            frames = self.stack_axis_values
        enhancements = make_itterable(enhancements)
        if not self._setup_enhanced_movie(enhancements) and is_subset(frames, self.enhanced_frames):
            logger.warning('Enhancements {} already applied to {}'.format(enhancements, repr(self)))
            return
        frames = to_array(frames)
        # TODO: check all frame values in stack axis
        for enhancement in enhancements:
            self._apply_enhancement(frames, enhancement)
        # Set frames as having been enhanced
        self._enhancements = enhancements
        mask = self._meta['n'].isin(frames)
        self._meta.loc[mask, 'enhanced'] = True
        if not keep_raw:
            self._data = None
        # raise NotImplementedError

    def to_hdf5(self, fn=None):
        raise NotImplementedError

    def from_hdf5(self, fn=None):
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
        return self.source_settings['pulse']

    @pulse.setter
    def pulse(self, value):
        if value is not None:
            assert isinstance(value, numbers.Number)
            self.source_settings['pulse'] = value

    @property
    def machine(self):
        return self.source_settings['machine']

    @machine.setter
    def machine(self, value):
        if value is not None:
            if value not in self.compatibities:
                raise ValueError('Movie class is not compatible with machine "{}". Options: {}'.format(
                        value, self.compatibities.keys()))
            self.source_settings['machine'] = value

    @property
    def camera(self):
        return self.source_settings['camera']

    @camera.setter
    def camera(self, value):
        if value is not None:
            assert self.machine is not None, 'Machine must be set before camera'
            if value not in self.compatibities[self.machine]:
                raise ValueError('Movie class is not compatible with camera "{}". Options: {}'.format(
                        value, self.compatibities[self.machine].keys()))
            self.source_settings['camera'] = value

    @property
    def movie_frame_range(self):
        if self._movie_meta is None:
            return None
        else:
            return self._movie_meta['frame_range']

    @property
    def nframes(self):
        if self._meta is not None:
            return len(self._meta)
        else:
            return None
        
    @property
    def frame_numbers(self):
        if self._meta is not None:
            return self._meta['n'].values

    @property
    def frame_times(self):
        if self._meta is not None:
            return self._meta['t'].values

    @property
    def image_resolution(self):
        return self._movie_meta['frame_shape']
    
    @property
    def is_enhanced(self):
        return self._enhancements is not None
    
    @property
    def enhanced_frames(self):
        mask = self._meta['enhanced'] == True
        return self._meta.loc[mask, 'n'].values

    @property
    def fn_path_0(self):
        if self.fn_path is not None:
            return self.fn_path.format(n=0)
        else:
            return None

class Enhancer(object):
    """Class to apply image enhancements to arrays of data"""
    desciptions = {'bgsub': {'requires_window': True}}
    default_settings = {}
    from ccfepyutils.image import threshold, reduce_noise, sharpen, extract_bg, extract_fg
    functions = {'threshold': threshold, 'reduce_noise': reduce_noise, 'sharpen': sharpen, 'extract_bg': extract_bg,
                 'extract_fg': extract_fg}

    def __init__(self, setting='default'):
        self.settings = Settings.get('Enhancer', setting)
        # self.movie = movie

    def __repr__(self):
        return '<Enhancer: {}>'.format(self.settings.name)

    def get_func_args(self, func, func_name=None, **kwargs):
        """Get arguments for enhancement function from settings object"""
        if func_name is None:
            func_name = func.__name__
        args, kws = [], {}
        sig = inspect.signature(func)
        for i, kw in enumerate(sig.parameters.values()):
            name = kw.name
            # if name == 'image':
            #     continue
            # if setting == 'movie':
            #     if self.movie is None:
            #         raise ValueError('Enhancer must have a Movie object assigned to perform {}'.format(func_name))
            #     kws['movie'] = self.movie
            setting = '{func}_{arg}'.format(func=func_name, arg=name)
            if setting in self.settings:
                kws[name] = self.settings[setting].value
            if setting in kwargs:
                kws[name] = kwargs[name]
        return args, kws

    def __call__(self, enhancements, data, **kwargs):
        out = copy(data)
        funcs = self.functions
        enhancements = make_itterable(enhancements)
        if np.max(data) == 0:
            logging.warning('Enhancement(s) {} is being applied to an empty frame'.format(enhancements))
        for enhancement in enhancements:
            if enhancement not in funcs:
                raise ValueError('Enhancement {} not recognised'.format(enhancement))
            ## TODO:
            func = funcs[enhancement]
            if len(kwargs) == 0:  # if keyword arguments havent been passed, look them up
                kwargs.update(self.settings.get_func_args(func, **kwargs))
            out = funcs[enhancement](image=out, **kwargs)
        return out

    def prepare_arguments(self, enhancement, movie, n):
        """Prepare arguments for enhancement function. Get inputs ready for parallel processing."""
        kwargs = {}
        func = self.functions[enhancement]
        sig = inspect.signature(func).parameters.keys()
        if 'frame_stack' in sig:
            kws = self.settings.get_func_args(movie.extract_frame_stack_window, func_name=enhancement)
            frame_stack = movie._enhanced_movie.extract_frame_stack_window(n, raw=True, **kws)
            kwargs['frame_stack'] = frame_stack

        return kwargs

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

