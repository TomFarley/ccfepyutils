#!/usr/bin/env python

"""Classes for working with fusion camera data"""

import os, re, numbers, logging, inspect, glob, sys, pickle, socket, string, shutil
from collections import defaultdict, OrderedDict
from pathlib import Path
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import natsort

import matplotlib.pyplot as plt

from pyIpx.movieReader import ipxReader, mrawReader, imstackReader

from ccfepyutils.classes.data_stack import Stack, Slice
# from ccfepyutils.classes.settings import Settings
from ccfepyutils.classes.plot import Plot
from ccfepyutils.utils import return_none, is_number, is_numeric, none_filter, to_array, make_iterable, args_for, \
    is_subset, is_in, replace_in, dataframe_description_str
from ccfepyutils.io_tools import pickle_load, locate_file
from ccfepyutils.classes.plot import Plot
from ccfepyutils.mpl_tools import annotate_axis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def copy_movie_settings_templates():
    from setpy.settings.template_settings_files import copy_template_settings_files
    current_dir = Path(__file__).parent
    template_movie_dir = (current_dir / '../template_settings/data/').resolve()
    copy_template_settings_files(template_movie_dir,
                        template_subset=['movie', 'enhancer', 'movie_source', 'movie_range', 'movie_data_locations'],
                        overwrite_existing=False)

copy_movie_settings_templates()

def get_camera_data_path(machine, camera, pulse):
    """Return path to movie file, looking up settings for camera"""
    from setpy import Settings
    # TODO: get paths from settings/config file
    if is_number(pulse):
        pulse = str(int(pulse))
    # host_name = socket.gethostname()
    # host_name = host_name.rstrip(string.digits)  # remove number at end of name in case of cluster nodes

    camera_settings = Settings('movie_data_locations', settings_name='{}_{}'.format(machine, camera))
    if len(camera_settings) == 0:
        raise ValueError('No Movie_data_locations settings exist for camera="{}", machine="{}". Options: {}'.format(
            camera, machine, Settings.saved_settings_files_for_application('movie_data_locations')))
    
    path_options = camera_settings['path_options']
    fn_options = camera_settings['fn_options']
    
    path_kws = {'machine': machine, 'pulse': pulse, 'pulse_prefix': '0{prefix}'.format(prefix=str(pulse[0:2]))}
    fn_kws = {'n': 1, 'pulse': pulse}
    path, fn_format = locate_file(path_options, fn_options, path_kws=path_kws, fn_kws=fn_kws,
                          return_raw_path=False, return_raw_fn=True, _raise=True, verbose=True)
    # Sub in pulse number, but leave mraw file number "{n:02d}" in place
    fn_format = fn_format.replace('{pulse}', '{}'.format(pulse))
    if ('frame_transforms' in camera_settings) and (isinstance(camera_settings['frame_transforms'], list)):
        transforms = camera_settings['frame_transforms']
    else:
        transforms = [] 
    return path, fn_format, transforms


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
    def data(self):
        meta = self.stack._meta
        if not meta.loc[self.value, 'set']:
            self.stack.load_movie_data(n=self.value)
        data = super(Frame, self).data
        return data

    @property
    def raw(self):
        if self.stack._raw_movie is not None:
            raw = self.stack._raw_movie
            return raw(raw=True, **{self.dim: self.value})
            pass
        else:
            return self.stack(raw=True, **{self.dim: self.value})
            # raise ValueError('Frame does not have a raw movie')

    @property
    def meta(self):
        meta = self.stack._meta.loc[self.value]
        return meta

    @property
    def n(self):
        n = self.meta['n']
        return n

    @property
    def t(self):
        t = self.meta['t']
        return t

    def hist(self, ax=None, label=None, annotate=True, update_existing=False, show=True, **kwargs):
        if update_existing:
            raise NotImplementedError

        # xr.plot.hist(self.data, ax=ax)
        plot = Plot.get(ax=ax)
        data = self.data.values.flatten()
        plot.plot(data, mode='pdf', ax=ax, label=label, **kwargs)

        if annotate:
            annotation_str = dataframe_description_str(self.data.to_dataframe()['pix_intensity'])
            annotate_axis(ax, annotation_str, x=0.7, y=0.4, fontsize=7)
        plot.set_axis_labels(ax=ax, xlabel='Pixel intensity')

        if show:
            plt.show()

    def plot(self, ax=None, annotate=True, update_existing=True, **kwargs):
        # TODO: Add automatic axis labeling once parameter class is complete
        kws = {'mode': 'image', 'cmap': 'gray', 'show': False, 'vmin': 0, 'vmax': None}
        kws.update(kwargs)
        show = args_for(Plot.show, kws, exclude='tight_layout')
        show.update(args_for(Plot.show, kws, remove=False))  # pass tight layout to show and plot
        if (update_existing and hasattr(ax, 'ccfe_plot') and (ax in ax.ccfe_plot.ax_artists) and
                ('img' in ax.ccfe_plot.ax_artists[ax])):
            plot = ax.ccfe_plot
            img = plot.ax_artists[ax]['img']
            img.set_data(self.data.T)
            data_max = np.max(self.data.values)
            clim1 = img.get_clim()[1]
            if (clim1 is not None) and ((clim1 > data_max / 4) or (data_max > clim1)):
                # Update imshow dynamic range to cover data range or reduce it if loading different frame data with diff range
                img.set_clim(0, data_max)
            logger.debug('Updated existing image artist')
        else:
            plot = Slice.plot(self, ax=ax, show=False, **kws)

        t = self.stack.lookup('t', **{self.stack.stack_dim: self.value}) if 't' in self.stack._meta.columns else None
        self._annotate_plot(plot.ax(), self.value, t=t, annotate=annotate, update_existing=update_existing)

        plot.show(**show)
        return plot

    @classmethod
    def _annotate_plot(cls, ax, n, t=None, annotate=True, xy=(0.05, 0.95), update_existing=True):
        """Add text annotation with frame number/time"""
        try:
            n_str = 'Frame: {:d}'.format(int(n))
            # TODO: implement time string in annotation
            t_str = '  Time: {:0.5f}s'.format(t) if ((t is not None) and (not np.isnan(t))) else ''

            # TODO: Move to Plot class
            text = n_str + t_str
            if update_existing and hasattr(ax, 'ccfe_plot') and ('text_frame_no' in ax.ccfe_plot.ax_artists[ax]):
                # Keep record of text object for updating
                artist = ax.ccfe_plot.ax_artists[ax]['text_frame_no']
                if annotate:
                    artist.set_text(text)
                    artist.set_visible(True)
                else:
                    artist.set_visible(False)
            else:
                if annotate:
                    frametxt = ax.annotate(text, xy=xy, xycoords='axes fraction', color='white', fontsize=8)
                    frametxt.set_bbox(dict(color='k', alpha=0.5, edgecolor=None))
                    if hasattr(ax, 'ccfe_plot'):
                        # Keep record of text object for updating
                        ax.ccfe_plot.ax_artists[ax]['text_frame_no'] = frametxt
        except Exception as e:
            logger.exception('Cannot annotate frame plot. {}'.format(e))
        return text

class Movie(Stack):
    # compatibities = dict((
    #     ('MAST', dict((
    #         ('SA1.1', dict((
    #             ('get_path', get_mast_camera_data_path), ('transforms', get_mast_movie_transforms)
    #             ))),
    #         ('SynthCam', dict((
    #             ('get_path', get_synthcam_data_path), ('transforms', get_synthcam_transforms)
    #         ))),
    #         ),)),
    #     ))
    _frame_range_keys = ['frame_range', 'nframes', 'stride', 'frames', 't_range', 't']
    slice_class = Frame
    time_format = '{:0.5f}s'
    def __init__(self, pulse=None, machine=None, camera=None, movie_path=None, settings='repeat', source=None,
                 movie_range=None, enhancer=None, name=None, frames=None, create_new_settings=False, **kwargs):
        from setpy import Settings
        # TODO: load default machine and camera from config file
        # assert (fn is not None) or all(value is not None for value in (pulse, machine, camera)), 'Insufficient inputs'
        self._reset_stack_attributes()  # Initialise attributes to None
        self._reset_movie_attributes()  # Initialise attributes to None
        keys = ('pulse', 'machine', 'camera', 'movie_source', 'movie_range', 'enhancer')
        values = (pulse, machine, camera, source, movie_range, enhancer)
        kwargs.update({key: value for key, value in zip(keys, values)
                       if value is not None})
        # self.settings = Settings.collect('Movie', settings, {'Movie_source': source, 'Movie_range': range,
        #                                                      'Enhancer': enhancer}, **kwargs)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        for key in ['start_frame', 'end_frame']:
            if (key in kwargs) and (kwargs[key] is None):
                kwargs.pop(key)
        try:
            settings = Settings(application='movie', settings_name=settings, recursive=True, updated_values=kwargs,
                                create_new=create_new_settings)
        except FileNotFoundError:
            copy_movie_settings_templates()

        self.settings = settings
        # TODO: lookup parameter objects
        x = defaultdict(return_none, name='n')
        y = defaultdict(return_none, name='x_pix')
        z = defaultdict(return_none, name='y_pix')
        quantity = 'pix_intensity'
        
        # Initialise data xarray
        super().__init__(x, y, z, quantity=quantity, stack_axis='x', name=name)

        # kws = self.settings.get_func_args(self.set_movie_source)


        # Start tmp ***********************************************
        # # TODO: remove tmp
        # print(settings.view(cols='value'))
        # print(movie_path)
        # End tmp ***********************************************


        kws = settings.args_for(self.set_movie_source)
        self.set_movie_source(fn_path=movie_path, **kws)

        # kws = self.settings.get_func_args(self.set_frames) if frames is None else {}
        kws = settings.args_for(self.set_frames) if frames is None else {}
        self.set_frames(frames=frames, **kws)
        logger.debug('Initialised {}'.format(repr(self)))

    def _reset_movie_attributes(self):
        """Set all Stack class attributes to None"""
        self.source_info = {}
        # Dicts of info detailing which frames can be loaded and which frames the user wants to work with
        # (frames in memory may be superset of user frames depending on enhancer settings)
        self._frame_range_info_all = {key: None for key in self._frame_range_keys}
        self._frame_range_info_user = {key: None for key in self._frame_range_keys}
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

        self.gui = None
        self._current_frame = None

    def __repr__(self):
        enhanced = self._enhanced_movie is not None
        try:
            out = '<Movie "{name}" {n}x{res}, enhanced={enh}>'.format(name=self.name, n=self.nframes,
                                                                    res=self.image_resolution, enh=enhanced)
        except TypeError as e:
            out = '<Movie; initialising id="{id}">'.format(id=id(self))
        return out

    def set_movie_source(self, machine, camera, pulse, fn_path=None, transforms=(), **kwargs):
        """Locate movie file to load data from"""
        # TODO: remove duplication?
        self.pulse = pulse
        self.machine = machine
        self.camera = camera
        if fn_path is None:
            path, fn_format, self._transforms = self.locate_movie_file(self.pulse, self.machine, self.camera)
            fn_path = Path(path) / fn_format
        else:
            fn_path = Path(fn_path)
            self._transforms = transforms

        # Check file path exists
        if not fn_path.parent.is_dir():
            raise IOError('Path "{}" does not exist'.format(fn_path.parent))
        if not Path(str(fn_path).format(n=1, pulse=pulse)).is_file():
            raise IOError('Cannot locate file "{}" in path {}'.format(fn_path.name, fn_path.parent))

        self.fn_path = str(fn_path)
        self.path = str(fn_path.parent)
        self.fn = str(fn_path.name)
        # TODO: movie movie formnat to ._movie_meta?
        self.movie_format = str(fn_path.suffix)

        self._movie_meta = {'format': self.movie_format}
        if self.movie_format == '.mraw':
            self._movie_meta.update(self.get_mraw_file_info(self.fn_path))
        elif self.movie_format == '.ipx':
            self._movie_meta.update(self.get_ipx_file_info(self.fn_path.format(pulse=pulse),
                                                           transforms=self._transforms))
        elif self.movie_format == '.p':
            self._movie_meta.update(self.get_pickle_movie_info(self.fn_path, transforms=self._transforms))
        elif self.movie_format == '.npz':
            self._movie_meta.update(self.get_npz_movie_info(self.fn_path, transforms=self._transforms))
        else:
            raise NotImplementedError('set_movie_file for format "{}" not implemented'.format(self.movie_format))
        self._y['values'] = np.arange(self._movie_meta['frame_shape'][0])
        self._z['values'] = np.arange(self._movie_meta['frame_shape'][1])
        logger.debug('{} movie file set to: {}'.format(repr(self), self.fn_path))

    @classmethod
    def locate_movie_file(cls, pulse, machine, camera, **kwargs):
        """Locate movie file given movie info"""
        path, fn_patern, transforms = get_camera_data_path(machine, camera, pulse)
        return path, fn_patern, transforms

    def set_frames(self, frames=None, start_frame=None, end_frame=None, start_time=None, end_time=None,
                   nframes_user=None, duration=None, frame_stride=1, all=False, transforms=None):
        """Set frame range in file to read"""
        previously_loaded_frames = self._frame_range_info_all['frames']
        if self._movie_meta is None:
            assert RuntimeError('The movie file must be set before the frame range is set')

        self._frame_range_info_all = {key: None for key in self._frame_range_keys}
        self._frame_range_info_user = {key: None for key in self._frame_range_keys}

        self._transforms = none_filter(self._transforms, transforms)  # update transforms if passed

        frames_user, nframes_user, frame_range_user = self.get_frame_range(frames=frames,
                                                                      start_frame=start_frame, end_frame=end_frame,
                                                                      start_time=start_time, end_time=end_time,
                                                                      nframes=nframes_user, duration=duration, stride=frame_stride)
        
        self._frame_range_info_user.update({'frame_range': frame_range_user, 'nframes': nframes_user, 'stride': frame_stride,
                                       'frames': frames_user})
        self.check_user_frame_range()
        fps = self._movie_meta['fps']
        t_range_user = (self._movie_meta['t_range'][0] + frame_range_user / fps)
        t_user = self._movie_meta['t_range'][0] + frames_user / fps  # [::frame_stride]
        self._frame_range_info_user['t_range'] = t_range_user
        self._frame_range_info_user['t'] = t_user

        # Initialise loadable frame range as same as user specified frame range
        self._frame_range_info_all = deepcopy(self._frame_range_info_user)
        frames_all = self._frame_range_info_all['frames']
        t_all = self._frame_range_info_all['t']

        movie_range = self._movie_meta['frame_range']
        frame_stack_enhancements = np.array(['extract_fg', 'extract_bg'])
        if any(is_in(frame_stack_enhancements, self.settings['enhancements'])):
            # Enhacements require the frame range to be extended due to enhancement functions needing a 'frame_stack'
            frame_range_info_all = self._frame_range_info_all
            frame_range_all = deepcopy(frame_range_user)
            enhancement = frame_stack_enhancements[is_in(frame_stack_enhancements, self.settings['enhancements'])][0]
            kws = {}
            # Get specific frame stack settings for this function
            args = ['n_backwards', 'n_forwards', 'step_backwards', 'step_forwards', 'skip_backwards', 'skip_forwards']
            for arg in args:
                func_arg = '{func_name}::{arg}'.format(func_name=enhancement, arg=arg)
                if func_arg in self.settings:
                    kws[arg] = self.settings[func_arg].value
            if np.all(np.diff(frames_user) == 1):
                # Contiguous frame range so only consider adding frames to end of range
                frame_range_all[0] = np.min(np.concatenate([self.get_frame_list(frame_range_user[0],
                                                            limits=movie_range, **kws),frame_range_user]))
                frame_range_all[1] = np.max(np.concatenate([self.get_frame_list(frame_range_user[-1],
                                                            limits=movie_range, **kws), frame_range_user]))
                # Get additional frames to sadwitch user frames with
                buffer_front = np.arange(frame_range_all[0], frame_range_user[0])
                buffer_end = np.arange(frame_range_user[1] + 1, frame_range_all[1] + 1)
                frames_all = np.concatenate((buffer_front, frames_user, buffer_end))
                logger.info('Padded movie frame range with {}+{}={} additional frames for enhancement "{}"'.format(
                    len(buffer_front), len(buffer_end), len(frames_all) - nframes_user, enhancement))
            else:
                # Non-contiguous frame range so only consider adding frames throughout
                # TODO: improve efficiency - only consider jumps
                for n in frames_user:
                    frames_all = set(frames_all).union(self.get_frame_list(n, limits=movie_range, **kws))
                frames_all = np.sort(np.array(list(frames_all)))
                frame_range_all = [np.min(frames_all), np.max(frames_all)]
                logger.info('Added additional {} frames for enhancement "{}"'.format(
                                    len(frames_all) - nframes_user, enhancement))
            # Make sure extended frame range is within movie range
            if (movie_range[0] > frame_range_all[0]):
                logger.warning('Movie enhancement "{}" lacks preceeding frames for first {} frames'.format(
                        enhancement, movie_range[0]-frame_range_all[0]))
                frame_range_all[0] = movie_range[0]
            if (movie_range[1] < frame_range_all[1]):
                logger.warning('Movie enhancement "{}" lacks following frames for last {} frames'.format(
                        enhancement, frame_range_all[1] - movie_range[1]))
                frame_range_all[1] = movie_range[1]
            frame_range_info_all['frame_range'] = frame_range_all

            nframes_all = len(frames_all)
            t_all = self._movie_meta['t_range'][0] + frames_all / fps
            t_range_all = [np.min(t_all), np.max(t_all)]
            frame_range_info_all['frames'] = frames_all
            frame_range_info_all['nframes'] = nframes_all
            frame_range_info_all['t'] = t_all
            frame_range_info_all['t_range'] = t_range_all

            assert len(frames_all) == len(t_all) == nframes_all
        else:
            frame_range_all = frame_range_user

        assert len(frames_user) == len(t_user) == nframes_user, '{} != {} != {}'.format(
                                                            len(frames_user), len(t_user), nframes_user)

        self._x['values'] = frames_all

        self.initialise_meta()
        self._meta['t'] = t_all
        self._meta['enhanced'] = False
        self._meta['user'] = False
        self._meta.loc[frames_user, 'user'] = True
        if self.current_frame not in self.frame_numbers:
            self.current_frame = frames_user[0]
        # TODO: recover useful data from previous_frames_data when extending frame range etc
        self.settings['start_frame'] = frame_range_all[0]
        self.settings['end_frame'] = frame_range_all[1]
        stride = (np.ptp(frames_all)+1)/len(frames_all)
        # 0 -> no applicable stride length (avoid np.nan for SettingInt error)
        self.settings['stride'] = int(stride) if stride % 1 == 0 else 0
        self._data = None
        self._values = None
        self._enhanced_movie = None
        pass
    
    def set_enhancements(self, enhancements):
        """Set image processing enhancements"""
        # TODO change to property?
        if self.settings['enhancements'] != enhancements:
            self.settings['enhancements'] = enhancements
        if self._enhancements != enhancements:
            self._meta.loc[:, 'enhanced'] = False
            self._enhanced_movie = None
            self._enhancements = None  # Enhancements already applied to _ENHANCED_movie
        else:
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
            frame_inputs = [start_frame, end_frame, nframes]
            if np.sum([v is not None for v in frame_inputs]) < 2:
                raise ValueError('Insufficient input information to build frame range, {}-{} ({})'.format(
                        start_frame, end_frame, nframes))
            assert isinstance(stride, (int, float))
            stride = int(stride)
            # Handle -ve frame numbers
            movie_end_frame = self._movie_meta['frame_range'][1]
            if (start_frame is not None) and start_frame < 0:
                start_frame = movie_end_frame + start_frame + 1
            if (end_frame is not None) and end_frame < 0:
                end_frame = movie_end_frame + end_frame + 1
            # Complete missing info
            if start_frame is None:
                start_frame = end_frame - (nframes + 1)
            if end_frame is None:
                end_frame = start_frame + (nframes - 1)
            if nframes is None:
                nframes = int(np.ceil((end_frame - start_frame) / stride)) + 1

            assert start_frame <= end_frame
            assert end_frame == (start_frame + (nframes -1) * stride)

            frames = np.arange(start_frame, end_frame+1, stride)
        else:
            frames = np.array(frames)
            start_frame, end_frame, nframes = frames.min(), frames.max(), len(frames)
        nframes = len(frames)

        # TODO: change start stop to range list []
        frame_range = np.array([np.min(frames), np.max(frames)])
        return frames, nframes, frame_range

    def check_user_frame_range(self):
        # TODO: break into sub functions for each file format
        movie_range = self._movie_meta['frame_range']
        frame_range = self._frame_range_info_user['frame_range']
        if (movie_range[0] > frame_range[0]) or (movie_range[1] < frame_range[1]):
            raise ValueError('Frame range {} set outside of movie file frame range {}'.format(frame_range, movie_range))

    @classmethod
    def get_mraw_file_info(cls, fn_path, transforms=[]):
        """Get meta data from mraw file"""
        # TODO: unhardcode transforms info in frame_shape
        movie_meta = {'movie_format': '.mraw'}
        mraw_files = pd.DataFrame({'StartFrame': []})
        # Get frame range in each mraw file
        n = 0
        while Path(fn_path.format(n=n)).is_file():
            vid = mrawReader(filename=fn_path.format(n=n))
            header = vid.file_header
            start_frame = int(header['StartFrame'].strip())
            mraw_files.loc[n, 'StartFrame'] = start_frame
            for key in ['TotalFrame', 'StartFrame']:  # 'OriginalTotalFrame', 'CorrectTriggerFrame', 'ZeroFrame']:
                mraw_files.loc[n, key] = int(header[key].strip())
            # Get time ranges for each file
            mraw_files.loc[n, 'StartTime'] = np.round(vid.set_frame_number(0).read()[2]['time_stamp'], decimals=6)
            mraw_files.loc[n, 'EndTime'] = np.round(
                    vid.set_frame_number(int(header['TotalFrame'].strip())).read()[2]['time_stamp'], decimals=6)
            if (len(mraw_files) > 1):
                if (mraw_files.loc[n, 'StartTime'] < mraw_files.loc[n-1, 'EndTime']):
                    # If same start time as previous file, add on to previous end time
                    mraw_files.loc[n, 'StartTime'] += mraw_files.loc[n-1, 'EndTime']
                if (mraw_files.loc[n, 'EndTime'] <= mraw_files.loc[n-1, 'EndTime']):
                    # If same start time as previous file, add on to previous end time
                    mraw_files.loc[n, 'EndTime'] += mraw_files.loc[n-1, 'EndTime']
            vid.release()
            n += 1
        assert n > 0, 'No mraw files read'
        # Calcuate time interval stored in each file
        mraw_files.loc[:, 'TotalTime'] = mraw_files.loc[:, 'EndTime'] - mraw_files.loc[:, 'StartTime']
        # Mraw movie frames don't start at zero so record start frame and offset by it so start at 0
        movie_meta['frame0'] = int(mraw_files.loc[0, 'StartFrame'])
        mraw_files.loc[:, 'StartFrame'] -= movie_meta['frame0']
        mraw_files.loc[:, 'EndFrame'] = mraw_files.loc[:, 'StartFrame'] + mraw_files.loc[:, 'TotalFrame'] - 1
        # Reset column types and order
        mraw_files = mraw_files.astype({'StartFrame': int, 'EndFrame': int, 'TotalFrame': int})
        mraw_files = mraw_files[['StartFrame', 'EndFrame', 'TotalFrame', 'StartTime', 'EndTime', 'TotalTime']]
        # Get additional meta data eg frame shape
        movie_meta['mraw_files'] = mraw_files
        movie_meta['mraw_header'] = header
        movie_meta['frame_range'] = [int(mraw_files.loc[0, 'StartFrame']), int(mraw_files.loc[n-1, 'EndFrame'])]
        movie_meta['t_range'] = [mraw_files.loc[0, 'StartTime'], mraw_files.loc[n-1, 'EndTime']]
        movie_meta['frame_shape'] = (int(header['ImageHeight'].strip()), int(header['ImageWidth'].strip()))
        movie_meta['fps'] = int(header['RecordRate(fps)'].strip())
        logger.info('Readimg mraw movie file {}:\n{}'.format(fn_path, mraw_files))
        return movie_meta

    @classmethod
    def get_ipx_file_info(cls, fn_path, transforms=[]):
        """
        Ipx files can be tested on Freia using eg:
            ipxview /net/fuslsa/data/MAST_IMAGES/023/23400/rba023400.ipx
        :param fn_path:
        :param transforms:
        :return:
        """
        movie_meta = {'movie_format': '.ipx'}
        # n = 0
        # step = 100

        # Read header info for first frame
        vid = ipxReader(filename=fn_path)
        file_header = vid.file_header
        ret, frame0, frame_header_first = vid.read(transforms=transforms)

        # Read header info for last frame, working backwards from end until frame is successfully read
        ret = False
        n = file_header['numFrames'] if 'numFrames' in file_header else file_header['frames']
        while (n>=0):
            n -= 1
            try:
                vid.set_frame_number(n)
                ret, frame, frame_header_last = vid.read(transforms=transforms)
                if vid._file.closed:
                    vid = ipxReader(filename=fn_path)
                if ret:
                    break
            except ValueError as e:
                pass
        else:
            raise ValueError('Failed to load last frame of movie (n={n}): {fn_path}'.format(n=n, fn_path=fn_path))
        vid.release()

        movie_meta['ipx_header'] = file_header
        movie_meta['frame_range'] = [0, n]
        movie_meta['t_range'] = np.array([float(frame_header_first['time_stamp']), float(frame_header_last['time_stamp'])])
        movie_meta['frame_shape'] = frame0.shape
        movie_meta['fps'] = (n) / np.ptp(movie_meta['t_range'])
        logger.info('Readimg ipx movie file {} with frame range {}'.format(fn_path, movie_meta['frame_range']))
        return movie_meta

    @classmethod
    def get_pickle_movie_info(cls, fn_path, transforms=[]):
        """Get meta data from pickled frame data"""
        path, fn = os.path.split(fn_path)
        # TODO: Get filename formats from settings file
        fn_patterns = ['^Frame_(\d+)_?(.*).p$', '^(.*).p$']
        files_all = [os.path.basename(p) for p in glob.glob(path + '/*')]

        # Find filename format
        for pattern in fn_patterns:
            # Find pickled frame files (NOT 'filament_data')
            frame_files = natsort.natsorted([f for f in files_all if re.match(pattern, f) and
                                             not re.search('filament_data', f)])
            if len(frame_files) > 0:
                # If files don't have name format Frame_0.p etc take all pickle files without filament_data
                fn_pattern = pattern
                break
        else:
            raise IOError('No pickle frame files located in "{}" for fn_patterns {}'.format(path, fn_patterns))

        # Dict linking frame numbers to filenames
        frames_all = {int(re.match(fn_pattern, f).group(1)): f for f in frame_files}

        pickle_files = {'path': path, 'fn_pattern': fn_pattern, 'frames_all': frames_all}
        movie_meta = {'movie_format': '.p', 'pickle_files': pickle_files}
        movie_meta['frame_range'] = [np.min(list(frames_all.keys())), np.max(list(frames_all.keys()))]
        movie_meta['t_range'] = [np.nan, np.nan]
        movie_meta['fps'] = np.nan
        example_frame = pickle_load(os.path.join(path, frame_files[0]), encoding='latin1')
        movie_meta['frame_shape'] = transform_image(example_frame, transforms).shape
        movie_meta['frame0'] = 0
        return movie_meta

    @classmethod
    def get_npz_movie_info(cls, fn_path, transforms=()):
        """Get meta data from pickled frame data"""
        path, fn = os.path.split(fn_path)
        # TODO: Get filename formats from settings file
        fn_patterns = ['^Frame_data_(\d+).npz$']
        files_all = [os.path.basename(p) for p in glob.glob(path + '/*')]

        # Find filename format
        for pattern in fn_patterns:
            # Find pickled frame files (NOT 'filament_data')
            frame_files = natsort.natsorted([f for f in files_all if re.match(pattern, f)])
            if len(frame_files) > 0:
                # If files don't have name format Frame_0.p etc take all pickle files without filament_data
                fn_pattern = pattern
                break
        else:
            raise IOError('No npz frame files located in "{}" for fn_patterns {}'.format(path, fn_patterns))

        # Dict linking frame numbers to filenames
        frames_all = {int(re.match(fn_pattern, f).group(1)): f for f in frame_files}

        npz_files = {'path': path, 'fn_pattern': fn_pattern, 'frames_all': frames_all}
        movie_meta = {'movie_format': '.npz', 'npz_files': npz_files}
        movie_meta['frame_range'] = [np.min(list(frames_all.keys())), np.max(list(frames_all.keys()))]
        movie_meta['t_range'] = [np.nan, np.nan]
        movie_meta['fps'] = np.nan
        example_frame = np.load(os.path.join(path, frame_files[0]))
        if hasattr(example_frame, 'files'):
            # If numpy file is a compressed file archive, extract the frame array from the archive
            example_frame = example_frame['frame']

        # TODO: Remove tmp bodge!
        if transforms == []:
            transforms = ['transpose', 'reverse_y']

        movie_meta['frame_shape'] = transform_image(example_frame, transforms).shape
        movie_meta['frame0'] = np.min(list(frames_all.keys()))
        return movie_meta

    def get_mraw_file_number(self, **kwargs):
        assert self._movie_meta is not None
        mraw_files = self._movie_meta['mraw_files']
        frame_range = self._movie_meta['frame_range']
        n = self.lookup('n', **kwargs)
        mask = (mraw_files['StartFrame'] <=  n) & (n <= mraw_files['EndFrame'])
        if np.sum(mask) != 1:
            raise ValueError('Frame number {} is outside of mraw file frame range {}'.format(n, frame_range))
        file_number = mraw_files.loc[mask].index.values[0]
        file_info = mraw_files.loc[file_number, :].to_dict()
        return file_number, file_info

    def _fill_values(self, **kwargs):
        """Called by Stack when data is accessed to ensure self._values is not empty"""
        if self._values is None:
            if self.fn_path is None:
                Stack._fill_values(self)
            else:
                self.load_movie_data(**kwargs)

    def load_movie_data(self, n=None):  # , pulse=None, machine=None, camera=None):
        """Load movie data into xarray given previously loaded movie file information"""
        self._init_xarray()
        if self.fn_path is None:
            raise ValueError('No movie file has been set for reading')
        if self._frame_range_info_user is None:
            raise ValueError('A frame range must be set before a movie can be read')

        if self.movie_format == '.mraw':
            data = self.read_mraw_movie(n=n)
        elif self.movie_format == '.p':
            data = self.read_pickle_movie(n=n)
        elif self.movie_format == '.npz':
            data = self.read_npz_movie(n=n)
        elif self.movie_format == '.ipx':
            data = self.read_ipx_movie(n=n)
        else:
            raise ValueError('Movie class does not currently support "{}" format movies'.format(self.movie_format))
        
        if n is None:
            self.set_data(data, reset=True)  # whole dataset
            self._meta['set'] = True  # TODO: Check correct?
        else:
            n = make_iterable(n)
            if len(n) == 0:
                return
            self._data.loc[{'n': n}] = data
            self._meta.loc[n, 'set'] = True
        return self
        # logger.debug('{} loaded movie data from {}'.format(repr(self), self.fn_path))

    def read_mraw_movie(self, n=None):
        # Initialise array for data to be read into

        if n is None:
            # Loop over frames from start frame, including those in the frame set
            # TODO: change to only loading files that haven't already been loaded?
            frames = self._frame_range_info_all['frames']
            n = self._frame_range_info_all['frame_range'][0]
            end = self._frame_range_info_all['frame_range'][1]
            i_meta = 0
            i_data = 0
        else:
            assert is_numeric(n), 'frames to load are not numeric: {}'.format(n)
            frames = make_iterable(n)
            n = frames[0]
            end = frames[-1]
            i_data = 0
        data = np.zeros((len(frames), *self._movie_meta['frame_shape']))
        logger.debug('Reading {} frames from mraw movie file'.format(len(frames)))

        file_number, file_info = self.get_mraw_file_number(n=n)
        vid = mrawReader(filename=self.fn_path.format(n=file_number))
        vid.set_frame_number(n - file_info['StartFrame'])
        while n <= end:
            # If reached end of current file, switch to next file
            if n > file_info['EndFrame']:
                vid.release()
                file_number, file_info = self.get_mraw_file_number(n=n)
                vid = mrawReader(filename=self.fn_path.format(n=file_number))
                vid.set_frame_number(n - file_info['StartFrame'])
            if n in frames:
                # frames are read with 16 bit dynamic range, but values are 10 bit!
                ret, frame, header = vid.read(transforms=self._transforms)
                data[i_data, :, :] = frame
                # self._meta.loc[n, 'set'] = True
                i_data += 1
            else:
                # TODO: Increment vid frame number without reading data
                vid._skip_frame()
                # ret, frame, header = vid.read(transforms=self._transforms)
            n += 1
        vid.release()
        return data

    def read_ipx_movie(self, n=None):
        # Initialise array for data to be read into

        if n is None:
            # Loop over frames from start frame, including those in the frame set
            # TODO: change to only loading files that haven't already been loaded?
            frames = self._frame_range_info_all['frames']
            n = self._frame_range_info_all['frame_range'][0]
            end = self._frame_range_info_all['frame_range'][1]
            i_meta = 0
            i_data = 0
        else:
            assert is_numeric(n), 'frames to load are not numeric: {}'.format(n)
            frames = make_iterable(n)
            n = frames[0]
            end = frames[-1]
            i_data = 0
        data = np.zeros((len(frames), *self._movie_meta['frame_shape']))
        logger.debug('Reading {} frames from mraw movie file'.format(len(frames)))

        vid = ipxReader(filename=self.fn_path)
        vid.set_frame_number(n)
        while n <= end:
            if n in frames:
                # frames are read with 16 bit dynamic range, but values are 10 bit!
                ret, frame, header = vid.read(transforms=self._transforms)
                data[i_data, :, :] = frame
                self._meta.loc[n, 'set'] = True
                i_data += 1
            elif n > self._movie_meta['frame_range'][1]:
                logger.warning('n={} outside ipx movie frame range'.format(n))
                break
            else:
                # TODO: Increment vid frame number without reading data
                vid._skip_frame()
                # ret, frame, header = vid.read(transforms=self._transforms)
            n += 1
        vid.release()
        return data

    def read_pickle_movie(self, n=None):
        # Loop over frames from start frame, including those in the frame set
        frames = self._frame_range_info_all['frames'] if n is None else make_iterable(n)
        # Initialise array for data to be read into
        data = np.zeros((len(frames), *self._movie_meta['frame_shape']))
        
        transforms = self._transforms
        path = self._movie_meta['pickle_files']['path']
        frames_all = self._movie_meta['pickle_files']['frames_all']
        # frames = self._frame_range['frames']
        for i, n in enumerate(frames):
            if n not in frames_all:
                raise IOError('Cannot locate pickle file for frame n={}'.format(n))
            fn = frames_all[n]
            data_i = pickle_load(fn, path, encoding='latin1')
            data_i = transform_image(data_i, transforms)
            data[i] = data_i
            self._meta.loc[n, 'set'] = True
        return data

    def read_npz_movie(self, n=None):
        # raise NotImplementedError
        # Loop over frames from start frame, including those in the frame set
        frames = self._frame_range_info_all['frames'] if n is None else make_iterable(n)
        # Initialise array for data to be read into
        data = np.zeros((len(frames), *self._movie_meta['frame_shape']))

        transforms = self._transforms
        path = self._movie_meta['npz_files']['path']
        frames_all = self._movie_meta['npz_files']['frames_all']
        # frames = self._frame_range['frames']
        for i, n in enumerate(frames):
            if n not in frames_all:
                raise IOError('Cannot locate npz file for frame n={}'.format(n))
            fn = frames_all[n]
            data_i = np.load(os.path.join(path, fn))
            if hasattr(data_i, 'files'):
                data_i = data_i['frame']

            # TODO: remove tmp bodge for npz files which are saved in different format to pickle synth data!
            if transforms == []:
                transforms = ['transpose', 'reverse_y']

            data_i = transform_image(data_i, transforms)
            data[i] = data_i
            self._meta.loc[n, 'set'] = True
        return data

    def __getitem__(self, n, raw=None, load=True, keep_raw=False):
        """Return data slices from the stack of data
        :param raw: True=return unenhanced data, False=return enhanced data (applying enhancements if nessecary),
                    None=Return existing enhanced data if available (without enhancing) else raw data"""
        # If have enhaced data return that over raw data
        if n not in self._frame_range_info_all['frames']:
            raise IndexError('{} is not a valid frame number. Options: {}'.format(n, self._frame_range_info_all['frames']))
        if (raw is False):
            if (self._enhanced_movie is None):
                # Force enhanced, so make sure  _enhanced_movie is not None
                self._setup_enhanced_movie(self.enhancements, force=True)
            is_set = self._enhanced_movie.check_data(n)
            if (not is_set):
                self.enhance(enhancements=self.enhancements, frames=[n], keep_raw=keep_raw)
        if (not raw) and (self._enhanced_movie is not None) and (self._enhanced_movie.check_data(n)):
            # Return enhanced frame if it is already available
            movie = self._enhanced_movie
        else:
            movie = self
            is_set = movie.check_data(n)
            if (not is_set) and load:
                self.load_movie_data(n)

        # item = movie.lookup_slice_index(item)
        return movie.get_slice(n)

    def __call__(self, raw=None, load=True, keep_raw=False, **kwargs):
        assert len(kwargs) > 0, 'Stack.__call__ requires keyword arg meta data to select frame'
        item = self.lookup(self.stack_dim, **kwargs)
        return self.__getitem__(item, raw=raw, load=load, keep_raw=keep_raw)

    @classmethod
    def get_frame_list(cls, n, current=True, n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1, skip_backwards=0,
                       skip_forwards=0, limits=(0, None), unique=True, verbose=False):
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
                                 num=frame_list_settings['n_backwards']).astype(int)]
        # Include current frame
        if current:
            frame_nos.append(np.array([n]))
        # Frames after current frame
        frame_nos.append(np.linspace(n + frame_list_settings['skip_forwards'] + 1,
                                end_frame,
                                num=frame_list_settings['n_forwards']).astype(int))
        frame_nos = np.round(np.hstack(frame_nos)).astype(int)

        # Make sure frames are in frame range
        if limits is not None:
            frame_nos = frame_nos.clip(min=limits[0], max=limits[1])
        # frameMarks = frameNos + self.frameNumbers[0]
        unique_frames = np.sort(np.array(list(set(frame_nos))))
        if len(unique_frames) < len(frame_nos):
            logger.warning('Frame list contains duplicates')
        if unique:  # remove duplicates
            frame_nos = unique_frames
        # logger.debug('Frames in frame_list:  {}'.format(str(frame_nos)))
        return frame_nos

    def extract_frame_stack(self, frames):
        """Extract array of frame data for set of frames"""
        # TODO: move functionality to Stack class
        frames = np.array(frames)
        self._init_xarray()
        # Make sure all frames in stack have been load - avoid nans!
        nans = self._meta.loc[:, 'set'].isnull()
        self._meta.loc[nans, 'set'] = False
        self._meta['set'] = self._meta['set'].astype({'set': bool})
        frames_not_set = list(frames[~self._meta.loc[frames, 'set'].values])
        if self.data.loc[frames].isnull().any():
            for n in frames:
                if self.data.loc[n].isnull().any() and (n not in frames_not_set):
                    frames_not_set.append(n)
        if len(frames_not_set) > 0:
            self.load_movie_data(n=frames_not_set)
        frame_stack = self._data.loc[{'n': frames}].values
        return frame_stack

    def extract_frame_stack_window(self, current, n_backwards=10, n_forwards=0, step_backwards=1, step_forwards=1,
                                   skip_backwards=0, skip_forwards=0, unique=True, verbose=False, raw=False):
        frames = self.get_frame_list(current, n_backwards=n_backwards, n_forwards=n_forwards,
                                     step_backwards=step_backwards, step_forwards=step_forwards,
                                     skip_backwards=skip_backwards, skip_forwards=skip_forwards,
                                     unique=unique, limits=self._frame_range_info_all['frame_range'])
        # If movie is enhanced, but some of the required frames are not enhanced, enhance those that have been missed
        #  -> return consistent enhanced output
        if self.is_enhanced and (not raw):
            # Make sure all the required frames have been enhanced
            mask = (self._raw_movie._meta['enhanced'] == True).values
            not_enhanced = np.array(list(set(frames) - set(self._raw_movie._meta['n'][mask].values)))
            self.enhance(self._raw_movie._enhancements, not_enhanced)
            # Get frames from enhanced movie
            frame_stack = self._enhanced_movie.extract_frame_stack(frames)
        else:
            frame_stack = self.extract_frame_stack(frames)
        return frame_stack

    def _setup_enhanced_movie(self, enhancements, force=False):
        """Reset enhanced movie if enhancements are to change"""
        if (not self.is_enhanced) or (self._enhancements != enhancements) or force:
            # Update attributes of self
            self._enhancer = Enhancer(setting=self.settings['enhancer'])  # TODO: specify enhancement settings
            self._enhancements = []  # Enhancements applied to _ENHANCED_movie
            self._meta.loc[:, 'enhanced'] = False
            # Set fresh enhanced movie to copy of self
            # Settings objects don't like being deep coppied, so set all settings to None and reassign them afterwares
            # TODO: Fix deepcopy of Settings objects
            enhanced_movie = copy(self)
            enhanced_movie._data = deepcopy(self._data)  # Separate data
            enhanced_movie._raw_movie = self
            enhanced_movie._enhanced_movie = None
            enhanced_movie.settings = None
            enhanced_movie.source_info = None
            enhanced_movie.range_settings = None
            enhanced_movie._enhancer = None
            enhanced_movie._slices = {}
            enhanced_movie._enhancer = None
            enhanced_movie._enhancements = None
            enhanced_movie.name = self.name + '_enhanced'
            enhanced_movie._meta = deepcopy(self._meta)

            # TODO: leave raw values set to avoid them needing to be reloaded? Rely on raw['enhanced']?
            enhanced_movie._meta[['set']] = False
            
            enhanced_movie._meta[['enhanced']] = False

            self._enhanced_movie = enhanced_movie
            enhanced_movie = self._enhanced_movie
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
        # If this is the first enhancement to be applied make sure the enhanced data has been set to the raw data
        frames = np.array(frames)
        frames_not_set_enhanced = frames[~self._enhanced_movie._meta.loc[frames, 'set'].values]
        frames_not_set_raw = frames[~self._meta.loc[frames, 'set'].values]
        if len(frames_not_set_enhanced) > 0:
            if len(frames_not_set_raw) > 0:
                # Make sure raw data has been loaded for all the relevant frames
                self.load_movie_data(n=frames_not_set_raw)
            # Make sure enhanced movie starts out with all relevant frames set to raw data
            # frames_not_set = frames[~self._enhanced_movie._meta.loc[frames, 'set'].values]
            self._enhanced_movie._data.loc[{'n': frames_not_set_enhanced}] = self._data.loc[
                                                                                {'n': frames_not_set_enhanced}]
            self._enhanced_movie._meta.loc[frames_not_set_enhanced, 'set'] = True
        # TODO: Make parallel - threading?
        for n in frames:
            if self._meta.loc[n, 'enhanced'] == True:
                # Already enhanced
                logger.debug('Frame n={} already enhanced'.format(n))
                continue
            args.append(self._enhancer.prepare_arguments(enhancement, self, n))
            # Get current state of enhanced frame, so enhancements are applied sequentially
            # Will return enhanced frame if available but will not try to apply all enhancements before returning
            frame_arrays.append(self(n=n, raw=None)[:])
            # frame_arrays.append(self._enhanced_movie._data.sel(n=n).values)

        # TODO: Make parallel - processes
        j = 0
        for n in frames:
            if self._meta.loc[n, 'enhanced'] == True:
                # Skip frame if already enhanced
                logger.debug('Frame n={} already enhanced'.format(n))
                continue
            # NOTE: movie and n args only needed for fg and bg extraction
            self(n=n, raw=None, load=False)[:] = self._enhancer(enhancement, frame_arrays[j], **args[j])
            # self._meta.loc[n, 'enhanced'] = True
            j += 1
            pass

    def enhance(self, enhancements, frames='all', keep_raw=True, **kwargs):
        """Apply mutiple enhancements to a set of frames in order"""
        # TODO: make t ect valid input
        self._init_xarray()
        if ((isinstance(frames, str)) and (frames == 'all')):
            frames = self.frame_numbers
        frames = to_array(frames)
        enhancements = make_iterable(enhancements)
        for key, value in kwargs.items():
            if key in self.settings.items:
                self.settings[key] = value
        self.set_enhancements(enhancements)
        # TODO: Set frames again to include frame stack data for fg_extraction etc
        if not self._setup_enhanced_movie(enhancements) and is_subset(frames, self.enhanced_frames):
            logger.warning('Enhancements {} already applied to {}'.format(enhancements, repr(self)))
            return
        # TODO: check all frame values in stack axis
        if len(enhancements) > 0:
            for enhancement in enhancements:
                self._apply_enhancement(frames, enhancement)
        else:
            # No enhancements so reset enhanced frames equal to raw frames
            # TODO: Redundant if set up right
            for n in frames:
                self._enhanced_movie(n=n, raw=True)[:] = self(n=n, raw=True, load=True)[:]
        # # Set frames as having been enhanced
        mask = self._meta['n'].isin(frames)
        self._meta.loc[mask, 'enhanced'] = True
        self._enhancements = enhancements
        if not keep_raw:
            self._data = None
        # raise NotImplementedError

    def to_hdf5(self, fn=None):
        raise NotImplementedError

    def from_hdf5(self, fn=None):
        raise NotImplementedError

    def open_gui(self):
        from ccfepyutils.guis.movie_gui.movie_gui import MovieGUI
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication(sys.argv)

        app.aboutToQuit.connect(app.deleteLater)  # if using IPython Console
        self.gui = MovieGUI(self)

        sys.exit(app.exec_())
        self.gui = None

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
        return self.source_info['pulse']

    @pulse.setter
    def pulse(self, value):
        if value is not None:
            assert isinstance(value, (numbers.Number, str))
            self.source_info['pulse'] = value

    @property
    def machine(self):
        return self.source_info['machine']

    @machine.setter
    def machine(self, value):
        if value is not None:
            # if value not in self.compatibities:
            #     raise ValueError('Movie class is not compatible with machine "{}". Options: {}'.format(
            #             value, self.compatibities.keys()))
            self.source_info['machine'] = value

    @property
    def camera(self):
        return self.source_info['camera']

    @camera.setter
    def camera(self, value):
        if value is not None:
            assert self.machine is not None, 'Machine must be set before camera'
            # if value not in self.compatibities[self.machine]:
            #     raise ValueError('Movie class is not compatible with camera "{}". Options: {}'.format(
            #             value, self.compatibities[self.machine].keys()))
            self.source_info['camera'] = value

    @property
    def name(self):
        """Name identifying movie instance

        Overloads method in data Stack object"""
        if self._name is not None:
            return self._name
        else:
            try:
                name = '{}:{}:{}'.format(self.machine, self.pulse, self.camera)
                return name
            except Exception as e:
                return super().name

    @name.setter
    def name(self, value):
        """Name identifying movie instance

        Property setter inheritance, see: https://gitlab.gnome.org/GNOME/gnome-music/snippets/31
        """
        super(Movie, self.__class__).name.fset(self, value)
        # assert isinstance(value, (str, type(None)))
        # self._name = value

    @property
    def frame_range(self):
        return self._frame_range_info_user['frame_range']

    @property
    def nframes(self):
        return self._frame_range_info_user['nframes']
        
    @property
    def frame_numbers(self):
        return self._frame_range_info_user['frames']

    @property
    def frame_times(self):
        return self._frame_range_info_user['t']

    @property
    def image_resolution(self):
        return self._movie_meta['frame_shape']

    @property
    def enhancements(self):
        """Return currently set enhancements (may not be applied yet)"""
        if self.settings is None:
            return []
        try:
            enhancements = self.settings['enhancements'].value
        except KeyError as e:
            enhancements = []
        except Exception as e:
            raise e
        return enhancements

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

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, n):
        if n not in self.frame_numbers:
            n = replace_in(n, self.frame_numbers, tol=1e10)
        if self.current_frame == n:
            # Avoid cyclic call from gui
            return
        self._current_frame = n
        if self.gui is not None:
            self.gui.set_frame(n=n)

class Enhancer(object):
    """Class to apply image enhancements to arrays of data"""
    desciptions = {'bgsub': {'requires_window': True}}
    default_settings = {}
    from ccfepyutils.image import image_enhancement_functions

    def __init__(self, setting='default'):
        from setpy import Settings
        self.settings = Settings('enhancer', setting)
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
        data = copy(data)
        funcs = self.image_enhancement_functions
        enhancements = make_iterable(enhancements)
        if np.max(data) == 0:
            logging.warning('Enhancement(s) {} is being applied to an empty frame'.format(enhancements))
        for enhancement in enhancements:
            if enhancement not in funcs:
                raise ValueError('Enhancement {} not recognised'.format(enhancement))
            ## TODO:
            func = funcs[enhancement]
            if len(kwargs) == 0:  # if keyword arguments havent been passed, look them up
                kwargs.update(self.settings.args_for(func, kwargs=kwargs))
            out = funcs[enhancement](image=data, **kwargs)
        return out

    def prepare_arguments(self, enhancement, movie, n):
        """Prepare arguments for enhancement function. Get inputs ready for parallel processing."""
        kwargs = {}
        func = self.image_enhancement_functions[enhancement]
        func_name = func.__name__
        sig = inspect.signature(func).parameters.keys()
        if 'frame_stack' in sig:

            # NOTE: Currently only extracts raw frames for frame stack - ie frame stack enhancement must come first

            # TODO: Add extract_frame_stack_window args to enhancer settings
            kws = self.settings.args_for(movie.extract_frame_stack_window)

            # # Get specific frame stack settings for this function
            # args = ['n_backwards', 'n_forwards', 'step_backwards', 'step_forwards', 'skip_backwards', 'skip_forwards']
            # for arg in args:
            #     func_arg = '{func_name}::{arg}'.format(func_name=func_name, arg=arg)
            #     if func_arg in self.settings:
            #         kws[arg] = self.settings[func_arg]

            frame_stack = movie.extract_frame_stack_window(n, raw=True, **kws)
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

def transform_image(image, transforms):
    for trans in transforms:
        if trans == 'reverse_x':
            image = image[::-1]
        elif trans == 'reverse_y':
            image = image[..., ::-1]
        elif trans == 'transpose':
            image = image.T
        else:
            raise ValueError('Transform "{}" not recognised'.format(trans))
    return image

if __name__ == '__main__':
    # movie = Movie(29852, 'MAST', 'SA1.1', start_frame=0, end_frame=10)
    movie = Movie(29991, 'MAST', 'SA1.1', start_frame=2889, end_frame=4689)
    # movie = Movie(23085, 'MAST', 'StereoA', start_frame=0, end_frame=10)
    # movie = Movie(23400, 'MAST', 'StereoA', start_frame=0, end_frame=10)
    pass