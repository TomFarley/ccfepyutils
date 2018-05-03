import logging
# from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')

from ccfepyutils.classes.movie import Movie
from ccfepyutils.classes.settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('** Running test_enhance')

def enhance_movie():
    pulse = 29852
    machine = 'MAST'
    camera = 'SA1.1'
    start_frame = 13
    end_frame = 25

    # Set the movie file - this will load meta data, but not frame data
    movie = Movie(pulse, machine, camera, name='Movie_demo')

    # Now we can look at the structure of the mraw file
    print('Mraw file structures:\n{}'.format(movie._movie_meta['mraw_files']))
    # Notice the frame range has been set acording to previously set setttings but the frame data has not been loaded
    # - the 'set' column below is all False
    print('Movie frame meta data:\n{}'.format(movie._meta))

    # Now lets set the frame range we want to work with we want to work with (and allocate the memory for that
    # many frames)
    movie.set_frames(start_frame=start_frame, end_frame=end_frame)
    # Now the frame meta data has been updated
    print('Movie frame meta data:\n{}'.format(movie._meta))

    frame24 = movie(i=11)
    frame24.plot(show=True)


    movie.enhance(['extract_fg', 'reduce_noise', 'sharpen'], frames=[24], keep_raw=True)



    # Plotting frames
    movie[end_frame].plot(show=True)
    movie[start_frame].raw

    # Useful attributes
    movie_meta = movie._movie_meta
    movie_meta.pop('mraw_header')
    mraw_meta = movie_meta['mraw_files']
    movie_meta.pop('mraw_files')
    print('Movie header meta data (reduced):\n{}'.format(movie_meta))
    print('Frame numbers: {}'.format(movie.frame_numbers))
    print('Frame times: {}'.format(movie.frame_times))
    print('Frame ramge: {}'.format(movie._frame_range))
    # movie._enhancer.settings('reduce_noise_sigma_space', 75, name='sigma space',
    #                          description='Spacial width for blur (overriden by diameter)')
    pass

def read_npz_synthetic_movie():
    start_frame = 0
    end_frame = 1

    movie = Movie(source='synthcam_demo', name='Movie_demo', start_frame=start_frame, end_frame=end_frame)
    # movie.set_frames(start_frame=start_frame, end_frame=end_frame)
    movie(i=1).plot(show=True)

def edit_movie_source_settings():
    source = 'synth_for_tom'
    s = Settings.get('Movie_source', source)
    s['camera'] = 'SynthCam'
    s['pulse'] = 'for_tom'
    s.save()
    # s = Settings.get('Movie_source')
    print(s.view())
    pass

if __name__ == '__main__':
    # edit_movie_source_settings()
    read_npz_synthetic_movie()
    # enhance_movie()
