import logging
# from logging.config import fileConfig, dictConfig
# fileConfig('../logging_config.ini')

from ccfepyutils.classes.movie import Movie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('** Running test_enhance')

def enhance_movie():
    pulse = 29852
    machine = 'MAST'
    camera = 'SA1.1'
    start_frame = 13
    end_frame = 20
    nframes = end_frame - start_frame + 1

    movie = Movie(pulse, machine, camera, name='Movie_demo')
    movie.set_frames(start_frame=start_frame, end_frame=end_frame)
    # movie.enhance(['extract_fg'], frames='all', keep_raw=True)
    # movie.enhance(['reduce_noise', 'sharpen'], frames='all', keep_raw=True)
    movie.enhance(['extract_fg', 'reduce_noise', 'sharpen'], frames='all', keep_raw=True)
    movie[end_frame].plot(show=True)
    movie[start_frame].raw
    # movie._enhancer.settings('reduce_noise_sigma_space', 75, name='sigma space',
    #                          description='Spacial width for blur (overriden by diameter)')
    pass

def read_npz_synthetic_movie():
    pulse = 'for_tom'
    machine = 'MAST'
    camera = 'SynthCam'
    start_frame = 0
    end_frame = 1

    movie = Movie(source='synth_for_tom', name='Movie_demo', start_frame=start_frame, end_frame=end_frame)
    # movie.set_frames(start_frame=start_frame, end_frame=end_frame)
    movie(i=1).plot(show=True)

if __name__ == '__main__':
    read_npz_synthetic_movie()
    # enhance_movie()
