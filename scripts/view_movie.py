from ccfepyutils.classes.movie import Movie


def show_frames():
    pulse = 29840
    machine = 'MAST'
    camera = 'SA1.1'

    enhancements = ['extract_fg', 'reduce_noise', 'sharpen']

    movie = Movie(pulse, machine, camera, name='Movie_demo');

    start_frame = 0
    end_frame = 30000
    frame_stride = 1000
    movie.set_frames(start_frame=start_frame, end_frame=end_frame, frame_stride=frame_stride)
    movie.enhance(enhancements, frames='all')

    for n in movie.frame_numbers:
        movie[n].plot(show=True)


if __name__ == '__main__':
    show_frames()