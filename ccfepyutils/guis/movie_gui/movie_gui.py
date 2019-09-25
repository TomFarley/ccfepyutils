# -*- coding: utf-8 -*-

"""
Tom Farley
July 2018

TODO:
BUGS (important):
- Fix fg_extractor making strange changes evert 10 frames
- Fix bug with raw data being modified!
- Fix chained enhancements!
- Fix enhancements when frame changed

BUGS (may be resolved?)
- Fix apply enhancements to all frames?
- Fix empty enhancements bug - recursion error?
- Fix settings frame at end of range when stride!=1 ?

FEATURES (major):
- Add movie file length info/slider
- Create settings for movie_widget
- Add File menu bar
- Add progress bar for enhancements etc

FEATURES (minor):
- Load pulse on enter from movie selection widgets
- Add mpl toolbar buttons, brightness etc
- Make movie file panel hideable
- Add save button for remembering pulse/frame settings?

"""
import sys, os, random, logging, traceback
from pathlib import Path
import numpy as np
from ccfepyutils.classes.movie import Movie
from ccfepyutils.utils import str_to_number, make_iterable, replace_in
from ccfepyutils.image import image_enhancement_functions

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from PyQt5 import uic, QtWidgets, QtGui, QtCore
except ImportError as e:
    logger.exception('Failed to import PyQt5.')

icon_dir = os.path.abspath('../icons/')

class MovieGUI(QtWidgets.QMainWindow):

    def __init__(self, movie):
        super(MovieGUI, self).__init__()
        assert isinstance(movie, Movie)
        self.movie = movie
        # Set up main window and gridlayout for this window
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.gl_main = QtWidgets.QGridLayout(self.centralWidget)
        # Load QtCore.movie widget and add to grid layout
        pwd = Path(__file__).parent
        self.movie_widget = uic.loadUi(str(pwd/'movie.ui'))
        self.gl_main.addWidget(self.movie_widget, 0, 0, 0, 0)

        self._prev_dir = os.path.expanduser('~')  # Record of previous save dir

        self.initUI()
        self.show()

    def initUI(self):
        from mpltoolbar import MplToolbar
        self.setWindowTitle("CCFE Movie Viewer")
        self.movie_widget.gb_enhancements.clicked.connect(self.hide_enhancements_pannel)
        movie = self.movie
        widget = self.movie_widget

        ax = widget.frame_canvas.canvas.ax
        widget.frame_ax = ax
        fig = ax.figure
        self.frame_plot = movie[self.movie.current_frame].plot(ax=ax, show=False)
        widget.frame_image = self.frame_plot.ax_artists[ax]['img']
        # fig.set_facecolor(color)
        fig.patch.set_alpha(0)  # Hide canvas around figure
        fig.subplots_adjust(left=0.0, bottom=0.00, right=1, top=1)  # minimise figure margins
        widget.annotate_frame = True

        # self.connect(self.procGBox, QtCore.SIGNAL("clicked()"), toggle_procbox)
        # Actions
        # self.actionExit.triggered.connect(self.shut)
        # self.actionAbout.triggered.connect(self.about)
        # Signals
        # self.horizontalSlider.valueChanged.connect(self.slider_update)
        # self.mpl_widget.pushButton_plotData.clicked.connect(self.plot_data)
        # self.my_widget.lineEdit.valueCanged.connect(self.line_edit_1_update)

        widget.mpl_toolbar = MplToolbar(widget.frame_canvas.canvas, self, orientation='vertical')
        widget.hl_movie_canvas.addWidget(widget.mpl_toolbar)
        widget.mpl_toolbar.setMaximumSize(QtCore.QSize(40, 500))

        self.play_timer = QtCore.QTimer()
        self.dflt_play_speed = 10
        self.time_multiplier = 1
        self.playing = False
        self.init_movie_controls()

        self.set_movie_file()

        # self.set_frame(n=self.movie.current_frame)
        # self.movie_widget.frame_canvas.canvas.draw()

        # self.adjustSize((300, 300))
        self.setGeometry(100, 100, 400, 550)
        # self.movie._setup_enhanced_movie(movie.enhancements)


    def init_movie_controls(self):
        widget = self.movie_widget
        movie = self.movie
        frame_stride = widget.sb_frame_stride.value()

        widget.btn_play.clicked.connect(self.play_pause)
        self.play_timer.timeout.connect(lambda: self.set_frame(n=self.movie.current_frame +
                                        self.movie._frame_range_info_user['stride'] * np.sign(widget.sb_speed.value())))

        # Play speed
        widget.sb_speed.valueChanged.connect(self.set_play_speed)
        widget.label_speed.clicked.connect(lambda: widget.sb_speed.setValue(1.0))

        def need_update():
            widget.btn_load_meta.setStyleSheet('QPushButton {color: red;}')  #background-color: #A3C1DA;
            widget.btn_load_frames.setStyleSheet('QPushButton {color: red;}')

        def updated():
            widget.btn_load_frames.setStyleSheet('QPushButton {color: black;}')

        # Movie file controls starting values
        widget.le_machine.setText(movie.settings['machine'].value)
        widget.le_camera.setText(movie.settings['camera'].value)
        widget.le_pulse.setText(str(movie.settings['pulse'].value))
        widget.sb_start_frame.setValue(movie._frame_range_info_user['frame_range'][0])
        widget.sb_end_frame.setValue(movie._frame_range_info_user['frame_range'][1])
        widget.sb_frame_stride.setValue(movie._frame_range_info_user['stride'])
        widget.sb_start_frame.valueChanged.connect(
                lambda: (widget.sb_end_frame.setMinimum(widget.sb_start_frame.value()), need_update()))
        widget.sb_end_frame.valueChanged.connect(
                lambda: (widget.sb_start_frame.setMaximum(widget.sb_end_frame.value()), need_update()))
        widget.le_machine.textChanged.connect(need_update)
        widget.le_camera.textChanged.connect(need_update)
        widget.le_pulse.textChanged.connect(need_update)
        widget.sb_frame_stride.valueChanged.connect(need_update)

        # Load meta data/frames
        widget.btn_load_meta.clicked.connect(lambda: (self.set_movie_file()))
        widget.btn_load_frames.clicked.connect(lambda: (self.set_movie_file(), movie.load_movie_data(), updated()))

        # Enhancements - On/Off, apply all
        widget.cb_apply_enhancements.stateChanged.connect(
                lambda: self.set_frame(n=movie.current_frame, force_update=True))
        # TODO: Use Qthread!
        widget.btn_enhance_all_frames.clicked.connect(
                lambda: (movie.enhance(movie.enhancements, n='all', keep_raw=True), self.set_frame(force_update=True)))
        # Enhancements - toggle

        enhancement_widget_names = list(image_enhancement_functions.keys())  #['extract_fg', 'extract_bg', 'reduce_noise', 'sharpen', 'add_abs_gauss_noise']  #: 'Extract_foreground'}
        current_enhancements = movie.settings['enhancements'].value
        for enhancement in enhancement_widget_names:
            try:
                check_box = getattr(widget, 'cb_'+enhancement)
                if enhancement in current_enhancements:
                    check_box.setChecked(True)
                check_box.stateChanged.connect(self.update_enhancement(enhancement))
            except:
                logger.warning('No enhancement checkbox has been implement for enhancement: {}'.format(enhancement))

            # getattr(widget, 'tb_'+widget_name).stateChanged.connect(lambda: self.update_enhancements(toggle=enhancement))

        logger.debug('Starting enhancements: {}'.format(movie.settings['enhancements']))

        # Current frame controls
        widget.btn_next_frame.clicked.connect(
                lambda: self.set_frame(n=self.movie.current_frame+widget.sb_frame_stride.value()))
        widget.btn_prev_frame.clicked.connect(
                lambda: self.set_frame(n=self.movie.current_frame-widget.sb_frame_stride.value()))
        widget.sb_frame_no.valueChanged.connect(lambda: self.set_frame(n=widget.sb_frame_no.value()))
        widget.sb_time.setValue(movie.lookup('t', n=movie.current_frame) * self.time_multiplier)
        widget.sb_time.valueChanged.connect(lambda: self.set_frame(t=widget.sb_time.value()))
        # widget.sb_time.editingFinished.connect(lambda: self.set_frame(t=widget.sb_time.value()))
        sldr_frame = widget.sldr_frame
        sldr_frame.valueChanged.connect(lambda: self.set_frame(n=sldr_frame.value()))
        sldr_frame.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        sldr_frame.setTickInterval(10)
        sldr_frame.setSingleStep(1)

        # Frame toolbar buttons
        toggle_annotate_action = QtWidgets.QAction(QtGui.QIcon(os.path.join(icon_dir, 'appbar.edit.png')),
                                             'Toggle frame annotation', self)
        toggle_annotate_action.setShortcut('Ctrl+a')
        toggle_annotate_action.triggered.connect(self.toggle_annotate_frame)
        self.movie_widget.mpl_toolbar.addAction(toggle_annotate_action)

        save_image_action = QtWidgets.QAction(QtGui.QIcon(os.path.join(icon_dir, 'appbar.save.png')),
                                            'Save current frame image (preserves resolution, no annotations)', self)
        save_image_action.setShortcut('Ctrl+s')
        save_image_action.triggered.connect(self.save_frame_image)
        self.movie_widget.mpl_toolbar.addAction(save_image_action)

        save_figure_action = QtWidgets.QAction(QtGui.QIcon(os.path.join(icon_dir, 'appbar.camera.png')),
                                'Save current frame figure (includes annotations, does not preserve resolution)', self)
        save_figure_action.setShortcut('Ctrl+f')
        save_figure_action.triggered.connect(self.save_figure)
        self.movie_widget.mpl_toolbar.addAction(save_figure_action)
        pass

    def set_movie_file(self):
        movie = self.movie
        widget = self.movie_widget

        # Get values from input boxes
        machine = widget.le_machine.text()
        camera = widget.le_camera.text()
        pulse = str_to_number(widget.le_pulse.text())

        start_frame = widget.sb_start_frame.value()
        end_frame = widget.sb_end_frame.value()
        frame_stride = widget.sb_frame_stride.value()
        widget.sb_frame_stride.setMaximum(end_frame-start_frame-1)

        try:
            movie.set_movie_source(machine=machine, camera=camera, pulse=pulse)
        except OSError as e:
            QtWidgets.QMessageBox.about(self, "Failed to locate movie file", "{}".format(e))
            # TODO: color lineedit inputs to show they are bad
            return
        movie_frame_range = movie._movie_meta['frame_range']
        if start_frame < 0:
            start_frame = movie_frame_range[1] + 1 + start_frame
        if end_frame < 0:
            end_frame = movie_frame_range[1] + 1 + end_frame

        if (start_frame < 0) or (end_frame < 0):
            QtWidgets.QMessageBox.about(self, 'Invalid frame range input', 'Frame range numbers must be > 0')
            return
        if (end_frame > movie_frame_range[1]):
            QtWidgets.QMessageBox.about(self, 'Invalid frame range input', 'End frame outside of move frame range: {} > {}'.format(
                end_frame, movie_frame_range[1]))
            return
        if start_frame > end_frame:
            QtWidgets.QMessageBox.about(self, 'Invalid frame range input', 'Start frame must be before end frame')

        movie.set_frames(start_frame=start_frame, end_frame=end_frame, frame_stride=frame_stride)

        if movie.current_frame not in self.movie.frame_numbers:
            self.set_frame(n=movie.frame_numbers[0])
        else:
            self.set_frame(force_update=True)

        # Frame range spin boxes
        widget.sb_start_frame.setMinimum(-movie_frame_range[1]+1)
        widget.sb_end_frame.setMaximum(movie_frame_range[1])

        # Current frame spinbox
        widget.sb_frame_no.setMinimum(-movie.frame_range[1]+1)
        widget.sb_frame_no.setMaximum(movie.frame_range[1])
        widget.sb_frame_no.setSingleStep(frame_stride)
        # Time range spin boxes
        # TODO: Handle missing time values
        if not np.all(np.isnan(movie._movie_meta['t_range'])):
            widget.sb_time.setMinimum(movie._movie_meta['t_range'][0]*self.time_multiplier)
            widget.sb_time.setMaximum(movie._movie_meta['t_range'][1]*self.time_multiplier)
            widget.sb_time.setSingleStep(frame_stride*self.time_multiplier/movie._movie_meta['fps'])
            widget.sb_time.setDecimals(np.ceil(np.log10(movie._movie_meta['fps'])))
        else:
            widget.sb_time.setMinimum(-1)
            widget.sb_time.setMaximum(-1)
            widget.sb_time.setSingleStep(0)
            widget.sb_time.setDecimals(-1)
        # Frame slider range
        sldr_frame = self.movie_widget.sldr_frame
        sldr_frame.setMinimum(movie.frame_range[0])
        sldr_frame.setMaximum(movie.frame_range[1])
        sldr_frame.setSingleStep(movie._frame_range_info_user['stride'])

        self.set_play_speed()

        widget.btn_load_meta.setStyleSheet('QPushButton {color: black;}')
        pass

    def set_frame(self, force_update=False, **kwargs):
        widget = self.movie_widget
        movie = self.movie
        if len(kwargs) == 0:
            kwargs['n'] = movie.current_frame
        n = movie.lookup('n', _raise_on_missing=False,  **kwargs)
        t = movie.lookup('t', _raise_on_missing=False, **kwargs)
        enhance = widget.cb_apply_enhancements.isChecked()

        logger.debug('Setting frame to n={}, t={}; {}'.format(n, t, kwargs))
        if n == movie.current_frame and (not force_update):
            logger.debug('Frame already set to n={}. Returning...'.format(n))
            return
        if n is None:
            widget.sb_time.setStyleSheet("color: red")
            return
        else:
            widget.sb_time.setStyleSheet("color: black")
        if (n > movie.frame_range[1]):
            n = movie.frame_range[1]
            if self.playing:
                self.play_pause()
        elif (n < movie.frame_range[0]):
            n = movie.frame_range[0]
            if self.playing:
                self.play_pause()
        elif (n not in movie.frame_numbers):
            # TODO: Switch to try except for speed?
            n_close = int(replace_in(n, movie.frame_numbers, tol=movie.frame_range[1]))
            logger.warning('Frame n={} is not a valid frame number. This needs fixing! Setting to n={}'.format(
                    n, n_close))
            n = n_close

        movie.current_frame = n
        widget.sb_frame_no.setValue(n)
        widget.sldr_frame.setValue(n)
        if t is not None:
            if not np.isnan(t):
                widget.sb_time.setValue(t*self.time_multiplier)
                widget.sb_time.setEnabled(True)
            else:
                widget.sb_time.setEnabled(False)
        frame = self.movie(n=n, raw=(not enhance), keep_raw=True)
        # frame[:] = frame[:]**1.1
        frame.plot(ax=widget.frame_canvas.canvas.ax, show=False, annotate=widget.annotate_frame)
        # widget.frame_image.set_data(self.movie[n].data.T)
        widget.frame_canvas.canvas.draw()
        logger.debug('Frame changed to n={}, t={}; {}'.format(n, t, kwargs))

    def update_enhancement(self, enhancement):
        """Update movie frame processing enhancements"""
        movie = self.movie
        widget = self.movie_widget
        def _update_enhencement(apply, toggle=True, settings=None):
            logger.debug('Modifying enhancements: {}   (apply={})'.format(movie.settings['enhancements'].value, apply))
            if toggle:
                enhancements_new = movie.enhancements
                if enhancement in movie.enhancements:
                    enhancements_new.pop(enhancements_new.index(enhancement))
                    # assert not apply
                else:
                    enhancements_new.append(enhancement)
                    # assert apply
                movie.set_enhancements(enhancements_new)
            if settings is not None:
                raise NotImplementedError
            logger.debug('Modified enhancements: {}'.format(movie.settings['enhancements'].value))
            # Refresh current frame
            if widget.cb_apply_enhancements.isChecked():
                # movie.enhance(movie.enhancements, frames=movie.current_frame, keep_raw=True)
                self.set_frame(force_update=True)
            widget.gb_enhancements.setTitle('Image enhancements: {}'.format(', '.join(enhancements_new)))
        return _update_enhencement

    def play_pause(self):
        icon = QtGui.QIcon()
        if not self.playing:
            # start playing
            self.play_timer.start()
            icon.addPixmap(QtGui.QPixmap(os.path.join(icon_dir, 'appbar.control.pause.png')),
                           QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.movie_widget.btn_play.setIcon(icon)
            #TODO: toggle icon, disable buttons
        else:
            # pause
            self.play_timer.stop()
            icon.addPixmap(QtGui.QPixmap(os.path.join(icon_dir, 'appbar.control.play.png')),
                           QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.movie_widget.btn_play.setIcon(icon)
        self.playing = not self.playing

    def set_play_speed(self):
        fps = (self.dflt_play_speed * abs(self.movie_widget.sb_speed.value()))
        if fps == 0:
            fps = 0.00001

        self.play_timer.setInterval(1000. / fps)

    def toggle_annotate_frame(self):
        self.movie_widget.annotate_frame = not self.movie_widget.annotate_frame
        logger.debug('Annotate frame: {}'.format(self.movie_widget.annotate_frame))
        self.set_frame(force_update=True)

    def get_fn(self, title='Select file', fn=None):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        if fn is None:
            fn = self._prev_dir
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, fn,
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if fn != '':
            self._prev_dir = os.path.dirname(fn)
        return fn

    def save_figure(self, bbox_inches='tight'):
        """Save (annotated) figure. Not at original image resolution."""
        movie = self.movie
        fn = 'frame_figure_{}_p{}_n{}.png'.format(movie.machine, movie.pulse, movie.current_frame)
        fn = self.get_fn('Save figure', fn)
        if not fn:
            return
        self.frame_plot.fig.savefig(fn, bbox_inches=bbox_inches, transparent=True)

    def save_frame_image(self):
        """Save frame image at original resolution."""
        import imageio
        movie = self.movie
        fn = 'frame_image_{}_p{}_n{}.png'.format(movie.machine, movie.pulse, movie.current_frame)
        fn = self.get_fn('Save frame image', fn)
        if not fn:
            return
        imageio.imwrite(fn, self.movie(n=self.movie.current_frame)[:].T[::-1])

    # ------ PART 1 : MENU BAR --------

    def shut(self):  # File -> Exit
        self.close()

    def about(self):  # Help -> About

        self.AB = AboutBox()
        self.AB.show()

    # ------ PART 2 : GUI --------

    def hide_enhancements_pannel(self):
        widget = self.movie_widget.frm_enhancements
        if self.movie_widget.gb_enhancements.isChecked():
            widget.setVisible(True)
        else:
            widget.setVisible(False)

    def slider_update(self):
        cel = self.horizontalSlider.value()
        self.lcdNumber_2.display(int(cel * 3))



class AboutBox(QtWidgets.QDialog):

    def __init__(self):
        super(AboutBox, self).__init__()
        uic.loadUi('about.ui', self)


class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)


class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


if __name__ == '__main__':
    movie = Movie('29852', camera='SA1.1', machine='MAST', start_frame=10, end_frame=100, name='Movie_gui')

    app = QtWidgets.QApplication(sys.argv)

    app.aboutToQuit.connect(app.deleteLater)  # if using IPython Console
    window = MovieGUI(movie)

    sys.exit(app.exec_())
