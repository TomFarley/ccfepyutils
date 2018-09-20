#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
""" Custom PyQt widgets """

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
import matplotlib
from ccfepyutils import batch_mode
from ccfepyutils.mpl_tools import set_matplotlib_backend
set_matplotlib_backend(batch_mode, non_visual_backend='Agg', visual_backend='Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT  # NavigationToolbar2QTAgg
import numpy as np
import re
from collections import OrderedDict#, deque

class NavigationToolbar2QT_custom(NavigationToolbar2QT):
    def __init__(self, canvas, parent, orientation='horizontal'):
        # create the default toolbar
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            # ('Back', 'Back to  previous view', 'back', 'back'),
            # ('Forward', 'Forward to next view', 'forward', 'forward'),
            # (None, None, None, None),  # spacer
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            # (None, None, None, None),  # spacer
            # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
        # NavigationToolbar2QT.__init__(self, canvas, parent)
        super(NavigationToolbar2QT_custom, self).__init__(canvas, parent)

        # Remove "edit curves lines..." button
        actions = self.findChildren(QtGui.QAction)
        for a in actions:
            if a.text() == 'Customize':
                self.removeAction(a)
                break
        if orientation == 'vertical':
            self.setOrientation(Qt.Vertical)
        self.update()

def TeX_label(string, fontsize=16, width=None, hight=20, alpha=1, color='w'):
    # Add FigureCanvasQTAgg widget to form
    TeXfig = plt.Figure()
    TeXfig.patch.set_alpha(alpha)
    TeXfig.patch.set_facecolor(color)

    TeXcanvas = FigureCanvasQTAgg(TeXfig)
    # print('string: "{}"'.format(string))
    non_tex_string = re.sub(r'\\\w{1,6}[=$()_ ]', '~~', string)  # replace symbols: "\symbol "
    non_tex_string = re.sub('{.*}', '~~', non_tex_string)
    non_tex_string = re.sub('[$^_]', '', non_tex_string)
    if width==None: width = fontsize/2 * (len(non_tex_string)+1)
    # print('len({})={}, width={}'.format(non_tex_string, len(non_tex_string), width))
    TeXcanvas.setFixedWidth(width)
    TeXcanvas.setFixedHeight(hight)

    # Clear figure
    TeXfig.clear()

    # Set figure title
    TeXfig.suptitle(string, x=0.0, y=0.5, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
    TeXcanvas.draw()
    return TeXcanvas

class QSpinSlider(QtGui.QFrame):
    """ Combined label, spinbox and slider widget for entering floating point values
    """
    def __init__(self, label=None, startval=0.0, lims=(0,100,10), unit=None, decimal=True, ndp=None):
        """
        :param label:
        :param startval:
        :param lims: (start value, end value, scale for slider)
        :param unit:
        :param decimal:
        :return:
        """

        super(QSpinSlider, self).__init__()
        self.setMinimumHeight(30)  # 60

        self.lims = lims
        # self.scale = lims[2]/float(lims[1]-lims[0])

        self.layout = QtGui.QHBoxLayout()

        if label is not None:
            if '$' in label:
                label =  TeX_label(label)#, color=self.clColor, alpha=1)
            else:
                label =  QtGui.QLabel(label)
            self.layout.addWidget(label, 1)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.TickPosition(QtGui.QSlider.TicksBelow)
        self.slider.setMinimum(lims[0]*lims[2])
        self.slider.setMaximum(lims[1]*lims[2])
        self.slider.setValue(startval*lims[2])
        # self.slider.setMinimumHeight(500)

        self.layout.addWidget(self.slider, 3)

        if decimal:
            ndp = ndp if ndp is not None else np.round(np.log10(lims[2]))
            self.spinbox = QtGui.QDoubleSpinBox()
            self.spinbox.setDecimals(ndp)
            self.spinbox.setSingleStep(10**-(ndp-1))
        else:
            self.spinbox = QtGui.QSpinBox()
        self.spinbox.setValue(startval)
        self.spinbox.setMinimum(lims[0])
        self.spinbox.setMaximum(lims[1])

        self.layout.addWidget(self.spinbox, 1)

        if unit is not None:
            if '$' not in unit:
                self.spinbox.setSuffix(unit)
            else:
                unit_label = TeX_label(unit)#, color=self.clColor, alpha=1)
                self.layout.addWidget(unit_label, 1)

        self.slider.valueChanged[int].connect(self._update_spinbox)
        self.spinbox.valueChanged.connect(self._update_slider)

        self.layout.setAlignment(Qt.AlignTop)

        self.layout.setMargin(0)

        self.setLayout(self.layout)


    def _update_spinbox(self, value):
        # print('value_slider:', value)
        value = float(value)/self.lims[2]
        self.spinbox.setValue(value)
        # print('value_spin:', value)

    def _update_slider(self, value):
        # print('value_spin:', value)
        value = float(value)*self.lims[2]
        self.slider.setValue(value)
        # print('value_slider:', value)

    def setValue(self, value):
        self.spinbox.setValue(value)
        # self.slider.setValue(value*self.lims[2])

    def connect(self, func):
        self.spinbox.valueChanged.connect(func)
        pass

    def setFixedWidth(self):
        pass

class SettingsWidget(QtGui.QWidget):
    """ Popup window used to edit frame history settings
    """
    def __init__(self, dic, settings, title='Settings', popup=False, call=[]):
        QtGui.QWidget.__init__(self)

        # self.cw = QtGui.QWidget(self)
        # self.setCentralWidget(self.cw)

        # self.setGeometry(QtCore.QRect(100, 100, 400, 200))
        self.setWindowTitle(title)

        grid = QtGui.QGridLayout()
        if not popup: grid.setMargin(0)
        self.setLayout(grid)

        for i, (key, setting) in enumerate(settings.items()):
            if 'kwargs' not in setting:
                setting['kwargs'] = {}

            widget = setting['widget']()
            # hbox = QtGui.QHBoxLayout()
            # print('Adding widget to row {}'.format(i))

            if type(widget) == QSpinSlider:
                widget = setting['widget'](label=setting['label'], **setting['kwargs'])
                widget.setValue(dic[key]) # Set value before connecting slots!
                for func in call:
                    widget.connect(func)
                hbox = QtGui.QHBoxLayout()
                hbox.setMargin(0)
                hbox.addWidget(widget)
                grid.addWidget(widget, i*2, 0, i*2, 2)
                grid.addItem(QtGui.QSpacerItem(5,15), i+1, 0, i+1, 2)
                widget.connect(qstring2num(dic, key))
            elif type(widget) == QtGui.QComboBox:
                widget.addItems(setting['items'])
                widget.currentIndexChanged.connect(count2itemText(widget, dic, key))
                for func in call:
                    widget.currentIndexChanged.connect(func)
                if 'startval' in setting.keys():
                    widget.setCurrentIndex(widget.findText(setting['startval']))
                label = QtGui.QLabel(setting['label'])
                grid.addWidget(label, i, 0)
                grid.addWidget(widget, i, 1)
            else:
                widget.setValue(dic[key]) # Set value before connecting slots!
                widget.valueChanged.connect(update_dict(dic, key))
                if 'range' in setting.keys():
                    widget.setRange(setting['range'][0], setting['range'][1])
                for func in call:
                    widget.valueChanged.connect(func)
                label = QtGui.QLabel(setting['label'])
                grid.addWidget(label, i, 0)
                grid.addWidget(widget, i, 1)

        if popup: # add ok button to clear window if to be used as stand alone window
            btn1 = QtGui.QPushButton("OK", self)
            btn1.setGeometry(QtCore.QRect(0, 0, 100, 30))
            self.connect(btn1, QtCore.SIGNAL("clicked()"), self.close)
            grid.addWidget(btn1, 100, 1)  # add at bottom

        # grid.addLayout(QSpinSlider(label=r'$\phi_0 \times \sigma$', unit='\u00B0'), i+2, 0, i+2,2)
        self.setLayout(grid)

    def print_value(self, value):
        print(value)

def update_dict(dic, key):
    def _update_dict(value):
        dic[key] = value
    return _update_dict

def count2itemText(wid, dic, key):  # convert count returned from currentIndexChanged event to string
    def _count2itemText(count):
        update_dict(dic, key)(str(wid.itemText(count)))
    return _count2itemText

def qstring2num(dic, key):
    def _qstring2num(qstring):
        update_dict(dic, key)(float(qstring))
    return _qstring2num


class CustomButton(QtGui.QPushButton):
    ''' Custom button with left right and double click functionality
        Code adapted from: http://stackoverflow.com/questions/19247436/pyqt-mouse-mousebuttondblclick-event
     '''
    left_clicked= QtCore.pyqtSignal(int)
    right_clicked = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QtGui.QPushButton.__init__(self, *args, **kwargs)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(250) # clicks <0.25s apart are grouped
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timeout)
        self.left_click_count = self.right_click_count = 0

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.left_click_count += 1
            if not self.timer.isActive():
                self.timer.start()
        if event.button() == QtCore.Qt.RightButton:
            self.right_click_count += 1
            if not self.timer.isActive():
                self.timer.start()

    def timeout(self):
        if self.left_click_count >= self.right_click_count:
            self.left_clicked.emit(self.left_click_count)
        else:
            self.right_clicked.emit(self.right_click_count)
        self.left_click_count = self.right_click_count = 0

class CustomProgressBar():
    def __itit__(self, range = [0,100], value=0, show=True):
        self.pbar = QtGui.QProgressBar.__init__(self, *args, **kwargs)
        self.hbox = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel('Ready...')
        self.hbox.addWidget(self.label)
        self.hbox.addWidget(self.pbar)

class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        self.button1 = CustomButton("Button 1")
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.button1)
        self.setLayout(hbox)
        self.button1.left_clicked[int].connect(self.left_click)
        self.button1.right_clicked[int].connect(self.right_click)

    def left_click(self, nb):
        if nb == 1: print('Single left click')
        else: print('Double left click')

    def right_click(self, nb):
        if nb == 1: print('Single right click')
        else: print('Double right click')

def debug_trace_QT():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook

    # Or for Qt5
    #from PyQt5.QtCore import pyqtRemoveInputHook

    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()

########################################## Stuff coppied from elzar for Jack to see #######################################

# Handle Mouse click events on main frame canvas
def onClick(event):
    if event.button == 1:  # on double left click set starting point with red +
        if event.dblclick:
            if event.inaxes is self.frameAx:
                self.selectedPixel = (int(event.xdata), int(event.ydata))
                if self.pixelplot:
                    self.pixelplot.set_visible(False)
                self.pixelplot = self.frameAx.scatter(self.selectedPixel[0], self.selectedPixel[1], s=70, c='r',
                                                      marker='+')
                self.frameCanvas.draw()
    elif event.button == 3:  # on double right click set end point with yellow x
        if event.dblclick:
            if event.inaxes is self.frameAx:
                if self.selectedLine is None or len(self.selectedLine) == 2:
                    # Load in first pixel coordinate and draw
                    if self.linePlot is not None:
                        self.linePlot[0].set_visible(False)
                        self.linePlot[1].set_visible(False)
                        self.linePlot[2][0].remove()
                    self.selectedLine = [[int(event.xdata), int(event.ydata)]]
                    self.linePlot = [self.frameAx.scatter(int(event.xdata), int(event.ydata), s=70, c='y', marker='x')]
                    self.frameCanvas.draw()
                elif len(self.selectedLine) == 1:
                    self.selectedLine.append([int(event.xdata), int(event.ydata)])
                    self.linePlot.append(
                        self.frameAx.scatter(int(event.xdata), int(event.ydata), s=70, c='y', marker='x'))
                    self.linePlot.append(self.frameAx.plot([self.selectedLine[0][0], self.selectedLine[1][0]],
                                                           [self.selectedLine[0][1], self.selectedLine[1][1]], '-y',
                                                           lw=2))
                    self.frameCanvas.draw()

#  frameCanvas.mpl_connect('button_press_event', onClick)

#####################################################################################################################

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    # w = MyDialog()
    # w.show()

    bgsubHistory = {'Nbackwards': 3, 'Nforwards': 0, 'skipBackwards':2,'skipForwards':2,
                        'stepBackwards': 1, 'stepForwards': 1}
    settings = OrderedDict( [('Nbackwards', {'label':'Number of frames from future', 'widget':QtGui.QSpinBox}),
                            ('Nforwards',     {'label':'Number of frames from past', 'widget':QtGui.QSpinBox}),
                            ('skipBackwards', {'label':'Number of frames to skip from past', 'widget':QSpinSlider}),
                            ('skipForwards',  {'label':'Number of frames to skip from future', 'widget':QtGui.QSpinBox}),
                            ('stepBackwards', {'label':'Step size in past', 'widget':QtGui.QSpinBox}),
                            ('stepForwards',  {'label':'Step size in future', 'widget':QtGui.QSpinBox})] )
    print(bgsubHistory)
    FrameHistorySettings = SettingsWidget(bgsubHistory, settings, title='Frame History Settings')
    # FrameHistorySettings.setModal(True)
    FrameHistorySettings.show()

    sys.exit(app.exec_())
    print(bgsubHistory)
