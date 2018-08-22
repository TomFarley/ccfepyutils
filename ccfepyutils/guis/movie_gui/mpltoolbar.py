#!/usr/bin/env python

""" 
Author: T. Farley
"""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT  # NavigationToolbar2QTAgg
from PyQt5 import QtCore, QtWidgets

# NavigationToolbar2QT_custom
class MplToolbar(NavigationToolbar2QT):
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
        super(MplToolbar, self).__init__(canvas, parent)

        # Remove "edit curves lines..." button
        actions = self.findChildren(QtWidgets.QAction)
        for a in actions:
            if a.text() == 'Customize':
                self.removeAction(a)
                break
        if orientation == 'vertical':
            self.setOrientation(QtCore.Qt.Vertical)
        self.update()