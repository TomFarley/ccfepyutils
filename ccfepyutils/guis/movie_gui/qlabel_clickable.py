#!/usr/bin/env python
from PyQt5 import QtCore, QtWidgets


class QLabelClickable(QtWidgets.QLabel):
    clicked=QtCore.pyqtSignal()
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)

    def mousePressEvent(self, ev):
        self.clicked.emit()