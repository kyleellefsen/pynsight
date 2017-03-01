import numpy as np
import os
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from process.BaseProcess import BaseDialog, SliderLabel
import global_vars as g


class SLD_Histogram(QtWidgets.QWidget):
    def __init__(self, pynsight_pts, microns_per_pixel):
        super(SLD_Histogram, self).__init__()
        self.pynsight_pts = pynsight_pts
        self.microns_per_pixel = microns_per_pixel
        self.nTracksLabel = QtWidgets.QLabel('Number of Tracks: {}/{}'.format(0, 0))
        self.trackLengthsGroup = self.makeTrackLengthsGroup()
        self.ExportGroup = self.makeExportGroup()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plotWidget = SLD_Histogram_Plot(pynsight_pts, self)
        self.layout.addWidget(self.plotWidget)
        self.layout.addWidget(self.nTracksLabel)
        self.layout.addWidget(self.trackLengthsGroup)
        self.CDF_button = QtWidgets.QPushButton('Plot CDF')
        self.CDF_button.pressed.connect(self.showCDF)
        self.layout.addWidget(self.CDF_button)

        self.layout.addWidget(self.ExportGroup)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle('Flika - Pynsight Plugin - SLD')
        self.setWindowIcon(QtGui.QIcon('images/favicon.png'))
        self.show()

    def makeTrackLengthsGroup(self):
        trackLengthsGroup = QtWidgets.QGroupBox()
        trackLengthsGroup.setTitle('Track Lengths')
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(QtWidgets.QLabel('Only include particles with track lengths between'))
        min_track_spinbox = QtWidgets.QSpinBox()
        max_track_len = np.max([len(t) for t in self.pynsight_pts.tracks])
        min_track_spinbox.setRange(2, max_track_len)
        min_track_spinbox.valueChanged.connect(self.min_track_spinbox_updated)
        max_track_spinbox = QtWidgets.QSpinBox()
        max_track_spinbox.setRange(2, max_track_len)
        max_track_spinbox.setValue(max_track_len)
        max_track_spinbox.valueChanged.connect(self.max_track_spinbox_updated)
        layout.addWidget(min_track_spinbox)
        layout.addWidget(QtWidgets.QLabel('and'))
        layout.addWidget(max_track_spinbox)
        layout.addWidget(QtWidgets.QLabel('frames long (inclusive).'))
        layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        trackLengthsGroup.setLayout(layout)
        trackLengthsGroup.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return trackLengthsGroup

    def makeExportGroup(self):
        ExportGroup = QtWidgets.QGroupBox()
        ExportGroup.setTitle('Export Data')
        layout = QtWidgets.QHBoxLayout()
        select_file_button = QtWidgets.QPushButton('Select file name')
        self.fnameTextBox = QtWidgets.QLineEdit('')
        exportButton = QtWidgets.QPushButton('Export mean SLDs')
        select_file_button.pressed.connect(self.select_file)
        exportButton.pressed.connect(self.export)
        CDF_export_button = QtWidgets.QPushButton('Export CDF')
        CDF_export_button.pressed.connect(self.exportCDF)


        layout.addWidget(select_file_button)
        layout.addWidget(self.fnameTextBox)
        layout.addWidget(exportButton)
        layout.addWidget(CDF_export_button)
        ExportGroup.setLayout(layout)
        ExportGroup.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        return ExportGroup

    def select_file(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Select File Name')
        self.fnameTextBox.setText(filename)
        print(filename)

    def export(self):
        filename = self.fnameTextBox.text()
        if filename == '':
            g.alert('You must select a filename below before exporting the mean single lag displacements.')
        else:
            np.savetxt(filename, self.plotWidget.plot_data, fmt='%f')

    def min_track_spinbox_updated(self, val):
        self.plotWidget.min_tracklength = val
        self.plotWidget.tracks_updated()

    def max_track_spinbox_updated(self, val):
        self.plotWidget.max_tracklength = val
        self.plotWidget.tracks_updated()

    def exportCDF(self):
        slds2, y, _ = self.calcCDF()
        data = np.array([slds2,y]).T
        filename = self.fnameTextBox.text()
        if filename == '':
            g.alert('You must select a filename below before exporting the Cumulative Distribution Function (CDF) of the mean single lag displacements.')
        else:
            np.savetxt(filename, data, fmt='%f')

    def showCDF(self):
        # self = g.m.pynsight.SLD_histogram
        mean_squared_SLDs, y, nTracks = self.calcCDF()
        p = pg.plot(mean_squared_SLDs, y, title = 'Cumulative Distribution Function')
        p.plotItem.getAxis('bottom').setLabel('Single Lag Displacements squared (um^2)')
        p.plotItem.getAxis('left').setLabel('CDF')
        p.plotItem.setTitle('{} tracks,    {} mean squared SLDs'.format(nTracks, len(mean_squared_SLDs)))
        print("WHY IS THE nTracks VARIABLE ALWAYS THE SAME AS THE mean_squared_SLDs VARIABLE???")
        p.plotItem.getAxis('bottom').enableAutoSIPrefix(False)

    def calcCDF(self):
        mean_SLDs, nTracks = self.plotWidget.calc_mean_SLDs()
        mean_squared_SLDs = mean_SLDs**2
        mean_squared_SLDs = np.sort(mean_squared_SLDs)
        y = np.arange(len(mean_squared_SLDs), dtype=np.float)
        y /= np.max(y)
        return mean_squared_SLDs, y, nTracks


class SLD_Histogram_Plot(pg.PlotWidget):
    def __init__(self, pynsight_pts, parent):
        '''
        debug code:
        pynsight_pts = g.m.pynsight.points
        self = SLD_Histogram(pynsight_pts)
        '''
        super(SLD_Histogram_Plot, self).__init__(title='Mean Single Lag Distance Histogram',
                                                 labels={'left': 'Count', 'bottom': 'Mean SLD Per Track (microns)'})
        self.parent = parent
        self.pynsight_pts = pynsight_pts
        self.nTracksTotal = len([t for t in self.pynsight_pts.tracks if len(t)>=2])
        self.plotitem = pg.PlotDataItem()
        self.addItem(self.plotitem)
        self.bins = np.zeros(1)
        self.n_bins = 0
        self.minimum = 0.
        self.maximum = 0.
        self.min_tracklength = 2
        self.max_tracklength = np.max([len(t) for t in self.pynsight_pts.tracks])
        self.plot_data = []
        self.bd = None
        self.tracks_updated()

    def calc_mean_SLDs(self):
        tracks = self.pynsight_pts.tracks
        pts = self.pynsight_pts.txy_pts
        mean_SLDs = []
        nTracks = 0
        track_lens = []
        for track in tracks:
            if self.min_tracklength <= len(track) <= self.max_tracklength:
                track_lens.append(len(track))
                txy = pts[track, :]
                dt_dx = txy[1:, :] - txy[:-1, :]
                dt_dx = dt_dx[dt_dx[:, 0] == 1] # Remove all lags that were greater than one
                if len(dt_dx) == 0:
                    continue
                slds = np.sqrt(dt_dx[:, 1] ** 2 + dt_dx[:, 2] ** 2) # slds in pixels
                mean_sld = np.mean(slds)
                assert not np.isnan(mean_sld)
                mean_sld *= self.parent.microns_per_pixel # Convert from pixels to microns.
                mean_SLDs.append(mean_sld)
                nTracks += 1
        print('Total # tracks included: {}'.format(nTracks))
        print('Total # slds2 included: {}'.format(len(mean_SLDs)))
        print('Average track length: {}'.format(np.mean(track_lens)))
        mean_SLDs = np.array(mean_SLDs)
        return mean_SLDs, nTracks

    def tracks_updated(self):
        mean_SLDs, _ = self.calc_mean_SLDs()
        self.parent.nTracksLabel.setText('Number of Tracks: {}/{}'.format(len(mean_SLDs), self.nTracksTotal))
        if len(mean_SLDs) > 0:
            self.setData(mean_SLDs)

    def setData(self, data=np.array([]), n_bins=None):
        if n_bins is None:
            n_bins = data.size ** 0.5
        if len(data) == 0:
            data = self.plot_data

        minimum = 0
        maximum = np.max(data)

        # if this is the first histogram plotted, initialize settings
        self.minimum = minimum
        self.maximum = maximum
        self.n_bins = n_bins
        self.bins = np.linspace(self.minimum, self.maximum, int(self.n_bins + 1))

        # re-plot the other histograms with this new binning if needed
        re_hist = False
        if minimum < self.minimum:
            self.minimum = minimum
            re_hist = True
        if maximum > self.maximum:
            self.maximum = maximum
            re_hist = True
        if n_bins > self.n_bins:
            self.n_bins = n_bins
            re_hist = True

        if re_hist:
            self.reset()

        self._plot_histogram(data)

    def preview(self, n_bins=None):
        if n_bins is None:
            n_bins = len(self.plotitem.getData()[1])

        bins = np.linspace(np.min(self.plot_data), np.max(self.plot_data), int(n_bins + 1))
        y, x = np.histogram(self.plot_data, bins=bins)
        self.plotitem.setData(x=x, y=y, stepMode=True, fillLevel=0)

    def _plot_histogram(self, data):

        # Copy self.bins, otherwise it is returned as x, which we can accidentally modify
        # by x *= -1, leaving self.bins modified.
        bins = self.bins.copy()
        y, x = np.histogram(data, bins=bins)

        self.plotitem.setData(x=x, y=y, stepMode=True, fillLevel=0)

        self.plot_data = data

    def reset(self):
        self.bins = np.linspace(self.minimum, self.maximum, self.n_bins + 1)
        bins = self.bins.copy()
        y, x = np.histogram(self.plot_data, bins=bins)

        self.plotitem.setData(x, y)

    def mouseDoubleClickEvent(self, ev):
        ev.accept()
        super(SLD_Histogram_Plot, self).mouseDoubleClickEvent(ev)
        if len(self.plot_data) > 0:
            self.edit_histogram_gui()

    def edit_histogram_gui(self):
        items = []
        binSlider = SliderLabel(0)
        binSlider.setMinimum(1)
        binSlider.setMaximum(len(self.plot_data))
        binSlider.setValue(self.n_bins)
        items.append({'name': 'Bins', 'string': 'Bins', 'object': binSlider})
        bd = BaseDialog(items, "Histogram options", 'Set the number of bins in the histogram')
        bd.accepted.connect(lambda: self.setData(n_bins=binSlider.value()))
        bd.rejected.connect(self.reset)
        bd.changeSignal.connect(lambda: self.preview(n_bins=binSlider.value()))
        bd.setMinimumWidth(400)
        bd.show()
        self.bd = bd

#SLD_Histogram(g.m.pynsight.points)



#y,x =np.histogram(track_lens, bins = np.arange(np.max(track_lens)))
#x=x[:-1]
#pg.plot(x,y)