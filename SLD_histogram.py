import numpy as np
import os
from scipy.optimize import curve_fit
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from distutils.version import StrictVersion

import flika
from flika import global_vars as g
from flika.process.BaseProcess import BaseDialog, SliderLabel
from flika.utils.misc import save_file_gui


class SLD_Histogram(QtWidgets.QWidget):
    def __init__(self, pynsight_pts, microns_per_pixel, seconds_per_frame):
        super(SLD_Histogram, self).__init__()
        self.pynsight_pts = pynsight_pts
        self.microns_per_pixel = microns_per_pixel
        self.seconds_per_frame = seconds_per_frame
        self.nTracksLabel = QtWidgets.QLabel('Number of Tracks: {}/{}'.format(0, 0))
        self.trackLengthsGroup = self.makeTrackLengthsGroup()
        self.ExportGroup = self.makeExportGroup()
        self.cdf_group = self.make_cdf_group()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plotWidget = SLD_Histogram_Plot(pynsight_pts, self)
        self.layout.addWidget(self.plotWidget)
        self.layout.addWidget(self.nTracksLabel)
        self.layout.addWidget(self.trackLengthsGroup)
        self.layout.addWidget(self.cdf_group)
        self.layout.addWidget(self.ExportGroup)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle('Flika - Pynsight Plugin - SLD')
        self.show()

    def make_cdf_group(self):
        cdf_group = QtWidgets.QGroupBox()
        cdf_group.setTitle('Cumulative Distribuion Function (CDF)')
        layout = QtWidgets.QHBoxLayout()
        CDF_button = QtWidgets.QPushButton('Plot CDF')
        CDF_button.pressed.connect(self.showCDF)
        layout.addWidget(CDF_button)
        layout.addWidget(QtWidgets.QLabel('Number of lags:'))
        self.CDF_nlags = QtWidgets.QSpinBox()
        max_track_len = np.max([len(t) for t in self.pynsight_pts.tracks])
        self.CDF_nlags.setRange(1, max_track_len)
        layout.addWidget(self.CDF_nlags)
        cdf_group.setLayout(layout)
        return cdf_group

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
        filename = save_file_gui("Select file name")
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
        nlds2, y, _ = self.calcCDF()
        data = np.array([nlds2, y]).T
        filename = self.fnameTextBox.text()
        if filename == '':
            g.alert('You must select a filename below before exporting the Cumulative Distribution Function (CDF) of the mean single lag displacements.')
        else:
            np.savetxt(filename, data, fmt='%f')

    def showCDF(self):
        # self = g.m.pynsight.SLD_histogram
        mean_squared_SLDs, y, nTracks = self.calcCDF()
        lag_size = self.CDF_nlags.value()
        self.cdf = CDF(mean_squared_SLDs, y, nTracks, self.seconds_per_frame, lag_size)

    def calcCDF(self):
        """
        mean_SLDs, nTracks = self.plotWidget.calc_mean_SLDs()
        mean_squared_SLDs = mean_SLDs**2
        mean_squared_SLDs = np.sort(mean_squared_SLDs)
        y = np.arange(len(mean_squared_SLDs), dtype=np.float)
        y /= np.max(y)
        return mean_squared_SLDs, y, nTracks
        """
        nlags = self.CDF_nlags.value()
        all_nLDs, nTracks = self.plotWidget.get_all_nLDs(nlags)
        squared_nLDs = all_nLDs**2
        squared_nLDs = np.sort(squared_nLDs)
        y = np.arange(len(squared_nLDs), dtype=np.float)
        y /= np.max(y)
        return squared_nLDs, y, nTracks



class SLD_Histogram_Plot(pg.PlotWidget):
    def __init__(self, pynsight_pts, parent):
        '''
        debug code:
        pynsight_pts = g.pynsight.points
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

    def get_all_nLDs(self, nlags):
        """
        nLD = n- lag displacement. How far each particle has traveled in n lags.
        
        debug code: 
        self = g.pynsight.SLD_histogram.plotWidget
        """
        tracks = self.pynsight_pts.tracks
        pts = self.pynsight_pts.txy_pts
        all_nLDs = []
        nTracks = 0
        track_lens = []
        for track in tracks:
            if self.min_tracklength <= len(track) <= self.max_tracklength:
                track_lens.append(len(track))
                txy = pts[track, :]
                dt_dx = txy[nlags:, :] - txy[:-nlags, :]
                dt_dx = dt_dx[dt_dx[:, 0] == nlags] # Remove all lags that were greater than 'nlags', in case we are skipping frames
                if len(dt_dx) == 0:
                    continue
                nlds = np.sqrt(dt_dx[:, 1] ** 2 + dt_dx[:, 2] ** 2) # slds in pixels
                nlds *= self.parent.microns_per_pixel # Convert from pixels to microns.
                all_nLDs.extend(nlds)
                nTracks += 1
        print('Total # tracks included: {}'.format(nTracks))
        print('Total # nlds included: {}'.format(len(all_nLDs)))
        print('Average track length: {}'.format(np.mean(track_lens)))
        all_nLDs = np.array(all_nLDs)
        return all_nLDs, nTracks

    def tracks_updated(self):
        mean_SLDs, _ = self.calc_mean_SLDs()
        self.parent.nTracksLabel.setText('Number of Tracks: {}/{}'.format(len(mean_SLDs), self.nTracksTotal))
        if len(mean_SLDs) > 0:
            self.setData(mean_SLDs)

    def setData(self, data=np.array([]), n_bins=None):
        if n_bins is None:
            if self.n_bins is None or self.n_bins == 0:
                n_bins = data.size ** 0.5
            else:
                n_bins = self.n_bins
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


def exp_dec(x, A1, tau):
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x, A1, tau1, tau2):
    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x, A1, A2, tau1, tau2, tau3):
    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)

class CDF(QtWidgets.QWidget):

    def __init__(self, squared_SLDs, y, nTracks, seconds_per_frame, nlags):
        super().__init__()
        self.seconds_per_frame = seconds_per_frame
        self.squared_SLDs = squared_SLDs
        self.y = y
        self.nTracks = nTracks
        self.nlags = nlags
        self.exp_dec_1_curve = None
        self.exp_dec_2_curve = None
        self.exp_dec_3_curve = None
        cdf_plot = pg.PlotWidget()
        cdf_plot.plot(self.squared_SLDs, self.y, pen='w', name='Data')
        if nlags == 1:
            cdf_plot.plotItem.getAxis('bottom').setLabel('Single Lag Displacements squared (um^2)')
        else:
            cdf_plot.plotItem.getAxis('bottom').setLabel('{}-Lag Displacements squared (um^2)'.format(nlags))
        cdf_plot.plotItem.getAxis('left').setLabel('CDF')
        cdf_plot.plotItem.setTitle('{} tracks,    {} squared SLDs'.format(self.nTracks, len(self.squared_SLDs)))
        cdf_plot.plotItem.getAxis('bottom').enableAutoSIPrefix(False)
        self.left_bound_line = cdf_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine), movable=True, bounds=(0, np.max(self.squared_SLDs)))
        self.right_bound_line = cdf_plot.addLine(x=np.max(self.squared_SLDs), pen=pg.mkPen('y', style=QtCore.Qt.DashLine), movable=True, bounds=(0, np.max(self.squared_SLDs)))
        self.cdf_plot = cdf_plot
        self.legend = self.cdf_plot.plotItem.addLegend()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('Cumulative Distribution Function')
        self.l = QtWidgets.QGridLayout(self)
        self.setLayout(self.l)
        self.l.addWidget(self.cdf_plot)
        self.fit_exp_dec_1_button = QtWidgets.QPushButton('Fit with one component exponential')
        self.fit_exp_dec_1_button.pressed.connect(self.fit_exp_dec_1)
        self.fit_exp_dec_2_button = QtWidgets.QPushButton('Fit with two component exponential')
        self.fit_exp_dec_2_button.pressed.connect(self.fit_exp_dec_2)
        self.fit_exp_dec_3_button = QtWidgets.QPushButton('Fit with three component exponential')
        self.fit_exp_dec_3_button.pressed.connect(self.fit_exp_dec_3)
        self.l.addWidget(self.fit_exp_dec_1_button)
        self.l.addWidget(self.fit_exp_dec_2_button)
        self.l.addWidget(self.fit_exp_dec_3_button)
        self.show()

    def fit_exp_dec_1(self):
        if self.exp_dec_1_curve is not None:
            self.cdf_plot.removeItem(self.exp_dec_1_curve)
            self.legend.removeItem(self.exp_dec_1_curve.name())
        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
        xdata = self.squared_SLDs
        ydata = self.y
        x_fit_mask = (left_bound < xdata) * (xdata < right_bound)
        xfit = xdata[x_fit_mask]
        popt, pcov = curve_fit(exp_dec, xfit, ydata[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
        tau_fit = popt[1]
        D_fit = self.tau_to_D(tau_fit)
        print('D = {0:.4g} um^2 s^-1'.format(D_fit))
        yfit = exp_dec(xfit, *popt)
        self.exp_dec_1_curve = self.cdf_plot.plot(xfit, yfit, pen='g', name=' Fit. D = {0:.4g} um^2 s^-1'.format(D_fit))
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_2(self):
        if self.exp_dec_2_curve is not None:
            self.cdf_plot.removeItem(self.exp_dec_2_curve)
            self.legend.removeItem(self.exp_dec_2_curve.name())
        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
        xdata = self.squared_SLDs
        ydata = self.y
        x_fit_mask = (left_bound < xdata) * (xdata < right_bound)
        xfit = xdata[x_fit_mask]
        popt, pcov = curve_fit(exp_dec_2, xfit, ydata[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))
        A1 = popt[0]
        A2 = -1 - A1
        tau1_fit = popt[1]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[2]
        D2_fit = self.tau_to_D(tau2_fit)
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2. A1={2:.2g} A2={3:.2g}'.format(D1_fit, D2_fit, A1, A2)
        print(msg)
        yfit = exp_dec_2(xfit, *popt)
        self.exp_dec_2_curve = self.cdf_plot.plot(xfit, yfit, pen='r', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_3(self):
        if self.exp_dec_3_curve is not None:
            self.cdf_plot.removeItem(self.exp_dec_3_curve)
            self.legend.removeItem(self.exp_dec_3_curve.name())
        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
        xdata = self.squared_SLDs
        ydata = self.y
        x_fit_mask = (left_bound < xdata) * (xdata < right_bound)
        xfit = xdata[x_fit_mask]
        popt, pcov = curve_fit(exp_dec_3, xfit, ydata[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))
        A1 = popt[0]
        A2 = popt[1]
        A3 = -1 - A1 - A2
        tau1_fit = popt[2]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[3]
        D2_fit = self.tau_to_D(tau2_fit)
        tau3_fit = popt[4]
        D3_fit = self.tau_to_D(tau3_fit)
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2, D3 = {2:.4g} um2/2. A1={3:.2g} A2={4:.2g}, A3={5:.2g}'.format(D1_fit, D2_fit, D3_fit, A1, A2, A3)
        print(msg)
        yfit = exp_dec_3(xfit, *popt)
        self.exp_dec_3_curve = self.cdf_plot.plot(xfit, yfit, pen='y', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def tau_to_D(self, tau):
        """ 
        tau = 4Dt
        tau is decay constant of exponential fit
        D is diffusion coefficient
        t is duration of one lag (exposure time) in seconds
        """
        t = self.seconds_per_frame * self.nlags
        D = tau / (4 * t)
        return D




#SLD_Histogram(g.m.pynsight.points)



#y,x =np.histogram(track_lens, bins = np.arange(np.max(track_lens)))
#x=x[:-1]
#pg.plot(x,y)