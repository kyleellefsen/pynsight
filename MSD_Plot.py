import numpy as np
import itertools
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from flika.utils.misc import save_file_gui


class MSD_Plot(QtWidgets.QWidget):
    def __init__(self, pynsight_pts, microns_per_pixel):
        super(MSD_Plot, self).__init__()
        self.pynsight_pts = pynsight_pts
        self.microns_per_pixel = microns_per_pixel
        self.nTracksLabel = QtWidgets.QLabel('')
        self.plotWidget = MSDWidget(pynsight_pts, self)
        self.trackLengthsGroup = self.makeTrackLengthsGroup()
        self.SLDFilterGroup = self.makeSLDFilterGroup()
        self.ExportGroup = self.makeExportGroup()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.plotWidget)
        self.layout.addWidget(self.nTracksLabel)
        self.layout.addWidget(self.trackLengthsGroup)
        self.layout.addWidget(self.SLDFilterGroup)
        self.layout.addWidget(self.ExportGroup)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle('Flika - Pynsight Plugin - MSD')
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

    def makeSLDFilterGroup(self):
        SLDFilterGroup = QtWidgets.QGroupBox()
        SLDFilterGroup.setTitle('Filter by Mean Single Lag Displacement')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Only include particles with Mean Track SLD between'))
        min_SLD_spinbox = QtWidgets.QDoubleSpinBox()
        max_SLD = self.plotWidget.max_SLD
        min_SLD_spinbox.setRange(0, max_SLD)
        max_SLD_spinbox = QtWidgets.QSpinBox()
        max_SLD_spinbox.setRange(0, max_SLD)
        max_SLD_spinbox.setValue(max_SLD)
        min_SLD_spinbox.valueChanged.connect(self.min_SLD_updated)
        max_SLD_spinbox.valueChanged.connect(self.max_SLD_updated)
        layout.addWidget(min_SLD_spinbox)
        layout.addWidget(QtWidgets.QLabel('and'))
        layout.addWidget(max_SLD_spinbox)
        layout.addWidget(QtWidgets.QLabel('pixels.'))
        layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        SLDFilterGroup.setLayout(layout)
        SLDFilterGroup.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return SLDFilterGroup

    def makeExportGroup(self):
        ExportGroup = QtWidgets.QGroupBox()
        ExportGroup.setTitle('Export Data')
        layout = QtWidgets.QHBoxLayout()
        select_file_button = QtWidgets.QPushButton('Select file name')
        self.fnameTextBox = QtWidgets.QLineEdit('')
        exportButton = QtWidgets.QPushButton('Export Data')
        select_file_button.pressed.connect(self.select_file)
        exportButton.pressed.connect(self.export)
        layout.addWidget(select_file_button)
        layout.addWidget(self.fnameTextBox)
        layout.addWidget(exportButton)
        ExportGroup.setLayout(layout)
        ExportGroup.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        return ExportGroup

    def select_file(self):
        filename = save_file_gui("Select file name")
        self.fnameTextBox.setText(filename)
        print(filename)

    def export(self):
        filename = self.fnameTextBox.text()
        x, y, er = self.plotWidget.calculate_MSD_plot()
        data = np.array([x, y, er]).T
        np.savetxt(filename, data, fmt='%f')

    def min_track_spinbox_updated(self, val):
        self.plotWidget.min_tracklength = val
        self.plotWidget.updateMSD()

    def max_track_spinbox_updated(self, val):
        self.plotWidget.max_tracklength = val
        self.plotWidget.updateMSD()

    def min_SLD_updated(self, val):
        self.plotWidget.min_SLD = val
        self.plotWidget.updateMSD()

    def max_SLD_updated(self, val):
        self.plotWidget.max_SLD = val
        self.plotWidget.updateMSD()




class MSDWidget(pg.PlotWidget):
    def __init__(self, pynsight_pts, parent):
        super(MSDWidget, self).__init__(title='Mean Squared Displacement Per Lag', labels={'left': 'Mean Squared Disance (um^2)', 'bottom': 'Lag Count'})
        self.pynsight_pts = pynsight_pts
        self.parent = parent
        self.min_tracklength = 2
        self.tracks = [Track(t, self.pynsight_pts.txy_pts) for t in self.pynsight_pts.tracks]
        self.total_tracks_with_mult_particles = len([t for t in self.tracks if t.length>1])
        self.max_tracklength = np.max([t.length for t in self.tracks])
        self.min_SLD = 0
        self.max_SLD = int(np.ceil(np.max([t.mean_SLD for t in self.tracks if t.mean_SLD is not None])))
        self.plot_data = None
        self.updateMSD()
        self.plotItem.getViewBox().autoRange()

    def updateMSD(self):
        x, y, er = self.calculate_MSD_plot()
        range = self.plotItem.viewRange()
        self.clear()
        err = pg.ErrorBarItem(x=x, y=y, top=er, bottom=er, beam=0.5)
        self.plot_data = {'er': er, 'x': x, 'y': y}
        self.addItem(err)
        msd = pg.PlotDataItem(x=x, y=y, pen=(0, 255, 0))
        self.addItem(msd)
        self.plotItem.setXRange(min=range[0][0], max=range[0][1], padding=0)
        self.plotItem.setYRange(min=range[1][0], max=range[1][1], padding=0)
        self.plotItem.showGrid(x=True, y=True, alpha=1)

    def calculate_MSD_plot(self):
        selected_tracks = [t for t in self.tracks if self.min_tracklength <= t.length <= self.max_tracklength]
        selected_tracks = [t for t in selected_tracks if self.min_SLD <= t.mean_SLD <= self.max_SLD]
        self.parent.nTracksLabel.setText(' Tracks included: {}/{}'.format(len(selected_tracks), self.total_tracks_with_mult_particles))
        lags = np.arange(self.max_tracklength+1)
        d = [[] for l in lags]
        for track in selected_tracks:
            for SD in track.SDs:
                if SD[1] <= self.max_tracklength:
                    d[int(SD[1])].append(SD[0])
        counts = np.array([len(d[lag]) for lag in lags])
        means = np.array([np.mean(d[lag]) if len(d[lag])>0 else 0 for lag in lags], dtype=np.float)
        std_errs = np.array([np.std(d[lag]) / np.sqrt(counts[lag]) if len(d[lag])>0 else 0 for lag in lags], dtype=np.float)

        # Convert from pixels^2 to microns^2
        means *= self.parent.microns_per_pixel**2
        std_errs *= self.parent.microns_per_pixel ** 2

        return lags, means, std_errs


class Track:
    def __init__(self, track_pts_idx, all_pts):
        self.track_pts_idx = track_pts_idx
        self.length = len(self.track_pts_idx)
        self.all_pts = all_pts
        self.SDs = self.calc_SDs()# Squared displacements
        self.time_averaged_MSD = self.calc_time_averaged_MSDs()
        if len(track_pts_idx) <= 1:
            self.mean_SLD = None
        else:
            pts = self.all_pts[self.track_pts_idx]
            d_pts = pts[1:, :] - pts[:-1, :]
            SLDs = np.sum(d_pts[:, 1:]**2,1) / d_pts[:, 0]
            self.mean_SLD = np.mean(SLDs) # Mean Single Lag Displacement.

    def calc_SDs(self):
        SDs = []
        for pt1, pt2 in itertools.combinations(self.track_pts_idx, 2): #loops through all pairs
            dt, dx, dy = self.all_pts[pt2] - self.all_pts[pt1]
            dd2 = dx**2 + dy**2
            SDs.append([dd2, dt])
        SDs = np.array(SDs)
        return SDs

    def calc_time_averaged_MSDs(self):
        if len(self.SDs) == 0:
            return np.array([0])
        lags = np.arange(1+int(np.max(self.SDs[:, 1])))
        ddd = [[] for l in lags]
        for SD in self.SDs:
            ddd[int(SD[1])].append(SD[0])
        ddd[0] = [0]
        d_mean = [[] for lag in ddd]
        for i, lag in enumerate(ddd):
            d_mean[i] = np.mean(lag)
        d_mean = np.array(d_mean)
        return d_mean



#
#MSD_Plot(g.m.pynsight)
#tracks = [t for t in g.m.pynsight.MSD_plot.plotWidget.tracks if t.length>1]
#mean_SLDS = np.array([t.mean_SLD for t in tracks])
#lengths = np.array([t.length for t in tracks])
#pg.plot(mean_SLDS, lengths, pen=None, symbol='o')  ## setting pen=None disables line drawing
if __name__ == "__main__":
    from plugins.pynsight import pynsight
    from plugins.pynsight.particle_simulator import particle_simulator
    result = particle_simulator(1, .01, 15, 0, 50, 128, .1, .16)
    blurred = gaussian_blur(1, keepSourceWindow=True)
    binary = threshold(13)
    pynsight.pynsight.gui()

    tracks = pynsight.pynsight.MSD_plot.plotWidget.tracks
    track = tracks[0]
    track.calc_time_averaged_MSDs()

