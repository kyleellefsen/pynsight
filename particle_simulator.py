# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 16:19:30 2016

@author: kyle
"""
from numpy import random
import numpy as np
import sys, os
from qtpy import QtWidgets, QtGui, QtCore
from flika.process.BaseProcess import BaseProcess_noPriorWindow, WindowSelector, SliderLabel, CheckBox
from flika import global_vars as g
from .gaussianFitting import gaussian
from .pynsight import Points


def generate_model_particle(x_remander,y_remander, amp):
    x = np.arange(7)
    y = np.arange(7)
    xorigin = 3+x_remander
    yorigin = 3+y_remander
    sigma = 1
    model_particle_wo_noise = gaussian(x[:,None], y[None,:], xorigin, yorigin, sigma, amp)
    noise = np.random.normal(0, 1, model_particle_wo_noise.shape) * model_particle_wo_noise
    model_particle = model_particle_wo_noise + noise
    return model_particle


def addParticle(A, t, x, y, amp, mx, my):
    assert isinstance(A, np.ndarray)
    x_int = int(np.round(x))
    y_int = int(np.round(y))
    x_remander = x-x_int
    y_remander = y-y_int
    model_particle = generate_model_particle(x_remander,y_remander, amp)
    dx, dy = model_particle.shape
    assert dx % 2 == 1
    tt = np.array([t], dtype=np.int)
    dx = (dx-1)/2
    dy = (dy-1)/2
    yy = np.arange(y_int-dy,y_int+dy+1, dtype=np.int)
    xx = np.arange(x_int-dx,x_int+dx+1, dtype=np.int)
    if np.min(yy)<0 or np.min(xx)<0 or np.max(yy)>=my or np.max(xx)>=mx:
        return A
    A[np.ix_(tt, xx, yy)] = A[np.ix_(tt, xx, yy)] + model_particle
    return A


def get_accuracy(true_pts, det_pts):
    """
    Frames are independent, so I can group true points and detected points by frame and search that much smaller subset
    """
    true_pos_dist_cutoff=2
    true_pts_by_frame=[]
    det_pts_by_frame = []
    max_frame=int(np.max([np.max(txy_pts[:,0]), np.max(true_pts[:,0])]))
    for frame in np.arange(max_frame+1):
        true_pts_by_frame.append(np.where(true_pts[:,0]==frame)[0])
        det_pts_by_frame.append(np.where(det_pts[:,0]==frame)[0])
    
    linked_pts=[]  #list of lists where each entry is [true_pt_idx, det_pt_idx, distance]
    false_pos=[]   #list of detected point indicies which have no corresponding true_pt entry
    false_neg=[]   #list of true point indicies which have no corresponding det_pt entry
    for frame in np.arange(max_frame+1):
        tru=true_pts_by_frame[frame]
        det=det_pts_by_frame[frame]
        D=np.zeros((len(tru),len(det)))
        for i in np.arange(len(tru)):
            for j in np.arange(len(det)):
                D[i,j]=np.sqrt(np.sum((true_pts[tru[i]][1:]-det_pts[det[j]][1:])**2))
                
        used_det=[]
        for i in np.arange(len(tru)):
            if len(det)>0 and np.min(D[i,:])<true_pos_dist_cutoff:
                j=np.argmin(D[i,:])
                linked_pts.append([tru[i],det[j],D[i,j]])
                used_det.append(j)
                D[:,j]=true_pos_dist_cutoff #this prevents double counting, even though it might not be optimal
            else:
                false_neg.append(tru[i])
        remaining_det=np.array(list(set(range(len(det))).difference(set(used_det))))
        if len(remaining_det)>0:
            false_pos.extend(det[remaining_det])
    linked_pts=np.array(linked_pts)
    return linked_pts, false_pos, false_neg




class Particle_simulator(BaseProcess_noPriorWindow):
    """ particle_simulator()
    This function simulates particles undergoing 2D diffusion
    """
    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        s = dict()
        s['rate_of_appearance'] = .1
        s['rate_of_disappearance'] = .01
        s['amplitude'] = 50
        s['mx'] = 256
        s['mt'] = 500
        s['D'] = 1.0  # Delta determines the "speed" of the Brownian motion.  The random variable of the position at time t, X(t), has a normal distribution whose mean is the position at time t=0 and whose variance is delta**2*t.]
        s['frame_duration'] = 1  # seconds
        s['microns_per_pixel'] = 0.16
        return s

    def gui(self):
        self.gui_reset()
        rate_of_appearance = SliderLabel(3)
        rate_of_appearance.setRange(0,10)
        rate_of_disappearance = SliderLabel(3)
        rate_of_disappearance.setRange(0, 10)
        amplitude = SliderLabel(1)
        amplitude.setRange(0, 100)
        mx = SliderLabel(0)
        mx.setRange(0, 512)
        mt = SliderLabel(0)
        mt.setRange(0, 10000)
        D = SliderLabel(2)
        D.setRange(0, 10)
        frame_duration = SliderLabel(2)
        frame_duration.setRange(0, 1)
        microns_per_pixel = SliderLabel(2)
        microns_per_pixel.setRange(0, 1)

        self.items.append({'name': 'rate_of_appearance',    'string': 'Rate of Appearance',     'object': rate_of_appearance})
        self.items.append({'name': 'rate_of_disappearance', 'string': 'Rate of Disappearance',  'object': rate_of_disappearance})
        self.items.append({'name': 'amplitude',             'string': 'Amplitude',              'object': amplitude})
        self.items.append({'name': 'mx',                    'string': 'Image Width and Height', 'object': mx})
        self.items.append({'name': 'mt',                    'string': 'Number of Frames',       'object': mt})
        self.items.append({'name': 'D',                     'string': 'Diffusion Coefficient (um^2/s)',  'object': D})
        self.items.append({'name': 'frame_duration',        'string': 'Frame Duration (s)',     'object': frame_duration})
        self.items.append({'name': 'microns_per_pixel',     'string': 'microns per pixel',      'object': microns_per_pixel})

        super().gui()

    def __call__(self, rate_of_appearance, rate_of_disappearance, amplitude, D, mt, mx, frame_duration, microns_per_pixel):
        self.start()
        udc = {'rate_of_appearance': rate_of_appearance,
               'rate_of_disappearance': rate_of_disappearance,
               'amplitude': amplitude,
               'D': D,
               'mt': mt,
               'mx': mx,
               'frame_duration': frame_duration,
               'microns_per_pixel': microns_per_pixel}
        simulated_particles = Simulated_particles(udc)
        self.newtif = simulated_particles.image
        self.newname = 'Simulated Particles'
        particle_window = self.end()
        simulated_particles.setupUI(particle_window)
        g.simulated_particles = simulated_particles
        g.sim = simulated_particles
        return simulated_particles
        #return A, true_pts

particle_simulator = Particle_simulator()


class Simulated_particles(QtWidgets.QWidget):
    def __init__(self, udc):
        super().__init__()
        self.udc = udc
        image, points = self.simulate_particles()
        self.image = image
        self.points = points
        self.particle_window = None

    def simulate_particles(self):
        rate_of_appearance = self.udc['rate_of_appearance']
        rate_of_disappearance = self.udc['rate_of_disappearance']
        amplitude = self.udc['amplitude']
        frame_duration = self.udc['frame_duration']
        microns_per_pixel = self.udc['microns_per_pixel']
        D = self.udc['D']
        mt = self.udc['mt']
        mx = self.udc['mx']
        my = mx

        a = random.poisson(rate_of_appearance, mt)
        creation_times = []
        while len(np.where(a)[0])>0:
            creation_times.extend(np.where(a)[0])
            a[np.where(a)[0]]-=1
        creation_times = np.array(creation_times)
        creation_times.sort()
        nParticles = len(creation_times)
        lifetimes = random.exponential(1/rate_of_disappearance, nParticles)
        destroy_times = creation_times + lifetimes
        txy_pts = []
        tracks = []
        for i in np.arange(nParticles):
            x = random.random() * mx
            y = random.random() * my
            t = creation_times[i]
            txy_pts.append([t, x, y])
            track = [len(txy_pts)-1]
            t += 1
            while t < destroy_times[i] and t < mt:
                x += random.randn() * np.sqrt( 2* D * frame_duration) / microns_per_pixel
                y += random.randn() * np.sqrt( 2* D * frame_duration) / microns_per_pixel
                txy_pts.append([t, x, y])
                track.append(len(txy_pts)-1)
                t+=1
            tracks.append(track)
        txy_pts = np.array(txy_pts)
        points = Points(txy_pts)
        points.tracks = tracks
        points.get_tracks_by_frame()
        A = random.randn(mt, mx, my)
        for pt in txy_pts:
            frame, x, y = pt
            A = addParticle(A, frame, x, y, amplitude, mx, my)

        return A, points

    def setupUI(self, particle_window):
        self.particle_window = particle_window
        self.setWindowTitle('Simulated particles - {}'.format(self.particle_window.name))
        self.l = QtWidgets.QGridLayout(self)
        self.setLayout(self.l)
        self.l.addWidget(self.particle_window)
        self.load_into_pynsight_button = QtWidgets.QPushButton('Load into Pynsight')
        self.load_into_pynsight_button.pressed.connect(self.load_into_pynsight)
        self.l.addWidget(self.load_into_pynsight_button)
        self.show()

    def load_into_pynsight(self):
        g.pynsight.gui()
        g.pynsight.txy_pts = g.sim.points.txy_pts
        g.pynsight.algorithm_gui.showPointsButton.setEnabled(True)
        g.pynsight.skip_refinePoints()
        g.pynsight.points = g.sim.points
        tracks = g.pynsight.points.tracks
        nTracks = len(tracks)
        g.pynsight.algorithm_gui.num_tracks_label.setText(str(nTracks))
        g.pynsight.algorithm_gui.showTracksButton.setEnabled(True)
        g.pynsight.algorithm_gui.analyze_tab_widget.setCurrentIndex(3)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    true_pts=np.array(true_pts)
    np.savetxt(r'C:\Users\kyle\Desktop\true_points.txt',true_pts)
    I=Window(A)
    gaussian_blur(1,keepSourceWindow=True)
    threshold(2,keepSourceWindow=True)

    txy_pts=get_points(g.m.currentWindow.image)
    linked_pts, false_pos, false_neg = get_accuracy(true_pts, txy_pts)
    refined_pts=refine_pts(txy_pts,I.image)
    refined_pts_txy=np.vstack((refined_pts[:,0],refined_pts[:,3], refined_pts[:,4])).T
    linked_pts, false_pos, false_neg = get_accuracy(true_pts, refined_pts_txy)

    refined_pts_txy[:,1:]+=.5
    np.savetxt(r'C:\Users\kyle\Desktop\simulated.txt',refined_pts_txy)
    p=Points(refined_pts_txy)
    p.link_pts()
    tracks=p.tracks
    filename=r'C:\Users\kyle\Desktop\simulated.bin'
    write_insight_bin(filename, refined_pts, tracks)



    fig, ax = plt.subplots()
    bins=np.arange(0,2,.01)
    n, _, patches = ax.hist(linked_pts[:,2], bins=bins, facecolor='blue', alpha=0.3)
    sigma=.8/particle_amp
    r=bins
    P=r/sigma**2*np.exp(-(r**2/(2*sigma**2)))
    ax.plot(bins,P*26)
