# -*- coding: utf-8 -*-
"""
Created on Wed June 15 2016
@author: Kyle Ellefsen
"""


if __name__ == '__main__':
    import os, sys; flika_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'GitHub', 'flika'); sys.path.append(flika_dir); from flika import *; start_flika()
    from plugins.pynsight.pynsight import *
    from plugins.pynsight.particle_simulator import simulate_particles
    A, true_pts = simulate_particles()
    data_window = Window(A)
    data_window.setName('Data Window (F/F0)')
    blur_window = gaussian_blur(2, norm_edges=True, keepSourceWindow=True)
    blur_window.setName('Blurred Window')
    binary_window = threshold(.7, keepSourceWindow=True)
    binary_window.setName('Binary Window')
    txy_pts = get_points(g.m.currentWindow.image)
    np.savetxt(r'C:\Users\kyle\Desktop\simulated.txt', txy_pts)
    refined_pts = refine_pts(txy_pts, blur_window.image)
    refined_pts_txy = np.vstack((refined_pts[:, 0], refined_pts[:, 3], refined_pts[:, 4])).T
    p = Points(refined_pts_txy)
    p.link_pts()
    tracks = p.tracks

    #filename = r'C:\Users\kyle\Desktop\test_flika.bin'
    #write_insight_bin(filename, refined_pts, tracks)

    sys.exit(g.app.exec_())  # This is required to run outside of Spyder or PyCharm