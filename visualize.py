import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

np_file = 'bmvs_stone/cameras_sphere.npz'
cam_sphere = np.load(np_file)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#set_boundary(ax, 40)

i = 0
scale_mat = cam_sphere.get('scale_mat_{}'.format(i))
while scale_mat is not None:
    world_mat = cam_sphere['world_mat_{}'.format(i)]
    world_mat = world_mat@scale_mat
    K, R, t = get_cam_params(world_mat)
    plot_camera(ax, K, R, t, name = '{}'.format(i), color = 'red')
    plot_bounding_shpere(ax, np.eye(4), ratio = 1)
    i += 1
    scale_mat = cam_sphere.get('scale_mat_{}'.format(i))

plt.show()
