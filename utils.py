import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_cam_params(world_mat):
    out = cv2.decomposeProjectionMatrix(world_mat[:3,:4])
    K = out[0]; R = out[1]; t = out[2]
    K = K/K[2,2]
    t = t[:3] / t[3]
    return K, R, t

def plot_line(ax, vec_start, vec_end, name = None, color = 'blue'):
    ax.plot(xs = [vec_start[0], vec_end[0]],
            ys = [vec_start[1], vec_end[1]],
            zs = [vec_start[2], vec_end[2]],
            color = color)
    if name is not None:
        ax.text(vec_end[0], vec_end[1], vec_end[2], name)
    return ax

def plot_camera(ax, K, R, t, name = 'camera', color = 'red'):
    K = K/30000
    origin = np.array([0, 0, 0, 1])
    x_vec = np.array([K[0,0], 0, 0, 1])
    y_vec = np.array([0, K[0,0], 0, 1])
    z_vec = np.array([0, 0, K[0,0], 1])
    upper_left = np.array([-K[0,-1], -K[1,-1], K[0,0], 1])
    upper_right = np.array([K[0,-1], -K[1,-1], K[0,0], 1])
    lower_left = np.array([-K[0,-1], K[1,-1], K[0,0], 1])
    lower_right = np.array([K[0,-1], K[1,-1], K[0,0], 1])
    # transform
    T = np.eye(4)
    T[:3,:3] = R.T
    T[:3,-1:] = t#-R.T@t 
    origin = origin@T.T 
    x_vec = x_vec@T.T 
    y_vec = y_vec@T.T
    z_vec = z_vec@T.T 
    upper_left = upper_left@T.T 
    upper_right = upper_right@T.T 
    lower_left = lower_left@T.T 
    lower_right = lower_right@T.T
    # plot axis
    ax = plot_line(ax, origin, x_vec, name = 'x')
    ax = plot_line(ax, origin, y_vec, name = 'y')
    ax = plot_line(ax, origin, z_vec, name = 'z')
    # plot cone
    ax = plot_line(ax, origin, upper_left, color = color)
    ax = plot_line(ax, origin, lower_left, color = color)
    ax = plot_line(ax, origin, upper_right, color = color)
    ax = plot_line(ax, origin, lower_right, color = color)
    ax = plot_line(ax, upper_left, upper_right, color = color)
    ax = plot_line(ax, lower_right, upper_right, color = color)
    ax = plot_line(ax, upper_left, lower_left, color = color)
    ax = plot_line(ax, lower_right, lower_left, color = color)
    ax.text(T[0][-1], T[1][-1], t[2][-1], name)
    return 

def plot_bounding_shpere(ax, scale_mat, ratio = 1):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = scale_mat[0,0]*np.cos(u)*np.sin(v)*ratio + scale_mat[0,-1]
    y = scale_mat[1,1]*np.sin(u)*np.sin(v)*ratio + scale_mat[1,-1]
    z = scale_mat[2,2]*np.cos(v)*ratio + scale_mat[2,-1]
    ax.plot_wireframe(x, y, z, color="green")

def set_boundary(ax, range_ = 100):
    x = np.array([-1, -1, -1, -1,  1,  1,  1,  1])*range_
    y = np.array([-1, -1,  1,  1, -1, -1,  1,  1])*range_
    z = np.array([-1,  1,  1, -1, -1,  1,  1, -1])*range_
    ax.scatter(x, y, z)