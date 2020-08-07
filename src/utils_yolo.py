import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

def rot(phi, P):
    """
    Apply rotation of angle phi to the points in P of dimension (2, nb of points)
    """
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    return np.dot(P, R.T)

def opt_rect(xy, center):
    """
    Return the optimal width, height and rotation angle of the rectangle with minimal area containing a set a points
    xy is the set of points of dimension (2, nb of points)
    center are the coordinates of the center of the rectangle
    """
    xy_c = xy - center
    def fun(X):
        xy_rot = rot(X[0], xy_c)
        ext = np.amax(np.abs(xy_rot), axis=0)
        return ext[0]*ext[1]
    phi0 = [np.pi/4]
    res = minimize(fun, phi0, bounds=[(0, np.pi/2)])
    phi_opt = res.x[0]
    xy_rot = rot(phi_opt, xy_c)
    ext = np.amax(np.abs(xy_rot), axis=0)
    return np.array([2*ext[0], 2*ext[1], phi_opt])

def run_dbscan(trajs, eps=0.04, min_samples=3):
    """
    For each timestamp, run the DBSCAN clustering algorithm on the 2D points coming from filter_trajs
    """
    dbs = []
    for t in range(len(trajs)):
        dbs.append(DBSCAN(eps=eps, min_samples=min_samples).fit(trajs[t]))
    return dbs

def compute_cluster_boxes(trajs, dbs):
    """
    For each timestamp and for each cluster, compute the optimal rectangular box
    """
    box_list = []
    for t in range(len(trajs)):
        X = trajs[t]
        db = dbs[t]
        labels = db.labels_
        unique_labels = set(labels)
        box = np.zeros((len(unique_labels), 5))
        for k in unique_labels:
            if k != -1:
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                box[k, :2] = np.mean(xy, axis=0)
                box[k, 2:] = opt_rect(xy, box[k, :2])
        box_list.append(box)
    return box_list

def compute_grid(box_list, nb_steps=20):
    """
    For each timestamp, compute a square grid
    Each cell of the grid contains either:
         - a six dimensional zero vectors if no cluster has its center point in this cell
         - the biggest cluster such that its center point lies in this cell, defined as a six dimensional vector
           the 1st element is 1, and the five remaining elements are the cluster's box parameters
    """
    T = len(box_list)
    grid = np.zeros((T, nb_steps, nb_steps, 6))
    for t in range(T):
        boxes = box_list[t].copy()
        for box in boxes:
            if (box[0]!=0) or (box[1]!=0):
                x = int(box[0]*nb_steps)
                y = int(box[1]*nb_steps)
                box[0] = box[0]*nb_steps - x
                box[1] = box[1]*nb_steps - y
                box[2] = box[2]*nb_steps
                box[3] = box[3]*nb_steps
                box[4] = 2*box[4]/np.pi
                if grid[t, x, y, 0] == 1:
                    if box[2]*box[3] > grid[t, x, y, 2]*grid[t, x, y, 3]:
                        grid[t, x, y, 1:] = box[:]
                else:
                    grid[t, x, y, 0] = 1
                    grid[t, x, y, 1:] = box[:]
    return grid

def plot_cluster_boxes(X, db, box, ax):
    """
    NOT MAINTAINED
    Plot on ax:
        - 2D points X with colors associated with clusters db
        - corresponding cluster's rectangles box
    """
    labels = db.labels_
    unique_labels = set(labels)
    colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], '+', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
        R = np.sqrt(box[k,2]**2+box[k,3]**2)/2
        theta = box[k,4] + np.arctan(box[k,3]/box[k,2])
        corner = (box[k,0]-R*np.cos(theta), box[k,1]-R*np.sin(theta))
        ax.add_patch(patches.Rectangle(corner, box[k,2], box[k,3], angle=box[k,4]*180/np.pi, linewidth=1, edgecolor=tuple(col), facecolor='none'))
    return ax

def plot_predict_clusters(pred_tensor, true_tensor, t_init=0):
    """
    Plot the predicted cluster's rectangles and the true cluster's rectangles
    """
    pred_box = pred_tensor[t_init, :, :, :]
    true_box = true_tensor[t_init, :, :, :]
    nb_steps = true_box.shape[1]
    eps = 1e-6
    
    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[0].set_title("Predicted")
    ax[1].set_title("True")
    
    for x in range(nb_steps):
        for y in range(nb_steps):
            box0 = pred_box[x, y].copy()
            box1 = true_box[x, y].copy()
            if box0[0] > 0.5:
                box0[1] = (box0[1] + x)/nb_steps
                box0[2] = (box0[2] + y)/nb_steps
                box0[3] = box0[3]/nb_steps
                box0[4] = box0[4]/nb_steps
                box0[5] = box0[5]*np.pi/2
                R = np.sqrt(box0[3]**2+box0[4]**2)/2
                theta = box0[5] + np.arctan(box0[4]/(box0[3]+eps))
                corner = (box0[1]-R*np.cos(theta), box0[2]-R*np.sin(theta))
                ax[0].add_patch(patches.Rectangle(corner, box0[3], box0[4], angle=box0[5]*180/np.pi, linewidth=1, edgecolor='r', facecolor='none'))
            if box1[0] > 0.5:
                box1[1] = (box1[1] + x)/nb_steps
                box1[2] = (box1[2] + y)/nb_steps
                box1[3] = box1[3]/nb_steps
                box1[4] = box1[4]/nb_steps
                box1[5] = box1[5]*np.pi/2
                R = np.sqrt(box1[3]**2+box1[4]**2)/2
                theta = box1[5] + np.arctan(box1[4]/(box1[3]+eps))
                corner = (box1[1]-R*np.cos(theta), box1[2]-R*np.sin(theta))
                ax[1].add_patch(patches.Rectangle(corner, box1[3], box1[4], angle=box1[5]*180/np.pi, linewidth=1, edgecolor='r', facecolor='none'))
        
    axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
    slider = Slider(axe_slider, 'Time', 0, true_tensor.shape[0]-1, valinit=t_init, valstep=1)
                         
    def update_cluster(val):
        ax[0].clear()
        ax[1].clear()
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[0].set_title("Predicted")
        ax[1].set_title("True")
        time = int(slider.val)
        pred_box = pred_tensor[time, :, :, :]
        true_box = true_tensor[time, :, :, :]
        for x in range(nb_steps):
            for y in range(nb_steps):
                box0 = pred_box[x, y].copy()
                box1 = true_box[x, y].copy()
                if box0[0] > 0.5:
                    box0[1] = (box0[1] + x)/nb_steps
                    box0[2] = (box0[2] + y)/nb_steps
                    box0[3] = box0[3]/nb_steps
                    box0[4] = box0[4]/nb_steps
                    box0[5] = box0[5]*np.pi/2
                    R = np.sqrt(box0[3]**2+box0[4]**2)/2
                    theta = box0[5] + np.arctan(box0[4]/(box0[3]+eps))
                    corner = (box0[1]-R*np.cos(theta), box0[2]-R*np.sin(theta))
                    ax[0].add_patch(patches.Rectangle(corner, box0[3], box0[4], angle=box0[5]*180/np.pi, linewidth=1, edgecolor='r', facecolor='none'))
                if box1[0] > 0.5:
                    box1[1] = (box1[1] + x)/nb_steps
                    box1[2] = (box1[2] + y)/nb_steps
                    box1[3] = box1[3]/nb_steps
                    box1[4] = box1[4]/nb_steps
                    box1[5] = box1[5]*np.pi/2
                    R = np.sqrt(box1[3]**2+box1[4]**2)/2
                    theta = box1[5] + np.arctan(box1[4]/(box1[3]+eps))
                    corner = (box1[1]-R*np.cos(theta), box1[2]-R*np.sin(theta))
                    ax[1].add_patch(patches.Rectangle(corner, box1[3], box1[4], angle=box1[5]*180/np.pi, linewidth=1, edgecolor='r', facecolor='none'))
        plt.draw()

    slider.on_changed(update_cluster)

    return fig, ax, slider