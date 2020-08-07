from utils_plots import *
from utils_yolo import *

class Window:
    """
    Store window boundaries for filtering functions
    """
    def __init__(self, _lon, _lat, _alt, lon_, lat_, alt_):
        self._lon = _lon
        self._lat = _lat
        self._alt = _alt
        self.lon_ = lon_
        self.lat_ = lat_
        self.alt_ = alt_

class Dataset:
    """
    Super class to manage data
    """
    
    def __init__(self, directory, file_name, input_type, metric_type):
        """
        input_type:
            1 -> congestion map
            2 -> lon, lat, alt, speed, head, vz
            3 -> lon, lat, alt, speed, head, vz, congestion, mat0, ..., mat8
        metric_type:
            0 -> metric value in index 11, for 'robust convergence' and 'linear dynamical system' (except NEW dataset)
            1 -> metric value in index 10, only for NEW dataset (extended neighborhood) and 'interaction'
        """
        self.path = directory + file_name
        self.input = input_type
        self.metric = metric_type
        self.read()
        
    def read(self):
        """
        Abstract method
        Read the data from files
        """
        raise NotImplementedError()
    
    def smoothing(self):
        """
        Apply a 3x3 gaussian kernel to the y_tensor attribute
        """
        tensor = np.zeros((self.nb_times, self.nb_steps+2, self.nb_steps+2))
        tensor[:,1:-1,1:-1] = self.y_tensor
        W = np.array([[[1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]] for t in range(nbTime)])
        for x in range(nb_steps):
            for y in range(nb_steps):
                self.y_tensor[:, x, y] = np.average(tensor[:,x:x+3,y:y+3], axis=(1,2), weights=W)
        self.max_cong = np.max(self.y_tensor)
        self.y_tensor /= self.max_cong
        
    def norm_coord(self, value, coord):
        """
        Normalize a value corresponding to a given coordinate
        coord = 0 -> longitude
        coord = 1 -> latitude
        coord = 2 -> altitude
        coord = 3 -> ground speed
        coord = 4 -> heading
        coord = 5 -> vertical speed
        coord = 6 -> congestion metric
        coord = 7 -> congestion matrix
        """
        if coord == 0:
            return (value - self.min_lon)/(self.max_lon - self.min_lon)
        elif coord == 1:
            return (value - self.min_lat)/(self.max_lat - self.min_lat)
        elif coord == 2:
            return (value - self.min_alt)/(self.max_alt - self.min_alt)
        elif coord == 3:
            return (value - self.min_spd)/(self.max_spd - self.min_spd)
        elif coord == 4:
            return (value - self.min_head)/(self.max_head - self.min_head)
        elif coord == 5:
            return (value - self.min_vz)/(self.max_vz - self.min_vz)
        elif coord == 6:
            return (value - self.min_cong)/(self.max_cong - self.min_cong)
        elif coord == 7:
            return (value - self.min_mat)/(self.max_mat - self.min_mat)
        
    def filter_trajs(self):
        """
        Rescale the congestion with log_thr
        Keep the a/c above cong_thr
        Keep longitude and latitude only
        """
        trajs = []
        max_val = np.log(1 + self.max_cong/self.log_thr)
        for t in range(self.nb_times):
            trajs.append([])
            for i in range(len(self.raw_tensor[t][0])):
                state = self.raw_tensor[t][0][i]
                congestion = self.raw_tensor[t][1][i]
                log_cong = np.log(1 + congestion/self.log_thr)/max_val
                if log_cong > self.cong_thr:
                    trajs[t].append([self.norm_coord(state[0], 0), self.norm_coord(state[1], 1)])
            if len(trajs[t]) == 0:
                trajs[t].append([0,0])
            trajs[t] = np.array(trajs[t])
        return trajs
    
    def filter_space(self, window):
        """
        Filter self.raw_tensor to keep only the a/c in the window
        Keep longitude, latitude, longitude speed and latitude speed
        """
        new_dict = {}
        max_val = np.log(1 + self.max_cong/self.log_thr)
        for time in range(self.nb_times):
            new_dict[time] = [[[window._lon, window._lat, 0, 0]], [0]]
            for i in range(len(self.raw_tensor[time][0])):
                state = self.raw_tensor[time][0][i]
                congestion = self.raw_tensor[time][1][i]
                if state[2] >= window._alt and state[2] <= window.alt_:
                    new_dict[time][0].append([state[0], state[1], np.sin(state[4]*np.pi/180), np.cos(state[4]*np.pi/180)])
                    val = np.log(1 + congestion/self.log_thr)/max_val
                    new_dict[time][1].append(val)
        self.raw_tensor = np.array([new_dict[time] for time in range(self.nb_times)], dtype='object')
        
    def filter_binary(self):
        """
        Transform the y_tensor attribute into a tensor of binary variables using a percentile of non-zero values as threshold
        """
        thr = np.percentile(self.y_tensor[self.y_tensor!=0].flatten(), self.percentile)
        self.y_tensor[self.y_tensor>=thr] = 1
        self.y_tensor[self.y_tensor<thr] = 0
        
    def plot_trajs(self, t_init=0, window=None, bounds=None):
        """
        Plot 2D trajectories with speed vectors and congestion values as colors
        """
        colors = ['blue', 'green', 'yellow', 'orange', 'red']
        colormap = LinearSegmentedColormap.from_list('name', colors)
        norme = Normalize(0, 1)
        
        if bounds is not None:
            t_init = bounds[0]

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        ax.set_aspect('equal')
        states = np.array(self.raw_tensor[t_init][0])
        emp = np.array([])
        if states.shape[0] == 1:
            p = ax.scatter(emp, emp, c=emp, s=5, cmap=colormap, norm=norme)
            q = ax.quiver(emp, emp, emp, emp, width=0.002, scale=50)
        else:
            p = ax.scatter(states[1:, 0], states[1:, 1], c=np.array(self.raw_tensor[t_init][1])[1:], s=10, cmap=colormap, norm=norme)
            q = ax.quiver(states[1:, 0], states[1:, 1], states[1:, 2], states[1:, 3], width=0.002, scale=30)
        if window is not None:
            ax.set_xlim(window._lon, window.lon_)
            ax.set_ylim(window._lat, window.lat_)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norme)
        plt.colorbar(sm)

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        if bounds is None:
            slider = Slider(axe_slider, 'Time', 0, len(self.raw_tensor)-1, valinit=t_init, valstep=1)
        else:
            slider = Slider(axe_slider, 'Time', bounds[0], bounds[1], valinit=t_init, valstep=1)

        def update_traj(val):
            time = int(slider.val)
            states = np.array(self.raw_tensor[time][0])
            if states.shape[0] == 1:
                p.set_offsets([emp,emp])
                p.set_array(emp)
                q.set_offsets([emp,emp])
                q.set_UVC(emp, emp)
            else:
                p.set_offsets(states[1:, :2])
                p.set_array(np.array(self.raw_tensor[time][1])[1:])
                q.set_offsets(states[1:, :2])
                q.set_UVC(states[1:, 2], states[1:, 3])
            plt.draw()

        slider.on_changed(update_traj)
        return fig, ax, slider
    
    def plot_ytensor(self, t_init=0):
        """
        Plot self.y_tensor
        """
        colors = ['blue', 'green', 'yellow', 'orange', 'red']
        colormap = LinearSegmentedColormap.from_list('name', colors)
        norme = Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norme)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.1, bottom=0.2)
        im = ax.imshow(self.y_tensor[t_init, :, :].T, cmap=colormap, norm=norme)
        plt.colorbar(sm)
        ax.set_xlim(0, self.nb_steps)
        ax.set_ylim(0, self.nb_steps)

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        slider = Slider(axe_slider, 'Time', 0, self.y_tensor.shape[0]-1, valinit=t_init, valstep=1)

        def update(val):
            time = int(slider.val)
            im.set_data(self.y_tensor[time, :, :].T)
            plt.draw()

        slider.on_changed(update)
        return fig, ax, slider
    
    def plot_hist(self):
        """
        Plot the histogram of the nonzero values of self.y_tensor
        """
        tensor = self.y_tensor[self.y_tensor!=0].flatten()
        fig, ax = plt.subplots()
        ax.hist(tensor, bins=50)
        print(stats.describe(tensor), flush=True)
        return fig, ax

    def plot_metric(self, coords):
        """
        Plot the metric over time in a given grid cell
        """
        fig, ax = plt.subplots()
        tensor = self.y_tensor[:, coords[0], coords[1]]
        ax.plot(tensor)
        ax.set_xlim(coords[2], coords[3])
        ax.set_ylim(coords[4], coords[5])
        return fig, ax
    
    def cluster_by_neighbors(self):
        """
        NOT MAINTAINED
        Return the rectangles containing the a/c clusters for each timestamp
        The congestion is rescaled by x -> log(1 + x/log_thr) = log(x + log_thr) - log(log_thr)
            if x << log_thr => log(1 + x/log_thr) ~= x/log_thr -> linear in x for small values
            if x >> log_thr => log(1 + x/log_thr) ~= log(x) - log(log_thr) -> log in x for high values
            if x = 0 => log(1 + x/log_thr) = 0
        Only the a/c with a rescaled congestion above cong_thr are kept
        The clusters are defined as the connex components of the graph defined by the neighboring relation between a/c
        """
        max_val = np.log(1 + self.max_cong/self.log_thr)

        def DFS(currentindex, time, neighbors, labels, ids, visited, cluster):
            """
            Depth-first search
            """
            visited[currentindex] = True
            congestion = self.raw_tensor[time][1][currentindex]
            log_cong = np.log(1 + congestion/self.log_thr)/max_val
            res = False
            if log_cong > self.cong_thr and len(neighbors[currentindex]) > 0:
                labels[currentindex] = cluster
                res = True
            else:
                labels[currentindex] = -1 #outliers -> below cong_thr or no neighbors
            for ident in neighbors[currentindex]:
                index = np.where(ids==ident)
                index = index[0][0]
                if not(visited[index]):
                    rec_res = DFS(index, time, neighbors, labels, ids, visited, cluster)
                    res = res or rec_res
            return res

        box_list = [] # for all t, an array of dimension (nb of clusters, 5)
        for t in range(self.nb_times):
            neighbors = self.raw_tensor[t][3]
            labels = np.zeros((len(neighbors),), dtype='int')
            ids = np.zeros((len(neighbors),))
            visited = [False for i in range(len(neighbors))]
            cluster = 0
            for i in range(len(neighbors)):
                ids[i] = self.raw_tensor[t][0][i][6]
            for i in range(len(neighbors)):
                if not(visited[i]):
                    res = DFS(i, t, neighbors, labels, ids, visited, cluster)
                    if res:
                        cluster += 1

            points = np.array([[self.norm_coord(self.raw_tensor[t][0][i][0], 0), self.norm_coord(self.raw_tensor[t][0][i][1], 1)] for i in range(len(neighbors))])
            unique_labels = set(labels)
            box = np.zeros((len(unique_labels), 5))
            for k in unique_labels:
                if k != -1:
                    class_member_mask = (labels == k)
                    xy = points[class_member_mask]
                    box[k, :2] = np.mean(xy, axis=0)
                    box[k, 2:] = opt_rect(xy, box[k, :2])
            box_list.append(box)

        return box_list
             
    def plot_clusters(self, t_init=0, plot_grid=False):
        """
        For each timstamp, plot the points clustered by colors and the rectangles associated with each cluster
        """
        X = self.trajs[t_init]
        db = self.dbs[t_init]
        box = self.box_list[t_init]
        eps = 1e-6

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        ax.set_aspect('equal')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        if plot_grid:
            minor_ticks = np.arange(0, 1, 1/self.nb_steps)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            plt.rc('grid', color='black')
            ax.grid(which='both')

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
            theta = box[k,4] + np.arctan(box[k,3]/(box[k,2]+eps))
            corner = (box[k,0]-R*np.cos(theta), box[k,1]-R*np.sin(theta))
            ax.add_patch(patches.Rectangle(corner, box[k,2], box[k,3], angle=box[k,4]*180/np.pi, linewidth=1, edgecolor=tuple(col), facecolor='none'))

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        slider = Slider(axe_slider, 'Time', 0, len(self.trajs)-1, valinit=t_init, valstep=1)

        def update_cluster(val):
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            time = int(slider.val)
            X = self.trajs[time]
            db = self.dbs[time]
            box = self.box_list[time]
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
                theta = box[k,4] + np.arctan(box[k,3]/(box[k,2]+eps))
                corner = (box[k,0]-R*np.cos(theta), box[k,1]-R*np.sin(theta))
                ax.add_patch(patches.Rectangle(corner, box[k,2], box[k,3], angle=box[k,4]*180/np.pi, linewidth=1, edgecolor=tuple(col), facecolor='none'))
            plt.draw()

        slider.on_changed(update_cluster)

        return fig, ax, slider