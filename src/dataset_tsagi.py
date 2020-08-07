from dataset import *

class DatasetTsagi(Dataset):
    """
    Manage data from a TSAGI_COMP file
    """
    
    def __init__(self,
                 directory="../data/",
                 tsagi_file="20200718_C_NEW.TSAGI_COMP",
                 input_type=2,
                 metric_type=0):
        super().__init__(directory, tsagi_file, input_type, metric_type)
        
    def get_input(self):
        """
        Return the input attribute
        Print a short description of the corresponding input type
        """
        print("Inputs are:", end=" ")
        if self.input_type == 1:
            print("congestion map")
        elif self.input_type == 2:
            print("lon, lat, alt, speed, head, vz")
        elif self.input_type == 3:
            print("lon, lat, alt, speed, head, vz, congestion, mat0, ..., mat8")
        return self.input
    
    def is_smoothed(self):
        """
        Return True if the data has been smoothed, False if not
        """
        return self.apply_smooth
    
    def is_rescaled(self):
        """
        Return the log_thr if the data has been rescaled, False if not
        """
        if self.rescale:
            return self.log_thr
        else:
            return self.rescale
        
    def get_y_criterion(self):
        """
        Return the criterion attribute
        Print the corresponding criterion
        """
        print("y criterion is:", end=" ")
        if criterion == 0:
            print("density")
        elif criterion == 1:
            print("max")
        elif criterion == 2:
            print("mean")
        return self.criterion
            
    def read(self):
        """
        Load the data from the TSAGI_COMP file into the raw_tensor attribute
        data_dict -> key is time, value is a list of length 4
        1st element is the list of a/c states at this time (used in the inputs)
        idac is used to cluster the a/c according to their neighboring relation
        2nd element is the list of congestion values corresponding to the a/c states (used in the ground truth)
        3rd element is the number of a/c at this time
        4th element (only for input!=3) is the list of neighbors lists of each a/c
        """
        with open(self.path) as f:
            line = f.readline()
            nb_trajs = int(line)
            data_dict = {}
            lon_list = []
            lat_list = []
            alt_list = []
            spd_list = []
            head_list = []
            vz_list = []
            cong_list = []
            mat_list = []
            line = f.readline()
            while line != "":
                point = line.split()
                idac = float(point[1])
                time = float(point[3])
                lat = float(point[4])
                lon = float(point[5])
                alt = float(point[6])
                speed = float(point[7])
                head = float(point[8])
                vz = float(point[9])
                if self.metric == 0:
                    congestion = float(point[11])
                else:
                    congestion = float(point[10])
                if self.input == 3:
                    mat = [float(point[i]) for i in range(12,21)]
                    for i in range(9):
                        mat_list.append(mat[i])
                else:
                    neighbors = [] # used to cluster the a/c according to their neighboring relation
                    for i in range(12,len(point)):
                        neighbors.append(float(point[i])) # non-empty only if the TSAGI_COMP file was created with the neighbors list
                lon_list.append(lon)
                lat_list.append(lat)
                alt_list.append(alt)
                spd_list.append(speed)
                head_list.append(head)
                vz_list.append(vz)
                cong_list.append(congestion)
                if time in data_dict:
                    if self.input == 3:
                        data_dict[time][0].append([lon, lat, alt, speed, head, vz, congestion, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]])
                    else:
                        data_dict[time][0].append([lon, lat, alt, speed, head, vz, idac])
                        data_dict[time][3].append(neighbors)
                    data_dict[time][1].append(congestion)
                    data_dict[time][2] += 1
                else:
                    if self.input == 3:
                        data_dict[time] = [[[lon, lat, alt, speed, head, vz, congestion, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]]], [congestion], 1]
                    else:
                        data_dict[time] = [[[lon, lat, alt, speed, head, vz, idac]], [congestion], 1, [neighbors]]
                line = f.readline()
        times = list(data_dict.keys())
        self.raw_tensor = np.array([data_dict[time] for time in sorted(times)], dtype='object')
        self.nb_times = len(times)
        self.min_lon = min(lon_list)
        self.max_lon = max(lon_list)
        self.min_lat = min(lat_list)
        self.max_lat = max(lat_list)
        self.min_cong = min(cong_list)
        self.max_cong = max(cong_list)
        self.min_alt = min(alt_list)
        self.max_alt = max(alt_list)
        self.min_spd = min(spd_list)
        self.max_spd = max(spd_list)
        self.min_head = min(head_list)
        self.max_head = max(head_list)
        self.min_vz = min(vz_list)
        self.max_vz = max(vz_list)
        if self.input == 3:
            self.min_mat = min(matList)
            self.max_mat = max(matList)
        
    def compute_y(self, nb_steps=100, apply_smooth=False, log_thr=1e-2, criterion=1, rescale=True):
        """
        Compute the ground truth congestion map
        nb_steps -> size of the square grid
        apply_smooth -> if True, apply a 3x3 gaussian kernel to the map
        criterion:
            0 -> density (nb of a/c per cell)
            1 -> max congestion per cell
            2 -> mean congestion per cell
        rescale -> if True, apply the log rescale
        Rescale the congestion by x -> log(1 + x/log_thr) = log(x + log_thr) - log(log_thr)
        If x << log_thr => log(1 + x/log_thr) ~= x/log_thr -> linear in x for small values
        If x >> log_thr => log(1 + x/log_thr) ~= log(x) - log(log_thr) -> log in x for high values
        If x = 0 => log(1 + x/log_thr) = 0
        Normalize the congestion between 0 and 1
        """
        self.lon_step = (self.max_lon - self.min_lon)/nb_steps
        self.lat_step = (self.max_lat - self.min_lat)/nb_steps
        self.nb_steps = nb_steps
        self.apply_smooth = apply_smooth
        self.log_thr = log_thr
        self.criterion = criterion
        self.rescale = rescale

        self.y_tensor = np.zeros((self.nb_times, nb_steps, nb_steps))
        max_val = np.log(1 + self.max_cong/log_thr)

        for t in range(self.nb_times):
            if criterion == 2: total = np.zeros((nb_steps, nb_steps))
            for i in range(len(self.raw_tensor[t][0])):
                s_lon = int((self.raw_tensor[t][0][i][0] - self.min_lon)/self.lon_step)
                s_lat = int((self.raw_tensor[t][0][i][1] - self.min_lat)/self.lat_step)
                if s_lon > nb_steps-1 or s_lat > nb_steps-1 or s_lon < 0 or s_lat < 0:
                    continue
                if rescale:
                    val = np.log(1 + self.raw_tensor[t][1][i]/log_thr)/max_val
                else:
                    val = self.raw_tensor[t][1][i]/self.max_cong
                if criterion == 0:
                    self.y_tensor[t, s_lon, s_lat] += 1
                if criterion == 1 and val > self.y_tensor[t, s_lon, s_lat]:
                    self.y_tensor[t, s_lon, s_lat] = val
                if criterion == 2:
                    self.y_tensor[t, s_lon, s_lat] += val
                    total[s_lon, s_lat] += 1
            if criterion == 2:
                total[total==0] = 1
                self.y_tensor[t, :, :] /= total
        if criterion == 0:
            self.min_cong = np.min(self.y_tensor)
            self.max_cong = np.max(self.y_tensor)
            self.y_tensor = (self.y_tensor - self.min_cong)/(self.max_cong - self.min_cong)
        if apply_smooth:
            self.smoothing()
            
    def compute_x(self, input_struct=0):
        """
        Compute the inputs for the models
        input_struct:
            0 -> by position (with zero padding)
            1 -> by a/c (with zero padding)
            2 -> by position (without zero padding)
        """
        self.input_struct = input_struct
        if input_struct == 0:
            max_ac = np.max(self.raw_tensor[:,2])*(16 if self.input==3 else 6)
            self.x_tensor = np.zeros((self.nb_times, max_ac))
        elif input_struct == 1:
            max_ac = np.max(self.raw_tensor[:,2])
            self.x_tensor = np.zeros((self.nb_times, max_ac, 6))
            ids_array = -np.ones((maxAc,))
        elif input_struct == 2:
            inputs_list = []
        for t in range(self.nb_times):
            if input_struct == 0 or input_struct == 2:
                states = np.array(sorted(self.raw_tensor[t,0], key=lambda l:l[0]+10*l[1]))
            elif input_struct == 1:
                states = np.array(self.raw_tensor[t][0])
                ids = states[:,6]
            states[:,0] = self.norm_coord(states[:,0], 0)
            states[:,1] = self.norm_coord(states[:,1], 1)
            states[:,2] = self.norm_coord(states[:,2], 2)
            states[:,3] = self.norm_coord(states[:,3], 3)
            states[:,4] = self.norm_coord(states[:,4], 4)
            states[:,5] = self.norm_coord(states[:,5], 5)
            if self.input == 3:
                states[:,6] = self.norm_coord(states[:,6], 6)
                states[:,7:16] = self.norm_coord(states[:,7], 7)
            if input_struct == 0:
                states = states[:,:6].flatten()
                self.x_tensor[t,:states.shape[0]] = states
            elif input_struct == 1 and t >= 1:
                for i in range(len(ids_array)):
                    if ids_array[i] != -1 and ids_array[i] not in ids:
                        ids_array[i] = -1
                for (ident,j) in zip(ids,range(states.shape[0])):
                    index = np.where(ids_array==ident)
                    if len(index[0]) > 0:
                        i = index[0][0]
                        self.x_tensor[t,i,:] = states[j,:6]
                    else:
                        index = np.where(ids_array==-1)
                        i = index[0][0]
                        ids_array[i] = states[j,6]
                        self.x_tensor[t,i,:] = states[j,:6]
            elif input_struct == 1 and t == 0:
                ids_array[:states.shape[0]] = states[:,6]
                self.x_tensor[0,:states.shape[0],:] = states[:,:6]
            elif input_struct == 2:
                input_list.append(states.tolist())
        if input_struct == 1: self.x_tensor.reshape((nb_times, max_ac*6))
        elif input_struct == 2: self.x_tensor = np.array(input_list)
            
    def load(self, input_struct=0, nb_steps=100, apply_smooth=False, log_thr=1e-2, criterion=1, rescale=True, apply_yolo=False, cong_thr=0.2, eps=0.04, min_samples=3, apply_binary=False, percentile=95, comp_y=True, comp_x=True):
        """
        Load x_tensor and y_tensor
        """
        self.input_struct = input_struct
        self.nb_steps = nb_steps
        self.apply_smooth = apply_smooth
        self.log_thr = log_thr
        self.criterion = criterion
        self.rescale = rescale
        self.apply_yolo = apply_yolo
        self.cong_thr = cong_thr
        self.eps = eps
        self.min_samples = min_samples
        self.apply_binary = apply_binary
        self.percentile = percentile
        
        print("Start preprocessing", flush=True)
        if comp_x:
            self.compute_x(input_struct)
        if comp_y:
            if not(apply_yolo):
                self.compute_y(nb_steps, apply_smooth, log_thr, criterion, rescale)
                if apply_binary:
                    self.filter_binary(percentile)
            else:
                self.trajs = self.filter_trajs()
                self.dbs = run_dbscan(self.trajs, eps, min_samples)
                self.box_list = compute_cluster_boxes(self.trajs, self.dbs)
                self.y_tensor = compute_grid(self.box_list, nb_steps)
        print("Preprocessing done", flush=True)
                 