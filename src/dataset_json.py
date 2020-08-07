from dataset import *
import json

class DatasetJson(Dataset):
    """
    NOT MAINTAINED
    Manage data from a json file
    """
    
    def __init__(self,
                 directory = "../data/",
                 filter_file="FRANCE_FIR.FILTER",
                 json_file="TEST_10000.json",
                 input_type=2,
                 metric_type=0):
        self.filter_path = directory + filter_file
        self.read_filter()
        super().__init__(directory, json_file, input_type, metric_type)
        
    def read_filter(self):
        """
        Load a filter from a FILTER file
        """
        with open(self.filter_path) as file:
            for i in range(6):
                file.readline()
            nb_box_points = int(file.readline())
            self.x_box_points = []
            self.y_box_points = []
            for i in range(nb_box_points):
                line = file.readline()
                tab = line.split(';')
                self.x_box_points.append(float(tab[0]))
                self.y_box_points.append(float(tab[1]))
            nb_polygon_points = int(file.readline())
            self.x_poly_points = []
            self.y_poly_points = []
            for i in range(nb_polygon_points):
                line = file.readline()
                tab = line.split(';')
                self.x_poly_points.append(float(tab[0]))
                self.y_poly_points.append(float(tab[1]))
        self.x_box_points.append(self.x_box_points[0])
        self.y_box_points.append(self.y_box_points[0])
        self.x_poly_points.append(self.x_poly_points[0])
        self.y_poly_points.append(self.y_poly_points[0])
        self.min_lon_fil = min(self.y_poly_points)
        self.max_lon_fil = max(self.y_poly_points)
        self.min_lat_fil = min(self.x_poly_points)
        self.max_lat_fil = max(self.x_poly_points)

    def read(self):
        """
        Load the data from the json file into the raw_tensor attribute
        """
        with open(self.path) as f:
            data = json.load(f)
        nb_trajs = len(data)
        data_dict = {}
        lon_list = []
        lat_list = []
        cong_list = []
        for i in range(nb_trajs):
            traj = data[i]['trajectory points']
            for point in traj:
                time = point["time"]
                lon = point["lon"]
                lat = point["lat"]
                congestion = point["congestion"]
                lon_list.append(lon)
                lat_list.append(lat)
                cong_list.append(congestion)
                if time in data_dict:
                    data_dict[time][0].append([lon, lat])
                    data_dict[time][1].append(congestion)
                else:
                    data_dict[time] = [[[lon, lat]], [congestion]]
        times = list(data_dict.keys())
        self.raw_tensor = np.array([data_dict[time] for time in sorted(times)], dtype='object')
        self.nb_times = len(times)
        self.min_lon = min(lon_list)
        self.max_lon = max(lon_list)
        self.min_lat = min(lat_list)
        self.max_lat = max(lat_list)
        self.min_cong = min(cong_list)
        self.max_cong = max(cong_list)

    def writeJson(self, file_path, tensor):
        """
        Save the tensor into a json file
        """
        with open(file_path, 'w') as f:
            json.dump(tensor.tolist(), f)
                 