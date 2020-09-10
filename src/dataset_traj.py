from dataset import *
from os import walk
#import pygrib
#import resource

class DatasetTraj(Dataset):
    """
    Manage data from traj files
    """
    
    def __init__(self,
                trajs_directory="../data/trajs/",
                plns_directory="../data/plns/",
                weather_directory="../data/weather/",
                load_data = True):
        self.trajs = trajs_directory
        self.plns = plns_directory
        self.weather = weather_directory
        if load_data:
            self.read()
        
    def read(self):
        """
        Create two numpy arrays:
        self.index_tensor has dimension (nb_of_trajs,) and contains the index of the flight plan corresponding to each trajectory
        self.y_tensor has dimension (nb_of_trajs, max_len_of_trajs, 7) and contains the trajectories
        Each point of the trajectory is a state vector of dimension 7:
        lat, lon, altitude, heading, vtas, u-wind, v-wind
        """
        self.load_weather()
        self.load_pln()
        (_, _, filenames) = next(walk(self.trajs))
        traj_list = []
        pln_list = []
        max_len = 0
        for f in filenames:
            trajectory = []
            with open(self.trajs+f) as file:
                line = file.readline()
                line = file.readline()
                point = line.split(',')
                pln_list.append(int(point[0][-1]) - 1)
                time = 0
                error = 0
                line_nb = 2
                while line != "":
                    if time == 0:
                        state = [float(point[2]), float(point[3]), float(point[4]), float(point[6]), float(point[8])]
                        hdg = state[3]
                        state[3] = np.cos(hdg)*state[4]
                        state[4] = np.sin(hdg)*state[4]
                        uwind, vwind = self.match_weather(f, state)
                        state.append(uwind)
                        state.append(vwind)
                        trajectory.append(state)
                        time += 10
                    elif time - 5 <= float(point[1]) <= time + 5:
                        error = 0
                        previous = trajectory[-1]
                        if len(point) < 10:
                            print(f)
                            print(point)
                            print(line_nb, flush=True)
                        else:
                            state = [self.interpolate(time, point[1], previous[0], point[2]),
                                     self.interpolate(time, point[1], previous[1], point[3]),
                                     self.interpolate(time, point[1], previous[2], point[4]),
                                     self.interpolate(time, point[1], previous[3], point[6]),
                                     self.interpolate(time, point[1], previous[4], point[8])]
                            hdg = state[3]
                            state[3] = np.cos(hdg)*state[4]
                            state[4] = np.sin(hdg)*state[4]
                            uwind, vwind = self.match_weather(f, state)
                            state.append(uwind)
                            state.append(vwind)
                            trajectory.append(state)
                            time += 10
                    else:
                        error += 1
                    if error >= 2:
                        raise RuntimeError("Too much skipped lines")
                    line = file.readline()
                    point = line.split(',')
                    line_nb += 1
            traj_list.append(trajectory)
            max_len = max(max_len, len(trajectory))
        self.index_tensor = np.array(pln_list)    
        self.y_tensor = np.zeros((len(traj_list), max_len, 7))
        for i in range(len(traj_list)):
            self.y_tensor[i, :len(traj_list[i]), :] = np.array(traj_list[i])
        self.maxs = []
        self.mins = []
        for k in range(7):
            self.maxs.append(np.max(self.y_tensor[...,k]))
            self.mins.append(np.min(self.y_tensor[...,k]))
            self.y_tensor[...,k] = (self.y_tensor[...,k] - self.mins[k])/(self.maxs[k] - self.mins[k])
    
    def load_weather(self):
        """
        Load all the weather numpy arrays in a dictionnary self.weather_dict
        """
        self.weather_dict = {}
        (_, _, filenames) = next(walk(self.weather))
        for f in filenames:
            if len(f) > 19:
                self.weather_dict[f[7:19]] = np.load(self.weather+f)
            else:
                self.weather_dict[f[:4]] = np.load(self.weather+f)
        
    def load_pln(self):
        """
        Load all the flight plans in an array self.x_tensor
        """
        temp_list = []
        max_len = 0
        (_, _, filenames) = next(walk(self.plns))
        for f in filenames:
            with open(self.plns+f) as file:
                plns = []
                for i in range(18):
                    file.readline()
                line = file.readline()
                while line != "":
                    point = line.split()
                    plns.append([float(point[2]), float(point[3])])
                    line = file.readline()
                temp_list.append(plns)
                max_len = max(max_len, len(plns))
        self.x_tensor = np.zeros((len(temp_list), max_len, 2))
        for i in range(len(temp_list)):
            self.x_tensor[i, -len(temp_list[i]):, :] = np.array(temp_list[i])
        self.maxs_pln = []
        self.mins_pln = []
        for k in range(2):
            self.maxs_pln.append(np.max(self.x_tensor[...,k]))
            self.mins_pln.append(np.min(self.x_tensor[...,k]))
            self.x_tensor[...,k] = (self.x_tensor[...,k] - self.mins_pln[k])/(self.maxs_pln[k] - self.mins_pln[k])
        
    def match_weather(self, f, state):
        """
        Match the position of the state with the closest weather point
        Return the state augmented with the u and v wind components
        """
        key = f[15:27]
        time = int(f[28])//3
        dlat = np.abs(self.weather_dict["lats"] - state[0])
        dlon = np.abs(self.weather_dict["lons"] - state[1])
        dlvl = np.abs(self.weather_dict["lvls"] - self.pressure(state[2]))
        ilat = np.argmin(dlat)
        ilon = np.argmin(dlon)
        ilvl = np.argmin(dlvl)
        uwind = self.weather_dict[key][time, ilat, ilon, ilvl, 0]
        vwind = self.weather_dict[key][time, ilat, ilon, ilvl, 1]
        return uwind, vwind
    
    def interpolate(self, x0, x1, y0, y1):
        """
        Linear interpolation 10 seconds after x0
        """
        x0 = x0 - 10
        x1 = float(x1)
        y1 = float(y1)
        a = (y1 - y0)/(x1 - x0)
        b = y0 - a*x0
        return float(a*(x0+10) + b)
    
    def temperature(self, alt):
        isaT0 = 288.15
        beta = -0.0065
        htrop = 11000.0
        if alt < 0.0:
            alt = 0.0
        if alt < htrop:
            return isaT0 + beta*alt
        return isaT0 + beta*htrop

    def pressure(self, alt):
        isaT0 = 288.15
        isaP0 = 101325
        htrop = 11000.0
        g = 9.80665
        R = 287.05287
        if alt < 0.0:
            alt = 0.0
        if alt <= htrop:
            t = self.temperature(alt)/isaT0
            return isaP0*t**5.25583
        t = self.temperature(htrop)
        h = alt - htrop
        u = g/(R*t)
        p = self.pressure(htrop)
        return p*np.exp(-u*h)
    
    def grib2npy(self, directory="../data/weather/grib/"):
        """
        Parse the grib files from the directory into npy files
        For each grib file, wind data are saved over France for 28 isobaric levels and 9 forecast times
        Message index:
            U component: 505-588
            V component: 757-840
        Lons: -6° / 10°
        Lats: 41° / 52°
        """
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        (_, _, filenames) = next(walk(directory))
        grib_info = True
        for f in filenames:
            if f.lower().endswith('.grib2'):
                grbs = pygrib.open(directory+f)
                print("Process file:", f, flush=True)
                tensor = np.zeros((9, 23, 33, 28, 2))
                for time in range(9):
                    tensor[time, ..., 0] = self.parse_grib(grbs, 504+28*time, grib_info)
                    grib_info = False
                    tensor[time, ..., 1] = self.parse_grib(grbs, 756+28*time, grib_info)
                np.save(self.weather+f+".npy", tensor)
                grbs.close()
                
    def parse_grib(self, grbs, message, grib_info):
        """
        Parse 28 messages (one for each level) into a tensor
        """
        grbs.seek(message)
        grb_list = grbs.read(28)
        tensor = np.zeros((23, 33, 28))
        for lvl in range(len(grb_list)):
            data, lats, lons = grb_list[lvl].data(lat1=41,lat2=52)
            tensor[..., lvl] = np.concatenate((data[:,708:], data[:,:21]), axis=1)
        if grib_info:
            lats = np.arange(41, 52.1, 0.5)
            lons = np.arange(-6, 10.1, 0.5)
            print(lats.shape)
            print(lons.shape, flush=True)
            np.save(self.weather+"lats.npy", lats)
            np.save(self.weather+"lons.npy", lons)
        return tensor
        
    def plot_plns(self, t_init=0):
        """
        Plot the flight plans from self.x_tensor
        """
        tensor = self.x_tensor.copy()
        nb_times = tensor.shape[0]

        for k in range(2):
            tensor[...,k] = tensor[...,k]*(self.maxs_pln[k] - self.mins_pln[k]) + self.mins_pln[k]
        
        fig, ax = plt.subplots()

        m = Basemap(projection='cyl', llcrnrlat=41, urcrnrlat=52, llcrnrlon=-6, urcrnrlon=10, resolution='c', area_thresh=1000.)
        #m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        #m.drawstates(linewidth=0.5)

        #Draw parallels and meridians

        m.drawparallels(np.arange(35.,55.,5.))
        m.drawmeridians(np.arange(-5.,10.,5.))
        m.drawmapboundary()

        #Convert latitude and longitude to coordinates X and Y

        x0, y0 = m(tensor[t_init, :, 1], tensor[t_init, :, 0])

        #Plot the points on the map

        sp0 = ax.scatter(x0,y0,c='g',marker='+',s=20.0)

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        slider = Slider(axe_slider, 'Example', 0, nb_times-1, valinit=t_init, valstep=1)

        def update_tp(val):
            time = int(slider.val)
            x0, y0 = m(tensor[time, :, 1], tensor[time, :, 0])
            sp0.set_offsets(np.c_[x0,y0])
            plt.draw()

        slider.on_changed(update_tp)

        return fig, ax, slider

    def plot_traj(self, t_init=0):
        """
        Plot the trajectories from self.y_tensor
        """
        tensor = self.y_tensor.copy()
        nb_times = tensor.shape[0]

        for k in range(7):
            tensor[...,k] = tensor[...,k]*(self.maxs[k] - self.mins[k]) + self.mins[k]
        
        fig, ax = plt.subplots()

        m = Basemap(projection='cyl', llcrnrlat=41, urcrnrlat=52, llcrnrlon=-6, urcrnrlon=10, resolution='c', area_thresh=1000.)
        #m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        #m.drawstates(linewidth=0.5)

        #Draw parallels and meridians

        m.drawparallels(np.arange(35.,55.,5.))
        m.drawmeridians(np.arange(-5.,10.,5.))
        m.drawmapboundary()

        #Convert latitude and longitude to coordinates X and Y

        x0, y0 = m(tensor[t_init, :, 1]*180/np.pi, tensor[t_init, :, 0]*180/np.pi)

        #Plot the points on the map

        sp0 = ax.scatter(x0,y0,c='r',marker='+',s=1.0)

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        slider = Slider(axe_slider, 'Example', 0, nb_times-1, valinit=t_init, valstep=1)

        def update_tp(val):
            time = int(slider.val)
            x0, y0 = m(tensor[time, :, 1]*180/np.pi, tensor[time, :, 0]*180/np.pi)
            sp0.set_offsets(np.c_[x0,y0])
            plt.draw()

        slider.on_changed(update_tp)

        return fig, ax, slider
    
    def plot_profile(self, t_init=0):
        """
        Plot the vertical profile of the trajectories from self.y_tensor
        """
        tensor = self.y_tensor.copy()
        nb_times = tensor.shape[0]

        for k in range(7):
            tensor[...,k] = tensor[...,k]*(self.maxs[k] - self.mins[k]) + self.mins[k]
        
        fig, ax = plt.subplots()

        sp0 = ax.scatter(np.arange(0, 10*len(tensor[t_init, :, 2]), 10),tensor[t_init, :, 2], c='r',marker='+',s=1.0)

        axe_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
        slider = Slider(axe_slider, 'Example', 0, nb_times-1, valinit=t_init, valstep=1)

        def update_tp(val):
            time = int(slider.val)
            sp0.set_offsets(np.c_[np.arange(0, 10*len(tensor[time, :, 2]), 10),tensor[time, :, 2]])
            plt.draw()

        slider.on_changed(update_tp)

        return fig, ax, slider
    
    def plot_wind(self, date, time, level):
        """
        Plot the wind components from self.weather_dict
        """
        lats = self.weather_dict["lats"]
        lons = self.weather_dict["lons"]
        tensor = self.weather_dict[date]
        tensor = np.reshape(tensor[time, :, :, level, :], (-1, 2))
        
        fig, ax = plt.subplots()

        m = Basemap(projection='cyl', llcrnrlat=41, urcrnrlat=52, llcrnrlon=-6, urcrnrlon=10, resolution='c', area_thresh=1000.)
        #m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        #m.drawstates(linewidth=0.5)

        #Draw parallels and meridians

        m.drawparallels(np.arange(35.,55.,5.))
        m.drawmeridians(np.arange(-5.,10.,5.))
        m.drawmapboundary()

        #Convert latitude and longitude to coordinates X and Y
        coords = np.transpose([np.tile(lons, len(lats)), np.repeat(lats, len(lons))])
        x0, y0 = m(coords[:,0], coords[:,1])

        #Plot the points on the map

        q = ax.quiver(x0, y0, tensor[:,0], tensor[:,1], width=0.002, scale=200)
        
        return fig, ax