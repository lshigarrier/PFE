from dataset_tsagi import *
from dataset_json import *
import sys
import yaml

def test_json(directory, filter_file, json_file, input_type, metric_type):
    ds = DatasetJson(directory, filter_file, json_file, input_type, metric_type)
    return ds

def test_tsagi(directory, tsagi_file, input_type, metric_type):
    ds = DatasetTsagi(directory, tsagi_file, input_type, metric_type)
    return ds

def test_dbscan(ds, input_struct, nb_steps, log_thr, cong_thr, eps, min_samples):
    ds.load(input_struct=input_struct,
            nb_steps=nb_steps,
            log_thr=log_thr,
            apply_yolo=True,
            cong_thr=cong_thr,
            eps=eps,
            min_samples=min_samples,
            comp_x=False)
    fig, ax, slider = ds.plot_clusters()
    return fig, ax, slider

def test_cluster_by_neighbors(ds, nb_steps, log_thr, cong_thr, comp_x):
    """
    NOT MAINTAINED
    """
    ds.load(nb_steps=nb_steps, log_thr=log_thr, cong_thr=cong_thr, comp_y=False, comp_x=comp_x)
    ds.box_list = ds.cluster_by_neighbors()
    ds.y_tensor = compute_grid(ds.box_list, ds.nb_steps)
    fig, ax, slider = plot_predict_clusters(ds.y_tensor, ds.y_tensor)
    return fig, ax, slider

def test_plot_trajs(ds, log_thr, window, bounds):
    ds.load(log_thr=log_thr, comp_y=False, comp_x=False)
    ds.filter_space(window)
    fig, ax, slider = ds.plot_trajs(t_init=1000, window=window, bounds=bounds)
    return fig, ax, slider

def test_plot_y(ds, nb_steps, apply_smooth, log_thr, criterion, apply_binary, percentile, comp_x):
    ds.load(nb_steps=nb_steps, apply_smooth=apply_smooth, log_thr=log_thr, criterion=criterion, apply_binary=apply_binary, percentile=percentile, comp_x=comp_x)
    fig, ax, slider = ds.plot_ytensor()
    return fig, ax, slider

def test_plot_hist(ds, nb_steps, apply_smooth, log_thr):
    ds.load(nb_steps=nb_steps, apply_smooth=apply_smooth, log_thr=log_thr, comp_x=False)
    fig, ax = ds.plot_hist()
    return fig, ax

def test_plot_metric(ds, nb_steps, apply_smooth, log_thr, coords):
    ds.load(nb_steps=nb_steps, apply_smooth=apply_smooth, log_thr=log_thr, comp_x=False)
    fig, ax = ds.plot_metric(coords)
    return fig, ax

def test_nb_ac(ds):
    ac_list = []
    for t in range(ds.nb_times):
        ac_list.append(ds.raw_tensor[t][2])
    print("Nb of timestamps:", len(ac_list), flush=True)
    fig, ax = plt.subplots()
    ax.plot(ac_list)
    return fig, ax

def main():
    """
    Tests all the main preprocessing and plotting functions
    """
    yaml_path = sys.argv[1]
    with open(yaml_path+".yaml", 'r') as f:
        param = yaml.safe_load(f)
    
    test = sys.argv[2]
    if test == "json":
        test_json(param["json_directory"], param["json_filter"], param["json_file"], param["input_type"], param["metric_type"])
    else:
        ds = test_tsagi(param["tsagi_directory"], param["tsagi_file"], param["input_type"], param["metric_type"])
        if test == "dbscan":
            fig, ax, slider = test_dbscan(ds, param["input_struct"], param["nb_steps"], param["log_thr"], param["cong_thr"], param["eps"], param["min_samples"])
        elif test == "neighbors":
            fig, ax, slider = test_cluster_by_neighbors(ds, param["nb_steps"], param["log_thr"], param["cong_thr"], param["comp_x"])
        elif test == "traj":
            win = param["window"]
            window = Window(win["_lon"], win["_lat"], win["_alt"], win["lon_"], win["lat_"], win["alt_"])
            bounds = tuple(param["bounds"])
            fig, ax, slider = test_plot_trajs(ds, param["log_thr"], window, bounds)
        elif test == "ytensor":
            fig, ax, slider = test_plot_y(ds, param["nb_steps"], param["apply_smooth"], param["log_thr"], param["criterion"], param["apply_binary"], param["percentile"], param["comp_x"])
        elif test == "hist":
            fig, ax = test_plot_hist(ds, param["nb_steps"], param["apply_smooth"], param["log_thr"])
        elif test == "metric":
            fig, ax = test_plot_metric(ds, param["nb_steps"], param["apply_smooth"], param["log_thr"], param["coords"])
        elif test == "ac":
            fig, ax = test_nb_ac(ds)
    plt.show()
        
if __name__ == '__main__':
    main()