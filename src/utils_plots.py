import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap, Normalize
import glob
import copy
import pickle
import pandas
from scipy import stats
from sklearn.metrics import precision_recall_curve

def plot_predict_ytensor(pred_tensor, true_tensor, t_init=0):
    """
    Plot the predicted congestion values and the true congestion values
    """
    nb_times = pred_tensor.shape[0]
    nb_steps = pred_tensor.shape[1]
    
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    colormap = LinearSegmentedColormap.from_list('name', colors)
    norme = Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norme)

    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.subplots_adjust(bottom=0.2)
    im_pred = ax[0].imshow(pred_tensor[t_init, :, :].T, cmap=colormap, norm=norme, aspect="auto")
    im_true = ax[1].imshow(true_tensor[t_init, :, :].T, cmap=colormap, norm=norme, aspect="auto")
    plt.colorbar(sm, ax=ax[0])
    plt.colorbar(sm, ax=ax[1])
    ax[0].set_xlim(0, nb_steps)
    ax[0].set_ylim(0, nb_steps)
    ax[1].set_xlim(0, nb_steps)
    ax[1].set_ylim(0, nb_steps)
    ax[0].set_title("Predicted")
    ax[1].set_title("True")

    axeSlider = plt.axes([0.1, 0.01, 0.8, 0.05])
    slider = Slider(axeSlider, 'Example', 0, nb_times-1, valinit=t_init, valstep=1)

    def update_eval(val):
        time = int(slider.val)
        im_pred.set_data(pred_tensor[time, :, :].T)
        im_true.set_data(true_tensor[time, :, :].T)
        plt.draw()

    slider.on_changed(update_eval)
    return fig, ax, slider

def plot_error_dist(pred_tensor, true_tensor):
    """
    Plot the distribution of the non-zero prediction errors
    """
    diff_tensor = np.abs(pred_tensor - true_tensor)
    error_tensor = diff_tensor[diff_tensor!=0].flatten()
    fig, ax = plt.subplots()
    ax.hist(error_tensor, bins=50)
    print(stats.describe(error_tensor), flush=True)
    return fig, ax, 0

def binary_classification_analysis(pred_tensor, true_tensor, threshold=0.1):
    """
    NOT MAINTAINED
    Compute: Precision, Recall, MCC, accuracy, F1 score
    Plot: Recall-Precision curve
    Only if model.ds.apply_binary is True
    """
    decision_tensor = pred_tensor.copy()
    decision_tensor[decision_tensor < threshold] = 0
    decision_tensor[decision_tensor >= threshold] = 1
    dtensor = decision_tensor - true_tensor
    fn = len(dtensor[dtensor == -1].flatten())
    fp = len(dtensor[dtensor == 1].flatten())
    atensor = decision_tensor + true_tensor
    tp = len(atensor[atensor == 2].flatten())
    tn = len(atensor[atensor == 0].flatten())
    print("tp =", tp)
    print("fp =", fp)
    print("tn =", tn)
    print("fn =", fn, flush=True)
    mcc = (tp*tn - fp*fn)/np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    pr = tp/(fp+tp)
    rc = tp/(fn+tp)
    print("Precision =", pr)
    print("Recall =", rc)
    print("MCC =", mcc)
    print("accuray =", (tp+tn)/(fn+fp+tp+tn))
    print("F1 =", 2*pr*rc/(pr+rc))
    print("datasize =", fn+fp+tp+tn)
    print("len(index) =", len(index), flush=True)

    precision, recall, thresholds = precision_recall_curve(true_tensor.flatten(), pred_tensor.flatten())
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    
    return fig, ax

def plot_sensitivity_analysis(directory=""):
    """
    NOT MAINTAINED
    Plot the final validation loss (averaged over the last 50 values) for various hyperparameters settings from directory
    """
    data = {'hparams':[], 'error':[]}
    for path in glob.glob(directory+'val_loss_*'):
        with open(path, 'rb') as f:
            val_loss = pickle.load(f)
        val_mean = np.mean(np.array(val_loss[-50:]))
        data['hparams'].append(path)
        data['error'].append(val_mean)
    df = pandas.DataFrame(data)
    ax = df.plot.bar(x='hparams', y='error', rot=45, figsize=(12,6))
    return ax
    
def plot_loss(directory):
    """
    NOT MAINTAINED
    Plot the model loss curves from directory
    """
    plt.figure()
    for path in glob.glob(directory+'val_loss_*'):
        with open(path, 'rb') as f:
            val_loss = pickle.load(f)
            plt.plot(val_loss, color='orange')
    for path in glob.glob(directory+'train_loss_*'):
        with open(train, 'rb') as f:
            train_loss = pickle.load(f)
            plt.plot(train_loss, color='blue')    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')