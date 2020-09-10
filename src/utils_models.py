import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv1D, LSTM, GRU, TimeDistributed, Dense, Reshape, Concatenate, Conv2DTranspose, BatchNormalization
from keras.regularizers import l2 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit, KFold

def batch_generator_tp(model, is_training):
    """
    x_encoder_batch is a batch of sequences of length model.ds.x_tensor.shape[1]
    x_decoder_batch is a batch of sequences of length model.ds.y_tensor.shape[1]
        (the first element of the sequences is a zero vector)
    y_batch is a batch of sequences of length model.ds.y_tensor.shape[1] (the wind components are not part of y_batch)
    """
    x = model.ds.x_tensor
    y = model.ds.y_tensor
    index_map = model.train_index if is_training else model.val_index
    m = len(index_map)
    bs = model.batch_size
    n_batches_per_epoch = m//bs
    while True:
        index = np.arange(m)
        np.random.shuffle(index)
        for i in range(n_batches_per_epoch):
            current_bs = min(m-i*bs, bs)
            x_encoder_batch = np.array([x[model.ds.index_tensor[index_map[index[j]]]] for j in range(i*bs,i*bs+current_bs)])
            x_decoder_batch = np.array([y[index_map[index[j]], :-1, :] for j in range(i*bs,i*bs+current_bs)])
            x_decoder_batch = np.concatenate((np.zeros((x_decoder_batch.shape[0], 1, x_decoder_batch.shape[2])), x_decoder_batch), axis=1)
            y_batch = np.array([y[index_map[index[j]], :, :5] for j in range(i*bs,i*bs+current_bs)])
            yield [x_encoder_batch, x_decoder_batch], y_batch

def batch_generator_seq2vec(model, is_training):
    """
    x_batch is a batch of sequences of length model.input_dim
    y_batch is a a batch of vectors such that their timestamp is model.output_dim after the end of x_batch
    """
    x = model.ds.x_tensor
    y = model.ds.y_tensor
    index_map = model.train_index if is_training else model.val_index
    m = len(index_map)
    bs = model.batch_size
    n_batches_per_epoch = m//bs
    while True:
        index = np.arange(m)
        np.random.shuffle(index)
        for i in range(n_batches_per_epoch):
            current_bs = min(m-i*bs, bs)
            x_batch = np.array([x[index_map[index[j]]:index_map[index[j]]+model.input_dim] for j in range(i*bs,i*bs+current_bs)])
            y_batch = np.array([y[index_map[index[j]]+model.input_dim+model.output_dim-1] for j in range(i*bs,i*bs+current_bs)])
            yield x_batch, y_batch
            
def batch_generator_seq2seq(model, is_training):
    """
    x_encoder_batch is a batch of sequences of length model.input_dim
    x_decoder_batch is a batch of sequences of length model.output_dim following x_encoder_batch
        (the first element of the sequences is a zero vector)
    y_batch is a batch of sequences of length model.output_dim following x_encoder_batch
    """
    x = model.ds.x_tensor
    y = model.ds.y_tensor
    index_map = model.train_index if is_training else model.val_index
    m = len(index_map)
    bs = model.batch_size
    n_batches_per_epoch = m//bs
    while True:
        index = np.arange(m)
        np.random.shuffle(index)
        for i in range(n_batches_per_epoch):
            current_bs = min(m-i*bs, bs)
            x_encoder_batch = np.array([x[index_map[index[j]]:index_map[index[j]]+model.input_dim] for j in range(i*bs,i*bs+current_bs)])
            x_decoder_batch = np.array([y[index_map[index[j]]+model.input_dim:index_map[index[j]]+model.input_dim+model.output_dim-1] for j in range(i*bs,i*bs+current_bs)])
            x_decoder_batch = np.concatenate((np.zeros((x_decoder_batch.shape[0], 1, x_decoder_batch.shape[2])), x_decoder_batch), axis=1)
            y_batch = np.array([y[index_map[index[j]]+model.input_dim:index_map[index[j]]+model.input_dim+model.output_dim] for j in range(i*bs,i*bs+current_bs)])
            yield [x_encoder_batch, x_decoder_batch], y_batch

def custom_loss_yolo(y_true, y_pred):
    """
    Custom loss for the 'yolo' model
    MSE is used for center position, rotation angle, and presence probability
    A weight is used to increase the importance of predicting a cluster when there is one
    Square error of the square root is used for width and height
    lbd is a boolean tensor corresponding to the presence or absence of cluster in each grid cell
    """
    y_pred = K.reshape(y_pred, (K.shape(y_true)[0], -1, 6))
    y_true = K.reshape(y_true, (K.shape(y_true)[0], -1, 6))
    lbd = y_true[..., 0]
    ly=K.sum(K.square(y_pred[..., 1]-y_true[..., 1])*lbd)
    lx=K.sum(K.square(y_pred[..., 2]-y_true[..., 2])*lbd)
    lt=K.sum(K.square(y_pred[..., 5]-y_true[..., 5])*lbd)
    la=K.sum(K.square(K.sqrt(y_pred[..., 3])-K.sqrt(y_true[..., 3]))*lbd)
    lb=K.sum(K.square(K.sqrt(y_pred[..., 4])-K.sqrt(y_true[..., 4]))*lbd)
    lno=K.sum(K.square(y_true[..., 0]-y_pred[..., 0])*(1-lbd))
    lp=K.sum(K.square(y_true[..., 0]-y_pred[..., 0])*lbd*10)
    return (ly+lx+lt+la+lb+lp+lno)/(K.cast(K.shape(y_true)[0],"float32")*K.cast(K.shape(y_true)[1],"float32"))           
            
def custom_loss_wmse(y_true, y_pred):
    """
    Weighted Mean Squared Error
    The weights are exp(k*y_true) where k > 0
    If there is not congestion (y_true=0), the weight is equal to 1
    If there is congestion (0<y_true<=1), the weight grows exponentially with y_true
    The model is forced to be accurate when the congestion is high (hot-spots)
    """
    temp = K.exp(5*y_true)
    weights = temp/K.sum(temp)
    loss=K.sum(K.square(y_pred - y_true)*weights)
    return loss

def custom_loss_tp(y_true, y_pred):
    """
    Mean Squared Error with coefficients for each variable to increase the errors
    """
    coeff = [1e1, 1e0, 1e1, 1e0, 1e1]
    dx2 = K.square(y_true[...,0] - y_pred[...,0])*coeff[0]
    dy2 = K.square(y_true[...,1] - y_pred[...,1])*coeff[1]
    dz2 = K.square(y_true[...,2] - y_pred[...,2])*coeff[2]
    dh2 = K.square(y_true[...,3] - y_pred[...,3])*coeff[3]
    dv2 = K.square(y_true[...,4] - y_pred[...,4])*coeff[4]
    loss=K.mean(dx2 + dy2 + dz2 + dh2 + dv2)
    return loss

def build_custom_loss_ml(model):
    """
    Build the negative log likelihood loss function
    """
    c = 5
    k = model.k_mixture
    def custom_loss_ml(y_true, y_pred):
        """
        Compute the negative log likelihood of gaussian mixture
        """
        splits = [k, k*c, k*c]

        phi, mu, sigma = tf.split(y_pred, num_or_size_splits=splits, axis=2)
        sigma_sq = sigma*sigma

        y_true = K.expand_dims(y_true, axis=3)
        mu = K.reshape(mu, [K.shape(y_true)[0], K.shape(y_true)[1], c, k])
        sigma_sq = K.reshape(sigma_sq, [K.shape(y_true)[0], K.shape(y_true)[1], c, k])
        dist = K.sum(K.square(y_true - mu)/sigma_sq, axis=2)

        exponent = K.log(phi) - (c/2.0)*np.log(2*np.pi) - (1/2.0)*K.sum(K.log(sigma_sq), axis=2) - (1/2.0)*dist

        return -K.sum(K.logsumexp(exponent, axis=2), axis=1)
    return custom_loss_ml

def AKF(model, X_, Cov_, X, Cov):
    """
    Adaptive Kalman Filter
    """
    dt = 10
    A = np.array([[1, 0, 0, dt, 0],
                  [0, 1, 0, 0, dt],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    Q = np.diag([1e-3, 1e-3, 1, 1e-6, 1e-6])
    
    for k in range(5):
        X[k] = X[k]*(model.ds.maxs[k] - model.ds.mins[k]) + model.ds.mins[k]
        X_[k] = X_[k]*(model.ds.maxs[k] - model.ds.mins[k]) + model.ds.mins[k]
        Cov[k, k] = Cov[k, k]*(model.ds.maxs[k] - model.ds.mins[k])**2
        Cov_[k, k] = Cov_[k, k]*(model.ds.maxs[k] - model.ds.mins[k])**2
    
    '''print("X", X)
    print("X_", X_)
    print("Cov", Cov)
    print("Cov_", Cov_, flush=True)'''
    # Predict
    X_pred = np.dot(A, X_)
    Cov_pred = np.dot(A, np.dot(Cov_, A.T)) + Q
    # Update
    S = Cov_pred + Cov
    K = np.dot(Cov_pred, np.linalg.inv(S))
    R = X - X_pred
    new_X = X_pred + np.dot(K, R)
    new_Cov = np.dot(np.eye(5) - K, Cov_pred)
    
    for k in range(5):
        X[k] = (X[k] - model.ds.mins[k])/(model.ds.maxs[k] - model.ds.mins[k])
        X_[k] = (X_[k] - model.ds.mins[k])/(model.ds.maxs[k] - model.ds.mins[k])
        Cov[k, k] = Cov[k, k]/(model.ds.maxs[k] - model.ds.mins[k])**2
        Cov_[k, k] = Cov_[k, k]/(model.ds.maxs[k] - model.ds.mins[k])**2
    
    return new_X, new_Cov

def compute_metric_wmse(y_true, y_pred):
    """
    Weighted MSE: grid cells with higher congestion have higher weights
    """
    #temp = (1 + y_true)**2
    temp = np.exp(5*y_true)
    weights = temp/np.sum(temp)
    metric = np.sum(weights*(y_pred - y_true)**2)
    print("MSE:", np.mean((y_pred-y_true)**2))
    print("Weighted MSE:", metric, flush=True)
    return metric

def compute_metric_tp(y_true, y_pred):
    """
    Compute the MSE of each variable
    """
    metric = [np.mean((y_true[...,i] - y_pred[...,i])**2) for i in range(5)]
    print("MSE for each variable:", metric, flush=True)
    return metric
    