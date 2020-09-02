from model import *

class ModelTP(CongestionModel):
    """
    Class for Trajectory Prediction models
    """
    
    def __init__(self,
             dataset,
             name,
             layers_list,
             directory="",
             batch_size=256,
             epochs=200,
             learning_rate=1e-3,
             beta_1=0.9,
             beta_2=0.999,
             loss='mean_squared_error',
             l2_lambd=0,
             nb_splits=0,
             train_val_split=True,
             index_file=""):
        """
        nb_splits: number of splits in the k fold validation, 0 if k fold validation is not used
        train_val_split:
            True -> creates its own training/validation split
            False -> uses the split saved in index_file 
        """
        self.ds = dataset
        self.name = name
        self.layers_list = layers_list
        self.directory = directory
        self.batch_size = batch_size
        self.nepochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss = loss
        self.l2_lambd = l2_lambd
        self.nb_splits = nb_splits
        self.train_val_split = train_val_split
        if self.train_val_split:
            self.index_file = self.name
        else:
            self.index_file = self.index_file
        self.n_input = self.ds.x_tensor.shape[1]
        self.m_total = self.ds.y_tensor.shape[0]
        self.batch_generator = batch_generator_tp
        
    def create_layers(self):
        """
        layers_list:
            layers_list[0] -> RNN encoder ((hidden state dimensions tuple))
            layers_list[1] -> RNN decoder ((hidden state dimensions tuple))
            layers_list[2] -> Dense layers ((output dimensions tuple), (activations tuple))
        """
        layers = {}
                
        self.nb_rnn_enc = len(self.layers_list[0])
        for i in range(self.nb_rnn_enc):
            layers["rnn_encoder"+str(i)] = LSTM(self.layers_list[0][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_rnn_dec = len(self.layers_list[1])
        for i in range(self.nb_rnn_dec):
            layers["rnn_decoder"+str(i)] = LSTM(self.layers_list[1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_dense = len(self.layers_list[2][0])
        for i in range(self.nb_dense):
            layers["dense"+str(i)] = Dense(self.layers_list[2][0][i], activation=self.layers_list[2][1][i], kernel_regularizer=l2(self.l2_lambd))
        
        self.layers = layers
        
    def training_model(self):
        """
        Build the encoder model and the full model
        """
        X_encoder = Input(shape=(self.ds.x_tensor.shape[1], self.ds.x_tensor.shape[2]), name='X_encoder')
        X = X_encoder
        encoder_states = []
        for i in range(self.nb_rnn_enc):
            X, state_h, state_c = self.layers["rnn_encoder"+str(i)](X)
            encoder_states.append(state_h)
            encoder_states.append(state_c)
        decoder_inputs = Input(shape=(self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2]), name='decoder_inputs') 
        X = decoder_inputs
        for i in range(self.nb_rnn_dec):
            X, _, _ = self.layers["rnn_decoder"+str(i)](X, initial_state=[encoder_states[2*(i-1)],encoder_states[2*(i-1)+1]])
        for i in range(self.nb_dense):
            X = TimeDistributed(self.layers["dense"+str(i)])(X)
        outputs = X
        self.full_model = Model([X_encoder, decoder_inputs], outputs)
        self.encoder_model = Model(X_encoder, encoder_states)
        
    def inference_model(self):
        """
        Build the inference model
        """
        decoder_inputs = Input(shape=(None, self.ds.y_tensor.shape[2]), name='decoder_inputs')
        decoder_states = []
        for i in range(self.nb_rnn_dec):
            decoder_states.append(Input(shape=(self.layers_list[1][i],)))
            decoder_states.append(Input(shape=(self.layers_list[1][i],)))
        decoder_states_outputs = []
        X = decoder_inputs
        for i in range(self.nb_rnn_dec):
            X, states_h, states_c = self.layers["rnn_decoder"+str(i)](X, initial_state=[decoder_states[2*(i-1)], decoder_states[2*(i-1)+1]])
            decoder_states_outputs.append(states_h)
            decoder_states_outputs.append(states_c)
        for i in range(self.nb_dense):
            X = TimeDistributed(self.layers["dense"+str(i)])(X)
        outputs = X
        self.decoder_model = Model([decoder_inputs]+decoder_states, [outputs]+decoder_states_outputs)
        
    def create_model(self, inference):
        """
        Create the full_model, the encoder_model and the decoder_model
        """
        self.training_model()
        if inference:
            self.inference_model()
            
    def predict(self, input_seq, wind_tensor):
        """
        Recursively decode a sequence given the inputs of the encoder model
        """
        bs = input_seq.shape[0]
        states_value = self.encoder_model.predict(input_seq)
        full_output = np.zeros((bs, self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2] - 2))
        target_seq = np.zeros((bs, 1, self.ds.y_tensor.shape[2]))
        for t in range(self.ds.y_tensor.shape[1]):
            out = self.decoder_model.predict([target_seq]+states_value)
            outputs = out[0]
            target_seq = np.zeros((bs, 1, self.ds.y_tensor.shape[2]))
            target_seq[:, 0, :5] = outputs[:, 0, :]
            target_seq[:, 0, -2:] = wind_tensor[:, t, :]
            states_values = out[1:]
            full_output[:, t, :] = outputs[:, 0, :]
        return full_output
    
    def reshape_ytensor(self, reshape_type):
        """
        Do nothing (for compatibility)
        """
        pass
    
    def plot_predict(self, pred_tensor, true_tensor):
        """
        Plot the predicted and true values
        """
        return plot_predict_tp(pred_tensor, true_tensor)
    
    def prediction(self, use_val=True):
        """
        Prediction using the validation set is use_val is True, the training set if False
        """
        print("Start inference:", self.name, "\nDataset size:", self.m_total, "\nInput dimension:", self.n_input, flush=True)
        
        weights_path = self.directory+"model_weights_"+self.index_file+".h5"
        if use_val:
            index_path = self.directory+"val_index_"+self.index_file
        else:
            index_path = self.directory+"train_index_"+self.index_file
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        
        self.create_layers()
        self.create_model(inference=True)
        self.full_model.load_weights(weights_path)
            
        input_tensor = np.array([self.ds.x_tensor[self.ds.index_tensor[index[t]]] for t in range(len(index))])
        true_tensor = np.array([self.ds.y_tensor[index[t],:,:5] for t in range(len(index))])
        wind_tensor = np.array([self.ds.y_tensor[index[t],:,5:] for t in range(len(index))])
        pred_tensor = self.predict(input_tensor, wind_tensor)
        
        for k in range(5):
            pred_tensor[...,k] = pred_tensor[...,k]*(self.ds.maxs[k] - self.ds.mins[k]) + self.ds.mins[k]
            true_tensor[...,k] = true_tensor[...,k]*(self.ds.maxs[k] - self.ds.mins[k]) + self.ds.mins[k]
        
        print(np.mean(np.sqrt(np.sum((pred_tensor[...,:3]-true_tensor[...,:3])**2, axis=-1)))*60, flush=True)
        
        fig, ax, slider = self.plot_predict(pred_tensor, true_tensor)

        return fig, ax, slider