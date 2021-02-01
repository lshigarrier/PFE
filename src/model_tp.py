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
             index_file="",
             k_mixture = 3):
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
        self.k_mixture = k_mixture
        
    def create_layers(self):
        """
        layers_list:
            layers_list[0] -> RNN encoder ((hidden state dimensions tuple))
            layers_list[1] -> RNN decoder ((hidden state dimensions tuple))
            if self.k_mixture <= 0:
                layers_list[2] -> Dense layers ((output dimensions tuple), (activations tuple))
        """
        layers = {}
                
        self.nb_rnn_enc = len(self.layers_list[0])
        for i in range(self.nb_rnn_enc):
            layers["rnn_encoder"+str(i)] = LSTM(self.layers_list[0][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_rnn_dec = len(self.layers_list[1])
        for i in range(self.nb_rnn_dec):
            layers["rnn_decoder"+str(i)] = LSTM(self.layers_list[1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        if self.k_mixture <= 0:
            self.nb_dense = len(self.layers_list[2][0])
            for i in range(self.nb_dense):
                layers["dense"+str(i)] = Dense(self.layers_list[2][0][i], activation=self.layers_list[2][1][i], kernel_regularizer=l2(self.l2_lambd))
        else:
            layers["dense_mixture"] = Dense(self.k_mixture, activation="softmax", kernel_regularizer=l2(self.l2_lambd))
            layers["dense_mean"] = Dense(self.k_mixture*(self.ds.state_dim), activation="elu", kernel_regularizer=l2(self.l2_lambd))
            layers["dense_var"] = Dense(self.k_mixture*(self.ds.state_dim), activation="elu", kernel_regularizer=l2(self.l2_lambd))
        
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
            #encoder_states.append(state_h)
            #encoder_states.append(state_c)
            if i == self.nb_rnn_enc-1:
                for i in range(self.nb_rnn_enc):
                    encoder_states.append(state_h)
                    encoder_states.append(state_c)
        decoder_inputs = Input(shape=(self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2]), name='decoder_inputs')
        #decoder_inputs = Input(shape=(self.ds.y_tensor.shape[1], 2), name='decoder_inputs')
        X = decoder_inputs
        for i in range(self.nb_rnn_dec):
            #X, _, _ = self.layers["rnn_decoder"+str(i)](X, initial_state=[encoder_states[2*i],encoder_states[2*i+1]])
            X, _, _ = self.layers["rnn_decoder"+str(i)](X, initial_state=[encoder_states[0],encoder_states[1]])
        if self.k_mixture <= 0:
            for i in range(self.nb_dense):
                X = TimeDistributed(self.layers["dense"+str(i)])(X)
            outputs = X
        else:
            phi = TimeDistributed(self.layers["dense_mixture"])(X)
            mu = TimeDistributed(self.layers["dense_mean"])(X)
            sigma = TimeDistributed(self.layers["dense_var"])(X)
            outputs = Concatenate(axis=-1)([phi, mu, sigma])
        self.full_model = Model([X_encoder, decoder_inputs], outputs)
        self.encoder_model = Model(X_encoder, encoder_states)
    
    '''def training_model(self):
        """
        Build the encoder model and the full model
        """
        X_encoder = Input(shape=(self.ds.x_tensor.shape[1], self.ds.x_tensor.shape[2]), name='X_encoder')
        X = X_encoder
        encoder_states = []
        for i in range(self.nb_rnn_enc):
            X, state_h, state_c = self.layers["rnn_encoder"+str(i)](X)
            if i == self.nb_rnn_enc-1:
                for i in range(self.nb_rnn_enc):
                    encoder_states.append(state_h)
                    encoder_states.append(state_c)
        decoder_inputs = Input(shape=(self.ds.y_tensor.shape[1], 2), name='decoder_inputs')
        decoder_states = []
        for i in range(self.nb_rnn_dec):
            decoder_states.append(encoder_states[0])
            decoder_states.append(encoder_states[1])
        for t in range(self.ds.y_tensor.shape[1]):
            X = Lambda(lambda x : x[:,t,:])(decoder_inputs)
            X = Reshape((1, 2))(X)
            for i in range(self.nb_rnn_dec):
                X, state_h, state_c = self.layers["rnn_decoder"+str(i)](X, initial_state=[decoder_states[2*i], decoder_states[2*i+1]])
                decoder_states[2*i] = state_h
                decoder_states[2*i+1] = state_c
            for i in range(self.nb_dense):
                X = TimeDistributed(self.layers["dense"+str(i)])(X)
            if t==0:
                outputs = X
            else:
                outputs = Concatenate(axis=1)([outputs, X])
        self.full_model = Model([X_encoder, decoder_inputs], outputs)
        self.encoder_model = Model(X_encoder, encoder_states)'''
        
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
            X, states_h, states_c = self.layers["rnn_decoder"+str(i)](X, initial_state=[decoder_states[2*i], decoder_states[2*i+1]])
            decoder_states_outputs.append(states_h)
            decoder_states_outputs.append(states_c)
        if self.k_mixture <= 0:
            for i in range(self.nb_dense):
                X = TimeDistributed(self.layers["dense"+str(i)])(X)
            outputs = X
        else:
            phi = TimeDistributed(self.layers["dense_mixture"])(X)
            mu = TimeDistributed(self.layers["dense_mean"])(X)
            sigma = TimeDistributed(self.layers["dense_var"])(X)
            outputs = Concatenate(axis=-1)([phi, mu, sigma])
        
        self.decoder_model = Model([decoder_inputs]+decoder_states, [outputs]+decoder_states_outputs)
        
    def create_model(self, inference):
        """
        Create the full_model, the encoder_model and the decoder_model
        """
        self.training_model()
        if inference:
            self.inference_model()
            
    def get_loss(self):
        """
        Return the training loss function
        """
        if self.loss == "tp":
            return custom_loss_tp
        elif self.loss == "ml":
            return build_custom_loss_ml(self)
        else:
            return super().get_loss()
            
    def predict(self, input_seq, true_tensor, wind_tensor, pred_dim=10):
        """
        Recursively decode a sequence given the inputs of the encoder model and the wind data
        For the first pred_dim steps, the true_tensor values are given as inputs
        """
        if self.k_mixture <= 0:
            
            #full_output = self.full_model.predict([input_seq, np.concatenate((np.zeros((true_tensor.shape[0], 1, true_tensor.shape[2]+2)), np.concatenate((true_tensor[:,:-1,:], wind_tensor[:,:-1,:]), axis=-1)), axis = 1)])
            bs = input_seq.shape[0]
            states_value = self.encoder_model.predict(input_seq)
            full_output = np.zeros((bs, self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2] - 2))
            target_seq = np.zeros((bs, 1, self.ds.y_tensor.shape[2]))
            for t in range(self.ds.y_tensor.shape[1]):
                out = self.decoder_model.predict([target_seq]+states_value)
                outputs = out[0]
                target_seq[:, 0, :self.ds.state_dim] = true_tensor[:, t, :] if t < pred_dim else outputs[:, 0, :]
                target_seq[:, 0, self.ds.state_dim:] = wind_tensor[:, t, :]
                states_value = out[1:]
                full_output[:, t, :] = outputs[:, 0, :]
            return full_output
                
        else:
            
            pi1 = 0.8
            pi2 = 0.2
            Nbs = self.k_mixture**2
            states_value = self.encoder_model.predict(input_seq)
            full_outputs = {-1e9 : [np.zeros((1, self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2] - 2)),
                                    np.zeros((1, 1, self.ds.y_tensor.shape[2])),
                                    states_value,
                                    np.eye(self.ds.state_dim)]}
            for t in range(self.ds.y_tensor.shape[1]):
                temp_dict = {}
                for (L, full_output) in full_outputs.items():
                    traj = full_output[0]
                    target_seq = full_output[1]
                    states_value = full_output[2]
                    Cov_ = full_output[3]

                    out = self.decoder_model.predict([target_seq]+states_value)
                    outputs = out[0]
                    phi = outputs[0, 0, :self.k_mixture]
                    mu = outputs[0, 0, self.k_mixture:self.k_mixture*6]
                    sigma = outputs[0, 0, self.k_mixture*6:]**2

                    if t >= pred_dim:
                        for i in range(self.k_mixture):
                            Cov = np.diag(sigma[i*self.ds.state_dim:(i+1)*self.ds.state_dim]**2)
                            new_X, new_Cov = AKF(self, traj[0, t-1, :], Cov_, mu[i*self.ds.state_dim:(i+1)*self.ds.state_dim], Cov)
                            #new_X = mu[i*self.ds.state_dim:(i+1)*self.ds.state_dim]
                            #new_Cov = Cov
                            new_traj = np.zeros((1, self.ds.y_tensor.shape[1], self.ds.y_tensor.shape[2] - 2))
                            new_traj[:, :t, :] = traj[:, :t, :]
                            new_traj[0, t, :] = new_X
                            new_target_seq = np.zeros((1, 1, self.ds.y_tensor.shape[2]))
                            new_target_seq[0, 0, :self.ds.state_dim] = new_X
                            new_target_seq[0, 0, self.ds.state_dim:] = wind_tensor[0, t, :]
                            new_states_value = out[1:]
                            new_sigma = np.diagonal(new_Cov)
                            new_L = L + pi1*np.log(phi[i]) + pi2*(-np.sum(np.log(new_sigma)))
                            temp_dict[new_L] = [new_traj, new_target_seq, new_states_value, new_Cov]
                        L_list = temp_dict.keys()
                        if len(L_list) > Nbs:
                            L_list = sorted(L_list, reverse=True)
                            for i in range(Nbs+1, len(L_list)):
                                del temp_dict[L_list[i]]

                    else:
                        traj[:, t, :] = true_tensor[:, t, :]
                        target_seq[:, 0, :self.ds.state_dim] = true_tensor[:, t, :]
                        target_seq[:, 0, self.ds.state_dim:] = wind_tensor[:, t, :]
                        states_value = out[1:]
                        Cov_ = np.diag(sigma[:self.ds.state_dim]**2)
                        temp_dict[L] = [traj, target_seq, states_value, Cov_]
                full_outputs = copy.deepcopy(temp_dict)

            L_list = full_outputs.keys()
            Lmax = max(L_list)
            return full_outputs[Lmax][0]
    
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
    
    def prediction(self, use_val=True, pred_dim=10):
        """
        Prediction using the validation set is use_val is True, the training set if False
        """
        print("Start inference:", self.name, "\nDataset size:", self.m_total, "\nInput dimension:", self.n_input, flush=True)
        
        #with open(self.directory+"model_archi_"+self.index_file+".json", 'r') as json_file:
        #    test_model = model_from_json(json_file.read())
        #test_model.summary()
        
        weights_path = self.directory+"model_weights_"+self.index_file+".h5"
        #weights_path = self.directory+"TP_ML_2LSTM512_2LSTM512_500epochs-14--97.83.hdf5"
        if use_val:
            index_path = self.directory+"val_index_"+self.index_file
        else:
            index_path = self.directory+"train_index_"+self.index_file
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        
        self.create_layers()
        self.create_model(inference=True)
        
        #self.full_model.summary()
        
        self.full_model.load_weights(weights_path)
        
        if self.k_mixture <= 0:
            input_tensor = np.array([self.ds.x_tensor[self.ds.index_tensor[index[t]]] for t in range(len(index))])
            true_tensor = np.array([self.ds.y_tensor[index[t],:,:self.ds.state_dim] for t in range(len(index))])
            wind_tensor = np.array([self.ds.y_tensor[index[t],:,self.ds.state_dim:] for t in range(len(index))])
        else:
            input_tensor = np.array([self.ds.x_tensor[self.ds.index_tensor[index[t]]] for t in range(1)])
            true_tensor = np.array([self.ds.y_tensor[index[t],:,:self.ds.state_dim] for t in range(1)])
            wind_tensor = np.array([self.ds.y_tensor[index[t],:,self.ds.state_dim:] for t in range(1)])
        #pred_tensor = self.predict(input_tensor, true_tensor, wind_tensor, pred_dim)
        pred_tensor = self.full_model.predict([input_tensor, wind_tensor])
        
        compute_metric_tp(true_tensor, pred_tensor)
        
        for k in range(self.ds.state_dim):
            #print("k=", k, self.ds.mins[k], self.ds.maxs[k])
            pred_tensor[...,k] = pred_tensor[...,k]*(self.ds.maxs[k] - self.ds.mins[k]) + self.ds.mins[k]
            true_tensor[...,k] = true_tensor[...,k]*(self.ds.maxs[k] - self.ds.mins[k]) + self.ds.mins[k]
        
        compute_metric_tp(true_tensor, pred_tensor)
        
        fig, ax, slider = self.plot_predict(pred_tensor, true_tensor)
        #fig, ax, slider = plot_predict_profile(pred_tensor, true_tensor)
        #fig, ax = plot_all_predict_profile(self, index, pred_tensor, true_tensor)
        #slider = 0

        return fig, ax, slider