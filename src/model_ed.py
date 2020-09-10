from model import *

class ModelEncoderDecoder(CongestionModel):
    """
    Class for encoder-decoder RNN models
    """
    
    def __init__(self,
                 dataset,
                 name,
                 layers_list,
                 directory="",
                 input_seq_len=160,
                 output_seq_len=160,
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
        super().__init__(
                 dataset=dataset,
                 name=name,
                 layers_list=layers_list,
                 directory=directory,
                 input_seq_len=input_seq_len,
                 output_seq_len=output_seq_len,
                 batch_size=batch_size,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 beta_1=beta_1,
                 beta_2=beta_2,
                 loss=loss,
                 l2_lambd=l2_lambd,
                 nb_splits=nb_splits,
                 train_val_split=train_val_split,
                 index_file=index_file)
        self.batch_generator = batch_generator_seq2seq
            
    def create_layers(self):
        """
        GRU NOT MAINTAINED
        layers_list:
            layers_list[0] -> Conv1D layers ((filters tuple), (kernels tuple), (activations tuple))
            layers_list[1] -> RNN encoder ("LSTM" or "GRU", (hidden state dimensions tuple))
            layers_list[2] -> RNN decoder ("LSTM" or "GRU", (hidden state dimensions tuple))
            layers_list[3] -> Dense layers
            layers_list[4] -> Conv2DTranspose layers
        """
        layers = {}
                
        self.nb_conv = len(self.layers_list[0][0])
        for i in range(self.nb_conv):
            layers["conv"+str(i)] = Conv1D(self.layers_list[0][0][i], self.layers_list[0][1][i], padding='same', activation=self.layers_list[0][2][i], strides=1)
        self.nb_rnn_enc = len(self.layers_list[1][0])
        for i in range(self.nb_rnn_enc):
            if self.layers_list[1][0] == "GRU":
                layers["rnn_encoder"+str(i)] = GRU(self.layers_list[1][1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
            else:
                layers["rnn_encoder"+str(i)] = LSTM(self.layers_list[1][1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_rnn_dec = len(self.layers_list[2][0])
        for i in range(self.nb_rnn_dec):
            if self.layers_list[2][0] == "GRU":
                layers["rnn_decoder"+str(i)] = GRU(self.layers_list[2][1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
            else:
                layers["rnn_decoder"+str(i)] = LSTM(self.layers_list[2][1][i], return_sequences=True, return_state=True, kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_dense = len(self.layers_list[3][0])
        for i in range(self.nb_dense):
            layers["dense"+str(i)] = Dense(self.layers_list[3][0][i], activation=self.layers_list[3][1][i], kernel_regularizer=l2(self.l2_lambd))
        self.nb_convtrans = len(self.layers_list[4][0])
        for i in range(self.nb_convtrans):
            layers["convtrans"+str(i)] = Conv2DTranspose(self.layers_list[4][0][i], self.layers_list[4][1][i], strides=2, padding="valid", output_padding=self.layers_list[4][2][i], activation=self.layers_list[4][3][i], kernel_regularizer=l2(self.l2_lambd))
        
        self.layers = layers
        
    def training_model(self):
        """
        Build the encoder model and the full model
        """
        X_encoder = Input(shape=(self.input_dim, self.n_input), name='X_encoder')
        X = X_encoder
        for i in range(self.nb_conv):
            X = self.layers["conv"+str(i)](X)
        encoder_states = []
        for i in range(self.nb_rnn_enc):
            X, state_h, state_c = self.layers["rnn_encoder"+str(i)](X)
            encoder_states.append(state_h)
            encoder_states.append(state_c)
        if self.ds.apply_yolo:
            decoder_inputs = Input(shape=(self.output_dim, self.ds.nb_steps*self.ds.nb_steps*6), name='decoder_inputs') 
        else:
            decoder_inputs = Input(shape=(self.output_dim, self.ds.nb_steps*self.ds.nb_steps), name='decoder_inputs') 
        X = decoder_inputs
        for i in range(self.nb_rnn_dec):
            X, _, _ = self.layers["rnn_decoder"+str(i)](X, initial_state=[encoder_states[2*(i-1)],encoder_states[2*(i-1)+1]])
        for i in range(self.nb_dense):
            X = TimeDistributed(self.layers["dense"+str(i)])(X)
        for i in range(self.nb_convtrans):
            if i == 0:
                dim = int(np.sqrt(self.layers_list[3][0][-1]))
                X = Reshape((self.output_dim, dim, dim, 1))(X)
            X = TimeDistributed(self.layers["convtrans"+str(i)])(X)
            if i == self.nb_convtrans - 1:
                X = Reshape((self.output_dim, self.ds.nb_steps*self.ds.nb_steps,))(X)
        outputs = X
        self.full_model = Model([X_encoder, decoder_inputs], outputs)
        self.encoder_model = Model(X_encoder, encoder_states)
    
    def inference_model(self):
        """
        Build the inference model
        """
        if self.ds.apply_yolo:
            decoder_inputs = Input(shape=(None, self.ds.nb_steps*self.ds.nb_steps*6), name='decoder_inputs')
        else:
            decoder_inputs = Input(shape=(None, self.ds.nb_steps*self.ds.nb_steps), name='decoder_inputs')
        decoder_states = []
        for i in range(self.nb_rnn_dec):
            decoder_states.append(Input(shape=(self.layers_list[2][1][i],)))
            decoder_states.append(Input(shape=(self.layers_list[2][1][i],)))
        decoder_states_outputs = []
        X = decoder_inputs
        for i in range(self.nb_rnn_dec):
            X, states_h, states_c = self.layers["rnn_decoder"+str(i)](X, initial_state=[decoder_states[2*(i-1)], decoder_states[2*(i-1)+1]])
            decoder_states_outputs.append(states_h)
            decoder_states_outputs.append(states_c)
        for i in range(self.nb_dense):
            X = TimeDistributed(self.layers["dense"+str(i)])(X)
        for i in range(self.nb_convtrans):
            if i == 0:
                dim = int(np.sqrt(self.layers_list[3][0][-1]))
                X = Reshape((-1, dim, dim, 1))(X)
            X = TimeDistributed(self.layers["convtrans"+str(i)])(X)
            if i == self.nb_convtrans - 1:
                X = Reshape((-1, self.ds.nb_steps*self.ds.nb_steps,))(X)
        outputs = X
        self.decoder_model = Model([decoder_inputs]+decoder_states, [outputs]+decoder_states_outputs)
    
    def create_model(self, inference):
        """
        Create the full_model, the encoder_model and the decoder_model
        """
        self.training_model()
        if inference:
            self.inference_model()
            
    def reshape_ytensor(self, reshape_type):
        """
        Reshape ds.y_tensor for training and inference
        """
        if self.ds.apply_yolo:
            if reshape_type == 0:
                self.ds.y_tensor = np.reshape(self.ds.y_tensor, (self.ds.y_tensor.shape[0], self.ds.nb_steps*self.ds.nb_steps*6))
            elif reshape_type == 1:
                self.ds.y_tensor = np.reshape(self.ds.y_tensor, (self.ds.y_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps, 6))
        else:
            super().reshape_ytensor(reshape_type)
    
    def get_loss(self):
        """
        Return the training loss function
        """
        if self.loss == "yolo":
            return custom_loss_yolo
        else:
            return super().get_loss()
     
    def predict(self, input_seq, pred_dim):
        """
        Recursively decode a sequence given the inputs of the encoder model
        """
        bs = input_seq.shape[0]
        states_value = self.encoder_model.predict(input_seq)
        if self.ds.apply_yolo:
            full_output = np.zeros((bs, pred_dim, self.ds.nb_steps*self.ds.nb_steps*6))
            target_seq = np.zeros((bs, 1, self.ds.nb_steps*self.ds.nb_steps*6))
        else:
            full_output = np.zeros((bs, pred_dim, self.ds.nb_steps*self.ds.nb_steps))
            target_seq = np.zeros((bs, 1, self.ds.nb_steps*self.ds.nb_steps))
        for t in range(pred_dim):
            out = self.decoder_model.predict([target_seq]+states_value)
            outputs = out[0]
            target_seq[:, 0, :] = outputs[:, 0, :]
            states_values = out[1:]
            full_output[:, t, :] = outputs[:, 0, :]
        return full_output
    
    def reshape_prediction(self, pred_tensor, true_tensor):
        """
        Reshape the predicted and true tensors
        """
        pred_tensor = pred_tensor[:, -1, :]
        if self.ds.apply_yolo:
            pred_tensor = np.reshape(pred_tensor, (pred_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps, 6))
            true_tensor = np.reshape(true_tensor, (true_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps, 6))
        else:
            pred_tensor, true_tensor = super().reshape_prediction(pred_tensor, true_tensor)
        return pred_tensor, true_tensor
        
    def plot_predict(self, pred_tensor, true_tensor):
        """
        Plot the predicted and true values
        """
        if self.ds.apply_yolo:
            return plot_predict_clusters(pred_tensor, true_tensor)
        else:
            return super().plot_predict(pred_tensor, true_tensor)
