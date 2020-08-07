from model import *

class ModelSeq2Vec(CongestionModel):
    """
    Class for Seq2Vec models
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
        self.batch_generator = batch_generator_seq2vec
            
    def create_layers(self):
        """
        GRU NOT MAINTAINED
        layers_list:
            layers_list[0] -> RNN layers ("LSTM" or "GRU", (hidden state dimensions tuple))
            layers_list[1] -> Dense layers ((output dimensions tuple), (activations tuple))
            layers_list[2] -> Conv2DTranspose layers ((filters tuple), (kernels tuple), (padding tuple), (activations tuple))
        """
        layers = {}
        
        self.nb_rnn = len(self.layers_list[0][1])
        for i in range(self.nb_rnn):
            if self.layers_list[0][0] == "GRU":
                layers["rnn"+str(i)] = GRU(self.layers_list[0][1][i], return_sequences=(i!=self.nb_rnn-1), kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
            else:
                layers["rnn"+str(i)] = LSTM(self.layers_list[0][1][i], return_sequences=(i!=self.nb_rnn-1), kernel_regularizer=l2(self.l2_lambd), recurrent_regularizer=l2(self.l2_lambd))
        self.nb_dense = len(self.layers_list[1][0])
        for i in range(self.nb_dense):
            layers["dense"+str(i)] = Dense(self.layers_list[1][0][i], activation=self.layers_list[1][1][i], kernel_regularizer=l2(self.l2_lambd))
        self.nb_convtrans = len(self.layers_list[2][0])
        for i in range(self.nb_convtrans):
            layers["convtrans"+str(i)] = Conv2DTranspose(self.layers_list[2][0][i], self.layers_list[2][1][i], strides=2, padding="valid", output_padding=self.layers_list[2][2][i], activation=self.layers_list[2][3][i], kernel_regularizer=l2(self.l2_lambd))
        
        self.layers = layers
    
    def create_model(self, inference):
        """
        Build the model
        """
        X_input = Input(shape=(self.input_dim, self.n_input), name='X_input')
        X = X_input
        for i in range(self.nb_rnn):
            X = self.layers["rnn"+str(i)](X)
        for i in range(self.nb_dense):
            X = self.layers["dense"+str(i)](X)
        for i in range(self.nb_convtrans):
            if i == 0:
                dim = int(np.sqrt(self.layers_list[1][0][-1]))
                X = Reshape((dim,dim,1))(X)
            X = self.layers["convtrans"+str(i)](X)
            if i == self.nb_convtrans - 1:
                X_output = Reshape((self.ds.nb_steps*self.ds.nb_steps,))(X)
        self.full_model = Model(X_input, X_output)
    
    def get_loss(self):
        """
        Return the training loss function
        """
        if self.loss == "weighted_mse":
            return custom_loss_wmse
        else:
            return super().get_loss()
        
    def predict(self, input_tensor, pred_dim):
        """
        Return the predicted values
        """
        return self.full_model.predict(input_tensor)

    
