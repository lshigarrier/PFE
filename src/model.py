from dataset_tsagi import *
from dataset_json import *
from utils_models import *

class CongestionModel:
    """
    Super class managing RNN models for congestion prediction
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
        self.ds = dataset
        self.name = name
        self.layers_list = layers_list
        self.directory = directory
        self.input_dim = input_seq_len
        self.output_dim = output_seq_len
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
        self.m_total = self.ds.y_tensor.shape[0] - self.input_dim - self.output_dim
            
    def create_layers(self):
        """
        Abstract method
        Create all the layers used in the models
        """
        raise NotImplementedError()
    
    def create_model(self, inference):
        """
        Abstract method
        Create all the models used for training and inference
        """
        raise NotImplementedError()
    
    def reshape_ytensor(self, reshape_type):
        """
        Reshape ds.y_tensor for training and inference
        """
        if reshape_type == 0:
            self.ds.y_tensor = np.reshape(self.ds.y_tensor, (self.ds.y_tensor.shape[0], self.ds.nb_steps*self.ds.nb_steps))
        elif reshape_type == 1:
            self.ds.y_tensor = np.reshape(self.ds.y_tensor, (self.ds.y_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps))
    
    def get_loss(self):
        """
        Return the training loss function
        """
        return self.loss

    def training(self):
        """
        Train the model
        """
        print("Start training:", self.name, "\nDataset size:", self.m_total, "\nInput dimension:", self.n_input, flush=True)
        
        self.reshape_ytensor(0)
        
        self.create_layers()
        self.create_model(inference=False)
        #self.full_model.summary()

        checkpoint_path = "../checkpoints/"+self.name+"-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        opt = Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, decay=self.learning_rate/self.nepochs)
        self.full_model.compile(optimizer=opt, loss=self.get_loss())

        if self.nb_splits != 0:

            self.full_model.save_weights(self.directory+'temp.h5')
            cv = KFold(n_splits = self.nsplits, shuffle=True, random_state=0)
            loss = np.zeros((self.nepochs,)).flatten()
            val_loss = np.zeros((self.nepochs,)).flatten()
            i = 1
            for self.train_index, self.val_index in cv.split(np.zeros((self.m_total, 1))):
                self.full_model.load_weights(self.directory+'temp.h5')
                history = self.full_model.fit_generator(generator=self.batch_generator(self, True),
                                              steps_per_epoch=(len(self.train_index)//self.batch_size+1),
                                              epochs=self.nepochs,
                                              validation_data=self.batch_generator(self, False),
                                              validation_steps=(len(self.val_index)//self.batch_size+1),
                                              callbacks=callbacks_list,
                                              verbose=1)
                loss += history.history['loss']
                val_loss += history.history['val_loss']

                plt.figure()
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Val'], loc='upper left')
                plt.savefig(self.directory+"model_loss_"+self.name+"_"+str(i)+".png")

                with open(self.directory+"train_loss_"+self.name+"_"+str(i), 'wb') as f:
                    pickle.dump(history.history['loss'], f)
                with open(self.directory+"val_loss_"+self.name+"_"+str(i), 'wb') as f:
                    pickle.dump(history.history['val_loss'], f)
                i += 1
            loss /= self.nsplits
            val_loss /= self.nsplits

        else:

            if self.train_val_split:
                self.m_train = int(self.m_total*0.9)
                index = np.arange(self.m_total)
                np.random.shuffle(index)
                self.train_index = index[:self.m_train]
                self.val_index = index[self.m_train:]
                with open(self.directory+"train_index_"+self.name, 'wb') as f:
                    pickle.dump(self.train_index, f)
                with open(self.directory+"val_index_"+self.name, 'wb') as f:
                    pickle.dump(self.val_index, f)
            else:
                with open(self.directory+"train_index_"+self.index_file, 'rb') as f:
                    self.train_index = pickle.load(f)
                with open(self.directory+"val_index_"+self.index_file, 'rb') as f:
                    self.val_index = pickle.load(f)
                self.m_train = len(self.train_index)
            
            history = self.full_model.fit_generator(generator=self.batch_generator(self, True),
                                          steps_per_epoch=(len(self.train_index)//self.batch_size+1),
                                          epochs=self.nepochs,
                                          validation_data=self.batch_generator(self, False),
                                          validation_steps=(len(self.val_index)//self.batch_size+1),
                                          callbacks=callbacks_list,
                                          verbose=1)

            model_json = self.full_model.to_json()
            with open(self.directory+"model_archi_"+self.name+".json", "w") as json_file:
                json_file.write(model_json)
            self.full_model.save_weights(self.directory+"model_weights_"+self.name+".h5")
            loss = history.history['loss']
            val_loss = history.history['val_loss']

        plt.figure()
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(self.directory+"model_loss_"+self.name+".png")

        with open(self.directory+"train_loss_"+self.name, 'wb') as f:
            pickle.dump(loss, f)
        with open(self.directory+"val_loss_"+self.name, 'wb') as f:
            pickle.dump(val_loss, f)
            
        self.reshape_ytensor(1)
    
    def predict(self, input_tensor):
        """
        Abstract method
        Return the predicted values
        """
        raise NotImplementedError()
        
    def reshape_prediction(self, pred_tensor, true_tensor):
        """
        Reshape the predicted and true tensors
        """
        pred_tensor = np.reshape(pred_tensor, (pred_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps))
        true_tensor = np.reshape(true_tensor, (true_tensor.shape[0], self.ds.nb_steps, self.ds.nb_steps))
        return pred_tensor, true_tensor
        
    def plot_predict(self, pred_tensor, true_tensor):
        """
        Plot the predicted and true values
        """
        return plot_predict_ytensor(pred_tensor, true_tensor)
    
    def prediction(self, use_val=True, plot_type=0, pred_dim=160):
        """
        Prediction using the validation set is use_val is True, the training set if False
        plot_type:
            0 -> plot_predict_ytensor
            1 -> plot_error_dist
        """
        print("Start inference:", self.name, "\nDataset size:", self.m_total, "\nInput dimension:", self.n_input, flush=True)
        self.reshape_ytensor(0)
        
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
            
        input_tensor = np.array([self.ds.x_tensor[index[t]:index[t]+self.input_dim] for t in range(len(index))])
        true_tensor = np.array([self.ds.y_tensor[index[t]+self.input_dim+pred_dim-1] for t in range(len(index))])
        pred_tensor = self.predict(input_tensor, pred_dim)
        
        pred_tensor, true_tensor = self.reshape_prediction(pred_tensor, true_tensor)
        self.reshape_ytensor(1)
        
        if plot_type == 0:
            fig, ax, slider = self.plot_predict(pred_tensor, true_tensor)
        elif plot_type == 1:
            fig, ax, slider = plot_error_dist(pred_tensor, true_tensor)        
        if self.ds.apply_binary:
            fig_, ax_ = binary_classification_analysis(pred_tensor, true_tensor)

        return fig, ax, slider
