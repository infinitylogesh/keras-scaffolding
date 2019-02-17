from keras import Model
from keras.callbacks import CSVLogger,ModelCheckpoint,TensorBoard
import os

class Training(object):

    def __init__(self,
                 model,
                 X_train,
                 Y_train,
                 X_test,
                 Y_test,
                 epochs=10,
                 batch_size=128,
                 summaries_directory=None,
                 tensorboard_embedding_layernames = None,
                 tensorboard_embedding_metadata = None,
                 tensorboard_write_graph = False,
                 tensorboard_write_grad = False,
                 tensorboard_write_image = False
                 ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.batch_size = batch_size
        self.epochs  = epochs
        if summaries_directory is None:
            summaries_directory = "./summaries"
        self.summaries_directory = summaries_directory
        self.tensorboard_embedding_layernames = tensorboard_embedding_layernames
        self.tensorboard_embedding_metadata = tensorboard_embedding_metadata
        self.tensorboard_write_graph = tensorboard_write_graph
        self.tensorboard_write_grad = tensorboard_write_grad
        self.tensorboard_write_image = tensorboard_write_image
        self.run_name = self.training_run_name()
        self.run_folder = os.path.join(self.summaries_directory,self.run_name)
        self.log_dir = os.path.join(self.run_folder,"logs")
        self._create_folder(summaries_directory)
        self._create_folder(self.run_folder)
    
    def training_run_name(self):
        #TODO : Logic to form run name.
        return "default_run_name"

    def callbacks(self):
        csvlogger = CSVLogger(os.path.join(self.run_folder,'training_log.csv'), separator=',', append=True) # Logger to log all the training log for each epoch.
        checkpoint = ModelCheckpoint(os.path.join(self.run_folder,'model.{epoch:02d}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5'), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
        tensorboard = TensorBoard(log_dir=self.log_dir,write_grads=self.tensorboard_write_grad,write_graph=self.tensorboard_write_graph,write_images=self.tensorboard_write_image)
        #TODO : add Evaluate best model custom callback
        return [csvlogger,checkpoint,tensorboard]
    
    def train(self):
        self.model.fit(self.X_train,self.Y_train,batch_size=self.batch_size,epochs=self.epochs,callbacks=self.callbacks(),validation_data=(self.X_test,self.Y_test))
    
    def _create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    


