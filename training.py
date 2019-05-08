from keras import Model
from keras.callbacks import CSVLogger,ModelCheckpoint,TensorBoard
from callbacks.evaluate_the_best import RunForEveryBest
from shutil import copyfile
from custom_metrics import map_metrics
import json
import os

class Training(object):

    def __init__(self,
                 model,
                 metrics,
                 optimizer,
                 loss,
                 run_name,
                 epochs=10,
                 batch_size=128,
                 change_details=None,
                 summaries_directory=None,
                 tensorboard_embedding_layernames = None,
                 tensorboard_embedding_metadata = None,
                 tensorboard_write_graph = False,
                 tensorboard_write_grad = False,
                 tensorboard_write_image = False,
                 evaluate_on_best_function = None,
                 evaluate_on_best_monitor_value = "val_loss",
                 evaluate_on_best_monitor_mode = "auto",
                 config_json_path = None
                 ):
        self.model = model
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
        self.run_name = run_name
        self.run_folder = os.path.join(self.summaries_directory,self.run_name)
        self.log_dir = os.path.join(self.run_folder,"logs")
        self._create_folder(summaries_directory)
        self._create_folder(self.run_folder)
        self.evaluate_on_best_function = evaluate_on_best_function
        self.evaluate_on_best_monitor_value = evaluate_on_best_monitor_value
        self.evaluate_on_best_monitor_mode = evaluate_on_best_monitor_mode
        self.metrics = map_metrics(metrics)
        self.optimizer = optimizer
        self.loss = loss
        self._persists_params(config_json_path) # let it be at last always
    
    def training_run_name(self):
        #TODO : Logic to form run name.
        return "Baseline"

    def _persists_params(self,config_json_path):
        if config_json_path is not None:
            copyfile(config_json_path,os.path.join(self.run_folder,"config.json"))
            # write summary to file
            with open(os.path.join(self.run_folder,'summary.txt'),'w') as fh:
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def callbacks(self):
        _callbacks = []
        _callbacks.append(CSVLogger(os.path.join(self.run_folder,'training_log.csv'), separator=',', append=True)) # Logger to log all the training log for each epoch.
        _callbacks.append(ModelCheckpoint(os.path.join(self.run_folder,'model.{epoch:02d}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5'), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=2)) # Saves all model for every best score.
        _callbacks.append(TensorBoard(log_dir=self.log_dir,write_grads=self.tensorboard_write_grad,write_graph=self.tensorboard_write_graph,write_images=self.tensorboard_write_image))
        if self.evaluate_on_best_function:
            # Callback to run evaluation for every best run.
            _callbacks.append(RunForEveryBest(evaluation_callback=self.evaluate_on_best_function,monitor=self.evaluate_on_best_monitor_value,mode=self.evaluate_on_best_monitor_mode,execute_on_best_only=True))
        return _callbacks
    
    def compile(self):
        pass
        #self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
    
    def train(self):  
        pass
        # self.compile()
        # self.model.fit(self.X_train,self.Y_train,batch_size=self.batch_size,epochs=self.epochs,callbacks=self.callbacks(),validation_data=(self.X_test,self.Y_test))
    
    def _create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    


class TrainingFitGenerator(Training):

    def __init__(self,
                model,
                train_generator,
                val_generator,
                steps_per_epoch,
                val_steps,
                metrics,
                optimizer,
                loss,
                run_name,
                epochs=10,
                batch_size=128,
                change_details=None,
                summaries_directory=None,
                tensorboard_embedding_layernames = None,
                tensorboard_embedding_metadata = None,
                tensorboard_write_graph = False,
                tensorboard_write_grad = False,
                tensorboard_write_image = False,
                evaluate_on_best_function = None,
                evaluate_on_best_monitor_value = "val_loss",
                evaluate_on_best_monitor_mode = "auto",
                config_json_path = None):
        

        super(TrainingFitGenerator,self).__init__(
                model,
                metrics,
                optimizer,
                loss,
                run_name,
                epochs,
                batch_size,
                change_details,
                summaries_directory,
                tensorboard_embedding_layernames,
                tensorboard_embedding_metadata,
                tensorboard_write_graph,
                tensorboard_write_grad,
                tensorboard_write_image,
                evaluate_on_best_function,
                evaluate_on_best_monitor_value,
                evaluate_on_best_monitor_mode,
                config_json_path)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        
    def train(self):
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        self.model.fit_generator(self.train_generator,
                                steps_per_epoch=self.steps_per_epoch,
                                validation_data=self.val_generator,
                                validation_steps=self.val_steps,
                                callbacks=self.callbacks(),
                                epochs=self.epochs,
                                max_queue_size=20,
                                workers=4)

class TrainingFit(Training):

    def __init__(self,
                model,
                x,
                y,
                validation_data,
                metrics,
                optimizer,
                loss,
                run_name,
                epochs=10,
                batch_size=128,
                change_details=None,
                summaries_directory=None,
                tensorboard_embedding_layernames = None,
                tensorboard_embedding_metadata = None,
                tensorboard_write_graph = False,
                tensorboard_write_grad = False,
                tensorboard_write_image = False,
                evaluate_on_best_function = None,
                evaluate_on_best_monitor_value = "val_loss",
                evaluate_on_best_monitor_mode = "auto",
                config_json_path = None):
        

        super(TrainingFit,self).__init__(
                model,
                metrics,
                optimizer,
                loss,
                run_name,
                epochs,
                batch_size,
                change_details,
                summaries_directory,
                tensorboard_embedding_layernames,
                tensorboard_embedding_metadata,
                tensorboard_write_graph,
                tensorboard_write_grad,
                tensorboard_write_image,
                evaluate_on_best_function,
                evaluate_on_best_monitor_value,
                evaluate_on_best_monitor_mode,
                config_json_path)
        self.train_X = x
        self.train_Y = y
        self.validation_data = validation_data
        
    def train(self):
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        self.model.fit(self.train_X,self.train_Y,
                                validation_data=self.validation_data,
                                callbacks=self.callbacks(),
                                epochs=self.epochs)

            

            
    

