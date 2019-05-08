from pipeline import Pipeline
from training import TrainingFit
from datasetOps import DatasetOps
import numpy as np
import random as rn
import math
import os
import keras
import os
import sys
from keras.datasets import mnist
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
from model import baseline_model

class experiment(Pipeline):

    def __init__(self,model,config_file):
        super(experiment,self).__init__(model,config_file)
    
    def run(self):
        (X_train, y_train), (X_test, y_test)= DatasetOps().fetch_data()
        # print(self.mdl.summary())
        TrainingFit(model=self.model,
                 x=np.expand_dims(X_train,axis=-1),
                 y=y_train,
                 validation_data=(np.expand_dims(X_test,-1),y_test),
                 optimizer=keras.optimizers.RMSprop(lr=1e-4),
                 loss=self.params["loss"],
                 metrics=self.params["metrics"],
                 epochs=self.params["epochs"],
                 summaries_directory=self.params["summaries_directory"],
                 tensorboard_write_grad=True,
                 run_name="test",
                 config_json_path=self.config_file
                 ).train()
        return self.model

if __name__ == "__main__":
    experiment(baseline_model(),"config/config.json").run()
    


