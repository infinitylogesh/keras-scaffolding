from pipeline import Pipeline
from training import Training
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
from hyperopt import hp,fmin, tpe, space_eval,STATUS_OK,Trials


class experiment(Pipeline):

    def __init__(self,model,config_file):
        super(experiment,self).__init__(model,config_file)
    
    def fetch_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        y_train = [ np.eye(9)[val-1] for val in y_train]
        y_test = [np.eye(9)[val-1] for val in y_test]
        return np.expand_dims(X_train,axis=-1), np.array(y_train), np.expand_dims(X_test,axis=-1), np.array(y_test)
    
    def _wrapper(self,args):
        acc =  Training(model=self.model,
                            X_train=self.X_train,
                            Y_train=self.y_train,
                            X_test=self.X_test,
                            Y_test=self.y_test,
                            optimizer=args['optimizer'],
                            loss=args['loss'],
                            metrics=args['metrics'],
                            epochs=args['epochs'],
                            summaries_directory=args['summaries_directory'],
                            tensorboard_write_grad=args['tensorboard_write_grad'],
                            config_json_path=args['config_json_path']
                            ).train()
        return {'loss': -acc, 'status': STATUS_OK}

    def run(self):
        self.X_train, self.y_train, self.X_test, self.y_test= self.fetch_dataset()
        # print(self.mdl.summary())
        space =  {          'optimizer':hp.choice('optimizer_options',[keras.optimizers.RMSprop(lr=1e-4),keras.optimizers.Adam(lr=1e-4)]),
                            'loss':self.params["loss"],
                            'metrics':self.params["metrics"],
                            'epochs':self.params["epochs"],
                            'summaries_directory':self.params["summaries_directory"],
                            'tensorboard_write_grad':True,
                            'config_json_path':self.config_file}
        print space
        trails = Trials()
        best = fmin(self._wrapper, space, algo=tpe.suggest, max_evals=3,trials=trails)
        print best
        return self.model

if __name__ == "__main__":
    experiment(baseline_model(),"config/config.json").run()
    


