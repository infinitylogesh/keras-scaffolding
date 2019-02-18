import os,json


class Pipeline(object):

    def __init__(self,model,config_file_path):
        self.model = model
        self.config_file = config_file_path
        self.params = self.read_config()
    
    def read_config(self):
        with open(self.config_file, 'r') as fh:
            data = json.load(fh)
        return data

    def fetch_dataset(self):
        pass # fetch dataset from the path specified in the config.
    
    def pre_process(self):
        pass # pre_process dataset if required 

    def run(self):
        pass # call Training and feed the read params and start training.
