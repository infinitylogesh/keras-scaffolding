import keras
from keras.datasets import mnist

class DatasetOps(object):

    def __init__(self,dummy=0):
        self.dummy = dummy

    def fetch_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        return (x_train,y_train),(x_test,y_test)

