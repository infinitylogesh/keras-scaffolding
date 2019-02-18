import numpy as np
import tensorflow as tf
import random as rn
import math
import os
import keras
import os
import sys
from keras import backend as K
from keras import layers
# tf.set_random_seed(1234)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
from keras.preprocessing.image import array_to_img,img_to_array
from keras.layers import BatchNormalization, Convolution2D, Input, merge,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation, Layer
from keras.models import Model

def baseline_model():
        # https://www.kaggle.com/morenoh149/keras-imagedatagenerator-validation-split
        model = keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(28, 28,1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(9, activation='softmax'))
        return model