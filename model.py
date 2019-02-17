from __future__ import absolute_import

# To reproduce results

import numpy as np
import tensorflow as tf
import random as rn
import math
import os
import keras
import os
import sys
from PIL import Image,ImageChops
from keras.datasets import mnist
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
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
from sklearn.metrics import f1_score
from callbacks.metrics import Metrics
from training import Training

summaries_directory = "summaries/"
run_name = "datasetv2_residual_dropout_cv_border_constant_with_dropout"




class model(object):
    def __init__(self,train_data_dir,val_data_dir,img_width=150,img_height=150,batch_size=128):
        self.hyperparams = {}
        self.hyperparams["img_width"] = img_width
        self.hyperparams["img_height"] = img_height
        self.hyperparams["batch_size"] = batch_size
        self.hyperparams["train_data_dir"] = train_data_dir
        self.hyperparams["val_data_dir"] = val_data_dir
        self._create_folder(os.path.join(summaries_directory,run_name))

    """
    Function to rescale the images and retain the aspect ration by padding the remaining area with black overlay.
    image : image - numpy array returned by ImageDataGenerator
    """
    def rescale_image_retain_aspect_ratio(self,image):
        image.thumbnail((self.hyperparams["img_width"],self.hyperparams["img_height"]), Image.ANTIALIAS)
        image_size = image.size
        size = (self.hyperparams["img_width"],self.hyperparams["img_height"])
        thumb = image.crop( (0, 0, size[0], size[1]) )
        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )
        image = image.convert("RGB")
        color = [101, 52, 152] # 'cause purple!
        thumb = cv.copyMakeBorder(img_to_array(image), offset_y,offset_y+size[1] - (image_size[1]+(offset_y*2)),offset_x,offset_x+size[0] - (image_size[0]+(offset_x*2)), cv.BORDER_CONSTANT,value=color)
        return array_to_img(thumb).convert("L")


    """
    Data generator definition and extraction happens here. If input normalization is switched on, input for bottleneck generation is
    normalized. Pre-processing function is used to rescale the images.
    """
    def dataset_ops(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        y_train = [ np.eye(9)[val-1] for val in y_train]
        y_test = [np.eye(9)[val-1] for val in y_test]
        return np.expand_dims(X_train,axis=-1), np.array(y_train), np.expand_dims(X_test,axis=-1), np.array(y_test)
    
    def conv_block(self,feat_maps_out, prev):
        prev = BatchNormalization(axis=1)(prev) # Specifying the axis and mode allows for later merging
        prev = Activation('relu')(prev)
        prev = Dropout(0.5)(prev)
        prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
        prev = BatchNormalization(axis=1)(prev) # Specifying the axis and mode allows for later merging
        prev = Activation('relu')(prev)
        prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
        prev = Dropout(0.5)(prev)
        return prev
    
    """
    All the images in the generator are extracted and returned.
    generater : Keras generator - either training generator or validation generator.
    """
    def _extract_generator(self,generator):
        x = []
        y = []
        batch_index = 0
        while batch_index <= generator.batch_index:
            data = generator.next()
            x.extend(data[0])
            y.extend(data[1])
            batch_index = batch_index + 1
        print(" Generator extraction completed")
        return (x,y)

    def skip_block(self,feat_maps_in, feat_maps_out, prev):
        if feat_maps_in != feat_maps_out:
            # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
            prev = Convolution2D(feat_maps_out, 1, 1, border_mode='same')(prev)
        return prev 


    def Residual(self,feat_maps_in, feat_maps_out, prev_layer):
        '''
        A customizable residual unit with convolutional and shortcut blocks
        Args:
        feat_maps_in: number of channels/filters coming in, from input or previous layer
        feat_maps_out: how many output channels/filters this block will produce
        prev_layer: the previous layer
        '''

        skip = self.skip_block(feat_maps_in, feat_maps_out, prev_layer)
        conv = self.conv_block(feat_maps_out, prev_layer)

        print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
        return merge([skip, conv], mode='sum') # the residual connection


    def f1(self,y_true, y_pred):
        '''
        metric from here 
        https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
        '''
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def baseline_model(self):
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
        model.compile(loss='binary_crossentropy',
                    optimizer=keras.optimizers.RMSprop(lr=1e-4),
                    metrics=['acc'])
        return model
    
    def residual_model(self):
        inp = Input((150, 150, 1))
        conv1 = layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 1))(inp)
        r = self.Residual(64, 128, conv1)
        r = self.Residual(128, 128, r)
        # r3 = self.Residual(128, 256, r2)
        flat = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid')(r)
        flat = layers.Flatten()(flat)
        dense = layers.Dense(64, activation='relu')(flat)
        out = layers.Dense(2, activation='softmax')(flat)
        model = Model(input=inp, output=out)
        model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.RMSprop(lr=1e-4),
                    metrics=['acc',self.f1])
        return model
    
    def callbacks(self):
        csvlogger = keras.callbacks.CSVLogger(summaries_directory+run_name+'/'+'training_log.csv', separator=',', append=True) # Logger to log all the training log for each epoch.
        checkpoint = keras.callbacks.ModelCheckpoint(summaries_directory+run_name+'/'+'model.{epoch:02d}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
        return [csvlogger,checkpoint]


    def train(self):
        X_train, y_train, X_test, y_test= self.dataset_ops()
        self.mdl = self.baseline_model() 
        # print(self.mdl.summary())
        Training(model=self.mdl,X_train=X_train,Y_train=y_train,X_test=X_test,Y_test=y_test,epochs=3,summaries_directory="./summaries",tensorboard_write_grad=True).train()
        return self.mdl
    
    def _create_folder(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
       
if __name__ == '__main__':
    m = model("/home/ubuntu/datasets/button_classification/datasetv2_with_pseudo/train","/home/ubuntu/datasets/button_classification/datasetv2_with_pseudo/val",150,150,32)
    m.train()


