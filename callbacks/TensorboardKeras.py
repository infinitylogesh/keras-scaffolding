from keras import backend as K
import tensorflow as tf
import numpy as np
import keras
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
from keras.models import Sequential,Model
from elasticsearch import Elasticsearch
from datetime import datetime
import cv2


class TensorboardKeras(keras.callbacks.Callback):
    def __init__(self,model,validation_input,actual,label_names,val_folder_name,log_dir):
        self.log_dir = log_dir
        self.session = K.get_session()
        self.label_names = label_names
        self.model = model
        self.actual = actual
        self.label_names = label_names
        self.validation_input = validation_input
        self.label_acc_ph = []
        self.val_folder_name = val_folder_name
        #self.es = Elasticsearch(["54.244.82.97:9201"])
        # -- Switch off - TensorBoard embedding. --
        # self.embedding_layer_names = ['image_input','text_input','dense_1024','dense_13']

        self.val_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('val/loss', self.val_loss_ph)

        self.val_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('val/acc', self.val_acc_ph)

        self.train_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('train/loss', self.train_loss_ph)

        self.train_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('train/acc', self.train_acc_ph)

        for label in label_names:
            ph = tf.placeholder(shape=(),dtype=tf.float32,name="_".join(label.split(" ")))
            self.label_acc_ph.append(ph)
            tf.summary.scalar('class-wise-accuracy/category/'+label,ph)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir,self.session.graph)
        # -- Switch off - TensorBoard embedding. --
        # with open(os.path.join(self.log_dir,'metadata.tsv'), 'w+') as metadata_file:
        #     for y in self.actual:
        #         metadata_file.write('%s\n' % str(self.label_names[y]))

    def _get_class_wise_accuracy(self):
        class_count = len(self.label_names)
        predicted = np.argmax(self.model.predict(self.validation_input),axis=1)
        result = np.zeros(class_count,dtype=np.float32)
        for p,a in zip(predicted,self.actual):
            if p == a:
                result[p]+=1
        actual_count = np.asarray(map(lambda i:len(filter(lambda x:x==i,self.actual)),xrange(class_count)),dtype=np.float32)
        # self.write_image_summary()
        return np.divide(result,actual_count)
    
    """def write_image_summary(self):
        val_filenames = np.load(self.val_folder_name+"_filenames.npy")
        for filename in val_filenames:
            img = cv2.imread(os.path.join(self.val_folder_name,filename))
            tf.summary.image("validation_misses",np.expand_dims(img,0),10)"""

    def _define_embedding_placeholder(self):
        self.embedding_tensors = {}
        self.keras_embedding_tensors = {}
        config = projector.ProjectorConfig()
        for layer_name in self.embedding_layer_names:
            self.keras_embedding_tensors[layer_name] = self.model.get_layer(layer_name).output
            tensor_shape = self.keras_embedding_tensors[layer_name].shape[-1]
            self.embedding_tensors[layer_name] = tf.Variable(tf.zeros([123,tensor_shape]), dtype=tf.float32,name=layer_name)
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embedding_tensors[layer_name].name
            embedding.metadata_path = os.path.join(self.log_dir,'metadata.tsv')
        projector.visualize_embeddings(self.writer, config)


    def _embedding_run(self):
        feed_dict = {}
        for layer_name in self.embedding_layer_names:
            embedding_model = Model(inputs=self.model.input,outputs=self.keras_embedding_tensors[layer_name])
            layer_output = embedding_model.predict(self.validation_input)
            feed_dict[layer_name] = layer_output
        K.batch_set_value(list(zip(self.embedding_tensors.values(),feed_dict.values())))

    def on_epoch_end(self,epoch,logs):
        # --- Switch off tensorboard embedding ---
        # self._define_embedding_placeholder()
        # self._embedding_run()
        cwa = self._get_class_wise_accuracy();
        feed_dict = {self.val_loss_ph: logs["val_loss"],
                   self.train_loss_ph: logs["loss"],
                   self.val_acc_ph: logs["val_categorical_accuracy"],
                   self.train_acc_ph: logs["categorical_accuracy"]
                   }
        cwa_feed_dict = {ph:acc for ph,acc in zip(self.label_acc_ph,cwa)}
        final_feed_dict = feed_dict.update(cwa_feed_dict)
        summary = self.session.run(self.merged,
                                   feed_dict=feed_dict)
        #es_doc = {'val_loss': logs["val_loss"],'train_loss':logs["loss"],'val_acc':logs["val_categorical_accuracy"],'train_acc':logs["categorical_accuracy"],'run_name':self.log_dir.split("/")[-1],'timestamp': datetime.now()}
        #self.es.index(index="app-log-ml-stats", doc_type='json', id=1, body=es_doc)
        # --- Switch off tensorboard embedding ---
        # saver = tf.train.Saver(self.embedding_tensors.values())
        # saver.save(self.session, os.path.join(self.log_dir, "model.ckpt"), epoch)
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_epoch_end_cb(self):
        return LambdaCallback(on_epoch_end=lambda epoch, logs:self.on_epoch_end(epoch, logs))
