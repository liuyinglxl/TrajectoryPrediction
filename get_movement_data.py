'''
Using the trained RF model, the data is divided into static and movement, and the movement data is used to train the model.
'''

import sys
import os
import numpy as np
import tensorflow as tf
import time
from seq2seq_model_utils import create_model
from dataset import Data
from sklearn.externals import joblib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("encoder_size", 30, "the steps of input")
tf.app.flags.DEFINE_integer("decoder_size", 10, "the steps of prediction")
tf.app.flags.DEFINE_string("dataset", "geolife", "the city of dataset")

train_data_path =  "./data/little_train.txt"
test_data_path = "./data/little_test.txt"
save_data_dir =  "./data/"

if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)

def train():
    print "dataset: %s \t prediction_steps: %d" % (FLAGS.dataset, FLAGS.decoder_size)
    # load data
    dataset = Data(FLAGS.encoder_size, FLAGS.decoder_size, train_data_path, test_data_path)
    train_X, train_y, test_X, test_y = dataset.train_inputs, dataset.train_labels, dataset.test_inputs, \
        dataset.test_labels

    print "load model ..."
    rf = joblib.load(".RF/model/rf_train_%d_model.m" % FLAGS.decoder_size)

    print "start predicting ..."
    predict_static = rf.predict(train_X[:, :, 0:2].reshape(train_X.shape[0], -1))
    static_train_X = train_X[predict_static == 1]
    static_train_y = train_y[predict_static == 1]

    movement_train_X = train_X[predict_static == 0]
    movement_train_y = train_y[predict_static == 0]

    print "sava train data"
    np.save(os.path.join(save_data_dir, "%s_%d_static_train_X.npy" % (FLAGS.dataset, FLAGS.decoder_size)), static_train_X)
    np.save(os.path.join(save_data_dir, "%s_%d_static_train_y.npy" % (FLAGS.dataset, FLAGS.decoder_size)), static_train_y)
    np.save(os.path.join(save_data_dir, "%s_%d_movement_train_X.npy" % (FLAGS.dataset, FLAGS.decoder_size)), movement_train_X)
    np.save(os.path.join(save_data_dir, "%s_%d_movement_train_y.npy" % (FLAGS.dataset, FLAGS.decoder_size)), movement_train_y)
    print "save done!"


    print "start predicting ..."
    predict_static = rf.predict(test_X[:, :, 0:2].reshape(test_X.shape[0], -1))
    static_test_X = test_X[predict_static == 1]
    static_test_y = test_y[predict_static == 1]

    movement_test_X = test_X[predict_static == 0]
    movement_test_y = test_y[predict_static == 0]

    print "sava train data"
    np.save(os.path.join(save_data_dir, "%s_%d_static_test_X.npy" % (FLAGS.dataset, FLAGS.decoder_size)), static_test_X)
    np.save(os.path.join(save_data_dir, "%s_%d_static_test_y.npy" % (FLAGS.dataset, FLAGS.decoder_size)), static_test_y)
    np.save(os.path.join(save_data_dir, "%s_%d_movement_test_X.npy" % (FLAGS.dataset, FLAGS.decoder_size)), movement_test_X)
    np.save(os.path.join(save_data_dir, "%s_%d_movement_test_y.npy" % (FLAGS.dataset, FLAGS.decoder_size)), movement_test_y)
    print "save done!"

if __name__ == '__main__':
    train()