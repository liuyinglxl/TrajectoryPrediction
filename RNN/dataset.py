# ----- coding:utf-8 ------
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

class Data(object):
    def __init__(self, batch_size, input_steps, prediction_steps, train_data_path, test_data_path):
        self.batch_size = batch_size
        self.input_steps = input_steps
        self.prediction_steps = prediction_steps
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        self.train_inputs = list()
        self.train_labels = list()
        self.test_inputs = list()
        self.test_labels = list()

        # read data
        print "start read data ..."
        train_data = pd.read_table(self.train_data_path, sep=' ', header=None)
        test_data = pd.read_table(self.test_data_path, sep=' ', header=None)
        
        # normalization
        print "start normalization ..."
        total_data = pd.concat([train_data,test_data])
        self.scaler = StandardScaler().fit(total_data.get([2,3]).values)
        train_data[2], train_data[3] = np.split(self.scaler.transform(train_data[[2,3]]), 2, axis=1)
        test_data[2], test_data[3] = np.split(self.scaler.transform(test_data[[2,3]]), 2, axis=1)
        
        # orignise train and test data
        print "start orginize data ..."
        train_data.groupby([0]).apply(self.orignise_train)
        test_data.groupby([0]).apply(self.orignise_test)
        
        print "start transfrom numpy ... "
        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = np.array(self.train_inputs, dtype=np.float32), \
            np.array(self.train_labels, dtype=np.float32), np.array(self.test_inputs, dtype=np.float32), np.array(self.test_labels, dtype=np.float32)
        
        print "start shuffle ..."
        self.train_inputs, self.train_labels = shuffle(self.train_inputs, self.train_labels, random_state=0)
        self.test_inputs, self.test_labels = shuffle(self.test_inputs, self.test_labels, random_state=0)
        
    def orignise_train(self, group):
        values = group.get([2,3]).values
        for i in range(0, group.shape[0] - self.input_steps - self.prediction_steps):
            self.train_inputs.append(values[i : i+self.input_steps])
            self.train_labels.append(values[i+self.input_steps : i+self.input_steps+self.prediction_steps, 0:2])

    def orignise_test(self, group):
        values = group.get([2,3]).values
        for i in range(0, group.shape[0] - self.input_steps - self.prediction_steps):
            self.test_inputs.append(values[i : i+self.input_steps])
            self.test_labels.append(values[i+self.input_steps : i+self.input_steps+self.prediction_steps, 0:2])