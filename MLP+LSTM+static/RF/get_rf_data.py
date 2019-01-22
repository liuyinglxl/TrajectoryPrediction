'''
Get RF training data.
If the node of input sequence are all the same, then the RF is used to judge whether the next node is same
'''

import sys
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm_notebook as tqdm

input_length = 30
predict_length = 10
train_inputs = []
train_labels = []
test_inputs = []
test_labels = []

train_data_path =  "../../data/sample_train.txt"
test_data_path =   "../../data/sample_test.txt"

def organise_train_data(group):
    values = group.get([2,3]).values
    for i in range(0, group.shape[0] - input_length - predict_length):
        train_inputs.append(values[i : i + input_length])
        out = values[i + input_length : i + input_length + predict_length, -2:]
        if (np.std(out, axis=0) == [0, 0]).all():
            train_labels.append(1)
        else:
            train_labels.append(0)


def organise_test_data(group):
    values = group.get([2,3]).values
    for i in range(0, group.shape[0] - input_length - predict_length):
        test_inputs.append(values[i : i + input_length])
        out = values[i + input_length : i + input_length + predict_length, -2:]
        if (np.std(out, axis=0) == [0, 0]).all():
            test_labels.append(1)
        else:
            test_labels.append(0)


print "load data ..."
train_data = pd.read_table(train_data_path, sep=' ', header=None)
test_data = pd.read_table(test_data_path, sep=' ', header=None)

print "normalization ..."
scaler = StandardScaler().fit(pd.concat([train_data,test_data]).get([2,3]).values)
train_data[2], train_data[3] = np.split(scaler.transform(train_data[[2,3]]), 2, axis=1)
test_data[2], test_data[3] = np.split(scaler.transform(test_data[[2,3]]), 2, axis=1)

train_data.groupby([0]).apply(organise_train_data)
test_data.groupby([0]).apply(organise_test_data)

print "get 0 and 1 data ..."

train_input_0 = []
train_input_1 = []
test_input_0 = []
test_input_1 = []

for i in range(len(train_labels)):
    if train_labels[i] == 0:
        train_input_0.append(train_inputs[i])
    else:
        train_input_1.append(train_inputs[i])

for i in range(len(test_labels)):
    if test_labels[i] == 0:
        test_input_0.append(test_inputs[i])
    else:
        test_input_1.append(test_inputs[i])

print "start sample ..."
train_slice = random.sample(train_input_0, len(train_input_1))
test_slice = random.sample(test_input_0, len(test_input_1))

new_train_inputs = train_slice + train_input_1
new_test_inputs = test_slice + test_input_1

new_train_labels = [0] * len(train_input_1) + [1] * len(train_input_1)
new_test_labels = [0] * len(test_input_1) + [1] * len(test_input_1)

print "start shuffle ..."
new_train_inputs, new_train_labels = shuffle(new_train_inputs, new_train_labels, random_state=0)
new_test_inputs, new_test_labels = shuffle(new_test_inputs, new_test_labels, random_state=0)

np.save("train_%d_inputs.npy" % predict_length, new_train_inputs)
np.save("train_%d_labels.npy" % predict_length, new_train_labels)
np.save("test_%d_inputs.npy" % predict_length, new_test_inputs)
np.save("test_%d_labels.npy" % predict_length, new_test_labels)