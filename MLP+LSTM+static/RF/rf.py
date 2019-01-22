import sys
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

predict_length = 10
print "prediction length :", predict_length

load_model = 0

print "load data ...."
train_X = np.load("./data/train_%d_inputs.npy" % predict_length)
train_y = np.load("./data/train_%d_labels.npy" % predict_length)
test_X = np.load("./data/test_%d_inputs.npy" % predict_length)
test_y = np.load("./data/test_%d_labels.npy" % predict_length)

# only use the location information
train_X = train_X[:, :, 0:2].reshape(train_X.shape[0], -1)
test_X = test_X[:, :, 0:2].reshape(test_X.shape[0], -1)

if load_model == 0:
    rf = RandomForestClassifier()
    rf.fit(train_X, train_y)
    joblib.dump(rf, "./model/rf_train_%d_model.m" % predict_length)
else:
    rf = joblib.load("./model/rf_train_%d_model.m" % predict_length)

print "train accuracy: ", accuracy_score(train_y, rf.predict(train_X))
print "test accuracy: ", accuracy_score(test_y, rf.predict(test_X))