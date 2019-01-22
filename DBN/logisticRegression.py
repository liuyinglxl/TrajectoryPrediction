# --------coding: utf-8-----------
"""
Logistic Regression
"""
import numpy as np
import tensorflow as tf
import input_data

class LogisticRegression(object):
    """Multi-class logistic regression class"""
    def __init__(self, inpt, n_in, n_out):
        self.W = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32)
        self.output = tf.matmul(inpt, self.W) + self.b
        # prediction
        self.y_pred = tf.argmax(self.output, axis=1)
        # keep track of variables
        self.params = [self.W, self.b]

    def lr_output(self, y):
        return self.output

    def cost(self, y):
        # rmse
        tmp = tf.cast(self.output, tf.float32)
        loss = 0.0
        for i in range(0, tmp.shape[1]):
            loss += tf.sqrt(tf.reduce_mean(tf.square(y[:,i]-tmp[:,i])))
        return loss

    def accuarcy(self, y):
        """errors"""
        tmp = tf.cast(self.output, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.square(y-tmp)))
        
    def rmse(self, y):
        """rmse"""
        tmp = tf.cast(self.output, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.square(y-tmp)))
