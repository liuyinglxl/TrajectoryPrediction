import random

import numpy as np
import tensorflow as tf
import sys

np.random.seed(1000)

class Seq2SeqModel(object):
    def __init__(self, encoder_size, decoder_size, hidden_dim, input_dim, output_dim):
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder_inputs = []
        for i in range(encoder_size):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_dim], name="encoder{0}".format(i)))
        self.decoder_inputs = []
        for i in range(decoder_size):
            self.decoder_inputs.append(
                tf.placeholder(tf.float32, shape=[None, output_dim], name="decoder{0}".format(i)))

        encoder_cell_fw = tf.contrib.rnn.GRUCell(hidden_dim)
        encoder_cell_bw = tf.contrib.rnn.GRUCell(hidden_dim)

        encoder_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_cell_fw, output_keep_prob=0.5)
        encoder_cell_bw = tf.contrib.rnn.DropoutWrapper(encoder_cell_bw, output_keep_prob=0.5)

        encoder_outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(encoder_cell_fw, encoder_cell_bw, self.encoder_inputs, dtype=tf.float32)
        
        state = tf.concat((state_fw, state_bw), 1)

        # construct gru basic cell
        decoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim*2)
        W = tf.Variable(tf.truncated_normal([hidden_dim*2, output_dim]))
        b = tf.Variable(tf.truncated_normal([output_dim]))

        self.decoder_outputs = []
        with tf.variable_scope("rnn_decoder"):
            for i, inp in enumerate(self.decoder_inputs):
                if i == 0:
                    prev = self.encoder_inputs[-1]
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                decoder_output, state = decoder_cell(prev, state)
                prev = tf.matmul(decoder_output, W) + b
                self.decoder_outputs.append(prev)

        self.loss = 0.0
        for i in range(len(self.decoder_inputs)):
            self.loss += tf.sqrt(tf.reduce_sum(tf.square(self.decoder_inputs[i] - self.decoder_outputs[i])))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())
        
    def step(self, sess, X, y, encoder_size, decoder_size, is_training):
        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.mlp_inputs[i].name] = X[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = y[i]
        # train
        if is_training:
            output_feed = [self.loss, self.decoder_outputs, self.optimizer]
            outputs = sess.run(output_feed, input_feed)
        # test
        else:
            output_feed = [self.loss, self.decoder_outputs]
            outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def get_batch(self, X, y, batch_size, step):
        if step == len(X) / batch_size - 1:
            batch_encode_inputs = X[len(X) - batch_size:]
            batch_decode_inputs = y[len(y) - batch_size:]
        else:
            batch_encode_inputs = X[step*batch_size : (step+1)*batch_size]
            batch_decode_inputs = y[step*batch_size : (step+1)*batch_size]
        batch_encode_inputs = np.transpose(batch_encode_inputs, (1, 0, 2))
        batch_decode_inputs = np.transpose(batch_decode_inputs, (1, 0, 2))
        return batch_encode_inputs, batch_decode_inputs