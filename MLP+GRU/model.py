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

        # the placeholder
        self.mlp_inputs = []
        for i in range(self.encoder_size):
            self.mlp_inputs.append(tf.placeholder(tf.float32, shape=[None, input_dim], name="mlp{0}".format(i)))
        self.decoder_inputs = []
        for i in range(self.decoder_size):
            self.decoder_inputs.append(
                tf.placeholder(tf.float32, shape=[None, output_dim], name="decoder{0}".format(i)))

        # MLP: encoding the input data
        self.encoder_inputs = []
        with tf.variable_scope("mlp"):
            for i in range(encoder_size):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                layer_1 = tf.layers.dense(inputs=self.mlp_inputs[i], units=self.hidden_dim, activation=tf.nn.relu, name="layer_1")
                layer_1 = tf.layers.dropout(inputs=layer_1, rate=0.5)
                layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dim, activation=tf.nn.relu, name="layer_2")
                layer_2 = tf.layers.dropout(inputs=layer_2, rate=0.5)
                logits = tf.layers.dense(inputs=layer_2, units=self.hidden_dim, name="logits")
                self.encoder_inputs.append(logits)

        # encoder: LSTM
        encoder_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
        encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=0.5)
        encoder_outputs, state = tf.contrib.rnn.static_rnn(encoder_cell, self.encoder_inputs, dtype=tf.float32)
        
        # decoder: LSTM
        decoder_cell = tf.contrib.rnn.GRUCell(hidden_dim)

        # decoding the middle vectors to output locations
        W = tf.Variable(tf.truncated_normal([hidden_dim, output_dim]))
        b = tf.Variable(tf.truncated_normal([output_dim]))

        self.decoder_outputs = []
        with tf.variable_scope("rnn_decoder"):
            for i, inp in enumerate(self.decoder_inputs):
                if i == 0:
                    prev = self.encoder_inputs[-1]
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                decoder_output, state = decoder_cell(prev, state)
                prev = decoder_output
                final_output = tf.matmul(decoder_output, W) + b
                self.decoder_outputs.append(final_output)

        self.loss = 0.0
        for i in range(len(self.decoder_inputs)):
            self.loss += tf.sqrt(tf.reduce_sum(tf.square(self.decoder_inputs[i] - self.decoder_outputs[i])))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
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