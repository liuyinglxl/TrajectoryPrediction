# --------coding: utf-8-----------
"""
Deep Belief Network
"""
import os
import sys
import numpy as np
import tensorflow as tf
from logisticRegression import LogisticRegression
from rbm import RBM
import data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from dataset import Data

class DBN(object):
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # normal sigmoid layer
        self.rbm_layers = []   # RBM layer
        self.params = [] 
        self.n_in = n_in
        self.n_out = n_out

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_in*2])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_out*2])

        total_data = np.loadtxt("/mnt/disk2/liuying/T-ITS/dataset/geolife/geolife_total")
        total_data = total_data[:, 2:4]
        self.scaler = MinMaxScaler().fit(total_data)
        
        self.checkpoint_times = 1

        # Contruct the layers of DBN
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                input_size = self.n_in*2
            else:
                layer_input = self.layers[i-1].output
                input_size = hidden_layers_sizes[i-1]
            # Sigmoid layer
            print("n_in:{0}   n_out:{1}".format(input_size, hidden_layers_sizes[i]))
            sigmoid_layer = tf.layers.dense(inputs=layer_input, units=hidden_layers_sizes[i], activation=tf.nn.sigmoid, name="layer_1")
            self.layers.append(sigmoid_layer)
            # Add the parameters for finetuning
            self.params.extend(sigmoid_layer.params)
            # Create the RBM layer
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                        W=sigmoid_layer.W, hbias=sigmoid_layer.b))
        # We use the LogisticRegression layer as the output layer
        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                n_out=n_out*2)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # The logistic regression output
        self.predictor = self.output_layer.output
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost, var_list=self.params)
        self.saver = tf.train.Saver(tf.global_variables())

    def mean_error(self, predict_output, real_output):
        """
        compute mean error
        """
        error = 0.0
        predict_output = predict_output.reshape(-1, 2)
        real_output = real_output.reshape(-1, 2)
        for i in range(predict_output.shape[0]):
            distance = np.sqrt(np.square(predict_output[i][0]-real_output[i][0]) + np.square(predict_output[i][1]-real_output[i][1]))
            error += distance
        error /= predict_output.shape[0]
        return error

    def process_data_for_batch(self, data):
        processed_data = []
        info_data = []
        for i in range(len(data)):
            processed_data.append([elem[2] for elem in data[i]])
            info_data.append([data[i][len(data[i]) - 1][0], data[i][len(data[i]) - 1][1]])
        return processed_data, info_data

    def next_batch(self, X, y, batch_size, step, forward_only=False):
        if step == len(X) / batch_size - 1:
            x_batch, X_info_data = self.process_data_for_batch(X[len(X) - batch_size:])
            y_batch, y_info_data = self.process_data_for_batch(y[len(y) - batch_size:])
        else:
            x_batch, X_info_data = self.process_data_for_batch(X[step * batch_size: (step + 1) * batch_size])
            y_batch, y_info_data = self.process_data_for_batch(y[step * batch_size: (step + 1) * batch_size])
        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)
        x_batch = x_batch.reshape(-1, 2)
        y_batch = y_batch.reshape(-1, 2)
        x_batch = self.scaler.transform(x_batch)
        y_batch = self.scaler.transform(y_batch)
        x_batch = x_batch.reshape(-1, self.n_in*2)
        y_batch = y_batch.reshape(-1, self.n_out*2)
        return x_batch, y_batch


    def pretrain(self, sess, dataset, batch_size=50, pretraining_epochs=10, lr=0.01, k=1,
                    display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        """
        train_X = dataset[0]
        train_y = dataset[1]
        print('Starting pretraining...\n')
        batch_num = len(train_X) / batch_size
        print("batch_num {0}".format(batch_num))
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for step in range(batch_num):
                    x_batch, _ = self.next_batch(train_X, train_y, batch_size, step)
                    # train
                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # compute cost
                    tmp_cost = sess.run(cost, feed_dict={self.x: x_batch,})
                    avg_cost += tmp_cost / batch_num
                    if (step+1) % 500 == 0:
                        print("\t\t\tPretraing layer {0} Epoch {1} Step {2} cost: {3}".format((i+1), (epoch+1), (step+1), tmp_cost))
                # output
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format((i+1), (epoch+1), avg_cost))

        
    def finetuning(self, sess, dataset, training_epochs=10, start=0, batch_size=100,
                   display_step=1, model_path="", model_name="model", load_model=0):
        """
        Finetuing the network
        """
        train_X = dataset[0]
        train_y = dataset[1]
        test_X = dataset[2]
        test_y = dataset[3]

        print("\nStart finetuning...\n")
        best_sess = sess
        global_test_error = 100000000
        tolerance_count = 0
        
        for epoch in range(start, training_epochs):

            avg_cost = 0.0
            batch_num = len(train_X) / batch_size
            for step in range(batch_num):
                x_batch, y_batch = self.next_batch(train_X, train_y, batch_size, step)
                # train
                sess.run(self.train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # compute cost
                avg_cost += sess.run(self.cost, feed_dict=
                {self.x: x_batch, self.y: y_batch}) / batch_num
            print "epoch:", epoch+1, "loss: ", avg_cost

            if (epoch+1) % self.checkpoint_times == 0:
                count = 0
                final_error = 0.0
                batch_num = len(test_X) / batch_size
                for step in range(batch_num):
                    x_batch, y_batch = self.next_batch(test_X, test_y, batch_size, step)
                    count += 1
                    predict = sess.run(self.predictor, feed_dict={self.x: x_batch, self.y: y_batch})
                    error = self.mean_error(predict, y_batch)
                    final_error += error

                test_error = (final_error/count) * 10000
                print "final mean error(x10000):", test_error

                if test_error < global_test_error:
                    tolerance_count = 0
                    global_test_error = test_error
                    self.saver.save(best_sess, os.path.join(model_path, model_name+"best_model.ckpt"))
                else:
                    tolerance_count += 1

                print "The global min test error:", global_test_error
                if tolerance_count >= 50:
                    break

        print 'The final final final global min test error:', global_test_error

    def test(self, sess, dataset, batch_size=100):
        test_X = dataset[2]
        test_y = dataset[3]
        count = 0
        final_error = 0.0
        batch_num = len(test_X) / batch_size
        for step in range(batch_num):
            x_batch, y_batch = self.next_batch(test_X, test_y, batch_size, step)
            count += 1
            predict = sess.run(self.predictor, feed_dict={self.x: x_batch, self.y: y_batch})

            error = self.mean_error(predict, y_batch)
            final_error += error
            print("\nTest step :{0}, mean_error:{1}".format(step, error))

        final_error /= count
        print "final mean error:", final_error
        

if __name__ == "__main__":
    input_size = 30
    output_size = 50
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.14)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # set random_seed
    tf.set_random_seed(seed=1111)
    train_X, train_y, test_X, test_y = dataset.train_inputs, dataset.train_labels, dataset.test_inputs, \
        dataset.test_labels
    datas = [train_X, train_y, test_X, test_y]

    dbn = DBN(n_in=input_size, n_out=output_size, hidden_layers_sizes=[128, 128, 128])

    init = tf.global_variables_initializer()
    sess.run(init)
    model_name = "%dstep_128hidden_"%output_size
    model_path = "./model/%dstep_128hidden_3layers"%output_size

    if os.path.isdir(model_path) is False:
        os.mkdir(model_path)
    # new train
    if sys.argv[1] == '0': # new train
        dbn.pretrain(sess, dataset=datas, batch_size=2000, pretraining_epochs=10, lr=0.2, k=10)
        dbn.finetuning(sess, dataset=datas, training_epochs=500, start=0, batch_size=2000, model_path=model_path, model_name=model_name, load_model=0)
    # load model
    elif sys.argv[1] == '1': # test
        checkpoint_dir = "./model/10step_128hidden_3layers/best_model.ckpt"
        dbn.saver.restore(sess, checkpoint_dir)
        dbn.test(sess, dataset=datas, batch_size=5000)
    elif sys.argv[1] == '2': #continue train
        checkpoint_dir = "./model/best_model.ckpt"
        dbn.saver.restore(sess, checkpoint_dir)
        dbn.finetuning(sess, dataset=datas, training_epochs=500, start=0, batch_size=2000, model_path=model_path, model_name=model_name, load_model=1)
    else:
        print "param1 error"
