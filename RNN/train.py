import sys
import os
import numpy as np
import tensorflow as tf
import time
from seq2seq_model_utils import create_model
from dataset import Data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("hidden_dim", 128, "hidden units of networks")
tf.app.flags.DEFINE_integer("batch_size", 2000, "batch size")
tf.app.flags.DEFINE_integer("input_dim", 2, "the dim of feature")
tf.app.flags.DEFINE_integer("output_dim", 2, "the dim of output")
tf.app.flags.DEFINE_integer("encoder_size", 30, "the steps of input")
tf.app.flags.DEFINE_integer("decoder_size", 10, "the steps of prediction")
tf.app.flags.DEFINE_integer("check_per_epoches", 10, "check_per_epoches")
tf.app.flags.DEFINE_integer("epoches", 500, "training epoches")
tf.app.flags.DEFINE_string("dataset", "geolife", "the city of dataset")

train_data_path =  "../data/sample_train.txt"
test_data_path =  "../data/sample_test.txt"
save_model_dir =  "./%s/%s/%d_steps/%d_units/" % (FLAGS.dataset, FLAGS.time_slot, FLAGS.decoder_size, FLAGS.hidden_dim)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
load_model = 1 if os.listdir(save_model_dir) else 0

def train():
    print FLAGS.time_slot, train_data_path, save_model_dir, FLAGS.hidden_dim
    # load data
    dataset = Data(FLAGS.batch_size, FLAGS.encoder_size, FLAGS.decoder_size, train_data_path, test_data_path)
    train_X, train_y, test_X, test_y = dataset.train_inputs, dataset.train_labels, dataset.test_inputs, \
        dataset.test_labels
    
    tolerance_count = 0
    checkpoint_dir = os.path.join(save_model_dir, "%dhidden_%ddecoder_bestmodel.ckpt" % (FLAGS.hidden_dim, FLAGS.decoder_size))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        global_test_error = 1000000
        best_model = sess
        model = create_model(sess, FLAGS.encoder_size, FLAGS.decoder_size, FLAGS.hidden_dim, FLAGS.input_dim, \
            FLAGS.output_dim, load_model, checkpoint_dir)

        # training
        for epoch in range(FLAGS.epoches):
            st = time.time()
            epoch_loss = 0.0
            total_step = train_X.shape[0] / FLAGS.batch_size
            for step in range(total_step):
                encode_inputs, decode_inputs = model.get_batch(train_X, train_y, FLAGS.batch_size, step)
                step_loss, predict_outputs = model.step(sess, encode_inputs, decode_inputs, FLAGS.encoder_size, \
                    FLAGS.decoder_size, is_training=True)
                epoch_loss += step_loss
                if step % 20 == 0:
                    print 'train(step:%d/%d epoch:%d/%d)'%(step+1,total_step,epoch+1,FLAGS.epoches), '\t', \
                    predict_outputs[0][0], '\t', decode_inputs[0][0], '\t loss:', step_loss
            print "train loss %.6f in epoch=%d, time=%f" % (epoch_loss, epoch+1, time.time() - st)

             # validation
            if (epoch + 1) % FLAGS.check_per_epoches == 0:
                print " validation (epoch:%d/%d)" % (epoch+1, FLAGS.epoches)
                # test dataset
                test_loss = 0.0
                times = 0.0
                for step_test in range(len(test_X) / FLAGS.batch_size):
                    encode_inputs, decode_inputs = model.get_batch(test_X, test_y, FLAGS.batch_size, step_test)
                    step_loss, predict_outputs = model.step(sess, encode_inputs, decode_inputs, FLAGS.encoder_size, \
                        FLAGS.decoder_size, is_training=False)
                    test_loss += step_loss
                    
                # update min test loss
                if test_loss < global_test_error:
                    tolerance_count = 0
                    global_test_error = test_loss
                    model.saver.save(sess, os.path.join(save_model_dir, "%dhidden_%ddecoder_bestmodel.ckpt"%(FLAGS.hidden_dim, FLAGS.decoder_size)))
                else:
                    tolerance_count += FLAGS.check_per_epoches
                print "test loss %.6f in epoch=%d" % (test_loss, epoch + 1)
                print "global min test loss %.6f in epoch=%d" % (global_test_error, epoch + 1)

                if tolerance_count >= 50:
                    break

        print 'The final final final global min test error: %f' % global_test_error

if __name__ == '__main__':
    train()