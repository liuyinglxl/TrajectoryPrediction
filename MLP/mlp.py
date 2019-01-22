# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import great_circle

# parmaters  
city = 'beijing'
learning_rate = 0.001
epochs = 200
display_step = 10
car_id_batch_size = 10
predict_steps = 10
r_threshold = 200
car_id_batch = int(10000 / car_id_batch_size)
keep_prob = 1
model_index = 1

n_input= 30*2
n_output= predict_steps*2
model_name = city + '_' + str(predict_steps) + 'steps_' + str(model_index)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True


def distance(true_x,true_y,predict_x,predict_y):
    return great_circle((true_x,true_y),(predict_x,predict_y)).m


if __name__ == '__main__':
    vec_distance = np.vectorize(distance)
    
    cdf = pd.read_csv("../data/sample_train.txt", index_col=0)
    ctdf = pd.read_csv("../data/sample_test.txt", index_col=0)

    tmp_cdf = np.array([cdf.x,cdf.y]).T
    min_max_scaler = MinMaxScaler().fit(tmp_cdf)
    tmp_1 = min_max_scaler.transform(tmp_cdf)
    cdf['x'] = tmp_1[:,0]
    cdf['y'] = tmp_1[:,1]

    tmp_ctdf = np.array([ctdf.x,ctdf.y]).T
    tmp_2 = min_max_scaler.transform(tmp_ctdf)
    ctdf['x'] = tmp_2[:,0]
    ctdf['y'] = tmp_2[:,1]

    print 'data has been loaded'

    print 'creating graph...'

    inputs_ = tf.placeholder("float32", [None, n_input])
    ys_ = tf.placeholder("float32", shape = [None, n_output])  
    keep_prob_ = tf.placeholder(tf.float32,name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32,name = 'learning_rate')

    layer_1 = tf.layers.dense(inputs=inputs_, units=128, activation=tf.nn.relu) 
    layer_1 = tf.layers.dropout(inputs=layer_1, rate=1-keep_prob)

    layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.relu) 
    layer_2 = tf.layers.dropout(inputs=layer_2, rate=1-keep_prob)

    logits = tf.layers.dense(inputs=layer_2, units=predict_steps*2)

    tmp = ys_ - logits
    tmp_1 = tf.strided_slice(tmp,[0,0],[(799 - 30 - predict_steps)*car_id_batch_size,predict_steps*2],[1,2])
    tmp_2 = tf.strided_slice(tmp,[0,1],[(799 - 30 - predict_steps)*car_id_batch_size,predict_steps*2],[1,2])

    loss = tf.reduce_mean(tf.sqrt(tf.square(tmp_1)+tf.square(tmp_2)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    print 'graph has been created'

    print 'initializing the session...'
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    print 'training the model...'
    all_loss = []
    all_error = []
    all_acc = []

    for i in range(epochs):
        losses = []
        for car_id in range(car_id_batch):
            fff = np.array([cdf.iloc[800*car_id:800*(car_id+car_id_batch_size)]['x'],
                            cdf.iloc[800*car_id:800*(car_id+car_id_batch_size)]['y']]).T.reshape((car_id_batch_size,800,2))
            batch_x = np.array([fff[:,start_point:start_point+30,:] for start_point in range(799 - 30 - predict_steps)])                  .reshape(((799 - 30 - predict_steps)*car_id_batch_size,30*2))
            batch_y = np.array([fff[:,start_point:start_point+predict_steps,:] for start_point in range(29+1,799-predict_steps)])                  .reshape(((799 - 30 - predict_steps)*car_id_batch_size,predict_steps*2))
            
            sess.run(optimizer, feed_dict={inputs_: batch_x, ys_: batch_y, keep_prob_: keep_prob,learning_rate_:learning_rate})
            losses.append(sess.run(loss,feed_dict={inputs_: batch_x, ys_: batch_y, keep_prob_: keep_prob,learning_rate_:learning_rate}))
             
            all_loss.append(mean(losses))
            if (i+1) % display_step == 0:
                print 'loss:',all_loss[-1]

    saver = tf.train.Saver()
    save_path = saver.save(sess, "../models/"+model_name+"/save_net.ckpt")
    print("Save to path: ", save_path)

    print ('predicting the vs...')
    predict_positions = []
    anwser_positions = []
    for car_id in tqdm(range(10000)):
        ttt = np.array([ctdf.iloc[200*car_id:200*(car_id+1)]['x'],
                        ctdf.iloc[200*car_id:200*(car_id+1)]['y']]).T.reshape((200,2))
        batch_x = np.array([ttt[start_point:start_point+30,:] for start_point in range(199 - 30 - predict_steps)])              .reshape((199 - 30 - predict_steps,30*2))
        batch_y = np.array([ttt[start_point:start_point+predict_steps,:] for start_point in range(29+1,199-predict_steps)])              .reshape((199 - 30 - predict_steps,predict_steps*2))
        predict_positions.append(sess.run(logits,feed_dict={inputs_: batch_x,keep_prob_:1.0}))
        anwser_positions.append(batch_y)
        
    predict_positions = np.array(predict_positions).reshape(10000*(199 - 30 - predict_steps)*predict_steps,2)
    anwser_positions = np.array(anwser_positions).reshape(10000*(199 - 30 - predict_steps)*predict_steps,2)

    predict_positions = min_max_scaler.inverse_transform(predict_positions)
    anwser_positions = min_max_scaler.inverse_transform(anwser_positions)

    errors = vec_distance(predict_positions[:,0],predict_positions[:,1],anwser_positions[:,0],anwser_positions[:,1])
    mean_error = np.mean(errors)
    acc = len(errors[errors<=r_threshold]) * 1.0 / len(errors)
    print (mean_error)
    print (acc)

    savez('./model/'+model_name+'.npz', all_loss=all_loss, all_error=all_error, all_acc=all_acc)

    mean_error = [mean_error]
    acc = [acc]

    savez('./model/'+model_name+'.npz', 
          all_loss=all_loss, 
          all_error=all_error, 
          all_acc=all_acc,
          mean_error=mean_error,acc=acc,
          predict_positions=predict_positions)