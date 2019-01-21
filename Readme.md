This Repository is for our paper "Trajectory Forecasting with Neural Networks: An Empirical Evaluation and A New Hybrid Model"

Our hybrid model consists of three parts: RF (Random Forest), MLP and LSTM:

1. RF is used to predict whether the input sequences are the stationary.
2. Depended on the prediction of RF, two different choices are taken:
   1. If the input sequences are predicted as stationary ones, the model directly outputs the sequences whose nodes are the same as the last node of input sequences.
   2. If the input sequences are predicted as unstationary ones, the model is trained via encoder-decoder framework to get the prediction sequences



--------

# Requires

Python == 2.7

Tensorflow => 1.4

-----

# How to use

1. Run ./RF/get_rf_data.py (get the training data for Random Forest.)
2. Run ./RF/tf/py (train a Random Forest classifier.)
3. Run ./get_movement_data.py (get the training data for seq2seq model.)
4. Run ./train.py 

