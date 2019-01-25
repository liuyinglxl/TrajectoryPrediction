This Repository is for our paper "Trajectory Forecasting with Neural Networks: An Empirical Evaluation and A New Hybrid Model"



---

# Method

## Deep Learning

1. MLP+LSTM+RF

   Our hybrid model consists of three parts: RF (Random Forest), MLP and LSTM:

   1. RF is used to predict whether the input sequences are the stationary.
   2. Depended on the prediction of RF, two different choices are taken:
      1. If the input sequences are predicted as stationary ones, the model directly outputs the sequences whose nodes are the same as the last node of input sequences.
      2. If the input sequences are predicted as unstationary ones, the model is trained via encoder-decoder framework to get the prediction sequences

2. MLP+LSTM

3. MLP+GRU

4. Long Short-Term Memory (LSTM)

5. Bi-directional LSTM (Bi_LSTM)

6. Gated Recurrent Unit (GRU)

7. Bi-directional GRU (Bi_GRU)

8. Recurrnt neural network (RNN)

9. Bi-directional RNN (Bi_RNN)

10. Stacked Autoencoder (SAE)

11. Convolution Neural Network (CNN)

12. Multi-layer Perceptron (MLP)

13. Deep Belief Network (DBN)

## Statistical Method

1. Kalman filter (KF)
2. Hidden Markov Model (HMM)
3. Autoregerssive intergrated moving average (ARIMA) 

--------



# Requires

Python == 2.7

Tensorflow => 1.4

