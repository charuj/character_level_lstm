'''
Character-level LSTM to generate text.

Uses built-in rnn_cell from Tensorflow

Characteristics :
    -2 layer lstm
    -512 hidden nodes
    -Dropout of 0.5 after each layer
    -Train with batches of 100 examples
    -Truncated BPTT length of 100 char


Based on:

https://github.com/hunkim/word-rnn-tensorflow/blob/master/model.py   [I'm not using seq2seq]

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py


'''


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

# Parameters

training_iters=100000

# TF Graph input
input_data= tf.placeholder(tf.float32, [batch_size, seq_length])
targets=tf.placeholder(tf.float32,[batch_size, seq_length])

class Model():
    def __init__(self, args):
        self.args= args
        args.batch_size= 50
        args.lstm_size= 512 # this is the number of nodes per hidden layer
        args.seq_length= 20 # number of steps to unroll the LSTM
        args.num_layers= 2
        args.learning_rate= 0.001

        # Create multi-layer LSTM cell
        lstm= rnn_cell.BasicLSTMCell(args.lstm_size) # size of a single LSTM cell
        self.lstm= rnn_cell.MultiRNNCell([lstm]*args.num_layers)  # creating a multi-layer LSTM; the advantage of this is that the number of layers can be changed easily
        self.initial_state= lstm.zero_state(args.batch_size,tf.float32)

        # Create placeholders for inputs and targets
        self.input_data= tf.placeholder(tf.float32,[args.batch_size, args.seq_length]) #TODO: should data be float or int?
        self.targets = tf.placeholder(tf.float32,[args.batch_size,args.seq_length])




