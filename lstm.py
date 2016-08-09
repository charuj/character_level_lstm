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


class Model():
    def __init__(self, args):
        self.args= args
        args.batch_size= 50
        args.lstm_size= 512 # this is the number of nodes per hidden layer
        args.seq_length= 20 # number of steps to unroll the LSTM
        args.num_layers= 2
        args.learning_rate= 0.001
        args.vocab_size= #todo: how should I represent vocab? 1-Hot?

        # Create multi-layer LSTM cell
        lstm= rnn_cell.BasicLSTMCell(args.lstm_size) # size of a single LSTM cell
        self.lstm= rnn_cell.MultiRNNCell([lstm]*args.num_layers)  # creating a multi-layer LSTM; the advantage of this is that the number of layers can be changed easily
        self.initial_state= lstm.zero_state(args.batch_size,tf.float32)

        # Create placeholders for inputs and targets
        self.input_data= tf.placeholder(tf.float32,[args.batch_size, args.seq_length]) #TODO: should data be float or int?
        self.targets = tf.placeholder(tf.float32,[args.batch_size,args.seq_length])

        with tf.variable_scope('lstm-'):
            ''' The purpose of using variable_scope is to easily share named variables when creating a graph.
            Instead of manually naming all the variables. This allows for scaling up (i.e. increase num_layers or size)

            see: https://www.tensorflow.org/versions/r0.10/how_tos/variable_scope/index.html
            '''

            # Variables created here will be named "lstm-/weights", "lstm-/biases"

            # Create variable named 'weights'. Uses an initializer instead of passing the value directly as in tf.Variable
            weights= tf.get_variable('weights',[args.lstm_size,args.vocab_size], initializer=tf.random_normal_initializer())

            # Create a variable named "biases". Initialize to value of 0
            biases = tf.get_variable('biases',args.vocab_size, initializer=tf.constant_initializer(0.0))

        






