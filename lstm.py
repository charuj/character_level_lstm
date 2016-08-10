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

ALSO check out the official Tensorflow PTB tutorial: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py

The parameters used in this model:

- learning_rate - the initial value of the learning rate
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- keep_prob - the probability of keeping weights in the dropout layer. **THIS LSTM USES DROPUT!**
- batch_size - the batch size



'''


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

# Parameters

training_iters=100000


class Model():
    def __init__(self, config, is_training):
        self.batch_size= batch_size= config.batch_size
        self.num_steps= num_steps= config.num_steps
        hidden_size= config.hidden_size
        vocab_size= config.vocab_size


        # Create placeholders for inputs and targets
        self.input_data= tf.placeholder(tf.int32,[batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        # Create multi-layer LSTM cell
        lstm_cell= tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # using default forget_bias=1.0, # TODO: try with forget_bias=0.0

        # Add dropout
        if is_training and config.keep_prob <1:
            lstm_cell= tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=config.keep_prob) #dropout of the outputs based on keep_prob

        # Creating a multi-layer LSTM; the advantage of this is that the number of layers can be changed easily
        self.lstm= lstm= tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers)
        self.initial_state = lstm.zero_state(batch_size, tf.int32)


        with tf.variable_scope('lstm-'):

            '''The purpose of using variable_scope is to easily share named variables when creating a graph.
            #Instead of manually naming all the variables. This allows for scaling up (i.e. increase num_layers or size)
            #see: https://www.tensorflow.org/versions/r0.10/how_tos/variable_scope/index.htm'''

            # Variables created here will be named "lstm-/weights", "lstm-/biases"

            # Create variable named 'weights'. Uses an initializer instead of passing the value directly as in tf.Variable
            weights= tf.get_variable('weights',[hidden_size, vocab_size], initializer=tf.random_normal_initializer())

            # Create a variable named "biases". Initialize to value of 0
            biases = tf.get_variable('biases', vocab_size, initializer=tf.constant_initializer(0.0))


            ########TODO: Understand this. Do I really need this to reshape the data? Can I write it as a function instead?
            embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            inputs = tf.split(1, num_steps, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            if is_training and config.keep_prob <1:
                inputs= tf.nn.dropout(inputs, config.keep_prob) # Apply dropout in inputs


        def data_shape(self, inputs, weights, biases):
            '''
            Reshape the data so that it can be used by the LSTM cell.
            Right now the input data has the shape (batch_size,
            '''

            #TODO: do I need to split the input into tensors of dimension (batch_size, seq_length)?
            # TODO: How does splitting the tensors consider 1-hot encoding of the data?

            # I believe the input embeddings are of shape [args.vocab_size, args.lstm_size]
            # where vocab_size represents the size with 1-hot encoding











