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

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.models.rnn import rnn
import process_text_lstm


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
        lstm_cell= tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # using default forget_bias=1.0

        # Add dropout
        if is_training and config.keep_prob <1:
            lstm_cell= tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=config.keep_prob) #dropout of the outputs based on keep_prob

        # Creating a multi-layer LSTM; the advantage of this is that the number of layers can be changed easily
        self.lstm= lstm= tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers)
        self.initial_state = lstm.zero_state(batch_size, tf.int32)


        with tf.variable_scope('lstm-'):

            '''The purpose of using variable_scope is to easily share named variables when creating a graph,
            instead of manually naming all the variables. This allows for scaling up (i.e. increase num_layers or size)
            see: https://www.tensorflow.org/versions/r0.10/how_tos/variable_scope/index.htm'''

            # Variables created here will be named "lstm-/weights", "lstm-/biases"

            # Create variable named 'weights'. Uses an initializer instead of passing the value directly as in tf.Variable
            weights= tf.get_variable('weights',[hidden_size, vocab_size], initializer=tf.random_normal_initializer())

            # Create a variable named "biases". Initialize to value of 0
            biases = tf.get_variable('biases', vocab_size, initializer=tf.constant_initializer(0.0))

            # Character IDs will be embedded into a dense vector representation before being fed into the LSTM
            # The embedding matrix 'embedding' is a tensor of shape [vocab_size, embedding size]
            embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data) #Character embeddings, where input_data =character IDs

            inputs = [tf.squeeze(inputs, [1]) for inputs in tf.split(1, num_steps, inputs)]  # build unrolled LSTM
            if is_training and config.keep_prob <1:
                inputs= tf.nn.dropout(inputs, config.keep_prob) # Apply dropout in inputs

        outputs, state = rnn.rnn(lstm, inputs, initial_state=self.initial_state)
        # TODO: May need to reshape the output

        self.logits= tf.matmul(outputs, weights) + biases
        #TODO: do I need self.probs = tf.nn.softmax(self.logits)  ???

        #Define loss and optimizer
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets))
        # ^^ The tensorflow tutorial uses seq2seq loss... TODO: see if this works or use seq2seq

        optimizer= tf.train.AdamOptimizer(learning_rate= config.learning_rate).minimize(loss)
        self.final_state= state

        # Evaluate model for accuracy
        correct_pred = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.targets,1))
        accuracy= tf.reduce_mean(tf.cast(correct_pred,tf.float32))


class config(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    keep_prob = 0.5
    num_batches=
    vocab_size = ### det vocab size of dataset in terms of characters, alphanumeric + symbols?


# Loading the data

# Initializing the variables (this may be redundant since I used get_variable()
init= tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range (num_epochs):
        state= Model.initial_state
        for j in range(num_batches):
            x, y = process_text_lstm.batch_generator(FILL)
            feed ={Model.input_data:x, Model.targets:y, Model.initial_state:state}
            











