'''
Code for processing text files for LSTM.

- Loads raw text data
- reads file
- converts character strings into integer IDs
- makes mini-batches with the inputs

Inspired by how Udacity's Tensorflow LSTM tutorial converts characters to IDs and vice versa.
See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb

'''

import numpy as np
import string
import cPickle as pickle

def read_date(filename):
    '''
    filename: textfile containing corpus.
    Returns: data; text written to string.
    '''

    with open(filename, 'r') as myfile:
        data= myfile.read().replace('\n', '') # remove new line. Prevents newlines from being turned into a list

    return data.lower()

def create_sets(data, valid_size):
    '''

    :param data: corpus in string format
    :param valid_size: desired size of validation set; a number of characters.
    :return: validation set, training set, size of training set
    '''

    valid_set =  data[:valid_size]  # the first valid_size characters become the validation set
    training_set= data[valid_size:]
    train_size= len(training_set)

    return valid_set, training_set, train_size

## Map characters to vocabulary IDs and back. Based on Udacity TF
'''
NOTE: I could alternatively use the tf tutorial Counter() method to create a dictionary that maps a character
to its number of occurrences, but that seems computationally unncessary.

'''

#vocabulary_size = len(string.ascii_lowercase) + len(string.digits) + len(string.punctuation) +  1   # [a-z] + [0123456789] + [punctuation] + ' '
# the vocab takes into account letters, numbers, and puncutation
letters= string.ascii_lowercase
digits= string.digits
puncuation= string.punctuation
vocabulary = letters + digits + puncuation
vocabulary_size= len(vocabulary) + 1 # 1 for ' ' space
first_char= ord(vocabulary[0])  #returns unicode value of first character

def char2id(vocabulary, data):
    '''

    :param char: character from corpus string
    :param vocabulary: list of characters in vocabulary
    :return: turns a character into an ID corresponding to its UNICODE value
    '''

    list_id=[]

    for char in data:

        if char in vocabulary:
            id= ord(char) - first_char + 1
        elif char==' ':
            id= 0
        else:
            #print 'unexpected character: %s' % char
            id= 0
        list_id.append(id)
    return list_id


def id2char(list_id):
    '''

    :param id: unicode ID of a character, obtained from the function char2id
    :return: the actual character corresponding to the ID
    '''
    id2char_readout=[]
    for id in list_id:
        if id > 0:
             id2char_readout.append(chr(id + first_char - 1))
        else:
            id2char_readout.append(' ')
    return id2char_readout


def batch_generator(data_as_id, batch_size, num_steps):
    '''

        Generates batches of the data and allows for minibatch iterations.
        This function is based on the function 'ptb_iterator' in the tensorflow PTB tutorial.

        :param data: string of data in ID format (i.e. either the validation or training set).
        :param batch_size: int, the batch size
        :param num_steps: int, number of unrolls in the LSTM. Should be the same as in config.num_steps in the file lstm.py
        :return: pairs of batched data in matrix of shape [batch_size, num_steps].
                The first set of batched data is the inputs, the second is the targets (time-shifted to the right by one).
    '''


    data_as_id= np.array(data_as_id, dtype=np.int32)
    data_len= len(data_as_id)
    batch_len= data_len // batch_size
    data= np.zeros([batch_size, batch_len], dtype=np.int32) # create an empty matrix, where each row will be a batch (batch_size x batch_len)
    for i in range(batch_size):
        data[i]= data_as_id[batch_len * i: batch_len * (i+1)] # populate the data array, where each row is a batch

    return data





### MAIN ###

data= read_date('nytimes_char_rnn.txt')
valid_set, training_set, train_size= create_sets(data, 100)
valid_id= char2id(vocabulary,valid_set)

valid_batch= batch_generator(valid_id,5,10)
print valid_batch
valid_readout = id2char(valid_id)
#pickle.dump( valid_batch, open( "valid_batch.p", "wb" ) )



'''
TODO: redo batch generation, maybe I don't need to create separate arrays for the inputs and targets
but can do that in the actual TF Graph

Todo: softmax for prediction ?

'''

