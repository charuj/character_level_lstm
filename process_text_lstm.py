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

def read_date(filename):
    '''
    filename: textfile containing corpus.
    Returns: data; text written to string.
    '''

    with open('filename.txt', 'r') as myfile:
        data= myfile.read().replace('\n', '') # remove new line. Prevents newlines from being turned into a list

    return data

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

def char2id(char, vocabulary):
    '''

    :param char: character from corpus string
    :param vocabulary: list of characters in vocabulary
    :return: turns a character into an ID corresponding to its UNICODE value
    '''
    if char in vocabulary:
        return ord(char) - first_char + 1
    elif char==' ':
        return 0
    else:
        print 'unexpected character: %s' % char
        return 0

def id2char(id):
    '''

    :param id: unicode ID of a character, obtained from the function char2id
    :return: the actual character corresponding to the ID
    '''
    if id > 0:
        return chr(id + first_char - 1)
    else:
        return ' '


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

    # batch size check, to see if it needs to be adjusted
    epoch_size= (batch_len -1) // num_steps
    if epoch_size ==0:
        raise ValueError("Epoch size = 0; decrease batch_size or num_steps")

    data = np.zeros([batch_size, batch_len], dtype =np.int32) # create an empty zero matrix to later fill with batches
    for i in range(batch_size):
        data[i]= data_as_id[batch_len*1: batch_len*(i+1)]

    for i in range(epoch_size): # TODO: write this with out generator? What shape/ form is it ??
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)







