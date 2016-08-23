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


# Parameters
batch_size= 5
num_unrollings = 10
valid_size= 100


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
vocabulary = " " + letters + digits + puncuation  # space is index 0
print vocabulary
vocabulary_size= len(vocabulary)
print vocabulary_size
first_char= ord(vocabulary[0])  #returns unicode value of first character


def one_hot(data, vocabulary, vocabulary_size):
    '''

    :param data: string of characters that needs to be converted into 1-hot encoding
    :param vocabulary_size: number of characters in the vocabulary
            vocabulary: string of characters that are in the vocabulary
    :return: data matrix as 1-hot encoding
    '''

    # First: define what's needed for the columns, that is create an array of numbers, where each number represents the index of the character in the vocabulary
    data_len= len(data)
    list_indexes = []
    for i in range(data_len):
        if data[i] in vocabulary:
            index_value= vocabulary.index(data[i])
        else:
            index_value = 0
        list_indexes.append(index_value)

    matrix_indexes = np.asarray(list_indexes) # convert the above-createdlist of indexes to an array

    #2nd: create zero matrix and fill
    one_hot_matrix= np.zeros([vocabulary_size, len(matrix_indexes)])
    for i in range(len(matrix_indexes)):
        one_hot_matrix[matrix_indexes[i],i] = 1

    return one_hot_matrix



def batch_generator(data_1hot, batch_size, num_unrollings, vocabulary_size):
    '''

        Generates batches of the data and allows for minibatch iterations.
        This function is based on the function 'ptb_iterator' in the tensorflow PTB tutorial.

        :param data_1hot: array of data as 1-hot encodings, dims vocabulary_size x data_len
        :param batch_size: int, the batch size
        :param num_unrollings: int, number of unrolls in the LSTM. Should be the same as in config.num_steps in the file lstm.py
        :return: Two 3D arrays  of batched data.
                Dimension = batch_len x vocabulary_size x num_unrollings
                The first set of batched data is the inputs, the second is the targets (time-shifted to the right by one).
    '''

    data_len= len(data_1hot[0])  # number of columns
    batch_len= data_len // batch_size
    # data= np.zeros([batch_size,vocabulary_size, num_unrollings])
    input_array_list= []
    target_array_list= []

    # Make a list of arrays for the inputs and targets, where each element of the list is of dim vocab_size x num_unrollings
    for i in range(batch_len):
        input_array = data_1hot[i*num_unrollings: (i+1) * num_unrollings]
        target_array = data_1hot[i*num_unrollings + 1 : (i+1) * num_unrollings + 1]
        input_array_list.append(input_array)
        target_array_list.append(target_array)


    stacked= np.concatenate(input_array_list, axis=0)

    # Stack the above-created arrays; dim = vocab_size x num_unrollings x batch_len















    #
    #
    # #data= np.zeros([batch_size, batch_len], dtype=np.int32) # create an empty matrix, where each row will be a batch (batch_size x batch_len)
    # for i in range(batch_size):
    #
    #     data[i]= data_as_id[batch_len * i: batch_len * (i+1)] # populate the data array, where each row is a batch
    #
    # num_epochs= (batch_len -1)//num_unrollings  # the number of time that num_unrollings fits into the big data matrix
    # if num_epochs == 0:
    #     raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    #
    # # Making array that is shape [batch_size, num_unrollings].
    # for i in range(num_epochs):
    #     x = data[:, i * num_unrollings:(i + 1) * num_unrollings] # inputs
    #     y = data[:, i* num_unrollings + 1: (i+1)*num_unrollings+1] # targets
    # return  x, y
    #


### MAIN ###

data= read_date('nytimes_char_rnn.txt')
valid_set, training_set, train_size= create_sets(data, valid_size)
valid_1hot= one_hot(valid_set, vocabulary,vocabulary_size)
train_1hot= one_hot(training_set, vocabulary, vocabulary_size)


# put training and validation 1hots into pickle files
pickle.dump(valid_1hot, open( "valid_1hot.p", "wb" ) )
pickle.dump(train_1hot, open( "train_1hot.p", "wb" ) )


valid_batch_input = batch_generator(valid_1hot,batch_size,num_unrollings,vocabulary_size)


#
# # Make and pickle validation batches
# validx, validy = batch_generator(valid_1hot,batch_size,num_unrollings) # batch_size = 5, num_unrollings = 10 (this should match lstm file )
# pickle.dump(validx, open( "validx.p", "wb" ) )
# pickle.dump(validy, open( "validy.p", "wb" ) )
#
# # Make and pickle training batches
# trainx, trainy= batch_generator(train_1hot,batch_size,num_unrollings)
# pickle.dump(trainx, open( "trainx.p", "wb" ) )
# pickle.dump(trainy, open( "trainy.p", "wb" ) )
#
#
#





#### APENDIX (CODE I NO LONGER NEED BUT WORKS) #####

'''

Instead of convering characters to IDs, I convert them to one-hot encoding based on the size of the vocabulary (69)

def char2id(vocabulary, data):


    # :param char: character from corpus string
    # :param vocabulary: list of characters in vocabulary
    # :return: turns a character into an ID corresponding to its UNICODE value


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

    #
    # :param id: unicode ID of a character, obtained from the function char2id
    # :return: the actual character corresponding to the ID

    id2char_readout=[]
    for id in list_id:
        if id > 0:
             id2char_readout.append(chr(id + first_char - 1))
        else:
            id2char_readout.append(' ')
    return id2char_readout
'''
