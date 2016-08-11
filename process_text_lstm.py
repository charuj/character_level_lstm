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


# Load text file/corpus; I am loading the text file as a string variable, and removing newline. May need to play around with this
with open('filename.txt', 'r') as myfile:
    data= myfile.read().replace('\n', '') # remove new line. Prevents newlines from being turned into a list





