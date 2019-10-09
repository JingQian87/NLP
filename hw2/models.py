"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<YOUR NAME HERE>
<YOUR UNI HERE>
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        raise NotImplementedError

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        raise NotImplementedError


class RecurrentNetwork(nn.Module):
    def __init__(self):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        raise NotImplementedError


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        raise NotImplementedError
