"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu> and Emily Allaway <eallaway@cs.columbia.edu>

Solution code
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

# define dense network
class DenseNetwork(nn.Module):
    def __init__(self, **params):
        super(DenseNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # create an Embedding layer using the pretrained embeddings; freeze=False allows us to keep training it
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(params['vecs']), freeze=False)
        # a two-layer feedforward network goes dense layer -> nonlinearity -> dense layer -> output
        # the first dense layer goes 100 -> 64 and the second goes 64 -> 4
        self.ffnn = nn.Sequential(nn.Linear(params['embed_dim'], params['hidden_dim']), 
                                  nn.Tanh(), 
                                  nn.Linear(params['hidden_dim'], params['num_labels']))

    def forward(self, x): # x = (B, L)
        ########## YOUR CODE HERE ##########
        embeds = self.embedding(x).float()
        # take the sum of all embeddings and pass it through the feedforward network
        pred = self.ffnn(embeds.sum(1))
        # we don't need to softmax this output as CrossEntropyLoss does that for us
        return pred

# define recurrent network
class RecurrentNetwork(nn.Module):
    def __init__(self, **params):
        super(RecurrentNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # same Embedding layer as above
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(params['vecs']), freeze=False)
        # create a recurrent layer (here, a GRU) which outputs a hidden state of 64
        self.rnn = nn.GRU(params['embed_dim'], params['hidden_dim'], num_layers=params['num_layers'],
                           batch_first=True, dropout=params['dropout'])
        # and the final projection layer (64 -> 4)
        self.ffnn = nn.Linear(params['hidden_dim'], params['num_labels'])


    # x is a PaddedSequence for an RNN, shape (B, L)
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        embeds = self.embedding(x).float()
        lens = (x != 0).sum(1)
        # pack the sequence instead of padding so that we no longer have a bunch of 0's to feed into our RNN
        # (Embedding layers don't like packed sequences, which is why we started with a padded sequence)
        p_embeds = rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(p_embeds)
        # take the final hidden state and feed it through the final dense layer
        hns = hn.split(1, dim=0)
        last_hn = hns[-1]
        pred = self.ffnn(last_hn.squeeze(0))
        return pred