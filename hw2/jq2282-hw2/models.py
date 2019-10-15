"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<Jing Qian>
<jq2282>
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, embed_dim, output_dim, hidden_dim, weight):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        # Define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(weight)
        # Define two dense layers
        self.dense1 = nn.Linear(embed_dim, hidden_dim) 
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        # Define the activation layer
        self.relu = nn.ReLU()     

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        x = self.embedding(x)
        # TODO: 2) Take the sum of all word embeddings in a sentence
        x = torch.sum(x,dim=1).float()
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class RecurrentNetwork(nn.Module):
    def __init__(self, sentence_len, output_dim, hidden_dim, weight):
        super(RecurrentNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        # Define dimensions
        self.hidden_dim = hidden_dim
        self.sentence_len = sentence_len
        self.embed_dim = weight.size(1)       
        # Define embeddings layer
        self.embedding = nn.Embedding.from_pretrained(weight)
        # Define 2-layer RNN
        self.rnn = nn.RNN(self.embed_dim,hidden_dim,num_layers=2,batch_first=True)     
        # Define the final dense layer 
        self.fc = nn.Linear(hidden_dim, output_dim)

    # Get the non-zero lengths of the PaddedSequence x.
    def get_len(self, x):
        x_len = []
        for ix in x:
            if ix[-1] != 0:
                x_len.append(len(ix))
            else:
                x_len.append((ix==0).nonzero()[0])           
        return x_len

    # Pad the sentence length of x to the parameter sentence_len.
    def pad(self, x):
        if x.size(1) > self.sentence_len:
            return x[:,:self.sentence_len]
        elif x.size(1) < self.sentence_len:
            tmp = torch.zeros(x.size(0), self.sentence_len-x.size(1), dtype=torch.long)
            return torch.cat((x,tmp), 1)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        batch_size = x.size(0)
        # Get the sentence length of x before padding
        x_lengths = self.get_len(x)
        # Pad x to the parameter sentence_len
        if x.size(1) != self.sentence_len:
            x = self.pad(x)
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        x = self.embedding(x).float()
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        out, _ = self.rnn(x)
        # Select the last output state according to the unpadded sentence length.
        selected = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float)
        for i, l in enumerate(x_lengths):
            selected[i,:] = out[i,l-1,:]
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        return self.fc(selected)


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    # extension-grading: Extension 2, architecture changes - flattening embeddings using the average of unpadded sentence words other than sum. 
    def __init__(self, embed_dim, output_dim, hidden_dim, weight):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.dense1 = nn.Linear(embed_dim, hidden_dim) 
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU() 

    # Get the non-zero lengths of the PaddedSequence x.
    def get_len(self, x):
        x_len = []
        for ix in x:
            if ix[-1] != 0:
                x_len.append(len(ix)*1.0)
            else:
                x_len.append((ix==0).nonzero()[0])           
        return x_len

    # x is a PaddedSequence 
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # Get the sentence length of x before padding
        x_lengths = self.get_len(x)
        # Put the words through an Embedding layer (which was initialized with the pretrained embeddings)
        x = self.embedding(x)
        # Take the averaged embeddings over sentence length
        y = torch.zeros(x.size(0), x.size(2), dtype=torch.float)
        for i in range(x.size(0)):
            for j in range(x.size(2)):
                y[i][j] = x[i,:,j].sum()/x_lengths[i]
        # Feed the result into 2-layer feedforward network which produces a 4-vector of values
        y = self.dense1(y)
        y = self.relu(y)
        y = self.dense2(y)
        return y
