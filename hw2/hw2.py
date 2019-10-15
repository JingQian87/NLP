"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<Jing Qian>
<jq2282>
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    EPOCHS = 20
    dev_losses = []
    for iepoch in range(EPOCHS): 
        # TODO: 1) Loop through the whole train dataset performing batch optimization with torch.optim.Adam
        for train_batch, train_label in train_generator:
            # Compute the loss
            loss = loss_fn(model(train_batch),train_label)
            # Zero the gradients
            model.zero_grad()
            # perform a backward pass (backpropagation)
            loss.backward()
            # Update the parameters
            optimizer.step()

        # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
        dev_loss = 0
        for ibatch, ilabel in dev_generator:
            dev_loss += loss_fn(model(ibatch), ilabel)

        # TODO: Make sure to print the dev set loss each epoch to stdout.
        print(iepoch+1, dev_loss)
        dev_losses.append(dev_loss)

        # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
        if iepoch > 1 and dev_losses[-2]-dev_loss < 0.01:
            break
    return model

def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main():
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test, BATCH_SIZE, EMBEDDING_DIM)
        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    HIDDEN_DIM = 64
    # TODO: for each of the two models, you should 1) create it,
    dnn = models.DenseNetwork(EMBEDDING_DIM, NUM_CLASSES, HIDDEN_DIM, embeddings)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dnn.parameters())
    # TODO 2) run train_model() to train it, and
    trained_dnn = train_model(dnn, loss_fn, optimizer, train_generator, dev_generator)
    DNN_PATH = 'dense.pth'
    torch.save(dnn, DNN_PATH)
    # TODO: 3) run test_model() on the result
    dnn_test = torch.load(DNN_PATH)
    test_model(dnn_test, loss_fn, test_generator)


if __name__ == '__main__':
    main()
