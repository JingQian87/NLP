"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu> and Alyssa Hwang <ahh2143@columbia.edu> and Emily Allaway <eallaway@cs.columbia.edu>

Solution code
"""

# Imports
import numpy as np
import pickle
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim

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
FRESH_START = True  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_gen, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    last_dev_loss = np.inf
    
    # loop until the dev loss stops improving
    while True:
        # set the model in train mode (for things like dropout)
        model.train()
        # for each batch in an epoch...
        for tx,ty in train_gen:
            # zero the gradients each batch (optimizer.zero_grad() also okay)
            model.zero_grad()

            # calculate predictions and loss
            pred = model(tx)
            loss = loss_fn(pred.double(), ty.long())

            # calculate all the gradients
            loss.backward()
            # apply all the parameter updates
            optimizer.step()

        # test mode
        model.eval()
        dev_loss = 0
        # evaluation does not need gradients since we don't plan to update anything -- no_grad() makes it faster
        with torch.no_grad():
            for dx,dy in dev_generator:
                pred = model(dx)
                # simply sum up all the batch losses to get the "dev loss" (average loss is also sometimes used)
                dev_loss += loss_fn(pred.double(), dy.long()).item()
        
        print("Dev loss:", dev_loss)
        # check if the loss has gotten worse since last epoch; stop if so
        if dev_loss > last_dev_loss:
            break
        last_dev_loss = dev_loss
    
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
            gold.extend(y_b.cpu().detach().numpy().astype(int).tolist())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy().astype(int).tolist())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("=" * 20)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))
    print("=" * 20)


def main():
    """
    Perform 4-way emotion classification using two kinds of neural networks.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

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

    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # create, train, and test dense model
    print("Dense model...")
    model = models.DenseNetwork(vecs=embeddings, embed_dim=EMBEDDING_DIM, hidden_dim=64, num_labels=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    test_model(model, loss_fn, test_generator)
    
    # create, train, and test recurrent model
    print("Recurrent model...")
    model = models.RecurrentNetwork(vecs=embeddings, embed_dim=EMBEDDING_DIM, dropout=0.1, hidden_dim=64, num_layers=2, num_labels=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    test_model(model, loss_fn, test_generator)



if __name__ == '__main__':
    main()
