Code for COMS 4705 HW3.
Jing Qian (jq2282) 

The code for section 1 in HW3 is "hw3_jq2282.py". This script generates two word embedding models (word2vec skip-gram with negative sampling and SVD on the positive PMI matrix) with different parameter sets. The corpus used in the script is "brown.txt" and the hyperparameters are:
* "windowList": Context window size
* "dimList": Dimension
* "numNS": Number of negative samples (applicable to SGNS only).

In the main function, run "train_word2vec()" to train word2vec and run "svd_ppmi()" to run SVD on the positive PMI matrix. The generated word vectors would be saved.

