import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import numpy as np
import sys

'''
COMS 4705 Fall 2019
Natural Language Processing - Kathleen McKeown
Homework 1 Solution -- DO NOT DISTRIBUTE
Author: Emily Allaway
'''


def load_data(data_name, topic):
    '''
    Loadsd the data for a specific topic.
    :param data_name: The name of the datafile.
    :param topic: The topic to load for.
    :return: the data for the given topic as a Dataframe,
            the input posts as a list,
            the labels as a numpy array
    '''
    df = pd.read_csv(data_name)

    top_df = df.loc[df['topic'] == topic]
    labels = np.array(top_df.label == 'pro').astype(int)

    return top_df, list(top_df['post_text']), labels


def get_author_features(data_in, train_index, test_index, aidx):
    '''
    Computes the author feature on the given data.
    :param data_in: all the data.
    :param train_index: the indices of the train data
    :param test_index: the indices of the test data
    :param aidx: the index of the author feature
    :return: the author feature for train,
            the author feature for test
    '''
    dict_vectorizer = DictVectorizer(sparse=False)

    # prepare authors from training data ONLY
    temp_trn = data_in[train_index, :][:, aidx]
    a2i = {w: i for i, w in enumerate(set(temp_trn))}
    auth_in = [{w: a2i[w]} for w in temp_trn]

    # fit vectorizer on training authors ONLY, and transform the training data
    X_a = dict_vectorizer.fit_transform(auth_in)

    # prepare authors from test data, using new index for unseen authors
    test_auth_in = [{w: a2i.get(w, len(temp_trn))} for w in data_in[test_index, :][:, aidx]]

    # transform the test data using the vectorizer fit on training data
    X_test_a = dict_vectorizer.transform(test_auth_in)
    return X_a, X_test_a


def get_features(feats, ngrams, data_in, train_index, test_index):
    '''
    Converts the data to features, using the specified ngram range.
    :param feats: the features to use.
    :param ngrams: the ngram range to use.
    :param data_in: the input data.
    :param train_index: the indices of the training data.
    :param test_index: the indices of the test data.
    :return: the train features,
            the test features.
    '''
    vectorizer = CountVectorizer(ngram_range=ngrams, min_df=30)

    # fit ngram vectorizer on the training data and transform the data to features
    X_txt = vectorizer.fit_transform(data_in[train_index, :][:, 0]).toarray()

    # transform the test data to features using the vectorizer fit on the training data
    X_test = vectorizer.transform(data_in[test_index, :][:, 0]).toarray()

    if len(feats) != 0:
        trn_feats = []
        test_feats = []
        if 'author' in feats:
            # load the author features and add them to the training and test data
            X_a, X_test_a = get_author_features(data_in, train_index, test_index, feats.index('author') + 1)
            trn_feats.append(X_a)
            test_feats.append(X_test_a)

        # load additional features for the data and add them to the training and test data
        for c in set(feats) - {'author'}:
            fidx = feats.index(c) + 1
            trn_feats.append(data_in[train_index, :][:, fidx].astype(float).reshape(-1, 1))
            test_feats.append(data_in[test_index, :][:, fidx].astype(float).reshape(-1, 1))

        # combine all additional features with ngram features
        X_feats = np.concatenate([X_txt] + trn_feats, axis=1)
        X_test_feats = np.concatenate([X_test] + test_feats, axis=1)
    else:
        X_feats = X_txt
        X_test_feats = X_test

    return X_feats, X_test_feats


def train_and_eval(data_name, topic):
    '''
    Trains and evaluates a model using 5-fold cross validation.
    Runs feature selection with various features.
    Prints the averaged accuracy and F1-score.
    :param data_name: The name of the data file to use.
    :param topic: The topic to use.
    '''
    liwc_feats = ['word_count', 'words_pronom', 'words_per_sen', 'words_over_6', 'pos_emo', 'neg_emo']
    pos_feats = ['count_noun', 'count_verb', 'count_adj']
    other_feats = ['author']

    np.random.seed(0)

    # load data and labels
    top_df, txt_data, labels = load_data(data_name, topic)

    print('TOPIC {}:'.format(topic))

    kf = KFold(n_splits=5, shuffle=True)

    for feats in [[], other_feats, liwc_feats, pos_feats]: # choose additional features to use
        for ngrams in [(1, 1), (1, 2), (1, 3)]: # choose ngrams to use
            print("Feats = {}".format(feats + ['ngrams = {}'.format(ngrams)]))

            # prepare data for splitting
            if len(feats) != 0:
                feat_cols = []
                for c in feats:
                    feat_cols.append(np.array(top_df[c]))
                data_in = np.vstack([np.array(txt_data)] + feat_cols).T
            else:
                data_in = np.array(txt_data).reshape(-1, 1)

            ########################
            # run cross-validation #
            ########################
            name2score = dict()
            for train_index, test_index in kf.split(X=data_in, y=labels):
                # get features
                X_feats, X_test_feats = get_features(feats=feats, ngrams=ngrams,
                                                    data_in=data_in, train_index=train_index,
                                                    test_index=test_index)

                # initialize models
                models = {'mnb': MultinomialNB(),
                          'svm rbf': SVC(gamma='scale'),
                          'svm linear': SVC(gamma='scale', kernel='linear'),
                          'svm poly': SVC(gamma='scale',  kernel='poly')}

                # fit models and predict
                for mname, m in models.items():
                    # train model
                    m.fit(X_feats, labels[train_index])
                    # make predictions with model
                    ypred = m.predict(X_test_feats)
                    # update scores
                    name2score[mname] = name2score.get(mname, [[], []])
                    name2score[mname][0] += [accuracy_score(labels[test_index], ypred)]
                    name2score[mname][1] += [f1_score(labels[test_index], ypred)]

            # print averaged scores
            for m, scores in name2score.items():
                print("{}: acc = {:.2f}, f1 = {:.2f}".format(m, np.mean(scores[0]), np.mean(scores[1])))
            print()
            ### end cross-validation ###


if __name__ == '__main__':
    '''
    Run with either:
    python hw1-classify.py stance-data.csv abortion
    or 
    python hw1-classify.py stance-data.csv "gay rights"
    '''
    train_and_eval(sys.argv[1], sys.argv[2])