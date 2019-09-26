import sys
from os import path
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction import stop_words
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy import sparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class ModelStats:
    """
    Representing a model attributes and its corresponding results
    """
    
    def __init__(self, estimator = None, ngram_model = (), feature_count = 0, 
                 additional_features = None, top20_feature_names= (),
                 accuracy = 0, f1_score = 0):
        """
        Constructor using relevant model parameters
        """
        self.estimator = estimator
        self.accuracy = accuracy
        self.f1_score = f1_score
        
        self.feature_count = 0
        self.ngram_model = ngram_model
        self.additional_features = additional_features
        self.top20_feature_names = top20_feature_names
        
    def print_info(self):
        """
        print the information for this model
        """
        
        print(f"""
        ============================
        Model: {str(self.estimator)},
        
        Top 20 Features: {str(self.top20_feature_names)},
        Features Count: {str(self.feature_count)},
        N-Gram: {str(self.ngram_model)},
        Additional Feature: {str(self.additional_features)}
        Accuracy: {self.accuracy},
        F1-Score: {self.f1_score},
        
        ============================
        """)



def read_dataset(filename):
    """
    read dataset from 'filename'
    
    @param filename: file path of the dataset
    
    @return: pd.DataFrame containing the data
    """
    return pd.read_csv(filename)


def split_topic_data(all_dataframe, topic):
    """
    Get a sub-dataframe with only one topic, 
    and get its corresponding labels with pro->1, con->0 in a ndarray
    
    @param all_dataframe: data with all topics in a pd.DataFrame
    @topic: a string that specifying the topic to be seperated out
    
    @return: (sub_dataframe, labels) tuple, the 'labels' is a numpy array
    """

    # split out the topic
    sub_df = all_dataframe[all_dataframe['topic'] == topic]
    
    # get the labels
    labels = (sub_df['label'] == 'pro').values
    labels = np.where(labels,1,0)
    return sub_df, labels

def vectorize_document(document, ngram_range = (1,1)):
    """
    Vectorize the input tokenized document using CountVectorizer
    
    @param document: natural language text, a 'string' or a Series with string typep
    @param ngram_range: 2-tuple in the form of '(min,max)', specifying 
                the range of ngram to be counted.
                
    @return: feature matrix and its feature names, in a 2-tuple
            'feature_matrix' would be a scipy.sparse matrix
    """
    
    # add stopwords
    s_words = stop_words.ENGLISH_STOP_WORDS
    new_s_words = ['ve', 'youre', 'ive', 'weve', 'youve']
    new_s_words.extend(list(s_words))
    
    # vectorize the document
    cv = CountVectorizer(stop_words=new_s_words, ngram_range=ngram_range)
    feature_matrix = cv.fit_transform(document)

    return feature_matrix, cv.get_feature_names()
    
def select_k_features(feature_matrix, labels, k = 20):
    """
    Select the best top k features using 'feature_matrix' and 'labels' 
    
    @param feature_matrix: feature matrix, numpy.ndarray type
    @param labels: labels of input data
    @param k: number of features to select, if its None then select all
    
    @return the filtered feature matrix and it's corresponding feature indices Boolean condition
    """
    
    # select all features
    if not k:
        return feature_matrix, np.arange(feature_matrix.shape[1])
    
    # select top k features
    selector = SelectKBest(chi2, k=k)
    # get the corresponding index list
    k_feature_matrix = selector.fit_transform(feature_matrix, labels)
    k_feature_index = selector.get_support()
    return k_feature_matrix, k_feature_index


def k_cross_validation(estimator, feature_matrix, labels,  kfold = 5, random_state = 13):
    """
    Run k-fold cross validation using estimator on features + labels
    return the average accuracy and f1-score for it
    
    @param estimator: estimator object to make prediction
    @param feature_matrix: feature matrix of ndarray type
    @param labels: labels for the feature matrix input
    @param kfold: number of fold
    @param random_state: seed value for the kfold shuffler
    
    @return: 2-tuple of average accuracy and average f1-score
    """
    
    # sum of all-fold ressult
    accuracy = 0
    f1 = 0
    
    kf = KFold(n_splits=kfold, shuffle= True, random_state=random_state)
    
    # run each cross validation
    for train_index, test_index in kf.split(feature_matrix):
        # split data into training set and testing set
        X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # train the model
        estimator.fit(X_train, y_train)
        # make prediction
        y_pred = estimator.predict(X_test)
        # accumulate the accuracy and f1_score
        f1 += f1_score(y_test, y_pred, average ='macro', labels = np.unique(y_pred))
        accuracy += estimator.score(X_test, y_test)
    
    # take average
    return accuracy / kfold, f1 / kfold


def get_sentiment_features(topic_dataframe):
    """
    Calculate the sentiment analysis result using the input dataframe's post_text
    
    @param topic_dataframe: input dataframe with values and 'post_text' column
    
    @return an numpy array containing the feature values
    """
    
    sid = SentimentIntensityAnalyzer()
    
    # temporary store the result
    result_list = []
    # perform sentiment analysis for all rows
    for _, row in topic_dataframe.iterrows():
        result = sid.polarity_scores(row['post_text'])
        result_list.append([result['neg'], result['neu'], result['pos']])
    return np.array(result_list)

def get_POS_features(topic_dataframe):
    """
    Get the POS column values for the input dataframe
    
    @param topic_dataframe: input dataframe with values and all POS columns
    
    @return POS feature values in a numpy array
    """
    
    return topic_dataframe.iloc[:,-3:].values


def get_LIWC_features(topic_dataframe):
    """
    Get the LIWC column values for the input dataframe
    
    @param topic_dataframe: input dataframe with values and all LIWC columns
    
    @return LIWC feature values in a numpy array
    """
    
    return topic_dataframe.iloc[:,5:-3].values


def get_all_feature_matrix(topic_dataframe, labels, ngram_model = (), feature_count = (), with_additional_features = False):
    """
    Get a list of feature matrices for the 'topic_dataframe'.
    It contains ngram models with different ngram ranges and feature numbers
    If with_additional_features is True, then additional features may be included, they are:
        1. sentiment analysis result
        2. POS count
        3. LIWC stats
        
    @param topic_dataframe: input dataframe with values and 'post_text',POS, LIWC columns
    @param labels: labels for the input data
    @param ngram_model: specifying the ngram range to perform vectorization. It's a 2-tuple in a form of (min, max)
    @param feature_count: a list specifying how many features to be considered, they will be calculated using 'get_k_best_features'
    @param with_additional_features: Boolean value that determine whether to consider additional features besides n-gram 
    
    @return a generator object that can get all feature matrix specified by the input settings
    """
        # set default values
    ngram_models = ngram_model if ngram_model else [
        (1,1),  # unigram only
        (2,2),  # bigram only
        (3,3),  # trigram only
        (1,2),  # unigram + bigram
        (2,3),  # bigram + trigram
        (1,3)   # unigram + bigram + trigram
    ]
    feature_count = feature_count if feature_count else [
        3000,
        2000,
        1000,
        500
    ]
    # yield additional features matrix only
    if with_additional_features:
        # record additional features
        additional_features = []
        
        sentiment_feature_matrix = get_sentiment_features(topic_dataframe)
        POS_feature_matrix = get_POS_features(topic_dataframe)
        LIWC_feature_matrix = get_LIWC_features(topic_dataframe)
        
        sentiment_feature_names = ["negative", "neutral", "positive"]
        POS_feature_names = ["count_noun", "count_verb", "count_adj"]
        LIWC_feature_names = ["word_count", "words_pronom", "words_per_sen", "words_over_6", "pos_emo", "neg_emo"]
        
        additional_features.append({
            "ngram": (),
            "feature_count": 0,
            "features" : "Sentiment",
            "feature_matrix": sentiment_feature_matrix,
            "feature_names": sentiment_feature_names
        })
        additional_features.append({
            "features" : "POS",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": POS_feature_matrix,
            "feature_names": POS_feature_names
        })
        additional_features.append({
            "features" : "LIWC",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": LIWC_feature_matrix,
            "feature_names": LIWC_feature_names
        })
        additional_features.append({
            "features" : "Sentiment+POS",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": np.hstack((sentiment_feature_matrix, POS_feature_matrix)),
            "feature_names": sentiment_feature_names + POS_feature_names
        })
        additional_features.append({
            "features" : "Sentiment+LIWC",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": np.hstack((sentiment_feature_matrix, LIWC_feature_matrix)),
            "feature_names": sentiment_feature_names + LIWC_feature_names
        })
        additional_features.append({
            "features" : "POS+LIWC",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": np.hstack((POS_feature_matrix, LIWC_feature_matrix)),
            "feature_names": POS_feature_names + LIWC_feature_names
        })
        additional_features.append({
            "features" : "Sentiment+POS+LIWC",
            "ngram": (),
            "feature_count": 0,
            "feature_matrix": np.hstack((sentiment_feature_matrix, POS_feature_matrix, LIWC_feature_matrix)),
            "feature_names": sentiment_feature_names + POS_feature_names + LIWC_feature_names
        })
        
            
    # n-gram only
    # ngram + additional
    for lang_model in ngram_models:
        # 1. vectorize
        ngram_features_matrix, ngram_feature_names =  vectorize_document(topic_dataframe['post_text'], ngram_range=lang_model)
        for count in feature_count:    
            # 2. get the right number of features
            sub_feature_matrix, sub_feature_index = select_k_features(ngram_features_matrix, labels, k = count)
            yield {
                "ngram": lang_model,
                "feature_count": count,
                "features" : None,
                "feature_matrix" : sub_feature_matrix.toarray(),
                "feature_names" : np.array(ngram_feature_names)[sub_feature_index]
            }
            if with_additional_features:
                # yield combined feature matrix
                for additional_input in additional_features:
                    yield {
                        "ngram": lang_model,
                        "feature_count": count,
                        "features" : additional_input["features"],
                        "feature_matrix" :  np.hstack((sub_feature_matrix.toarray(), additional_input["feature_matrix"])),
                        "feature_names": list(np.array(ngram_feature_names)[sub_feature_index]) +  additional_input["feature_names"]
                    }

def get_top_model_inner(feature_matrix, labels, verbose = False):
    """
    Iterate a predefined set of Bayes and SVC models using the 'feature_matrix' and 'label'
    return the model with highest accuracy in a ModeStats object
    
    @param feature_matrix: input feature matrix of numpy array type
    @param labels: labels for the feature matrix
    @param verbose: whether print tested model information or not(default)
    
    @return: ModelStat object containing the stats for the best classifier model for the input
    """
    
    # record the best model
    best_model = ModelStats()
    
    # hyper-parameters for SVC error penality
    c_param = [1, 3, 10, 30]
    # hyper-parameter for SVC kernel type
    kernels = ['rbf' ] #, 'poly', 'sigmoid']

    # hyper-parameters for Bayes smoothing
    alpha_param = [1.0e-10, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1]

    # SVC models with all predefined hyperparameters
    for ker in kernels:
        for c in c_param:
            estimator = SVC(C = c, kernel=ker, gamma='scale')
            # run 5-fold cross validation
            accuracy, f1score = k_cross_validation(estimator, feature_matrix, labels)
            if verbose:
                print(f"SVC Model (Kernel: {ker}, C: {c}): Accuracy( {accuracy}), F1-Score( {f1score})")
            # record the model stats
            if accuracy > best_model.accuracy:
                best_model = ModelStats(
                        estimator = estimator,
                        accuracy = accuracy,
                        f1_score = f1score,
                    )
                
    # Naive Bayes models with all predefined hyperparameters
    for alpha in alpha_param:
        estimator = MultinomialNB(alpha=alpha)
        # run 5-fold cross validation
        accuracy, f1score = k_cross_validation(estimator, feature_matrix, labels)
        if verbose:
            print(f"Naive Bayes Model (Alpha: {alpha}): Accuracy( {accuracy}), F1-Score( {f1score})")
        if accuracy > best_model.accuracy:
                best_model = ModelStats(
                        estimator = estimator,
                        accuracy = accuracy,
                        f1_score = f1score
                    )
                
    return best_model


def get_top_model(feature_matrix_dict, labels, verbose = False):
    """
    Get the best model with a predefined settings of SVC and Naive Bayes model
    
    @param feature_matrix_dict: a dictionary with the following key-value pairs:
                    {
                        "ngram": ngram range,
                        "feature_count": number of features,
                        "features" : additioanl feature used besides ngram,
                        "feature_matrix" : the overall feature matrix to do model selection
                    }
    @param labels: labels for the input data
    @verbose: a boolean value, whether to print testing result and best model result for all classifiers
    
    @return: a models list containing the best models for each (ngram, feature_count, features) combination
    """
    
    # perform cross validation on a set of SVC and Bayes classfiers and get the best one
    model = get_top_model_inner(feature_matrix_dict['feature_matrix'], labels, verbose = verbose)
    # record extra information
    model.feature_count = feature_matrix_dict['feature_count']
    model.ngram_model = feature_matrix_dict['ngram']
    model.additional_features = feature_matrix_dict['features']
    # get the top 20 
    _, top20_feature_index = select_k_features(feature_matrix_dict['feature_matrix'], labels, k = 20)
    model.top20_feature_names = np.array(feature_matrix_dict['feature_names'])[top20_feature_index]
    
    if verbose:
        model.print_info()
    
    return model



def get_best_model(top_models, with_additional_features = False):
    """
    Get the model with the highest accuracy from 'top_models'
    
    @param top_models: list of ModelStats objects
    @param with_additional_features: boolean value, whether to consider models with additional features for the best model
    
    @return: a ModelStats object with highest accuracy in the 'top_models' list
    """
    
    best_model = None
    best_accuracy = -1
    for model in top_models:
        # skip models containing additional features
        if not with_additional_features and 'features' in model:
            continue

        if model.accuracy > best_accuracy:
            best_model = model
            best_accuracy = model.accuracy
    return best_model



def get_best_features(topic_df, topic_y, ngram_range, feature_count):
    """
    Get a subset of the dataframe, according to the parameters

    @param topic_df: original dataframe
    @param topic_y: labels for the dataset
    @param ngram_range: a 2-tuple, n-gram vectorization range
    @param feature_count: how many best features in the result feature matrix

    @return: (sub_feature_matrix, top20_feature_names)
    """
    feature_matrix, feature_names =  vectorize_document(topic_df['post_text'], ngram_range= ngram_range)
    sub_feature_matrix, sub_feature_index = select_k_features(feature_matrix, topic_y, k = feature_count)
    top20_feature_matrix, top20_feature_index = select_k_features(sub_feature_matrix, topic_y, k = 20)
    top20_feature_names = np.array(feature_names)[sub_feature_index][top20_feature_index]
    return sub_feature_matrix, top20_feature_names

def get_best_performance(topic, train_from_scratch = False, topic_df = None, topic_y = None, verbose = False):
    """
    Get the best two models for the gay rights topic

    @param dataframe: input dataframe
    @param labels: input labels, a numpy array
    @param train_from_scratch: a boolean value, if it is true, the best model will be calculated from scratch, which takes a few hours
            otherwise it will initialize the classifiers using hard-coded parameters for the best ngram model and custom model, 
            then get the best one between this two by comparing their accuracies
    """

    # intialize classifiers using hard-coded parameters
    if not train_from_scratch:
        if topic == "abortion":
            # 1. best ngram model for "gay rights"
            ngram_classifier = MultinomialNB(alpha=1e-10)
            # 2. get the best feature selection result and top 20 feature names
            ngram_feature_matrix, ngram_top20_features = get_best_features(topic_df, topic_y, (1,2), 3000)

            best_accuracy, best_f1 = k_cross_validation(ngram_classifier, ngram_feature_matrix, topic_y, 5)
            best20_features = ngram_top20_features
            # 2. best model with additional features
            
            # print(ngram_accuracy, ngram_f1, ngram_top20_features)

        else:
            # 1. best ngram model for "gay rights"
            classifier = SVC(C=3, kernel='rbf', gamma = 'scale')
            # 2. get the best number of features
            ngram_feature_matrix, ngram_top20_features = get_best_features(topic_df, topic_y, (3,3), 2000)
            best_accuracy, best_f1 = k_cross_validation(classifier, ngram_feature_matrix, topic_y, 5)
            best20_features = ngram_top20_features
            # 2. best model with additional features: commented out because its worse than ngram model, and it runs slower as well
            # if you want to see the result, just remove the comment and print the result

            # a. initiate the classifier
            # custom_classifier = SVC(kernel = 'rbf', C = 3, gamma = 'scale')
            # # b. add additional features
            # sentiment_matrix = get_sentiment_features(topic_df)
            # custom_feature_matrix, custom_top20_features = get_best_features(topic_df, topic_y, (3,3), 3000)
            # custom_feature_matrix = np.hstack((custom_feature_matrix.toarray(), sentiment_matrix))
            # # c. get result
            # custom_accuracy, custom_f1 = k_cross_validation(custom_classifier, custom_feature_matrix, topic_y, 5)
                    
    else:
        # otherwise train from scratch
        feature_count = [3000,2500,2000,1500,1000,500,200,100]
        top_models = []
        # iterate all settings and get the best result    
        for feature_matrix in get_all_feature_matrix(topic_df, topic_y, feature_count= feature_count, with_additional_features= False):
            model = get_top_model(feature_matrix, topic_y, verbose = verbose)
            top_models.append(model)
        best_model = get_best_model(top_models, with_additional_features = True)
        # get the best result
        best_accuracy, best_f1 =  best_model.accuracy, best_model.f1_score
        best20_features = best_model.top20_feature_names
    
    # output
    print(f"""
    For topic {topic},
    The model's average accuracy is {best_accuracy},
    Average f1 score is {best_f1},
    Top 20 feature names are: {str(best20_features)}
    """)


def main(train_from_scratch = False):
    """
    Read dataset and perform the cross validation using the best model for the topic
    Print the accuracy, f1-score and top 20 feature names for this model
    """

    # 1. get input and sanity check
    file_name = sys.argv[1]
    if not path.isfile(file_name):
        raise Exception("File does not exist.")
    
    topic_name = sys.argv[2]
    if not topic_name or topic_name not in ('abortion', 'gay rights'):
        raise Exception("Topic value error.")
    
    # 2. read the data
    data_df = read_dataset(file_name)
    
    # 3. split out the topic
    topic_df, topic_y = split_topic_data(data_df, topic_name)
    
    # 4. output the best performance
    get_best_performance(topic_name, train_from_scratch = train_from_scratch, topic_df = topic_df, topic_y = topic_y, verbose= False)

if __name__ == "__main__":
    main(False)

