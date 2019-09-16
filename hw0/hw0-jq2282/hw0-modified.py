"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# Modified for Columbia's COMS4705 Fall 2019 by
# Elsbeth Turcan <eturcan@cs.columbia.edu>

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(4705)
kn = 30

if __name__ == "__main__":
    # collect the training data_tutorial
    movie_reviews_data_folder = "data/txt_sentoken/"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent and uses uni- and bigrams
    clf = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1, 2))),
        ('clf', KNeighborsClassifier(n_neighbors=kn)),
    ])

    # Train the classifier on the training set
    clf.fit(docs_train, y_train)

    # Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = clf.predict(docs_test)

    # Get the probabilities you'll need for the precision-recall curve
    y_probs = clf.predict_proba(docs_test)[:, 1]

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # TODO: calculate and plot the precision-recall curve
    # HINT: Take a look at scikit-learn's documentation linked in the homework PDF,
    from sklearn.metrics import precision_recall_curve
    from inspect import signature
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    #print(precision, recall)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('precision recall curve (%s neighbors)' %kn)
    plt.savefig('precision_recall_curve%s.png'%kn)


    # and/or find an example of this curve being plotted.
    # You should use the y_probs calculated above as an argument...
