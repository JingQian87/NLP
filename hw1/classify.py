# python classify.py train-data.csv abortion

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
np.random.seed(4705)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack, vstack

#def best_params(topic, method):
# topic == 'abortion' and method == 'Ngrams', MultinomialNB
# 0.6118245271046628
# clf__alpha: 1
# exf__k: 500	

# topic == 'abortion' and method == 'Ngrams', LinearSVC
# accuracy = 0.6229 with params:
# clf__C: 1
# clf__class_weight: None
# clf__loss: 'hinge'
# clf__max_iter: 1000
# exf__k: 500

# topic == 'gay rights' and method == 'Ngrams', MultinomialNB
# 0.6283767892623981
# clf__alpha: 1
# exf__k: 50

# topic == 'gay rights' and method == 'Ngrams', LinearSVC
# 0.6306178446399848
# clf__C: 100
# clf__class_weight: None
# clf__loss: 'hinge'
# clf__max_iter: 1000
# exf__k: 50

def load_data(filename, topic):
	raw = pd.read_csv(filename)
	print("Loading %s with %d records" %(filename,len(raw)))
	print("Column names: ", raw.columns)

	selected = raw[raw.topic == topic]
	selected = np.random.permutation(selected)
	return selected

def get_features(selected, method):
	X = CountVectorizer(stop_words='english', ngram_range=(1,3))
	X = X.fit_transform(selected[:,0])
	if method != "Ngrams":
		X = hstack((X, selected[:,5:8].astype(float)))
	return X

def train_Ngrams_GridSearch(x, y):
	# Use Naive Bayes classifier
	pipe = Pipeline([
    ('exf', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
])
	params = {
    'exf__k':(20,50,100,500,1000,2000),
    'clf__alpha':(1,0.5,1e-1,1e-2,1e-3),
}
	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
	gs = gs.fit(x, y)
	print(gs.best_score_)
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, gs.best_params_[param_name]))

	# Use SVM classifier
	pipe = Pipeline([
		('exf', SelectKBest(chi2)),
		('clf', LinearSVC()),
])
	params = {
		'exf__k':[50,100,500,1000],
		'clf__loss':['hinge','squared_hinge'],
		'clf__C': [1, 10, 50,100],
		'clf__max_iter':[1000,2000,3000],
		'clf__class_weight':[None,'balanced']}
	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
	gs = gs.fit(x, y)
	print(gs.best_score_)
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, gs.best_params_[param_name]))


def run_Ngrams(x, y, clf):
	k = 5
	test_size = len(y)//k
	score = 0
	f1 = 0
	for i in range(k):
		testX = x[i*test_size:(i+1)*test_size]
		testY = y[i*test_size:(i+1)*test_size]
		trainX = vstack((x[:i*test_size,:],x[(i+1)*test_size:,:]))
		trainY = np.concatenate((y[:i*test_size],y[(i+1)*test_size:]))
		print('fold: %d, train size: %d, test size: %d' %(i,len(trainY),len(testY)))
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		score += metrics.accuracy_score(testY, pred)
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')
	print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.4f." %(score/5, f1/5))



if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("\nusage: classify.py [data file] [topic]")
		exit(0)

	topic = sys.argv[2]
	data = load_data(sys.argv[1],topic)
	y = data[:,3]

	Ngrams = True
	if Ngrams:
		X = get_features(data, "Ngrams")
		# Find best model for Ngrams:
		#train_Ngrams_GridSearch(X, y)

		# run Ngrams
		if topic == 'abortion':
			X_selected = SelectKBest(chi2, k=500).fit_transform(X,y)
			run_Ngrams(X_selected, y, MultinomialNB(alpha=1))
			run_Ngrams(X_selected, y, LinearSVC(C=1, loss='hinge'))
		elif topic == 'gay rights':
			X_selected = SelectKBest(chi2, k=500).fit_transform(X,y)
			run_Ngrams(X_selected, y, MultinomialNB(alpha=1))
			run_Ngrams(X_selected, y, LinearSVC(C=1, loss='hinge'))





