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

def get_features(selected, method=1):
	X = CountVectorizer(stop_words='english', ngram_range=(1,3))
	X = X.fit_transform(selected[:,0])
	if method != 1:
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


def run_best(selected, k, clf, method=1):
	kfolds = 5
	test_size = len(selected)//kfolds
	score = 0
	f1 = 0
	for i in range(kfolds):
		test = selected[i*test_size:(i+1)*test_size]
		train = np.vstack((selected[:i*test_size,:],selected[(i+1)*test_size:,:]))
		print('fold: %d, train size: %d, test size: %d' %(i,len(train),len(test)))

		trainY = train[:,3]
		testY = test[:,3]
		trainX = get_features(train, method)
		kBest = SelectKBest(chi2, k=k)
		trainX = kBest.fit_transform(trainX, trainY)
		testX = get_features(test,method)
		testX = kBest.transform(testX)

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


	# Find best model for Ngrams:
	# if Ngrams:
	# 	X_search = get_features(data)
	#	y = data[:,3]
		#train_Ngrams_GridSearch(X_search, y)

	# run Ngrams
	if topic == 'abortion':
		run_best(data, 500, MultinomialNB(alpha=1))
			#run_Ngrams(X_selected, y, LinearSVC(C=1, loss='hinge'))
		# elif topic == 'gay rights':
		# 	run_Ngrams(data, k = 50, MultinomialNB(alpha=1))
		# 	run_Ngrams(X_selected, y, LinearSVC(C=1, loss='hinge'))


#run_best(selected, k, clf, method=1):


