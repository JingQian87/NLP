# python classify.py train-data.csv abortion

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
#np.random.seed(4705)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
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

# topic == 'gay rights' and method == 'Ngrams'
# MultinomialNB
# 0.6283767892623981
# clf__alpha: 1
# exf__k: 50
# LinearSVC
# 0.6320776394577132
# clf__C: 50
# clf__class_weight: None
# clf__loss: 'hinge'
# clf__max_iter: 3000
# exf__k: 50

def load_data(filename, topic):
	raw = pd.read_csv(filename)
	print("Loading %s with %d records" %(filename,len(raw)))
	print("Column names: ", raw.columns)

	selected = raw[raw.topic == topic]
	selected = np.random.RandomState(seed=4705).permutation(selected)
	return selected


def train_Ngrams_GridSearch(selected, method=1):
	# Use SVM classifier
	pipe = Pipeline([
		('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))),
		('exf', SelectKBest(chi2)),
		('clf', SVC(kernel='rbf')),
	])
	params =  {
		'exf__k':[50,100,500,1000],
		'clf__gamma': [1e-3, 5e-3, 1e-4],
		'clf__C': [0.1,1, 10, 50, 100]
	}
	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
	gs = gs.fit(selected[:,0], selected[:,3])
	print(gs.best_score_)
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, gs.best_params_[param_name]))
# python classify_trySGD.py stance-data.csv "abortion"
# 0.6129906878633534
# clf__C: 50
# clf__gamma: 0.001
# exf__k: 1000


# 	pipe = Pipeline([
# 		('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))),
# 		('exf', SelectKBest(chi2)),
# 		('clf', SGDClassifier())
# ])
# 	params = {
# 		'exf__k':[50,100,500,1000],
# 		'clf__alpha':(1,1e-1,1e-2,1e-3),
# 		'clf__penalty':('l1','l2','None'),
# 		'clf__loss':('hinge', 'log', 'modified_huber', 'squared_hinge'),
# 		#'clf__max_iter':(1000,3000,5000),
# 		}
# 	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
# 	gs = gs.fit(selected[:,0], selected[:,3])
# 	print(gs.best_score_)
# 	for param_name in sorted(params.keys()):
# 		print("%s: %r" % (param_name, gs.best_params_[param_name]))
#abortion
# 0.632937181663837
# clf__alpha: 0.1
# clf__loss: 'hinge'
# clf__penalty: 'l2'
# exf__k: 1000
# slightly higher than LinearSVC, longer running, but similar in the run_best.

#gay rights
# 0.6423881161888542
# clf__alpha: 0.1
# clf__loss: 'modified_huber'
# clf__penalty: 'l1'
# exf__k: 100

def run_best(selected, k, clf, method=1):
	kfolds = 5
	test_size = len(selected)//kfolds
	score = 0
	f1 = 0
	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	kBest = SelectKBest(chi2, k=k)
	for i in range(kfolds):
		test = selected[i*test_size:(i+1)*test_size,:]
		train = np.vstack((selected[:i*test_size,:],selected[(i+1)*test_size:,:]))
		print(np.shape(train),np.shape(test))
		#print('fold: %d, train size: %d, test size: %d' %(i,len(train),len(test)))

		trainY = train[:,3]
		testY = test[:,3]

		trainX = cv.fit_transform(train[:,0],trainY)
		trainX = kBest.fit_transform(trainX, trainY)
		testX = cv.transform(test[:,0])
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		tmp = metrics.accuracy_score(testY, pred)
		print(tmp)
		score += tmp
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')
	print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.2f." %(score/5, f1/5))



if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("\nusage: classify.py [data file] [topic]")
		exit(0)

	topic = sys.argv[2]
	data = load_data(sys.argv[1],topic)


	# Find best model for Ngrams:
	#train_Ngrams_GridSearch(data)

	# run Ngrams
	# LinearSVC is slightly better than MultinomialNB
	if topic == 'abortion':
		#run_best(data, 1000, SGDClassifier(alpha=0.1,loss='hinge',penalty='l2'))
	# # 	#run_best(data, 500, MultinomialNB(alpha=1))
		run_best(data, 500, LinearSVC(C=1, loss='hinge'))
	# elif topic == 'gay rights':
	# 	run_best(data, 100, SGDClassifier(alpha=0.1,loss='modified_huber',penalty='l1'))
	# 	#run_best(data, 50, MultinomialNB(alpha=1))
	# 	run_best(data, 50, LinearSVC(C=50, loss='hinge', max_iter=3000))


#run_best(selected, k, clf, method=1):


