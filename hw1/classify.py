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
	selected = np.random.permutation(selected)
	return selected


def train_Ngrams_GridSearch(selected, method=1):
	# Use Naive Bayes classifier
# 	pipe = Pipeline([
# 		('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))),
#     ('exf', SelectKBest(chi2)),
#     ('clf', MultinomialNB()),
# ])
# 	params = {
#     'exf__k':(20,50,100,500,1000,2000),
#     'clf__alpha':(1,0.5,1e-1,1e-2,1e-3),
# }
# 	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
# 	gs = gs.fit(selected[:,0], selected[:,3])
# 	print(gs.best_score_)
# 	for param_name in sorted(params.keys()):
# 		print("%s: %r" % (param_name, gs.best_params_[param_name]))
# 0.6118245271046628
# clf__alpha: 1
# exf__k: 500

	# Use SVM classifier
	pipe = Pipeline([
		('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))),
		('exf', SelectKBest(chi2)),
		('clf', LinearSVC()),
])
	params = {
		'exf__k':[50,100,250, 500,750,1000],
		'clf__loss':['hinge','squared_hinge'],
		'clf__C': [0.5,1, 5,10, 50,100],
		'clf__max_iter':[1000,2000,3000],}
		#'clf__class_weight':[None,'balanced']}
	gs = GridSearchCV(pipe, params, cv=5, iid=False, n_jobs=-1)
	gs = gs.fit(selected[:,0], selected[:,3])
	print(gs.best_score_)
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, gs.best_params_[param_name]))
# 0.6229339232734818
# clf__C: 1
# clf__class_weight: None
# clf__loss: 'hinge'
# clf__max_iter: 1000
# exf__k: 500

# 0.6229424979849428
# clf__C: 0.5
# clf__loss: 'hinge'
# clf__max_iter: 1000
# exf__k: 500

def run_best(selected, k, clf, method=1):
	analyze = False

	kfolds = 5
	test_size = len(selected)//kfolds
	score = 0
	f1 = 0
	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	kBest = SelectKBest(chi2, k=k)
	for i in range(kfolds):
		test = selected[i*test_size:(i+1)*test_size]
		train = np.vstack((selected[:i*test_size,:],selected[(i+1)*test_size:,:]))
		#print('fold',i,train[:2,0])
		#print('fold: %d, train size: %d, test size: %d' %(i,len(train),len(test)))

		trainY = train[:,3]
		testY = test[:,3]

		trainX = cv.fit_transform(train[:,0])
		testX = cv.transform(test[:,0])
		if method == 2:
			trainX = hstack((trainX, train[:,6:9].astype(float)))
			testX = hstack((testX, test[:,6:9].astype(float)))
		trainX = kBest.fit_transform(trainX, trainY)
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		score += metrics.accuracy_score(testY, pred)
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')

		# if analyze and i==2:
		# 	print(metrics.accuracy_score(testY, pred))


	print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.2f." %(score/5, f1/5))

def run(selected, paras, method=1, NB=True, search=False):
	kfolds = 5
	test_size = len(selected)//kfolds
	score,f1 = 0,0

	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	exf_k = paras[0]
	kBest = SelectKBest(chi2, k=exf_k)
	if NB:
		clf = MultinomialNB(alpha=paras[1])
	else:
		clf = LinearSVC(loss=paras[1],C=paras[2],max_iter=paras[3],class_weight=paras[4])

	for i in range(kfolds):
		test = selected[i*test_size:(i+1)*test_size]
		train = np.vstack((selected[:i*test_size,:],selected[(i+1)*test_size:,:]))

		trainY = train[:,3]
		testY = test[:,3]
		trainX = cv.fit_transform(train[:,0])
		testX = cv.transform(test[:,0])
		if method == 2:
			#tfidf = TfidfTransformer()
			#trainX = tfidf.fit_transform(trainX)
			trainX = hstack((trainX, train[:,6:9].astype(float)))
			#testX = tfidf.transform(testX)
			testX = hstack((testX, test[:,6:9].astype(float)))
		trainX = kBest.fit_transform(trainX, trainY)
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		score += metrics.accuracy_score(testY, pred)
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')
	if not search:
		print("The averaged accuracy score of 5-fold cv is %0.2f and f1 score is %0.2f." %(score/5, f1/5))
	else:
		return score/5

def MySearch(selected, NB=True, method=2):
	params_NB = {
		'exf_k':(20,50,100,500,1000,2000),
		'clf_alpha':(1,0.5,1e-1,1e-2,1e-3),

	}
	comb_NB = []
	for i in range(len(params_NB['exf_k'])):
		for j in range(len(params_NB['clf_alpha'])):
			comb_NB.append((params_NB['exf_k'][i], params_NB['clf_alpha'][j]))
	#print(comb_NB[:6])
	params_SVM = {
    	'exf_k':[50,100,500,1000],
		'clf_loss':['hinge','squared_hinge'],
		'clf_C': [1, 10, 50,100],
		'clf_max_iter':[1000,2000,3000],
		'clf_class_weight':[None,'balanced']
	}
	comb_SVM = []
	for i in params_SVM['exf_k']:
		for j in params_SVM['clf_loss']:
			for l in params_SVM['clf_C']:
				for m in params_SVM['clf_max_iter']:
					for t in params_SVM['clf_class_weight']:
						comb_SVM.append((i,j,l,m,t))

	best_score = 0
	if NB:
		comb_paras = comb_NB
	else:
		comb_paras = comb_SVM
	for ipar in comb_paras:
		tmp = run(selected, ipar, method=method,NB=NB,search=True)
		print("cross validation with paras:", ipar, ", score:", tmp)
		if tmp > best_score:
			best_score = tmp
			best_paras = ipar
	return best_score, best_paras



if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("\nusage: classify.py [data file] [topic]")
		exit(0)

	topic = sys.argv[2]
	data = load_data(sys.argv[1],topic)


	# Find best model for Ngrams:
	#train_Ngrams_GridSearch(data)

	# Find best model for other features:
	print(MySearch(data, NB=False, method=2))
	#abortion NB best, (0.6111436950146627, (500, 1))
	#abortion SVM best, (0.6129032258064516, (50, 'squared_hinge', 1, 2000, None))

	# run Ngrams
	# LinearSVC is slightly better than MultinomialNB
	# if topic == 'abortion':
	# 	# run_best(data, 500, MultinomialNB(alpha=1))
	# 	#run_best(data, 500, LinearSVC(C=0.5, loss='hinge'))
	# 	# print("Method2")
	# 	#run_best(data, 1000, MultinomialNB(alpha=1), method=2)
	# 	#run_best(data, 500, LinearSVC(C=1, loss='squared_hinge',max_iter=1000,class_weight=None),method=2)
	# 	#run_best(data, 20, SGDClassifier(alpha=0.01,loss='hinge',penalty=None),method=2)
	# 	#run_best(data, 1000, SGDClassifier(alpha=0.1,loss='hinge',penalty='l2'))
	# elif topic == 'gay rights':
	# 	#run_best(data, 100, SGDClassifier(alpha=0.1,loss='modified_huber',penalty='l1'))
	# 	#run_best(data, 50, MultinomialNB(alpha=1))
	# 	#run_best(data, 50, LinearSVC(C=50, loss='hinge', max_iter=3000))
	# 	run_best(data, 50, LinearSVC(C=1, loss='hinge',max_iter=1000,class_weight=None),method=2)


#run_best(selected, k, clf, method=1):


