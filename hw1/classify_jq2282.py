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
from sklearn.model_selection import StratifiedKFold

def load_data(filename, topic):
	raw = pd.read_csv(filename)
	print("Loading %s with %d records" %(filename,len(raw)))
	print("Column names: ", raw.columns)

	selected = raw[raw.topic == topic]
	selected = np.random.permutation(selected)
	return selected

def runold(selected, paras, method=1, NB=True, search=False):
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
		print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.4f." %(score/5, f1/5))
	else:
		return score/5

def run(selected, paras, method=1, NB=True, search=False):
	kfolds = 5
	score,f1 = 0,0

	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	exf_k = paras[0]
	kBest = SelectKBest(chi2, k=exf_k)
	if NB:
		clf = MultinomialNB(alpha=paras[1])
	else:
		clf = LinearSVC(loss=paras[1],C=paras[2],max_iter=paras[3],class_weight=paras[4])

	Y = selected[:,3]
	X = selected[:,0]
	skf = StratifiedKFold(n_splits=kfolds)#random_state=None, shuffle=False
	for train_index, test_index in skf.split(X, Y):
		#print("TRAIN:", train_index, "TEST:", test_index)
		trainX, testX = X[train_index], X[test_index]
		trainY, testY = Y[train_index], Y[test_index]

		trainX = cv.fit_transform(trainX)
		testX = cv.transform(testX)
		if method == 2:
			trainX = hstack((trainX, selected[train_index, 6:9].astype(float)))
			testX = hstack((testX, selected[test_index, 6:9].astype(float)))
		trainX = kBest.fit_transform(trainX, trainY)
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		score += metrics.accuracy_score(testY, pred)
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')
	if not search:
		print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.2f." %(score/5, f1/5))
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

def run_best(selected, paras, method=1, NB=False):
	kfolds = 5
	score,f1 = 0,0

	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	exf_k = paras[0]
	kBest = SelectKBest(chi2, k=exf_k)
	if NB:
		clf = MultinomialNB(alpha=paras[1])
	else:
		clf = LinearSVC(loss=paras[1],C=paras[2],max_iter=paras[3],class_weight=paras[4])

	X = selected[:,0]
	Y = selected[:,3]
	skf = StratifiedKFold(n_splits=kfolds)
	ifold = 0
	for train_index, test_index in skf.split(X, Y):
		trainX, testX = X[train_index], X[test_index]
		trainY, testY = Y[train_index], Y[test_index]

		trainX = cv.fit_transform(trainX)
		testX = cv.transform(testX)
		if method == 2:
			trainX = hstack((trainX, selected[train_index, 6:9].astype(float)))
			testX = hstack((testX, selected[test_index, 6:9].astype(float)))
		trainX = kBest.fit_transform(trainX, trainY)
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		scorei = metrics.accuracy_score(testY, pred)
		#print(scorei)

		ifold += 1
		score += scorei
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')

		#if ifold == 2:
			#top20(X[train_index], Y[train_index])

		# 他的k_feature_matrix = trainX, 
		# k_feature)index = kBest.get_support()
  #   # get the top 20 
  #   _, top20_feature_index = select_k_features(feature_matrix_dict['feature_matrix'], labels, k = 20)
  #   model.top20_feature_names = np.array(feature_matrix_dict['feature_names'])[top20_feature_index]
	print("The averaged accuracy score of 5-fold cv is %0.4f and f1 score is %0.4f." %(score/5, f1/5))

def top20(dataX, dataY):
	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	dataX = cv.fit_transform(dataX)
	names = cv.get_feature_names()
	#print(names)
	kBest = SelectKBest(chi2, k=20)
	k_feature = kBest.fit_transform(dataX, dataY)
	k_feature_index = kBest.get_support()
	#print(k_feature_index)

	res = []
	for i in k_feature_index:
		res.append(names[i])
	print(len(k_feature_index),len(res))
	return res

#def params():
"""
abortion-Ngrams
run_best(data, (500,1), method=1, NB=True)->0.6118
run_best(data, (500, 'hinge', 1, 1000, None),method=1,NB=False)->0.6194
abortion-Others
run_best(data, (500, 0.1), method=2, NB=True)->0.6177
run_best(data, (50, 'squared_hinge', 1, 2000, None),method=2,NB=False)->0.614

Gay rights-Others
0.6306178446399848 (50, 'squared_hinge', 50, 3000, 'balanced')

"""

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("\nusage: classify.py [data file] [topic]")
		exit(0)

	topic = sys.argv[2]
	data = load_data(sys.argv[1],topic)

	#best_score, best_paras = MySearch(data, NB=False, method=2)
	#print(best_score, best_paras)

	#run_best(data, (500, 0.1),method=2,NB=True)
	# if topic == "abortion":
	# 	#run_best(data, (500, 'hinge', 1, 1000, None),method=1,NB=False)
	# 	run(data, (500,1), method=2, NB=True, search=False)#0.6153


	# Find best model for Ngrams:
	#train_Ngrams_GridSearch(data)

	# Find best model for other features:
	#print(MySearch(data, NB=True, method=2))
	#abortion NB best, (0.6111436950146627, (500, 1))
	#abortion SVM best, (0.6129032258064516, (50, 'squared_hinge', 1, 2000, None))
	#gay rights SVM best: (0.6302583025830258, (50, 'hinge', 10, 3000, None))
	#gay rights NB best: (0.6273062730627307, (1000, 1))


# MySearch(data, NB=True, method=2)->0.617679340090206 (500, 0.1)
# MySearch(data, NB=False, method=2)->0.6182572756426745 (50, 'squared_hinge', 10, 3000, None)

	# run Ngrams
	# LinearSVC is slightly better than MultinomialNB
	if topic == 'abortion':
		#run_best(data, 500, MultinomialNB(alpha=1))
		run(data, (500,1), method=1, NB=True, search=False)
	# # 	#run_best(data, 500, LinearSVC(C=0.5, loss='hinge'))
	# 	print("Method2")
	# 	run_best(data, (500, 0.1), method=2, NB=True)
	# 	run(data, (50, 'squared_hinge', 1, 2000, None),method=2,NB=False)

	# 	#run_best(data, 20, SGDClassifier(alpha=0.01,loss='hinge',penalty=None),method=2)
	# 	#run_best(data, 1000, SGDClassifier(alpha=0.1,loss='hinge',penalty='l2'))
	elif topic == 'gay rights':
		run(data, (50, 'squared_hinge', 50, 3000, None), method=2, NB=False,search=False)
		run_best(data, (50, 'squared_hinge', 50, 3000, None),method=2, NB=False)
	# 	#run_best(data, 100, SGDClassifier(alpha=0.1,loss='modified_huber',penalty='l1'))
	# 	#run_best(data, 50, MultinomialNB(alpha=1))
	# 	#run_best(data, 50, LinearSVC(C=50, loss='hinge', max_iter=3000))
	# 	run_best(data, 50, LinearSVC(C=1, loss='hinge',max_iter=1000,class_weight=None),method=2)


#run_best(selected, k, clf, method=1):


