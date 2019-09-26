# python classify.py stance-data.csv "abortion"

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

def load_data(filename, topic):
	raw = pd.read_csv(filename)
	print("Loading %s with %d records" %(filename,len(raw)))
	print("Column names: ", raw.columns)

	selected = raw[raw.topic == topic]
	selected = np.random.permutation(selected)
	return selected


def MySearch(selected, NB=True, method=2):
	# Parameters search range
	params_NB = {
		'exf_k':(20,50,100,500,1000,2000),
		'clf_alpha':(1,0.5,1e-1,1e-2,1e-3),
	}
	params_SVM = {
		'exf_k':[50,100,500,1000],
		'clf_loss':['hinge','squared_hinge'],
		'clf_C': [1, 10, 50,100],
		'clf_max_iter':[1000,2000,3000],
		'clf_class_weight':[None,'balanced']
	}

	# Combine parameters into set which can be parsed into run()
	comb_NB = []
	for i in range(len(params_NB['exf_k'])):
		for j in range(len(params_NB['clf_alpha'])):
			comb_NB.append((params_NB['exf_k'][i], params_NB['clf_alpha'][j])))
	comb_SVM = []
	for i in params_SVM['exf_k']:
		for j in params_SVM['clf_loss']:
			for l in params_SVM['clf_C']:
				for m in params_SVM['clf_max_iter']:
					for t in params_SVM['clf_class_weight']:
						comb_SVM.append((i,j,l,m,t))

	# Start searching, find the parameter set that gets the highest average score from 5-fold cross validation.
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

	MySearch(data, NB=False, method=2)

	if topic == 'abortion':
		run(data, (500, 'hinge', 1, 1000, None),method=1,NB=False,search=False)
		run(data, (500, 'hinge', 0.5, 1000, None),method=2,NB=False,search=False)
	elif topic == 'gay rights':
		run(data, (50,1), method=1, NB=True,search=False)
		run(data, (50, 'hinge', 0.5, 1000, None),method=2,NB=False,search=False)

