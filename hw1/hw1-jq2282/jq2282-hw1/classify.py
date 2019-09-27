# python classify.py stance-data.csv "abortion"
"""
In this script, Naive Bayes and SVM classifiers are used for stance classification on given dataset. Two types of models are trained and tested: Ngrams and Other features. For each model, the parameters are searched in function mySearch(). The best parameter combination for each model is already implement in the main function.
09/26/2019 by Jing Qian (jq2282@columbia.edu).
"""

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
import string

def loadData(filename, topic):
	"""
	Load csv data from filename and select subsets of data according to the topic. 
	Return the shuffled subset data.
	"""
	raw = pd.read_csv(filename)
	print("Loading %s with %d records" %(filename,len(raw)))
	print("Column names: ", raw.columns)

	selected = raw[raw.topic == topic]
	selected = np.random.permutation(selected)
	return selected

def repeatedPunc(text):
	"""
	Add number of repeated punctuations as feature.
	"""
	rp = ['??', '??????', '!!!', '?!']
	counts = np.zeros((len(text),1))
	for i in range(len(text)):
		count = 0
		for j in rp:
			count += text[i].strip().count(j)
		counts[i] = count
	return counts

def mySearch(selected, NB=True, method=2):
	"""
	Search best combinations of parameters that achieves highest accuracy with 5-fold cross validation.
	Input:
		1. selected: incoming data.
		2. NB: classifier, True means MultinomialNB() and False means LinearSVC().
		3. method: model type, 1 means Ngrams and 2 means other features. 
	Return: best score and corresponding parameter combination.
	"""
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
			comb_NB.append((params_NB['exf_k'][i], params_NB['clf_alpha'][j]))
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
		#print("cross validation with paras:", ipar, ", score:", tmp)
		if tmp > best_score:
			best_score = tmp
			best_paras = ipar
	return best_score, best_paras

def run(selected, paras, method=1, NB=True, search=False, example=False):
	"""
	Train model with given parameters within 5-fold cross validations and return the average score.
	Input:
		1. selected: incoming data.
		2. paras: given parameters for model
		3. method: model type. 1 means Ngrams and 2 means other features. 
		4. NB: classifier. True means MultinomialNB() and False means LinearSVC().
		5. search: denotes whether this run is for parameter search or show performance for best parameter combination.
		6. example: denotes whether to output the wrongly-classified post texts.
	Output: 
		1. Top 20 features from the best model for each topic. In our case, both best models are Ngrams.
		2. When example is turned on, the classification report and confusion matrix are printed with 15 wrongly-classified post texts.
	"""

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
		# For other features model, add LIWC features.
		# After tuning, found choosing only 3 out of 6 LIWC features gives best accuracy: 'words_pronom', 'words_per_sen', 'words_over_6'. 
		if method == 2: 
			trainX = hstack((trainX, train[:,6:9].astype(float)))
			trainX = hstack((trainX, train[:,-3:].astype(float)))
			trainC = repeatedPunc(train[:,0])
			trainX = hstack((trainX, trainC[:,-1:].astype(float)))
			testX = hstack((testX, test[:,6:9].astype(float)))
			testX = hstack((testX, test[:,-3:].astype(float)))
			testC = repeatedPunc(test[:,0])
			testX = hstack((testX, testC[:,-1:].astype(float)))
		trainX = kBest.fit_transform(trainX, trainY)
		testX = kBest.transform(testX)

		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		score += metrics.accuracy_score(testY, pred)
		#average in f1 could be any of 'macro','micro','weighted'
		f1 += metrics.f1_score(testY, pred, average='micro')

	# Output top20 features of the last fold.
	if not search and method==1:
		print("The top20 features are: ")
		print(top20(train[:,0], trainY))
	# Output 5 wrongly-classified post texts.
	if example==True:
		print(metrics.classification_report(testY, pred, target_names=['con','pro']))
		print(metrics.confusion_matrix(testY, pred))
		analyze(test,testX,clf)
	
	print("cross validation with paras:", paras, ", score:%0.2f and f1 score:%0.2f." %(score/5, f1/5))
	return score/5		

def top20(dataX, dataY):
	"""
	Select top 20 features from the best model. In our case, best models of both topics are Ngrams.
	"""
	cv = CountVectorizer(stop_words='english', ngram_range=(1,3))
	dataX = cv.fit_transform(dataX)
	names = cv.get_feature_names()
	kBest = SelectKBest(chi2, k=20)
	kBest.fit_transform(dataX, dataY)
	k_feature_index = kBest.get_support(indices=True)

	res = []
	for i in k_feature_index:
		res.append(names[i])
	#print(len(k_feature_index),len(res))
	return res

def analyze(test, testX, clf):
	"""
	Output 15 wrongly-classified post texts from test data for error analysis.
	Input:
		1. test: test data.
		2. testX: features extracted from test data.
		3. clf: trained model. 
	"""
	testY = test[:,3]
	post = test[:,0]
	pred = clf.predict(testX)
	nexample = 0
	for i in range(len(testY)):
		if testY[i] != pred[i] and nexample < 15:
			print(test[i,4],pred[i],testY[i],post[i],'\n')
			nexample += 1
    
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("\nusage: classify.py [data file] [topic]")
		exit(0)
	topic = sys.argv[2]
	if topic not in ("abortion", "gay rights"):
		print('Topic must be "abortion" or "gay rights"!')
		exit(0)

	data = loadData(sys.argv[1],topic)

	# Uncomment the following line to enable parameter search.
	#mySearch(data, NB=False, method=2)

	# Run best two models for each topic.
	if topic == 'abortion':
		run(data, (500, 'hinge', 1, 1000, None),method=1, NB=False,search=False, example=False)
		run(data, (500, 'hinge', 1, 1000, None),method=2, NB=False,search=False)
	elif topic == 'gay rights':
		run(data, (50,1), method=1, NB=True, search=False, example=False)
		run(data, (50, 'hinge', 1, 1000, None),method=2, NB=False,search=False)



