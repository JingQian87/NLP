'''
Code for HW3 section 1
Parameter search for word embeddings.
Jing Qian (jq2282)
'''

from gensim.models.word2vec import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from scipy import sparse
import math
from scipy.sparse import linalg
import os

text = 'data/brown.txt'
windowList = [2,5,10]
dimList = [100,300,1000]
numNS = [1,5,15]
EPOCHS = 5

# generator for reading corpus from filename.
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename, 'r'):
            yield [word.lower() for word in tokenizer.tokenize(line)]

# train word2vec skip-gram with negative sampling.
def train_word2vec(corpus):
	for iwin in windowList:
		for idim in dimList:
			for inum in numNS:
				print("Training word2vec:")
				print("Window size = %d, dimension = %d, NS num = %d." %(iwin, idim, inum))
				model = Word2Vec(size=idim, # Dimension
				                 window=iwin, # Context Window
				                 sg=1, #Skip gram
				                 negative=inum, #Number of negative samples
				                 workers=4, #parallel
				                 iter=EPOCHS)
				model.build_vocab(corpus)
				model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
				model.wv.save("../../NLPdata/hw3/savedWV_word2vec/wv_win%d_dim%d_num%d" %(iwin, idim, inum))

# count unigrams
def vocab_index(sentences):
    word_counts = Counter()
    for x in sentences:
        word_counts.update(x)
    # print(len(word_counts))
    # print(word_counts.most_common(5))
    # print(word_counts.most_common()[-5:])
    
    # generate index for words in vocabulary
    index_dict = dict()
    ii = 0
    for i in word_counts:
        index_dict[i] = ii
        ii += 1
    
    return index_dict

# count co-occurance as a dictionary "joint"
def cooc_dict(sentences, iwin):
    joint = Counter()
    for isen in sentences:
        for j, word in enumerate(isen):
            index_min = max(0, j-iwin)
            index_max = min(len(isen), j+iwin+1)
            index = [ii for ii in range(index_min, index_max) if ii!=j]
            for iin in index:
                joint[(word, isen[iin])] += 1
    return joint

# transform count joint to sparse matrix
def cooc_matrix(joint_dict, index_dict):
    row_index = []
    col_index = []
    values = []
    for (wi,wj), count in joint_dict.items():
        row = index_dict[wi]
        col = index_dict[wj]
        value = count
        row_index.append(row)
        col_index.append(col)
        values.append(value)
    return sparse.csr_matrix((values, (row_index, col_index)))

# convert joint matrix to ppmi matrix.
def ppmi_matrix(joint_matrix):
   # calculate the column sum, row sum and total sum of the co-occur matrix.
    sum_a0 = joint_matrix.sum(axis=0)
    #print(np.shape(sum_a0))
    sum_a1 = joint_matrix.sum(axis=1)
    #print(np.shape(sum_a1))
    sum_total = joint_matrix.sum()
    #print(sum_total) 
   
    # find the non-zero elements in the sparse matrix
    nonzero_index = joint_matrix.nonzero()
    num_nonzero = np.shape(nonzero_index)[1]
    #print(num_nonzero)
    
    # calculate values for non-zero ppmi
    ppmi_values = []
    for i in range(num_nonzero):
        row = nonzero_index[0][i]
        col = nonzero_index[1][i]
        pwc_scaled = joint_matrix[row, col]
        pwpc_scaled = sum_a1[row,0]*sum_a0[0,col]/sum_total
        if pwc_scaled > pwpc_scaled:
            value = math.log(pwc_scaled/pwpc_scaled)
        else:
            value = 0
        ppmi_values.append(value)
    
    return sparse.csr_matrix((ppmi_values, nonzero_index)) 

# truncate ppmi matrix with svd.
def ppmi_svd(ppmi, idim):
    uu,ss,vv = linalg.svds(ppmi, idim)
    #print(np.shape(uu),np.shape(ss),np.shape(vv)) 
  
    sigma_sr = np.diag([x**0.5 for x in ss])
    return np.matmul(uu,sigma_sr)

# save the word vectors from SVD on the ppmi matrix.
def save_wv(word_vecs, index_dict, iwin, idim):
    keys = list(index_dict.keys())
    #print(np.shape(keys))
    
    f1 = open('../../NLPdata/hw3/savedWV_svd/wv_svd_win%d_dim%d.txt' %(iwin, idim),'w')
    for i in range(len(keys)):
        f1.write(keys[i]+' '+' '.join(str(x) for x in word_vecs[i,:]))
        f1.write('\r\n')
    f1.close()  

# generate word vectors from SVD on the ppmi matrix with different hyperparameters.
def svd_ppmi(corpus):
	index_dict = vocab_index(corpus)

	for iwin in windowList:
		for idim in dimList:
			print("SVD on the positive PMI matrix:")
			print("Window size = %d, dimension = %d." %(iwin, idim))
			joint = cooc_dict(corpus, iwin)
			joint_matrix = cooc_matrix(joint, index_dict)
			ppmi = ppmi_matrix(joint_matrix)
			word_vecs = ppmi_svd(ppmi, idim)
			save_wv(word_vecs, index_dict, iwin, idim)


if __name__ == "__main__":
    # load corpus
	tokenizer = RegexpTokenizer(r'\w+')
	sentence = MySentences(text)

	# train word2vec
	train_word2vec(sentence)

	# svd on the positive PMI matrix
	svd_ppmi(sentence)
