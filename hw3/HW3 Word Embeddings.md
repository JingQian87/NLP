# HW3 Word Embeddings

#### Jing Qian (jq2282)

### 1. Parameter search

#### (1) Table of the parameter search results

| Algorithm | Win. | Dim. | N.s. | WordSim | BATS 1 (encyclopedic-semantics | BATS 2 (antonyms - binary) | BATS 3 (total) | MSR   |
| --------- | ---- | ---- | ---- | ------- | ------------------------------ | -------------------------- | -------------- | ----- |
| word2vec  | 2    | 100  | 1    | 0.055   | 0.033                          | 0.023                      | 0.013          | 0.661 |
| word2vec  | 2    | 100  | 5    |         |                                |                            |                |       |
| word2vec  | 2    | 100  | 15   |         |                                |                            |                |       |
| word2vec  | 2    | 300  | 1    |         |                                |                            |                |       |
| word2vec  | 2    | 300  | 5    |         |                                |                            |                |       |
| word2vec  | 2    | 300  | 15   |         |                                |                            |                |       |
| word2vec  | 2    | 1000 | 1    |         |                                |                            |                |       |
| word2vec  | 2    | 1000 | 5    |         |                                |                            |                |       |
| word2vec  | 2    | 1000 | 15   |         |                                |                            |                |       |
| word2vec  | 5    | 100  | 1    |         |                                |                            |                |       |
| word2vec  | 5    | 100  | 5    |         |                                |                            |                |       |
| word2vec  | 5    | 100  | 15   |         |                                |                            |                |       |
| word2vec  | 5    | 300  | 1    |         |                                |                            |                |       |
| word2vec  | 5    | 300  | 5    |         |                                |                            |                |       |
| word2vec  | 5    | 300  | 15   |         |                                |                            |                |       |
| word2vec  | 5    | 1000 | 1    |         |                                |                            |                |       |
| word2vec  | 5    | 1000 | 5    |         |                                |                            |                |       |
| word2vec  | 5    | 1000 | 15   |         |                                |                            |                |       |
| word2vec  | 10   | 100  | 1    |         |                                |                            |                |       |
| word2vec  | 10   | 100  | 5    |         |                                |                            |                |       |
| word2vec  | 10   | 100  | 15   |         |                                |                            |                |       |
| word2vec  | 10   | 300  | 1    |         |                                |                            |                |       |
| word2vec  | 10   | 300  | 5    |         |                                |                            |                |       |
| word2vec  | 10   | 300  | 15   |         |                                |                            |                |       |
| word2vec  | 10   | 1000 | 1    |         |                                |                            |                |       |
| word2vec  | 10   | 1000 | 5    |         |                                |                            |                |       |
| word2vec  | 10   | 1000 | 15   | 0.291   | 0.029                          | 0.116                      | 0.018          | 0.668 |
| SVD       | 2    | 100  | -    |         |                                |                            |                |       |
| SVD       | 2    | 300  | -    |         |                                |                            |                |       |
| SVD       | 2    | 1000 | -    |         |                                |                            |                |       |
| SVD       | 5    | 100  | -    |         |                                |                            |                |       |
| SVD       | 5    | 300  | -    |         |                                |                            |                |       |
| SVD       | 5    | 1000 | -    |         |                                |                            |                |       |
| SVD       | 10   | 100  | -    |         |                                |                            |                |       |
| SVD       | 10   | 300  | -    |         |                                |                            |                |       |
| SVDs      | 10   | 1000 | -    |         |                                |                            |                |       |





#### (2) Written analysis of the results





### 2. Fun with objective functions.

#### 1) (Preliminaries)

**i)** 
$$
\sigma(-x) = \frac{1}{1+e^x}
$$
