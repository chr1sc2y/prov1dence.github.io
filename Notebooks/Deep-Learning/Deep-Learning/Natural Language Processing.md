# Natural Language Processing

## Word Embedding

### Transfer Learning

1. learn word embeddings from large text corpus
2. transfer embedding to new task with smaller training set
3. continue to finetune the word embeddings with new data

### Cosine Similarity

$ cos(u,v) = \frac{u^T v}{||u||_2||v||_2}$

### Word2Vec

#### Model

$ p(t|c)=\frac{e^{\theta_t^T e_c}}{\sum_{j=1}^n e^{\theta_j^T e_c}}  $

