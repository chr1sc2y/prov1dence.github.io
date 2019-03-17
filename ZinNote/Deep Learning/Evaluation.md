# Evaluation

## Regularization

- prevent overfitting
- Regularization Parameter: $ \lambda $

### L2 Norm

- $ \left \| \omega \right \|_2^2 = \sum_{i = 1}^{n} \omega_i^2 = \omega^T\omega $

### L1 Norm

-  $ \left \| \omega \right \|_1 = \sum_{i = 1}^{n} |\omega| $
- weight decay

### Dropout Regularization

### Early Stopping

- stop training earlier
  - prevent overfitting
  - not optimizing cost function
- cannot work out the two problems
  - not overfitting
  - optimize cost function



## Bias and Variance

| Train set error | Dev set error | bias and variance        |
| --------------- | ------------- | ------------------------ |
| low             | high          | high variance            |
| high            | high          | high bias, high variance |
| high            | low           | high bias                |

#### high bias

- underfitting
- large $ \lambda $
- high training error, high cross validation error 

##### solve

- bigger network
- NN architecture search
- train longer
- polynomial features
- decrease $ \lambda $

#### high variance

- overfitting
- small $ \lambda $
- low training error, high cross validation error

##### solve

- more data
  - flipping horizontally
  - rotate
  - distort
- NN architecture search
- regularization
- smaller sets of features
- increase $ \lambda $

### Training Set

- training set increase, training error increase, cross validation error decrease

## Error Analysis

### Learning Approach

- start with a simple algorithm
- Implement and test on cross-validation data
- plot learning curves
  - to decide data features
- Manually examine the examples that the algorithm made errors on

### Skewed Class

one kind much more than another kind in the training data



## Evaluation

### Precision/Recall

On cross-validation data:

- positive: predict 1
- negative: predict 0
- true: predict = actual
- false: predict != actual 

#### Precision

- true positives / predicted positives
- true positives / true positives + false positives

#### Recal

- true positives / actual positives
- true positives / true positives + false negatives

#### Trade off

- predict 1 if $ h_\theta(x) $ > p
  - if p = 0.9
    - high precision
    - low recall
  - if p = 0.1
    - high recall
    - low precision

#### F Score

- Average: $ \frac{Precision + Recall}{2} $
  - not good
- $ F_1 $ Score: $ 2 \frac{PR}{P+R} $

### Evaluation Metrics

- pick one to be optimizing
- the left to be satisficing
  - reach a threshhold

### Evaluation Methods

####hold-out

##### previous (Machine Learning)

- randomly order
- 70% traning set
- 30% test set

##### current (Deep Learning / Big Data)

- more data for training
- less for dev and test
  - dev set to evaluate different models
  - test set to evaluate final cost bias

#### cross validation 

##### Leave-One-Out Cross-Validation

- split the data into n sets
- 1 for testing, n-1 for training

##### K-Fold Cross-Validation

- split the data into n sets
- k for testing, n-k for training
  - if k is too small, underfitting
  - if k is too large, overfitting
    - high correlation



