# Logistic Regression

## Logistic Regression

Model: $ \hat{y} = H_\theta(x) = \sigma(\theta^Tx) = \frac{1}{1 + e^{-\theta^Tx}} = \sigma(z) = \frac{1}{1 + e^-z} ​$ ($z =\theta^Tx​$)

Parameters: $\theta^T = \theta_0, \theta_1, ...,  \theta_n​$

- $H_\theta(x) = P(y=1 | x; \theta)​$ 
- $H_\theta(x)$ is the estimated probability that $H_\theta(\vec{x})=1$ on features $\vec{x}$ and parameters $\theta$

### Cost Function

​	$ Cost(H_\theta(x), x) = \frac{1}{m} \sum_{i=1}^m(-y^i log(H_\theta(x^i))  -  (1-y^i)log(1 - H_\theta(x^i)) ​$

- $ Cost(H_\theta(x),y) = \left\{\begin{matrix} -log(H_\theta(x)), y = 1
  \\ -log(1 - H_\theta(x), y = 0
  \end{matrix}\right.$
  - For $ H_\theta(x) $ = 1, $ Cost(H_\theta(x),y) = 0 $ 
  - if $ H_\theta(x) $ = 0, $ Cost(H_\theta(x),y) = \infty  $

  - if $ H_\theta(x) $ = 0, $ Cost(H_\theta(x),y) = 0 $ 
  - if $ H_\theta(x) $ = 1, $ Cost(H_\theta(x),y) = \infty  ​

1. Cost function on single example

- $P(y|x) = \hat{y}^y (1 - \hat{y})^{(1-y)}​$
  - If y = 1, $P(y|x) = \hat{y}$
  - If y = 0, $P(y|x) = 1 - \hat{y} ​$
- $logP(y|x) = y log\hat{y} + {(1-y)} log(1 - \hat{y})​$
- minimise cost function $->$ maximize $logP(y|x)$, penalise by large cost

2. Cost function on m examples

- $P(Y|X) = \prod_{i=1}^m P(y^i|x^i)​$
- $log P(Y|X) = log \prod_{i=1}^m P(y^i|x^i)$
- maximum likelihood estimation
- $ Cost(H_\theta(x), y) = \frac{1}{m} \sum_{i=1}^m(-y^i log(H_\theta(x^i))  -  (1-y^i)log(1 - H_\theta(x^i)) ​$

### Gradient Descent

#### Predict

- $ Z = \omega ^ T X + \beta $

```Z = np.dot(w.T, X) + b```

- $ A = g(Z) $
- $dZ = A - Y$

#### Adjust Weights

$ d b = \frac{1}{m} \sum_{i=1}^{m} d z^i = \frac{1}{m}​$ ```np.sum(dz)```

$ d w = \frac{1}{m} X d z^T $

### BroadCasting

- matrix(m, n) + vector(1, n) -> matrix(m, n) + matrix(m, n)
- matrix(m, n) + x -> matrix(m, n) + matrix(m, n)

### Decision Boundary

1. linear

   $ H_\theta(\vec{x}) = g(\theta^T\vec{x}) = g(\theta_0 + \theta_1x_1 + ... + \theta_nx_n)​$

2. non-linear

   $ H_\theta(\vec{x}) = g(\theta^T\vec{x}) = g(\theta_0 + \theta_1x_1 + \theta_2x_2^2 + \theta_3x_3^3 ... + \theta_nx_n^n)$




## Regularization

### Cost Function

Cost Function: $ J(\omega, \beta) = \frac{1}{m} \sum_{i=1}^m(-y^i log\hat{y}^i)  -  (1-y^i)log(1 - \hat{y} ^ i) + \frac{\lambda}{2m} \left \| \omega \right \|$

Regularization Parameter: $ \lambda ​$

L2 regularization: $ \left \| \omega \right \|_2^2 = \sum_{i = 1}^{n} \omega_i^2 = \omega^T\omega ​$



## Softmax Regression

#### Activation Function

t = $ e^{z^{[l]}} $

$ a_i^{[l]} = \frac{t}{\sum_{j=1}^{number of class}t_j} $