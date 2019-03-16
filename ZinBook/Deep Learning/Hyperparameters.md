# Hyperparameters

## Hyperparameters

### level of importance

1. - learning rate $\alpha$

2. - momentum term $\beta$
   - mini-batch size
   - number of hidden units
3. - layers
   - learning rate decay

### random hyperparameters

- previous
  - sample the points in a grid
- current
  - choose the points at random
  - coarse to fine

### $\alpha$ learning rate 

$ r = -4 * $np.random.rand()

$\alpha = 10^r$

### $\beta$ exponentially weighted averages

- $ \beta = 0.9 ... 0.999 $
- $ 1 - \beta = 0.1 ... 0.001 $
- $ r = -3 *$ np.random.rand()
- $ 1 - \beta = 10^r $
- $ \beta = 1 - 10^r $

### Train Models

1. babysitting one model
2. training many models in parallel



## Batch Normalization

- reduce the changing of input values
  - more stable
- limit the amount of effect on distribution of values

### Formula

- normalize $z^{[l]i]}$

- Given some intermediate values in NN

- hidden units $z^{[i]1} ... z^{[i]n}$ 
- $ \mu =  \frac{1}{m} \sum_{i=1}^{n}z^i ​$
- $ \sigma^2 = \frac{1}{m} \sum_{i=1}^{n} (z^i - \mu)^2 $
- $ z^i_{norm} = \frac{z^i - \mu}{\sqrt{\sigma^2 + \epsilon}} ​$
- $ \tilde{z^i} = \gamma z^i_{norm} + \beta ​$
  - $ \gamma, \beta ​$: learnable parameters
    - arange the values
  - use $ \tilde{z^i} $ instead of $ z^i $

### Batch Normalization With Mini-Batch Gradient Descent

- handle one mini-batch at a time

- $ X^{\{1\}} -> z^{[1]} -> \tilde{z^1} -> ... -> \tilde{z^n} ​$
  - $z^{[l]} = w^{[l]} a^{[l - 1]}  ​$
  - eliminate $ b $
  - $ z^l_{norm} = \frac{z^l - \mu}{\sqrt{\sigma^2 + \epsilon}} ​$
  - $ \tilde{z^l} = \gamma z^l_{norm} + \beta ​$
    - $ \beta $ controls the shift or the biased t erms

- $ X^{\{2\}} -> z^{[1]} -> \tilde{z^1} -> ... -> \tilde{z^n} ​$
  - ...

### Implement 

- for t = 1 ... number of Mini-Batches
  - forward propagation on $X{\{t\}}​$
    - in each hidden layer
    - use $ z^i_{norm} $ to replace $z^i​$
    - compute $\tilde{z^l}$
  - backpropagation on $X{\{t\}}$
    - compute $ dw^{[l]}, d\beta^{[l]}, d\gamma^{[l]} ​$
    - update $ w^{[l]}, \beta^{[l]}, \gamma^{[l]} $

 ### At Test Time

- $ \mu =  \frac{1}{m} \sum_{i=1}^{n}z^i $
- $ \sigma^2 = \frac{1}{m} \sum_{i=1}^{n} (z^i - \mu)^2 $
- $ z^i_{norm} = \frac{z^i - \mu}{\sqrt{\sigma^2 + \epsilon}} $
- $ \tilde{z^i} = \gamma z^i_{norm} + \beta $

