# Gradient Descent

## Gradient Descent

Function: $F_\vec{\theta}​$

Goal: $ \underset{\vec{\theta}}{\operatorname{argmin}}F_\vec{\theta} $

Gradient Descent: $\theta_i = \theta_i - \alpha * \frac{\partial}{\partial\theta_i}J(\vec{\theta}) ​$

- $\theta_0 = \theta_0 - \alpha * \frac{\partial}{\partial\theta_0}J(\vec{\theta}) ​$
- $\theta_1 = \theta_1 - \alpha * \frac{\partial}{\partial\theta_1}J(\vec{\theta}) $
- ...
- $\theta_n = \theta_n - \alpha * \frac{\partial}{\partial\theta_n}J(\vec{\theta})$

Outline:

- Start with some $\vec{\theta}$
- Keep changing $\vec{\theta}$ to reduce $F(\vec{\theta})$
- End up at a minimum

### Simultaneous Update

Repeat Until Converge:

$\theta_i = \theta_i - \alpha * \frac{\partial}{\partial\theta_i}J(\vec{\theta}) $

- $\alpha​$: learning rate
  - for sufficiently small $\alpha$, $J(\vec{\theta})$ should decrease after each iteration
  - if $\alpha$ is too small:
    - slow decrease
  - if $\alpha$ is too large:
    - increase
    - overshoot the minimum
    - fail to converge, or even diverge
  - take smaller $\alpha$ automatically as it approaches a local minimum
- $\frac{\partial}{\partial\theta_i}J(\vec{\theta}) $: derivative

Note:

- Gradient Descent can only converge to a local minimum
- Declare convergence if $J(\vec{\theta})$ decreases by less than $10^{-3}$ in a single iteration

### Gradient Vanishing/Exploding

In deep network, activations end up increasing/decreasing exponentially.

$ \hat{Y} = X\omega^{[1]} \omega^{[2]} ... \omega^{[L]} ​$

- $ \omega{[i]} ​$ is bigger than 1
  - activations explode
- $ \omega{[i]} $ is smaller than 1
  - activations vanish

### Gradient Checking

- only to debug
- check components to identify bug
- use regularization
  - not work with dropout
- run at random initialization

#### Formula

$ f'(\theta) = \lim_{\epsilon->0} \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} = O(\epsilon^2) ​$

- $ f(\theta + \epsilon) - f(\theta - \epsilon) $: height
- $ 2\epsilon ​$: length

#### Grad Check

- Reshape $\omega{[1]}, \beta{[1]}, ... \omega{[L], \beta{[L]}}  ​$ into a big vector $ \theta ​$
  - $ J(\omega{[1]}, \beta{[1]}, ... \omega{[L], \beta{[L]}}  ) = J(\theta) = J(\theta_1, ..., \theta_i, .. , \theta_L)​$
- Reshape $d\omega{[1]}, d\beta{[1]}, ... d\omega{[L], d\beta{[L]}}  ​$ into a big vector $ \theta ​$
  - $ J(d\omega{[1]}, d\beta{[1]}, ... d\omega{[L], d\beta{[L]}}  ) = J(d\theta) = J(d\theta_1, ..., d\theta_i, .. , d\theta_L ​$
- for each i:
  - $ d\theta_{approx}^{[i]} = \frac{J(\theta_1, ..., \theta_i + \epsilon, .. , \theta_L) - J(\theta_1, ..., \theta_i - \epsilon, .. , \theta_L}{2\epsilon} \approx d\theta^{[i]} = \frac{\partial J}{\partial \theta^{[i]}}  ​$ 

- Check
  - check $ \frac{||d\theta_{approx} - d\theta||_2}{||d\theta_{approx}||_2 + ||d\theta||_2} $
    - $ < 10 ^{-7} $, grate
    - $> 10 ^{-3} ​$, wrong



## Batch Gradient Descent

### Batch Gradient Descent

Each step of Gradient Descent use all the training examples.

$ X= [x^1 x^2 ... x^n] ​$

$ Y= [y^1 y^2 ... y^n] $

### Mini-Batch Gradient Descent

Each step of Gradient Descent use a set of the training examples.

- $ X= [x^{\{1\}} x^{\{2\}} ... x^{\{n\}}] $
  - $ x{\{1\}} = [x^1 ... x ^m] $ (m=1000, e.g.)
  - $ x{\{2\}} = [x^{m+1} ... x ^{2m}] $ (m=1000, e.g.)
  - ...
  - $ x{\{n/m\}} = [x^{n-m} ... x ^{n}] ​$ (m=1000, e.g.)

- $ Y= [y^{\{1\}} y^{\{2\}} ... y^{\{n\}}] $

#### Mini-Batch Size

- size = n: Batch Gradient Descent
  - too long per iteration
- size = 1: Stochastic Gradient Descent
  - lose speedup from vectorization
- size = m: In-Between
  - speedup from vectorization
  - make progress without waiting

#### Mini-Batch Gradient Descent in Neural Network

for t = 1, ... , n/m	// 1 epoch: 1 pass through training set

​	Forward Propagation on $ X^{\{t\}} ​$

​	Compute cost $ J^{\{t\}} = \frac{1}{m} \sum_{i = 1}^{l} l(\hat{y}{i},y^{i}) + \frac{\lambda}{2m}\sum ||\omega^{[l]}||^2 $

​	Backward Propagation to compute gradient descent



## Adam

### Exponentially Weighted Average

#### Formula

$ v_t = \beta v_{t-1} + (1-\beta)\theta_t ​$

- $v_t​$: averaging over  $ \frac{1}{1-\beta}​$ day's temperature

##### $\beta$

- e.g  $\beta = 0.5 ​$: 2 day's average

- e.g. $\beta = 0.98 ​$: 50 day's average

- e.g. $\beta = 0.9 $: 10 day's average
  - $ v_{100} = 0.9 v_{99} + 0.1 \theta_{100} ​$
  - $ v_{99} = 0.9 v_{98} + 0.1 \theta_{99} $
  - ...
  - $ v_{100} =0.1 \theta_{100} + 0.9(0.1\theta_{99} + 0.9(0.1\theta_{98} + ... + 0.1 \theta_{2} + 0.9v_{1} )) $
  - $v_{100} = 0.1\theta_{100} + 0.1*0.9\theta_{99} + 0.1 *(0.9)^2\theta_{98} + ... + 0.1*(0.9)^{99}\theta_{1} $
- $ (1-\epsilon)^{(1/\epsilon)} \approx \frac{1}{e} $
- $ \beta = (1-\epsilon) $ 

#### Bias Correction

- not good estimate during initial phase
  - $ v_{1} =0.9 v_{0} + 0.1 \theta_{1}  = 0.1 \theta_{1} ​$
  - $ v_{2} =0.9 v_{1} + 0.1 \theta_{2}  = 0.1 \theta_{2} + 0.9 * 0.1\theta_1 $ 

- more accurate during initial phase
  - $ v_t = \beta v_{t-1} + (1-\beta)\theta_t $
  - $ \frac{V_t}{1-\beta^t} $
    - $ 1-\beta^t $: weighted average of data
    - remove the bias

#### Aim

- damp the oscillation

### Momentum

- for iteration = 1, ... , n=
  -  Forward Propagation on $ X^{\{t\}} ​$
  -  Compute cost $ J^{\{t\}} = \frac{1}{m} \sum_{i = 1}^{l} l(\hat{y}{i},y^{i}) + \frac{\lambda}{2m}\sum ||\omega^{[l]}||^2 $
  -  Backward Propagation to compute gradient descent
    - Compute $dw$, $ db $ on the current mini-batch
    - $ v_{dw} = \beta v_{dw} + (1-\beta)dw  ​$ 
    - $ v_{db} = \beta v_{db} + (1-\beta)db  ​$ 
    - $ w = w - \alpha v_{dw} ​$
    - $ b = b - \alpha v_{db} ​$

### RMSprop

- for iteration = 1, ... , n=
  -  Forward Propagation on $ X^{\{t\}} $
  -  Compute cost $ J^{\{t\}} = \frac{1}{m} \sum_{i = 1}^{l} l(\hat{y}{i},y^{i}) + \frac{\lambda}{2m}\sum ||\omega^{[l]}||^2 ​$
  -  Backward Propagation to compute gradient descent
    - Compute $dw​$, $ db ​$ on the current mini-batch
    - $ S_{dw} = \beta S_{dw} + (1-\beta)d^2w  ​$ 
    - $ S_{db} = \beta S_{db} + (1-\beta)d^2b  $ 
    - $ w = w - \alpha \frac{dw}{\sqrt{S_{dw}}+ \epsilon} ​$
    - $ b = b - \alpha \frac{db}{\sqrt{S_{db}}+ \epsilon} ​$

### Adam

Adaptive Moment Estimation

- $ v_{dw}, v_{db},  S_{dw}, S_{db} = 0, 0, 0, 0 ​$
- for t = 1, ... , n    // t: iteration
  -  Forward Propagation on $ X^{\{t\}} $
  -  Compute cost $ J^{\{t\}} = \frac{1}{m} \sum_{i = 1}^{l} l(\hat{y}{i},y^{i}) + \frac{\lambda}{2m}\sum ||\omega^{[l]}||^2 $
  -  Backward Propagation to compute gradient descent
    - Compute $dw​$, $ db ​$ on the current mini-batch
    - $ v_{dw} = \beta_1 v_{dw} + (1-\beta_1)dw  ​$
    - $ v_{db} = \beta_1 v_{db} + (1-\beta_1)db  $
    - $ s_{dw} = \beta_2 s_{dw} + (1-\beta_2)d^2w  $
    - $ s_{db} = \beta_2 s_{db} + (1-\beta_2)d^2b  $
    - $ v_{dw}^{corrected} = \frac{v_{dw}}{(1-\beta_1^t)} ​$
    - $ v_{db}^{corrected} = \frac{v_{db}}{(1-\beta_1^t)} $
    - $ s_{dw}^{corrected} = \frac{s_{dw}}{(1-\beta_2^t)} ​$
    - $ s_{db}^{corrected} = \frac{s_{db}}{(1-\beta_2^t)} ​$
    - $ w = w - \alpha \frac{v_{dw}^{corrected}}{\sqrt{s_{dw}^{corrected}}+ \epsilon} ​$
    - $ b = b - \alpha \frac{v_{db}^{corrected}}{\sqrt{s_{db}^{corrected}}+ \epsilon} $

#### Hyperparameters

- $\alpha$
  - needs to be tuned
- $\beta_1$: first moment
  - Momentum term
  - default 0.9
- $\beta_2$: second moment
  - RMSprop term
  - default 0.999
- $\epsilon​$
  - default $10^{-1}​$



## Learning Rate Decay

- bigger learning rate during the initial steps
- Slower learning rate as approaching convergence

### Decay Rate

$ \alpha = \frac{1}{1 + decay\_rate * epoch\_number} \alpha_0 $

- for $\alpha_0​$ = 0.2, decay_rate = 1

| epoch | $\alpha$ |
| ----- | -------- |
| 1     | 0.1      |
| 2     | 0.67     |
| 3     | 0.5      |
| 4     | 0.4      |
| ...   | ...      |

Other Rate Decay

- $  \alpha = decay\_rate^{epoch\_number} \alpha_0$
  - Exponential Decay
  - exponentially quickly decay

- $   \alpha = \frac{k}{\sqrt{epoch\_number}} \alpha_0 $



## Local Optimal

