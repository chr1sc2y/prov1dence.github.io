# Machine Learning

## Support Vector Machine

###  Cost Function

$ \min_{\theta} C\sum_{i=1}^m(y^{i}cost_1(\theta^T X^i) + (1 - y^{i})cost_0(\theta^T X^i)) + \frac{1}{2}\sum_{i = 0}^m \theta_j^2 $

- C: not too large

### Decision Boundary

#### Margin

$\min_{\theta} \frac{1}{2} \sum_{j=1}^n \theta_j^2 = \frac{1}{2}(\theta_1^2 + \theta_2^2) = \frac{1}{2}\sqrt{\theta_1^2 + \theta_2^2} = \frac{1}{2}||\theta||^2 $



## Recommender System

### Anomaly Detection

1. choose features $x_i​$ which might be indicative of anomalous examples
2. fit parameters $\mu_1, \mu_2, ... , \mu_n, \sigma_1^2, ... , \sigma_n^2 ​$

- $ \mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{i} ​$
- $ \sigma^2 = \frac{1}{m} \sum_{i=1}^{m}(x_j^{(i)} - \mu_j)^2 ​$

3. compute $p(x)​$

- $ p(x) = \prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2) = \prod_{j=1}^n(x \sim N(x;\mu, \sigma^2)) $

#### Gaussian (Normal) Distribution

$ x \sim N(x;\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(x-\mu)^2}{2\sigma^2}) ​$

#### 