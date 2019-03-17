# Linear Regression

## Linear Regression

Model: $F_\theta(\vec{x}) = \theta^T \vec{x} = \theta_0x_0 + \theta_1x_1^2 + ... +  \theta_nx_n^n​$

Parameters: $\theta^T = \theta_0, \theta_1, ...,  \theta_n​$

### Cost Function

$J(\vec{\theta}) =\frac{1}{2m}\sum_{i=1}^{m} (F_\theta x^{i} - y^{i}) ^ 2​$

## Square Loss

- 
- Least Square Method
- Minimise Square Loss

### Formula

- $ \omega, \beta = \underset{\omega, \beta}{argmin} \sum_{i=1}^{m} (F_\vec{\theta} x^i - y^i) ^ 2 ​$

- $ \omega =\frac{\sum_{i=1}^{m} y_i (x_i\bar{x})}{\sum_{i=1}^{m}x_i^2 - \frac{1}{m}(\sum_{i=1}^{m}x_i)^2} $
- $ \beta = \frac{1}{m} \sum_{i=1}^{m}(y_i - \omega x_i) $

## Normal Equation

Normal Equation: A method to solve for $\vec{\theta}$ analytically

For m examples, n features:

$X = \left[\begin{matrix} x_0^1 x_1^1 ... x_n^1 \\ x_0^2 x_1^2 ... x_n^2 \\ ... \\ x_0^m x_1^m ... x_n^m \end{matrix}\right]​$	$Y = \left[\begin{matrix} y^1\\y^2\\ ... \\y^m\end{matrix}\right]​$

$\vec{\theta} = (X^TX)^{-1}X^TY​$



## Comparison between Gradient Descent and Normal Equation

| Gradient Descent                     | Normal Equation                                              |
| ------------------------------------ | ------------------------------------------------------------ |
| choose $\alpha$<br />many iterations | no iteration                                                 |
| works well even if n is large        | need to Compute $(X^TX)^{-1}X^TY$<br />slow if n is very large |

