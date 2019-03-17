# Sequence Model

## Recurrent Neural Network

### Cons of Standard Network

- inputs and outputs can be different length in different examples
- does not share features learned across different positions

### Recurrent Neural Network

#### Notation

$ a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a) = g(W_a[a^{<t-1>}, x^{<t>}] + b_a) $

$ \hat{y}^{<t>} = g(W_{ya}a^{<t>} + b_y) $

- Forward Propagation
- Backpropagation Through Time

#### Examples

- many-to-many architecture
- many-to-one
- one-to-many

 

#### Bidirectional Recurrent Neural Network
