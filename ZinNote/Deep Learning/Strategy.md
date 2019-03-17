# Strategy

## Transfer Learning

- pre-training/pre-initialize low-level features
- fine tuning / retrain

### retrain

- just retrain the weights of the last layer
- retrain all the layers if you have enough data

### Usage

- low level features from the previous could be helpful for the current
- tasks have the same type of input
- have 
  - more data for the the problem transfer from
  - less data for the problem transfer to

## Orthogonalization

have a distinct set of knobs to adjust the parameters

### Workflow

- try a lot of ideas
- train up different models on the training set
- use the development set to evaluate
- pick one
- keep iterating to improve development set performance

### Set Distribution

- choose the dev set and test set to freflect data expected
- take all data from the same distribution (training/development/test)
- randomly shuffle data

### Human-Level Performance

#### ML worse than humans

- get labeled data from humans
- gain insight from manual error analysis
- better analysis of bias/variance

#### Avoidable Bias

The difference between Bayes Error and the training error

