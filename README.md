# Neural Network

### Training flow

![Preview](readme/training_flow.png)

### Testing flow

![Preview](readme/testing_flow.png)


## Single layer

Overview

![Preview](readme/overview.png)

Neural network

![Preview](readme/neural_network.png)

Training data set

![Preview](readme/training_set.png)

=> Single data set

![Preview](readme/single_set.png)

Result
```
Weights After Training:
[[ 9.67299303]
 [-0.2078435 ]
 [-4.62963669]]


TESTING
- Out1: without transfer function
- Out2: with transfer function

Input      Out1            Out2      
[0 0 0]    [0.5]           [0.]      
[0 0 1]    [0.009664]      [0.]      
[0 1 0]    [0.44822538]    [0.]      
[0 1 1]    [0.00786466]    [0.]      
[1 0 0]    [0.99993704]    [1.]      
[1 0 1]    [0.99358931]    [1.]      
[1 1 0]    [0.9999225]     [1.]      
[1 1 1]    [0.99211997]    [1.]
```

Explanation:

- numpy: https://docs.scipy.org/doc/numpy/reference/routines.html
- random.seed: https://en.wikipedia.org/wiki/Deterministic_algorithm
- sigmoid: https://en.wikipedia.org/wiki/Sigmoid_function
- slope: https://en.wikipedia.org/wiki/Slope
- sigmoid derivative: https://en.wikipedia.org/wiki/Logistic_function

