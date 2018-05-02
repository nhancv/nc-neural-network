import numpy as np


# transfer function
def transfer(matrix):
    for x in np.nditer(matrix, op_flags=['readwrite']):
        if x[...] > 0.5:
            x[...] = 1
        else:
            x[...] = 0

    return matrix


# sigmoid function
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input training
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output samples
Y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation deterministic
np.random.seed(1)

# initialize weights randomly [-1..1]
syn0 = 2 * np.random.random((3, 1)) - 1

# training
for i in range(10000):
    l0 = X
    l1 = sigmoid(l0.dot(syn0))
    l1_error = Y - l1
    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += l0.T.dot(l1_delta)

print("Output After Training:")
print(l1)

print("Weights After Training:")
print(syn0)

print("Testing After Training")
# input testing
TX = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
print("- Without transfer function:")
print(sigmoid(TX.dot(syn0)))
print("- With transfer function:")
print(transfer(sigmoid(TX.dot(syn0))))
