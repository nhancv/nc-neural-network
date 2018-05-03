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

# seed random numbers to make calculation deterministic (Deterministic algorithm)
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

print("Weights After Training:")
print(syn0)

print("\n")
print("TESTING")
print(""
      "- Out1: without transfer function\n"
      "- Out2: with transfer function\n"
      "")
print('{0:10} {1:15} {2:10}'.format("Input", "Out1", "Out2"))
# input testing
TX = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])


Res = sigmoid(TX.dot(syn0))
Res2 = transfer(sigmoid(TX.dot(syn0)))
for index in range(len(TX)):
    print('{0:10} {1:15} {2:10}'.format(TX[index], Res[index], Res2[index]))

