import numpy as np


# transfer function
def transfer(matrix):
    res = np.array(matrix)
    for x in np.nditer(res, op_flags=['readwrite']):
        if x[...] > 0.5:
            x[...] = 1
        else:
            x[...] = 0
    return res


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
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# training
for i in range(10000):
    l0 = X
    l1 = sigmoid(l0.dot(syn0))
    l2 = sigmoid(l1.dot(syn1))
    l2_error = Y - l2
    l2_delta = l2_error * sigmoid(l2, True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)


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

Res = sigmoid(sigmoid(TX.dot(syn0)).dot(syn1))
Res2 = transfer(Res)
for index in range(len(TX)):
    print('{0:10} {1:15} {2:10}'.format(TX[index], Res[index], Res2[index]))

