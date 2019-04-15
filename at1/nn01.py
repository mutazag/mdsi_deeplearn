# building NN from scratch using numpy and calculus 

import numpy as np


# follow along with https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3

# the sample data is the result of XOR operation 

#   A   B   A XOR B
#   0   0       0  
#   1   0       1
#   0   1       1
#   1   1       1

# first is the network structure 
# 1 x input layer 
# 1 x hidden layer 
# 1 x output layer 

# activation function 
# for the last layer, we will use a sigmoid function as it outputs values between 0 and 1 
# for hidden layers, will use a tanh() function 

# other functions for activation could be things like ReLU



# def a sigmoid function -- used as activation to the last layer 
def sigmoid(x): 
    return 1/(1 + np.exp(-x))

print("sigmoid of 5 is {}".format(sigmoid(5)))

# paramaters in this NN are W and B
# one hidden layer(W1 and B1)
# one output layer(W2 and B2)
# W1, W2
# B1, B2

np.random.seed(2)

# The 4 training examples by columns
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
print(X)
print(X.shape)
# x is (2,4) - that is 2 features and 4 samples 

# The outputs of the XOR for every example in X
Y = np.array([[0, 1, 1, 0]])
print(Y.shape)
print(Y)
# y, the output, is 1,4, as in 1 label and 4 examples 

for i in range(0,4): 
    print("{} {} --> {}".format(
        X[0,i], 
        X[1,i], 
        Y[0,i]
    ))




# n0 -> number of input neurons, that is the number of features in a sample 
#dimensions of X is: Sn x  n0
#   where Sn is number of samples 
#   n0 number of features in the sample  

# n1 is number of neurons in hidden layer 
#   dimensions of W1 == n0 x n1
#   Lh = XW1 + B1
#   dimensins of XW1 is (Sn x n0) dot (n0 x n1) ==> (Sn x n1)
# 

# n2 is number of neurons in the output layer  
# dimensions of W2 == n1 x n2
#     Lout = LhW2 + B2
#     dimonsions of LhW1 is (Sn x n1) dot (n1 x n2) ==> (Sn x n2)



def initialise_parameters(n_x, n_h, n_y): 
    # n_x - number of featurs in input 
    # n_h - number of neurons in hidden layer 
    # n_y - number of neurons in output 
    # inour exmaple that will be 2, 2, 1 
    print("input feautres {}\nhidden layer neurons {}\noutput layer neurons {}".format(
        n_x, 
        n_h, 
        n_y))

    # w1 is n_x by n_h, b1 must be n_h 
    w1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h,1))
    print("w1: {} x {}:\n {}".format(n_x, n_h, w1))
    print("b1: {} x {}:\n {}".format(n_h, 1, b1))

    # w2 is n_h x n_y, b must be b_y
    w2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    print("w2: {} x {}:\n {}".format(n_h, n_y, w2))
    print("b2: {} x {}:\n {}".format(n_y, 1, b2))

    parameters = {
        "w1": w1, 
        "b1": b1, 
        "W2": w2, 
        "b2": b2
    }
    return parameters


initialise_parameters(2,2,1)


