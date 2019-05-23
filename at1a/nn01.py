#%% 
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


#%% 
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
        "W1": w1, 
        "b1": b1, 
        "W2": w2, 
        "b2": b2
    }
    return parameters



# build the forward propogation function 
# takes inpout X and network parameters 
def forward_prop(X, p):
    W1 = p["W1"]
    b1 = p["b1"]
    W2 = p["W2"]
    b2 = p["b2"]

    # general formula 
    # An = ActivationFunction ( X.W + B)
    
    # first layer 
    Z1 = np.dot(W1, X)  + b1
    A1 = np.tanh(Z1) # activation for layer 1 is tanh

    #second layer 
    Z2 = np.dot(W2, Z1) + b2 # Z1 is input to Z2 
    A2 = sigmoid(Z2) # activationfor layer 2 is sigmoid 

    cache = {
        "A1": A1, 
        "A2": A2
    }

    # returning the final output abd the cache 
    return A2, cache



# cost function 
# now have compute the loss function. We will use the Cross
# Entropy Loss function. Calculate_cost(A2, Y) takes as input the result of the
# NN A2 and the groundtruth matrix Y and returns the cross entropy cost:


def calculate_cost(A2, Y):
    cost = -np.sum(
        np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2))
        )/m  # m is number of samples in X
    cost = np.squeeze(cost)
    return cost



# Now the most difficult part of the Neural Network algorithm, Back Propagation.
# The code here may seem a bit weird and difficult to understand but we will not
# dive into details of why it works here. This function will return the
# gradients of the Loss function with respect to the 4 parameters of our
# network(W1, W2, b1, b2):


def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads





# Nice, now we have the all the gradients of the Loss function, so we can
# procceed to the actual learning! We will use Gradient Descent algorithm to
# update our parameters and make our model learn with the learning rate passed
# as a parameter:

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    new_parameters = {
        "W1": W1,
        "W2": W2,
        "b1" : b1,
        "b2" : b2
    }

    return new_parameters







def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialise_parameters(n_x, n_h, n_y)

    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)

        if(i%100 == 0):
             print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters


def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    if(yhat >= 0.5):
        y_predict = 1
    else:
        y_predict = 0

    return y_predict



np.random.seed(2)

# The 4 training examples by columns
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# The outputs of the XOR for every example in X
Y = np.array([[0, 1, 1, 0]])

# No. of training examples
m = X.shape[1]
# Set the hyperparameters
n_x = 2     #No. of neurons in first layer
n_h = 2     #No. of neurons in hidden layer
n_y = 1     #No. of neurons in output layer
num_of_iters = 1000
learning_rate = .3



trained_parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)
# Test 2X1 vector to calculate the XOR of its elements. 
# You can try any of those: (0, 0), (0, 1), (1, 0), (1, 1)
# X_test = np.array([[1], [1]])
# y_predict = predict(X_test, trained_parameters)


def predict_and_print(Xin, trained_parameters): 
    predict_y = predict(Xin, trained_parameters)
    # Print the result
    print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(
    Xin[0][0], Xin[1][0], predict_y))
    return predict_y


X_test = np.array([[0],[0]])
y_predict = predict_and_print(X_test, trained_parameters)
X_test = np.array([[0],[1]])
y_predict = predict_and_print(X_test, trained_parameters)
X_test = np.array([[1],[0]])
y_predict = predict_and_print(X_test, trained_parameters)
X_test = np.array([[1], [1]])
y_predict = predict_and_print(X_test, trained_parameters)
# Print the result
# print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(
#     X_test[0][0], X_test[1][0], y_predict))

