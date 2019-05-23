#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os


#%%
import tensorflow as tf

#%% 
tf.enable_eager_execution() # Allows us to not need to initalize variables (more below)


#%%
import pandas as pd
import numpy as np

# Graphing
import matplotlib.pyplot as plt

#%%
# You can also generate data
one_eg = tf.ones(shape=(3,3))
random_eg = tf.random_uniform([3,3])


#%%
# Take a look
one_eg


#%%
# Take a look
random_eg

#%%
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)

# Can confirm on https://matrix.reshish.com/multCalculation.php


#%%
x = tf.Variable(3.0, name="x")
y = tf.Variable(2.0, name="y")
z = tf.Variable(4.0, name="z")

#%% 
with tf.GradientTape() as t:
    t.watch(x)
    
    # Our next layer
    a = tf.add(x,y, name="a")
    b = tf.multiply(a, z, name="b")
    f = tf.square(b, name="f")


#%%
# We could also just ask for b back to simplify the above:
b


#%%
# Now we can differentiate 
# See more here https://www.tensorflow.org/tutorials/eager/automatic_differentiation
df_da = t.gradient(f, a)
df_da


#%%
x = tf.Variable(3.0, name="x")
y = tf.Variable(2.0, name="y")
z = tf.Variable(4.0, name="z")

# Our next layer
a = tf.add(x,y, name="a")
b = tf.multiply(a, z, name="b")
f = tf.square(b, name="f")

b


#%%
# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

#%% 
graph = tf.get_default_graph()

print(graph.get_operations())

#%%
# Now we create a session
with tf.Session() as sess:
     
    # # initialize the variables
    # sess.run(init_op)
     
    # run the operation
    output = sess.run(f)
  
    print("Value of the equation is : {}".format(output))
    sess.close()

#%% [markdown]
# # 3. A practical example
#%% [markdown]
# Adapted from:
# * https://appliedmachinelearning.blog/2018/12/26/tensorflow-tutorial-from-scratch-building-a-deep-learning-model-on-fashion-mnist-dataset-part-1/    

#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
import os
import pickle
data_dir = os.getcwd() + "/new_data"


#%%
def read_file(filename):
    with open(filename, 'rb') as file_in:
        dataset = pickle.load(file_in)
        file_in.close()
    return dataset


#%%
def read_files():
    global X_train, X_test, X_val, y_train, y_test, y_val
    file_list = ["X_train.pkl", "X_test.pkl", 
                 "y_train.pkl", "y_test.pkl",
                "X_val.pkl", "y_val.pkl"]
    file_list = [data_dir + "/" + x for x in file_list]

    X_train = read_file(file_list[0])
    X_test = read_file(file_list[1])
    y_train = read_file(file_list[2])
    y_test = read_file(file_list[3])
    
    X_val = read_file(file_list[4])
    y_val = read_file(file_list[5])


#%%
read_files()


#%%
#Confirm it worked
print(X_train.shape,X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)

#%% [markdown]
# ## 3.1 Some setup

#%%
n_input = 784
n_hidden1 = 128
n_hidden2 = 128
n_class = 10
n_epoch = 20
learning_rate = 0.001
batch_size = 128
dropout = 0.20

#%% [markdown]
# ## 3.2 Forward Prop

#%%
# Our forward layer

def model(batch_x):
 
    """
    We will define the learned variables, the weights and biases,
    within the method ``model()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables. 
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    """
 
    b1 = tf.get_variable("b1", [n_hidden1], initializer = tf.zeros_initializer())
    h1 = tf.get_variable("h1", [n_input, n_hidden1], initializer = tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.add(tf.matmul(batch_x,h1),b1))
 
    b2 = tf.get_variable("b2", [n_hidden2], initializer = tf.zeros_initializer())
    h2 = tf.get_variable("h2", [n_hidden1, n_hidden2], initializer = tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,h2),b2))
 
    b3 = tf.get_variable("b3", [n_class], initializer = tf.zeros_initializer())
    h3 = tf.get_variable("h3", [n_hidden2, n_class], initializer = tf.contrib.layers.xavier_initializer())
    layer3 = tf.add(tf.matmul(layer2,h3),b3)
 
    return layer3


#%%
# One hot encode the labels
def one_hot(n_class, Y):
    """
    return one hot encoded labels to train output layers of NN model
    """
    return np.eye(n_class)[Y]

#%% [markdown]
# ## 3.3  Loss

#%%
# See here for the useful nn module https://www.tensorflow.org/api_docs/python/tf/nn

def compute_loss(predicted, actual):
    """
    This routine computes the cross entropy log loss for each of output node/classes.
    returns mean loss is computed over n_class nodes.
    """
 
    total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = predicted,labels = actual)
    avg_loss = tf.reduce_mean(total_loss)
    
    return avg_loss

#%% [markdown]
# ## 3.4  Optimiser

#%%
# Create an optimiser

def create_optimizer():
 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer


#%%
def one_hot(n_class, Y):
    """
    returns one hot encoded labels to train output layers of NN model
    """
    return np.eye(n_class)[Y]

#%% [markdown]
# ## 3.5 Train

#%%
def train(X_train, X_val, X_test, y_train, y_val, y_test, verbose = False):
    """
    Trains the network, also evaluates on test data finally.
    """
    # Creating place holders for image data and its labels
    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")
 
    # Forward pass on the model
    logits = model(X)
 
    # computing sofmax cross entropy loss with logits
    avg_loss = compute_loss(logits, Y)
 
    # create adams' optimizer, compute the gradients and apply gradients (minimize())
    optimizer = create_optimizer().minimize(avg_loss)
 
    # compute validation loss
    validation_loss = compute_loss(logits, Y)
 
    # evaluating accuracy on various data (train, val, test) set
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
    # initialize all the global variables
    init = tf.global_variables_initializer()
 
    # starting session to actually execute the computation graph
    with tf.Session() as sess:
 
        # all the global varibles holds actual values now
        sess.run(init)
 
        # looping over number of epochs
        for epoch in range(n_epoch):
 
            epoch_loss = 0.
 
            # calculate number of batches in dataset
            num_batches = np.round(X_train.shape[0]/batch_size).astype(int)
 
            # looping over batches of dataset
            for i in range(num_batches):
 
                # selecting batch data
                batch_X = X_train[(i*batch_size):((i+1)*batch_size),:]
                batch_y = y_train[(i*batch_size):((i+1)*batch_size),:]
 
                # execution of dataflow computational graph of nodes optimizer, avg_loss
                _, batch_loss = sess.run([optimizer, avg_loss],
                                                       feed_dict = {X: batch_X, Y:batch_y})
 
                # summed up batch loss for whole epoch
                epoch_loss += batch_loss
            # average epoch loss
            epoch_loss = epoch_loss/num_batches
 
            # compute validation loss
            val_loss = sess.run(validation_loss, feed_dict = {X: X_val ,Y: y_val})
 
            # display within an epoch (train_loss, train_accuracy, valid_loss, valid accuracy)
            if verbose:
                print("epoch:{epoch_num}, train_loss: {train_loss}, train_accuracy: {train_acc}, val_loss: {valid_loss}, val_accuracy: {val_acc} ".format(
                                                       epoch_num = epoch,
                                                       train_loss = round(epoch_loss,3),
                                                       train_acc = round(float(accuracy.eval({X: X_train, Y: y_train})),2),
                                                       valid_loss = round(float(val_loss),3),
                                                       val_acc = round(float(accuracy.eval({X: X_val, Y: y_val})),2)
                                                      ))
 
        # calculate final accuracy on never seen test data
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))
        sess.close()


#%%
# One hot encoding of labels for output layer training
y_train =  one_hot(n_class, y_train)
y_val = one_hot(n_class, y_val)
y_test = one_hot(n_class, y_test)


#%%
# Let's train and evaluate the fully connected NN model
train(X_train, X_val, X_test, y_train, y_val, y_test, True)

#%% [markdown]
# # 4. An extended tutorial
#%% [markdown]
# Work through this tutorial:
# 
# * https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#1
#%% [markdown]
# # 5. In Keras
#%% [markdown]
# Source:
# 
# * https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

#%%
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#%%
print(y_train.shape)
print(y_train[0:5])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_train[0:5])


#%%
# Build a simple model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


#%%
# Run it

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%



