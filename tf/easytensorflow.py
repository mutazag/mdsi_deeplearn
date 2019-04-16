# https://github.com/easy-tensorflow/easy-tensorflow/blob/master/1_TensorFlow_Basics/Tutorials/1_Graph_and_Session.ipynb

#%%
import tensorflow as tf

#%% 
a = 2
b = 3
c = tf.add(a, b, name='Add')
print(c)


#%% 
# to run the graph, put it in a session and run 
sess = tf.Session()
print(sess.run(c))
sess.close()


#%%
print("END")

