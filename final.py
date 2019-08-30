#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import glob, os
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class JumpingWindow:
    
    
    def __init__(self, size, overlap, strict_size=True):
        if (
            not isinstance(size, int) or
            not isinstance(overlap, int) or
            not isinstance(strict_size, bool) or
            overlap >= size
        ):
            title = "ERROR: Wrong values in JumpingWindow constructor."
            description = "Expecting: size(int), overlap(int), strict_size(bool), AND overlap < size"
            provided = "size({}), overlap({}), strict_size({}), overlap < size? {}".format(
                    type(size),
                    type(overlap),
                    type(strict_size),
                    overlap < size
                )
            raise Exception(title + "\n" + description + "\n" + provided)
        self.windows = []
        self.entries = []
        self.size = size
        self.overlap = overlap
        self.strict_size = strict_size
        self.transformed = False


    def add(self, element):
        if self.transformed:
            raise Exception("Cannot add the elements to the window, once it has been transformed!")
        self.entries.append(element)


    def transform_entries_to_windows(self):
        for i in range(0, len(self.entries), self.size-self.overlap):
            window = self.entries[i:i+self.size]
            if (self.strict_size and (len(window) != self.size)):
                continue
            self.windows.append(window)
        self.transformed = True
    
    
    def getWindow(self, window_number):
        return windows[window_number]
    
    
    def getAllWindows(self):
        return self.windows[:]


    def __str__(self):
        result = "\n---------- Jumping Window ----------\n"
        for i in range(len(self.windows)):
            result = result + "Window {}:\t{}\n".format(i, self.windows[i])
        result = result + "------------------------------------\n"
        return result
    
        


# In[ ]:


def shapeMatrix(one_dimensional_list, num_rows, num_cols):
    if (num_rows * num_cols != len(one_dimensional_list)):
        # not possible to transform the values
        return None
    else:
        matrix = []
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(j)
            matrix.append(row)
        return matrix


# In[ ]:


def getValuesFromFile(filename):
    values = []

    f = open("merged_test_0026.csv", "r")
    line = f.readline() # col titles
    line = f.readline() # first row
    while line:
        values.append(np.float32(line.split(",")[0]))
        line = f.readline()
    f.close()
    
    return values


# In[ ]:


# break down into overlapping jumping windows
def createJumpingWindows(values):
    jws = JumpingWindow(30, 5)
    for value in values:
        jws.add(value)
    jws.transform_entries_to_windows()
    return jws


# In[ ]:


#here I create default graph 
tf.reset_default_graph()


# In[ ]:


# parameters for my stacked autoencoders
num_inputs = 30
neurons_hid1 = 20
neurons_hid2 = 10
neurons_hid3 = neurons_hid1
num_output = num_inputs


# In[ ]:


learning_rate = 0.001


# In[ ]:


activation_fun = tf.nn.sigmoid


# In[ ]:


#now I begin everything I need for the session of my Graph
X = tf.placeholder(tf.float32,shape=[None,None])


# In[ ]:


#now here I set my weight: where I am trying to achieve the weight of the tensors
initializer = tf.variance_scaling_initializer()


# In[ ]:


#here I create my weight variables

w1 = tf.Variable(initializer([num_inputs,neurons_hid1]),dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hid1,neurons_hid2]),dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hid2,neurons_hid3]),dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hid3,num_output]),dtype=tf.float32)


# In[ ]:


#now I create my bias
b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_output))


# In[ ]:


activation_func = tf.nn.sigmoid


# In[ ]:


#now I create my hidden layers
hid_layer1 = activation_func(tf.matmul(X,w1)+b1)
hid_layer2 = activation_func(tf.matmul(hid_layer1,w2)+b2) 
hid_layer3 = activation_func(tf.matmul(hid_layer2,w3)+b3) 
output_layer = activation_func(tf.matmul(hid_layer3,w4)+b4) 


# In[ ]:


#now I define my cost functions(loss)
loss = tf.reduce_mean(tf.squared_difference(output_layer, X))


# In[ ]:


#now I define my optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)


# In[ ]:


#now I define my trianing operation attempting to minimize the loss function
train = optimizer.minimize(loss)


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


# Training
num_epochs = 20

total_training_losses = []

with tf.Session() as sess:
    
    sess.run(init)
    
    for filename in glob.glob("*.csv"):
        print("===== Learning file: {} =====".format(filename))
        
        values = getValuesFromFile(filename)
        jws = createJumpingWindows(values)
        
        training_losses_per_file = []
    
        for epoch in range(num_epochs):

            for jw in jws.getAllWindows():
                matrix = shapeMatrix(jw,1,30)
                sess.run(train, feed_dict={X:matrix})

            training_loss = loss.eval(feed_dict={X:matrix})
            training_losses_per_file.append(training_loss)
            
            print("\tEpoch: {}\tLoss: {}".format(epoch, training_loss))
        
        print(
            "Completed file: {}\tTraining Losses: [{:0.3f}..{:0.3f}]".format(
                filename,
                training_losses_per_file[0],
                training_losses_per_file[-1]
            )
        )
        
        total_training_losses.append(training_losses_per_file[:])
        
    model_filename = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, model_filename)
    print("===== ========== =====\nSaved model in file: {}".format(model_filename))
        


# In[ ]:


to_plot = []
for training_loss_per_file in total_training_losses:
    avg_loss = sum(training_loss_per_file) / len(training_loss_per_file)
    to_plot.append(avg_loss)
    
plt.plot(to_plot, 'r--')
plt.title('Avg Loss Per Trained File')
plt.xlabel('Number of files trained')
plt.ylabel('Losses')


# In[ ]:


# Evaluating

# Craete a list of random numbers representing a test
import random
new_test_values = []
for x in range(360):
    new_test_values.append(random.uniform(-100,100))

jws = createJumpingWindows(new_test_values)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_filename)
    losses_per_segment = []
    for jw in jws.getAllWindows():
        matrix = shapeMatrix(jw,1,30)
        segment_loss = loss.eval(feed_dict={X:matrix})
        losses_per_segment.append(segment_loss)
    
    print(
        "\n===== RESULT: =====\nAvg RSME for segments in the new test: {:0.5f}\n===== ======= =====".format(
            sum(losses_per_segment) / len(losses_per_segment)
        )
    )


# In[ ]:




