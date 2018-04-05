'''
This neural network is part of an assignment for Computational intelligence
It takes two inputs from a CSV file for training with tensorflow

@Author: Aaron Ward - B00079288
'''

import tensorflow as tf 
import csv
import time
import numpy as np
########################################################################################
'''
Hyper parameters for fine tuning the network

'''
# Number of training or evaluation examples
NUM_EXAMPLES = 500
 # Number of inputs
NUM_INPUTS = 2
# Number of output features or class labels
NUM_LABELS = 1	
# Number of hidden/middle layer nodes
NUM_HIDDEN = 3
# tells the optimizer how far to move the weights in the direction of the gradient
LEARNING_RATE = 0.01
# Number of iterations of traversing through the data
NUM_EPOCHS = 10000

########################################################################################
'''
Basic data retrieval and sorting into lists 

'''
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/eval.csv'
MODEL_PATH = 'data/trained_model.ckpt'

#input features and labels lists
x = []
y = []

# import training data
file  = open(TRAIN_DATA, "r")
input_data = csv.reader(file, delimiter=',')
for row in input_data:
    x.append([float(row[0]), float(row[1])])
    y.append([float(row[2])])

########################################################################################
'''
Set up the tensorflow placeholders, Create dictionaries for
the weights and biases and and network structure structure

'''

x_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_INPUTS])
y_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_LABELS])

weights = {
    #                       2 inputs, 2 hidden layer nodes
    'w_1': tf.Variable(tf.random_uniform([NUM_INPUTS, NUM_HIDDEN], -1, 1)),
    # 'w_2': tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_HIDDEN], -1, 1)),  
    'out': tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_LABELS], -1, 1))
}

biases = {
    #                       2 hidden
    'b_1': tf.Variable(tf.zeros([NUM_HIDDEN])),
    # 'b_2': tf.Variable(tf.random_normal([NUM_HIDDEN])),
    'out': tf.Variable(tf.zeros([NUM_LABELS]))
}

layer_1 = tf.sigmoid(tf.matmul(x_, weights['w_1']) + biases['b_1'])
# layer_2 = tf.sigmoid(tf.matmul(layer_1, weights['w_2']) + biases['b_2'])
output_layer = tf.sigmoid(tf.matmul(layer_1, weights['out']) + biases['out'])


########################################################################################
'''
Define cost and optimization function for training
the network.

'''

# Predicted output - expected output
cost = tf.reduce_sum(tf.square(output_layer - y_))
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

########################################################################################
'''
Run a tensorflow session and print the metrics to the console

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    error = 0
    interval = 100
    t_start = time.clock()

    # Feed the values into the network
    for epoch in range(0, NUM_EPOCHS):
        op, acc, ol, c = sess.run([optimizer, accuracy,output_layer, cost], feed_dict={x_: x, y_: y})

        if epoch % interval == 0:
            # op, ol, c = sess.run([optimizer,output_layer, cost], feed_dict={x_: x, y_: y})
            for i in range(len(ol)):
                    error = abs(ol[i] - y[i])
            print('Epoch: ', epoch,' - Accuracy: ', acc, ' - Cost: ', c,' - Error: ', error)

    t_end = time.clock()
    print('\n########################### FINAL RESULT ##################################')
    print('Cost: ', c,' - Error: ', error, ' - Elapsed time: ', t_end - t_start)
    print('###########################################################################\n')

    #Save the model to disk
    # save_path = saver.save(sess, MODEL_PATH)
    # print("Model saved in file: " , save_path)
