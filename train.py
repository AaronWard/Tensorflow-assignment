import tensorflow as tf 
import csv
import time
import numpy as np

NUM_EXAMPLES = 500 # Number of training or evaluation examples
NUM_FEATURES = 2 # Number of input features
NUM_LABELS = 1	# Number of output features or class labels
NUM_HIDDEN = 2 # Number of hidden/middle layer nodes
LEARNING_RATE = 0.45 # Speed of learning
NUM_EPOCHS = 1000 # Present the training data this number of times
OUTPUT_INTERVAL = 100 # Used for output during training

TRAIN_DATA = 'data/train.csv'
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

# print('Training data', x)
# print('Label', y)

########################################################################################
'''
Set up input variables and network structure

'''



x_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_LABELS], name = 'labels')

weights = {
    #                       2 inputs, 2 hidden layer nodes
    'w_1': tf.Variable(tf.random_uniform([NUM_FEATURES, NUM_HIDDEN], -1, 1)),
    'w_2': tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_LABELS], -1, 1))
}

biases = {
    #                       2 hidden
    'b_1': tf.Variable(tf.zeros([NUM_HIDDEN])),
    'b_2': tf.Variable(tf.zeros([NUM_LABELS]))
}

layer_1 = tf.sigmoid(tf.matmul(x_, weights['w_1']) + biases['b_1'])
output_layer = tf.sigmoid(tf.matmul(layer_1, weights['w_2']) + biases['b_2'])


########################################################################################
'''
Define functions for training 

'''
# Predicted output - expected output
cost = tf.reduce_sum(tf.square(output_layer - y_))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

########################################################################################
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    t_start = time.clock()
    for epoch in range(NUM_EPOCHS):
        _, c = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})
        print('Epoch : ', epoch, " - Cost: ", c)

    t_end = time.clock()
    print('Elapsed time: ', t_end - t_start)

    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: " , save_path)