#encoding:utf8

import tensorflow as tf 

INPUT_NODE = 128*128
OUTPUT_NODE = 5

IMAGE_SIZE = 128
NUM_CHANNELS = 3
NNUM_LABELS = 5

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

CONV3_DEEP = 128
CONV3_SIZE = 3

FC_SIZE = 512

def Flowers_Cnn(input_tensor, train = True, regularizer = None):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            'weight', [CONV1_SIZE,CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer = tf.contrib.layers.variance_scaling_initializer())
        conv1_biases = tf.get_variable(
            'bias',[CONV1_DEEP], initializer = tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights,strides = [1,1,1,1], padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weight', [CONV2_SIZE,CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer = tf.contrib.layers.variance_scaling_initializer())
        conv2_biases = tf.get_variable(
            'bias',[CONV2_DEEP], initializer = tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(
            pool1, conv2_weights,strides = [1,1,1,1], padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    with tf.variable_scope('layer5-conv3'):
        conv3_weights = tf.get_variable(
            'weight', [CONV3_SIZE,CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer = tf.contrib.layers.variance_scaling_initializer())
        conv3_biases = tf.get_variable(
            'bias',[CONV3_DEEP], initializer = tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(
            pool2, conv3_weights,strides = [1,1,1,1], padding = 'SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(
            relu3, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    pool1_shape = pool3.get_shape().as_list()
    nodes = pool1_shape[1] * pool1_shape[2] * pool1_shape[3]

    reshapeed = tf.reshape(pool3, [pool1_shape[0], nodes])

    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable(
            'weight',[nodes, FC_SIZE],
            initializer = tf.contrib.layers.variance_scaling_initializer())
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshapeed, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable(
            'weight',[FC_SIZE, NNUM_LABELS],
            initializer = tf.contrib.layers.variance_scaling_initializer())
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias',[NNUM_LABELS],initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit