#-----------------------------------------------------------------------

import tensorflow as tf

#-----------------------------------------------------------------------

# Handy synonyms

NOT_SPECIFIED1 = None
NOT_SPECIFIED2 = -1

# The input images are 28 pixels x 28 pixels. All pixels are black and
# white.

NUM_ROW_INPUT_IMG = 28
NUM_COL_INPUT_IMG = 28
NUM_CHANNEL_INPUT_IMG = 1

# The first convolutional layer's kernel

NUM_ROW_KERNEL_CONV_LAYER_1 = 5
NUM_COL_KERNEL_CONV_LAYER_1 = 5
NUM_CHANNEL_CONV_LAYER_1 =  64

# The output comprises 10 classes

NUM_CLASS = 10

#-----------------------------------------------------------------------

# Input

tmp = NUM_ROW_INPUT_IMG * NUM_COL_INPUT_IMG * NUM_CHANNEL_INPUT_IMG
raw_x = tf.placeholder (tf.float32,
                        [NOT_SPECIFIED, tmp])
x = tf.reshape (raw_x, shape = [-1,
                                NUM_ROW_INPUT_IMG,
                                NUM_COL_INPUT_IMG,
                                NUM_CHANNEL_INPUT_IMG])

#-----------------------------------------------------------------------
# First Convolutional Layer
# -------------------------

# w1 = coefficients of the convolutional kernel
# conv1 = the result of applying the convolutional kernel to x

w1 = tf.Variable (tf.random_normal ([NUM_ROW_KERNEL_CONV_LAYER_1,
                                     NUM_COL_KERNEL_CONV_LAYER_1,
                                     NUM_CHANNEL_INPUT_IMG,
                                     NUM_CHANNEL_CONV_LAYER_1]))
conv1 = tf.nn.conv2d (x, w1,
                      strides = [1, 1, 1, 1],
                      padding = 'SAME')

#-----------------------------------------------------------------------

# Add a bias to the first convolutional layer, then apply ReLU
# ------------------------------------------------------------

b1 = tf.Variable (tf.random_normal ([NUM_CHANNEL_CONV_LAYER_1])
conv1_with_bias = tf.nn.bias_add (conv1, b1)

relu1 = tf.nn.relu (conv1_with_bias)

#-----------------------------------------------------------------------
# First Max Pooling Layer
# -----------------------

pool1 = tf.nn.max_pool (relu1,
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME')

#-----------------------------------------------------------------------
# Second Convolutional Layer
# --------------------------

# w2 = coefficients of the convolutional kernel
# conv2 = the result of applying the convolutional kernel 

w2 = tf.Variable (tf.random_normal ([NUM_ROW_KERNEL_CONV_LAYER_2,
                                     NUM_COL_KERNEL_CONV_LAYER_2,
                                     NUM_CHANNEL_CONV_LAYER_1,
                                     NUM_CHANNEL_CONV_LAYER_2]))
conv2 = tf.nn.conv2d (x, w2,
                      strides = [1, 1, 1, 1],
                      padding = 'SAME')

#-----------------------------------------------------------------------

# Add a bias to the second convolutional layer, then apply ReLU
# -------------------------------------------------------------

b2 = tf.Variable (tf.random_normal ([NUM_CHANNEL_CONV_LAYER_2])
conv2_with_bias = tf.nn.bias_add (conv2, b2)

relu2 = tf.nn.relu (conv2_with_bias)

#-----------------------------------------------------------------------
# Second Max Pooling Layer
# ------------------------

pool2 = tf.nn.max_pool (relu2,
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME')

#-----------------------------------------------------------------------

# For each input image, we want the neural network to generate one
# probability per class. The number of input images has not been
# specified here, so the number of predictions cannot be specified

y = tf.placeholder (tf.float32, [NOT_SPECIFIED, 10])

#-----------------------------------------------------------------------

