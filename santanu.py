#-----------------------------------------------------------------------

import tensorflow as tf

#-----------------------------------------------------------------------

# Handy synonyms

NOT_SPECIFIED_1 = None
NOT_SPECIFIED_2 = -1

# The input images are 28 pixels x 28 pixels.
# All pixels are black and white.

NUM_ROW_IN_IMG = 28
NUM_COL_IN_IMG = 28
NUM_CHANNEL_IN_IMG = 1

# The first convolutional layer's kernel.
# The convolution will transform each input image into an image with
# the same number of pixels, with 64 channels.
# Eg, if the input image is 28 x 28 pixels and 3 channels, the image
# after convolution 1 will be 28 x 28 pixels with 64 channels

NUM_ROW_KERNEL_CONV_LAYER_1 = 5
NUM_COL_KERNEL_CONV_LAYER_1 = 5
NUM_CHANNEL_CONV_LAYER_1 = 64
PADDING_CONV_LAYER_1 = 'SAME'

# Max pooling decreases number of pixels along x-axis by a factor of 2 and
# along y-axis by a factor of 2.
# The result will be that each image will have 1/4 as many pixels as
# convolution layer 1, but the same number of channels.
# Eg if the image, after convolution 1, is 28 x 28 pixels with 64 channels
# the image after pooling 1, is 14 x 14 pixels with 64 channels

PIXEL_DECREASE_MAX_POOL_LAYER_1 = 2

# The second convolutional layer's kernel
# As an example let the image coming into this layer has 14 x 14 pixels
# with 64 channels.
# The result of this convolution will be 14 x 14 pixels with 128 channels

NUM_ROW_KERNEL_CONV_LAYER_2 = 5
NUM_COL_KERNEL_CONV_LAYER_2 = 5
NUM_CHANNEL_CONV_LAYER_2 = 128
PADDING_CONV_LAYER_2 = 'SAME'

# Max pooling decreases number of pixels along x-axis by a factor of 2 and
# along y-axis by a factor of 2.
# The result will be that each image will have 1/4 as many pixels as
# convolution layer 2, but the same number of channels.
# Eg, let the image coming into this layer be 14 x 14 pixels with 128
# channels. The result of this max pooling will be 7 x 7 pixels with
# 128 channels.

PIXEL_DECREASE_MAX_POOL_LAYER_2 = 2

# Fully connected layer
# Eg if the image coming in is 7 x 7 pixels with 128 channels, it is
# treated as a column vector with 7 * 7 * 128 = 6272 entries = 6272
# neurons.

NUM_IN_NEURON_FC_LAYER = NUM_ROW_IN_IMG
NUM_IN_NEURON_FC_LAYER /= PIXEL_DECREASE_MAX_POOL_LAYER_1
NUM_IN_NEURON_FC_LAYER /= PIXEL_DECREASE_MAX_POOL_LAYER_2

NUM_IN_NEURON_FC_LAYER *= NUM_COL_IN_IMG
NUM_IN_NEURON_FC_LAYER /= PIXEL_DECREASE_MAX_POOL_LAYER_1
NUM_IN_NEURON_FC_LAYER /= PIXEL_DECREASE_MAX_POOL_LAYER_2

NUM_OUT_NEURON_FC_LAYER = 1024

# The last layer comprises 10 classes

NUM_CLASS = 10

#-----------------------------------------------------------------------

# Placeholders

tmp = NUM_ROW_IN_IMG * NUM_COL_IN_IMG * NUM_CHANNEL_IN_IMG
raw_x = tf.placeholder (tf.float32,
                        [NOT_SPECIFIED_1, tmp])
x = tf.reshape (raw_x, shape = [NOT_SPECIFIED_2,
                                NUM_ROW_IN_IMG,
                                NUM_COL_IN_IMG,
                                NUM_CHANNEL_IN_IMG])

y = tf.placeholder (tf.float32, [NOT_SPECIFIED_1, 10])

keep_prob = tf.placelholder (tf.float32)

#-----------------------------------------------------------------------

# Coefficients of the convolutional kernel in convolutional layer 1

w_conv1 = tf.Variable (tf.random_normal ([NUM_ROW_KERNEL_CONV_LAYER_1,
                                          NUM_COL_KERNEL_CONV_LAYER_1,
                                          NUM_CHANNEL_IN_IMG,
                                          NUM_CHANNEL_CONV_LAYER_1]))

# Bias just after first convolution

b_conv1 = tf.Variable (tf.random_normal ([NUM_CHANNEL_CONV_LAYER_1])

# Coefficients of the convolutional kernel in convolutional layer 2

w_conv2 = tf.Variable (tf.random_normal ([NUM_ROW_KERNEL_CONV_LAYER_2,
                                          NUM_COL_KERNEL_CONV_LAYER_2,
                                          NUM_CHANNEL_CONV_LAYER_1,
                                          NUM_CHANNEL_CONV_LAYER_2]))

# Bias just after second convolution

b_conv2 = tf.Variable (tf.random_normal ([NUM_CHANNEL_CONV_LAYER_2])

# Weights of the full-connected layer

w_fc = tf.Variable (tf.random_normal ([NUM_IN_NEURON_FC_LAYER,
                                       NUM_OUT_NEURON_FC_LAYER]))

# Bias of the fully-connected layer

b_fc = tf.Variable (tf.random_normal ([NUM_OUT_NEURON_FC_LAYER]))

# Weights of the output layer

w_out = tf.Variable (tf.random_normal ([NUM_OUT_NEURON_FC_LAYER,
                                        NUM_CLASS]))

# Bias of the output layer

b_out = tf.Variable (tf.random_normal ([NUM_CLASS]))

#-----------------------------------------------------------------------

def cnn (x, weight, bias, dropout):

    # First convolutional layer

    conv1 = tf.nn.conv2d (x, w_conv1,
                          strides = [1, 1, 1, 1],
                          padding = 'PADDING_CONV_LAYER_1')

    # Add a bias to the first convolutional layer

    conv1_with_bias = tf.nn.bias_add (conv1, b_conv1)

    # Apply ReLU

    relu1 = tf.nn.relu (conv1_with_bias)

    # First max pooling layer

    pool1 = tf.nn.max_pool (relu1,
                            ksize   = [1, PIXEL_DECREASE_MAX_POOL_LAYER_1,
                                       PIXEL_DECREASE_MAX_POOL_LAYER_1, 1],
                            strides = [1, PIXEL_DECREASE_MAX_POOL_LAYER_1,
                                       PIXEL_DECREASE_MAX_POOL_LAYER_1, 1],
                            padding = 'PADDING_MAX_POOL_LAYER_1')

    # Second convolutional layer

    conv2 = tf.nn.conv2d (x, w_conv2,
                          strides = [1, 1, 1, 1],
                          padding = 'PADDING_CONV_LAYER_2')

    # Add a bias

    conv2_with_bias = tf.nn.bias_add (conv2, b_conv2)

    # Apply ReLU

    relu2 = tf.nn.relu (conv2_with_bias)

    # Max pooling

    pool2 = tf.nn.max_pool (relu2,
                            ksize   = [1, PIXEL_DECREASE_MAX_POOL_LAYER_2,
                                       PIXEL_DECREASE_MAX_POOL_LAYER_2, 1],
                            strides = [1, PIXEL_DECREASE_MAX_POOL_LAYER_2,
                                       PIXEL_DECREASE_MAX_POOL_LAYER_2, 1],
                            padding = 'PADDING_MAX_POOL_LAYER_2')

    # Fully Connected Layer

    fc = tf.reshape (conv2, [NOT_SPECIFIED_2, NUM_IN_NEURON_FC_LAYER])
    fc = tf.matmul (fc, w_fc)
    fc = tf.add (fc, b_fc)
    fc = tf.nn.relu (fc)
    fc = tf.nn.dropout (fc, keep_prob)

    # Output layer

    out = tf.matmul (fc, w_out)
    out = tf.add (out, b_out)

    # Return value

    return out

#-----------------------------------------------------------------------

