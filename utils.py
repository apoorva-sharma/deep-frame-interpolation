import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def deconv2d(x, width, height, outputdepth, W):
    deconv2d = tf.nn.conv2d_transpose(x, W,
     output_shape = [tf.shape(x)[0], width, height, outputdepth],
       strides=[1,2,2,1], padding='SAME')
    return deconv2d

def conv_layer(input, filtersize, inputdepth, outputdepth, name="conv2d"):
    with tf.variable_scope(name):
        W_conv_layer1 = weight_variable([filtersize, filtersize, inputdepth, outputdepth])
        b_conv_layer1 = bias_variable([outputdepth])
        h_conv_layer1 = conv2d(input, W_conv_layer1) + b_conv_layer1
        return h_conv_layer1

def deconv_layer(input, filtersize, inputdepth, outputdepth, name="deconv2d"):
    with tf.variable_scope(name):
        height = 2 * int(input.get_shape()[1])
        width = 2 * int(input.get_shape()[2])

        W_deconv_layer1 = weight_variable([filtersize, filtersize, outputdepth, inputdepth])
        b_deconv_layer1 = bias_variable([outputdepth])
        h_deconv_layer1 = deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1
        return h_deconv_layer1

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x  +  f2 * abs(x)

def linear(input_, output_size, stddev=0.02, bias_start=0.0, with_w=False, name="linear"):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class batch_norm(object):
    def __init__(self, epsilon = 1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                            decay = self.momentum,
                            updates_collections=None,
                            epsilon=self.epsilon,
                            scale=True,
                            is_training=train,
                            scope=self.name)
