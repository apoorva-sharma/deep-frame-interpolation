# Adapted from Parag K. Mital, Jan 2016 convolutional_autoencoder.py

import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt


def autoencoder(input_shape=[None, 16384], # [num_examples, num_pixels]
                n_filters=[1, 10, 10, 10], # number of filters in each conv layer
                filter_sizes=[3, 3, 3, 3]):
    """Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """

    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2: # assuming second dim of input_shape is num_pixels of an example
        # convert 1D image into 2D and add fourth dimension for num_filters
        x_dim = np.sqrt(x.get_shape().as_list()[1]) # assuming each image is square
        if x_dim != int(x_dim): # not a square image
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]]) # reshape input samples to m * 2D image * 1 layer for input
    elif len(x.get_shape()) == 4: # assuming we already did that
        x_tensor = x
    else: 
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]): # enumerate the number of filters in each hidden layer
        n_input = current_input.get_shape().as_list()[3] # number of filters in current input
        shapes.append(current_input.get_shape().as_list()) # append shape of this layer's input
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i], # a filter_size x filter_size filter
                n_input, n_output], # mapping n_inps to n_outs
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input))) # create Weight mx W_ij = rand([-1,1])
        b = tf.Variable(tf.zeros([n_output])) # create Bias vector 
        encoder.append(W)
        output = lrelu( # apply non-linearity
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b)) # add bias to output of conv(inps,W)
        current_input = output

    # store the latent representation
    z = current_input
    encoder.reverse() # going backwards for the decoder
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i] # using same weights as encoder
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]])) # but different biases
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose( # transpose conv is deconv
                current_input, W,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]), # output shape
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference between output and input
    cost = tf.reduce_sum(tf.square(y - x_tensor)) 

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost} # output of symbolic operations representing
        # input, intermediate, output, and cost


 # %%
def test_mandrill():
    """Test the convolutional autoencder using Mandrill Small image."""
    # %%
    import tensorflow as tf
    import numpy as np
    import scipy.io
    import matplotlib.pyplot as plt

    # Load Madrill Small data
    mandrill_small = scipy.io.loadmat('mandrill_small.mat')
    mandrill_small = mandrill_small['A']
    mandrill_small = np.array(mandrill_small)

    mandrill_small = np.transpose(mandrill_small, [2,0,1])
    mandrill_small = np.reshape(mandrill_small, [3,128*128])


    mean_img = np.mean(mandrill_small, axis=0)
    ae = autoencoder()

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    n_epochs = 1
    for epoch_i in range(n_epochs):
        batch_xs = mandrill_small
        train = np.array([img - mean_img for img in batch_xs])
        sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Plot example reconstructions
    test_xs = mandrill_small
    n_examples = 3
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (128, 128)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (128**2,)) + mean_img,
                (128, 128)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()



if __name__ == '__main__':
    test_mandrill()


