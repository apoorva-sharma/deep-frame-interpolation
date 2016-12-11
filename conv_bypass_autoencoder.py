
import tensorflow as tf
import numpy as np
import math
import msssim

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def deconv2d(x, width, height, outputdepth, W):
    deconv2d = tf.nn.conv2d_transpose(x, W,
     output_shape = [tf.shape(x)[0], width, height, outputdepth],
       strides=[1,2,2,1], padding='SAME')
    return deconv2d

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, filtersize, inputdepth, outputdepth):
    W_conv_layer1 = weight_variable([filtersize, filtersize, inputdepth, outputdepth])
    b_conv_layer1 = bias_variable([outputdepth])
    h_conv_layer1 = tf.nn.relu(conv2d(input, W_conv_layer1) + b_conv_layer1)
    max_pool1 = max_pool_2x2(h_conv_layer1)
    return max_pool1

def deconv_layer(input, filtersize, inputdepth, outputdepth):
    height = 2 * int(input.get_shape()[1])
    width = 2 * int(input.get_shape()[2])

    W_deconv_layer1 = weight_variable([filtersize, filtersize, outputdepth, inputdepth])
    b_deconv_layer1 = bias_variable([outputdepth])
    h_deconv_layer1 = tf.nn.relu(deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1)
    return h_deconv_layer1


def autoencoder(input_shape):
    x = tf.placeholder(tf.float32, input_shape, name='x') # m * n*n * 3
    print(tf.rank(x))

    # conv layers
    h_conv1 = conv_layer(x, 3, input_shape[3], 10) # m * n/2*n/2 * 10
    print(tf.rank(h_conv1))
    h_conv2 = conv_layer(h_conv1, 3, 10, 10) # m * n/4*n/4 * 10
    print(tf.rank(h_conv2))
    z = conv_layer(h_conv2, 3, 10, 10) # latent representation, m * n/8*n/8 * 10

    # deconv layers
    h_deconv1 = deconv_layer(z, 3, 10, 10) # m * n/4*n/4 * 10
    print(tf.rank(h_deconv1))
    h_deconv1_stacked = tf.concat(3, [h_deconv1, h_conv2]) # m * n/4*n/4 * 20
    print(tf.rank(h_deconv1_stacked))
    h_deconv2 = deconv_layer(h_deconv1_stacked, 3, 20, 10) # m * n/2*n/2 * 10
    h_deconv2_stacked = tf.concat(3, [h_deconv2, h_conv1]) # m * n/2*n/2 * 20
    y = deconv_layer(h_deconv2_stacked, 3, 20, input_shape[3]) # m * n*n * 3

    loss = tf.reduce_sum(tf.square(y-x));

    return {'x':x, 'z':z, 'y':y, 'loss':loss}



def test_bypass_autoencoder():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # import some data
    # load MNIST
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)

    ae = autoencoder([None,32,32,1])

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['loss'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    batch_size = 100
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            train = np.reshape(train, [-1,28,28,1]) 
            train = np.lib.pad(train, ((0,0),(2,2),(2,2),(0,0)),'constant')
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['loss'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    test_xs_norm = np.reshape(test_xs_norm, [-1,28,28,1]) 
    test_xs_norm = np.lib.pad(test_xs_norm,((0,0),(2,2),(2,2),(0,0)),'constant')
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...][2:-2,2:-2,:], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_bypass_autoencoder()






