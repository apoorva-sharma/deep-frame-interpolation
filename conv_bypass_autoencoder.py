
import tensorflow as tf
import numpy as np
import math

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


def frame_interpolator(image_shape):
    x = tf.placeholder(tf.float32, [image_shape[0], image_shape[1], image_shape[2], 2*image_shape[3]], name='x') # input is two images
    y = tf.placeholder(tf.float32, image_shape, name='y')

    layer_depths = [30, 30, 30]
    filter_sizes = [3, 3, 3]
    conv_outputs = []

    current_input = x
    current_inputdepth = 2*image_shape[3]
    # convolutional portion
    for i, outputdepth in enumerate(layer_depths):
        result = conv_layer(current_input, filter_sizes[i], current_inputdepth, outputdepth)
        conv_outputs.append(result)
        current_input = result
        current_inputdepth = outputdepth

    z = current_input

    layer_depths.reverse()
    filter_sizes.reverse()
    conv_outputs.reverse()

    # deconv portion
    for i, outputdepth in enumerate(layer_depths[:-1]): # reverse process exactly until last step
        result = deconv_layer(current_input, filter_sizes[i], current_inputdepth, outputdepth)
        stack = tf.concat(3,[result, conv_outputs[i+1]])
        current_input = stack
        current_inputdepth = 2*outputdepth

    yhat = deconv_layer(current_input, filter_sizes[-1], current_inputdepth, image_shape[3])

    # define the loss
    loss = tf.reduce_sum(tf.square(y-yhat))

    return {'x':x, 'y':y, 'z':z, 'yhat':yhat, 'loss':loss}



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


def test_frame_interpolator():
    import data_loader
    import matplotlib.pyplot as plt

    dataset = data_loader.read_data_set()
    mean_img = np.mean(dataset.train.labels, axis=0)

    fi = frame_interpolator([None,384,384,3])

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fi['loss'])

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())

    # Fit all the training data
    batch_size = 4
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(dataset.train.num_examples // batch_size):
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            train_xs = np.array([img - np.tile(mean_img,[1,1,2]) for img in batch_xs])
            train_ys = np.array([img - mean_img for img in batch_ys])
            sess.run(optimizer, feed_dict={fi['x']: train_xs, fi['y']: train_ys})
        print(epoch_i, sess.run(fi['loss'], feed_dict={fi['x']: train_xs, fi['y']: train_ys}))

    # %%
    # Plot example reconstructions
    n_examples = 4
    test_xs, test_ys = dataset.test.next_batch(n_examples)
    test_xs_norm = np.array([img - np.tile(mean_img,[1,1,2]) for img in test_xs])
    test_ys_norm = np.array([img - mean_img for img in test_ys])
    recon = sess.run(fi['yhat'], feed_dict={fi['x']: test_xs_norm})

    fig, axs = plt.subplots(3, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow((np.reshape(0.5*test_xs[example_i,:,:,0:3] + 0.5*test_xs[example_i,:,:,3:6], (384,384,3)))/255)
        axs[1][example_i].imshow((np.reshape(recon[example_i, ...] + mean_img, (384, 384, 3)))/255)
        axs[2][example_i].imshow((np.reshape(test_ys[example_i,:,:,:], (384, 384, 3)))/255)
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_frame_interpolator()






