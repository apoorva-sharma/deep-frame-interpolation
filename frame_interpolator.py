import tensorflow as tf
import numpy as np
import math
#import msssim
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def conv_layer(input, filtersize, inputdepth, outputdepth, name="conv2d"):
    with tf.variable_scope(name):
        W_conv_layer1 = weight_variable([filtersize, filtersize, inputdepth, outputdepth])
        b_conv_layer1 = bias_variable([outputdepth])
        h_conv_layer1 = tf.nn.relu(conv2d(input, W_conv_layer1) + b_conv_layer1)
        max_pool1 = max_pool_2x2(h_conv_layer1)
        return max_pool1

def deconv_layer(input, filtersize, inputdepth, outputdepth, name="deconv2d"):
    with tf.variable_scope(name):
        height = 2 * int(input.get_shape()[1])
        width = 2 * int(input.get_shape()[2])

        W_deconv_layer1 = weight_variable([filtersize, filtersize, outputdepth, inputdepth])
        b_deconv_layer1 = bias_variable([outputdepth])
        h_deconv_layer1 = tf.nn.relu(deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1)

        #W_conv_layer1 = weight_variable([filtersize, filtersize, outputdepth, outputdepth])
        #b_conv_layer1 = bias_variable([outputdepth])
        #h_conv_layer1 = tf.nn.relu(conv2d(h_deconv_layer1, W_conv_layer1) + b_conv_layer1)
        return h_deconv_layer1

def final_deconv_layer(input, filtersize, inputdepth, outputdepth):
    height = 2 * int(input.get_shape()[1])
    width = 2 * int(input.get_shape()[2])

    W_deconv_layer1 = weight_variable([filtersize, filtersize, outputdepth, inputdepth])
    b_deconv_layer1 = bias_variable([outputdepth])
    # h_deconv_layer1 = 255*tf.nn.tanh(deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1);
    h_deconv_layer1 = (deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1);
    # h_deconv_layer1 = (255/3)*(tf.nn.relu6(deconv2d(input, height, width, outputdepth, W_deconv_layer1) + b_deconv_layer1)-3);

    #W_conv_layer1 = weight_variable([filtersize, filtersize, outputdepth, outputdepth])
    #b_conv_layer1 = bias_variable([outputdepth])
    #h_conv_layer1 = tf.nn.relu(conv2d(h_deconv_layer1, W_conv_layer1) + b_conv_layer1)
    return h_deconv_layer1


def frame_interpolator(image_shape):
    print("I was called!")
    x = tf.placeholder(tf.float32, [image_shape[0], image_shape[1], image_shape[2], 2*image_shape[3]], name='x') # input is two images
    y = tf.placeholder(tf.float32, image_shape, name='y')

    layer_depths = [20, 40, 80, 160, 320]
    filter_sizes = [3, 3, 3, 3, 3]
    #layer_depths = [20]
    #filter_sizes = [3]
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
    for i, outputdepth in enumerate(layer_depths[1:]): # reverse process exactly until last step
        result = deconv_layer(current_input, filter_sizes[i], current_inputdepth, outputdepth)
        stack = tf.concat([result, conv_outputs[i+1]],3)
        current_input = stack
        current_inputdepth = 2*outputdepth

    yhat = final_deconv_layer(current_input, filter_sizes[-1], current_inputdepth, image_shape[3])

    # define the loss
    epsilon = 0.1
    loss = tf.sqrt( tf.nn.l2_loss(y - yhat) + (epsilon ** 2))
    noise_penalty = 10
    # loss = tf.add(tf.reduce_mean(tf.square(y-yhat)),
    #     noise_penalty*tf.reduce_mean(tf.mul( tf.square(yhat), tf.exp(-tf.square(yhat))  )))
    # loss = msssim.MultiScaleSSIM(np.array(y),np.array(yhat))

    return {'x':x, 'y':y, 'z':z, 'yhat':yhat, 'loss':loss}


def test_frame_interpolator():
    import data_loader

    downsample_factor = 2
    dataset = data_loader.read_data_set(downsample_factor)
    mean_img = 0*np.mean(dataset.train.labels, axis=0)
    img_width = mean_img.shape[1];

    fi = frame_interpolator([None,img_width,img_width,3])

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fi['loss'])

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # log learning performance
    train_set_size = [0]
    train_perf = [0]
    test_perf = [0]


    # Fit all the training data
    batch_size = 10
    n_epochs = 10
    n_examples = dataset.test.num_examples
    for epoch_i in range(n_epochs):
        for batch_i in range(dataset.train.num_examples // batch_size):
            train_set_size.append(train_set_size[-1]+batch_size);
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            train_xs = np.array([img - np.tile(mean_img,[1,1,2]) for img in batch_xs])
            train_ys = np.array([img - mean_img for img in batch_ys])
            # print(batch_xs.shape, batch_ys.shape)
            # fig, axs = plt.subplots(3, 1, figsize=(12, 8))
            # axs[0].imshow(batch_xs[0,:,:,0:3]/255)
            # axs[1].imshow(batch_xs[0,:,:,3:]/255)
            # axs[2].imshow(batch_ys[0,:,:,:]/255)
            # plt.show()
            sess.run(optimizer, feed_dict={fi['x']: train_xs, fi['y']: train_ys})

            # train_perf.append(sess.run(fi['loss'], feed_dict={fi['x']: train_xs, fi['y']: train_ys}))

            # test_xs, test_ys = dataset.test.next_batch(n_examples)
            # test_xs_norm = np.array([img - np.tile(mean_img,[1,1,2]) for img in test_xs])
            # test_ys_norm = np.array([img - mean_img for img in test_ys])
            # test_perf.append(sess.run(fi['loss'], feed_dict={fi['x']: test_xs_norm, fi['y']: test_ys_norm}))


        print(epoch_i, sess.run(fi['loss'], feed_dict={fi['x']: train_xs, fi['y']: train_ys}))

    # %%
    # Plot example reconstructions
    n_examples = 4
    test_xs, test_ys = dataset.test.next_batch(n_examples)
    test_xs_norm = np.array([img - np.tile(mean_img,[1,1,2]) for img in test_xs])
    test_ys_norm = np.array([img - mean_img for img in test_ys])
    recon = sess.run(fi['yhat'], feed_dict={fi['x']: test_xs_norm})

    fig, axs = plt.subplots(3, n_examples, figsize=(12, 8))
    for example_i in range(n_examples):
        axs[0][example_i].imshow((np.reshape(0.5*test_xs[example_i,:,:,0:3] + 0.5*test_xs[example_i,:,:,3:6], (img_width,img_width,3)))/255)
        axs[1][example_i].imshow((np.reshape(recon[example_i, ...] + mean_img, (img_width, img_width, 3)))/255)
        axs[2][example_i].imshow((np.reshape(test_ys[example_i,:,:,:], (img_width, img_width, 3)))/255)

    fig.savefig('yomama.pdf')

    # plt.subplot(111)
    # plt.plot(train_set_size[1:], train_perf[1:], 'r', train_set_size[1:], test_perf[1:], 'b')
    # plt.title('Learning Curve')
    # plt.ylabel('Loss')
    # plt.xlabel('Training Set Size')
    # plt.savefig('learning_curve.pdf')



if __name__ == '__main__':
    test_frame_interpolator()
