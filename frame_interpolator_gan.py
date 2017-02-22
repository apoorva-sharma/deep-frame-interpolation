from utils import *


class DCGAN(object):
    def __init__(self, sess, input_height=192, input_width=192, is_crop=True,
                    batch_size=64, sample_num=64, y_dim=None, z_dim=100, gf_dim=64,
                    df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3,
                    dataset_name="default", input_fname_pattern="*.jpg", # don't think we use these
                    checkpoint_dir=None, sample_dir=None):
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name = 'g_bn0')
        self.g_bn1 = batch_norm(name = 'g_bn1')
        self.g_bn2 = batch_norm(name = 'g_bn2')
        self.g_bn3 = batch_norm(name = 'g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        #self.build_model()

    # Generator takes two images and generates a middle frame
    def generator(self, input_frames):
        layer_depths = [20, 40, 80, 160, 320]
        filter_sizes = [3, 3, 3, 3, 3]
        #layer_depths = [20]
        #filter_sizes = [3]
        conv_outputs = []

        current_input = input_frames
        current_inputdepth = 6 # TODO FIX THIS 2*image_shape[3]
        # convolutional portion
        for i, outputdepth in enumerate(layer_depths):
            layer_name = "g_" + "conv_" + str(i)
            result = tf.nn.relu( conv_layer(current_input, filter_sizes[i],
                                 current_inputdepth, outputdepth, name=layer_name) )

            conv_outputs.append(result)
            current_input = result
            current_inputdepth = outputdepth

        z = current_input

        layer_depths.reverse()
        filter_sizes.reverse()
        conv_outputs.reverse()

        # deconv portion
        for i, outputdepth in enumerate(layer_depths[1:]): # reverse process exactly until last step
            layer_name = "g_" + "deconv_" + str(i)
            result = tf.nn.relu( deconv_layer(current_input, filter_sizes[i],
                                current_inputdepth, outputdepth, name=layer_name) )

            stack = tf.concat([result, conv_outputs[i+1]],3)
            current_input = stack
            current_inputdepth = 2*outputdepth

        yhat = 255*tf.nn.tanh( deconv_layer(current_input, filter_sizes[-1],
                               current_inputdepth, 3, name="g_final") )

        return yhat

    # Discriminator takes two images and a middle frame, and scores the "realness"
    # of the middle frame.
    def discriminator(self, input_frames, middle_frame, reuse=False):
        with tf.variable_scope("d_main", reuse=reuse):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()

            image_depth = 3 #middle_frame.shape[3].value
            n_pixels = self.input_height*self.input_width # middle_frame.shape[1].value * middle_frame.shape[2].value

            h_depth = 32 # number of conv filters for first hidden layer

            stack = tf.concat([input_frames,middle_frame],3)

            h0 = lrelu(conv_layer(stack,3,3*image_depth,h_depth, name='d_h0_conv'))
            h1 = lrelu(conv_layer(h0,3,h_depth,2*h_depth, name='d_h1_conv'))
            h2 = lrelu(conv_layer(h1,3,2*h_depth,4*h_depth, name='d_h2_conv'))
            h3 = lrelu(conv_layer(h2,3,4*h_depth,4*h_depth, name='d_h3_conv'))

            h3_depth = int(4*h_depth*n_pixels/4/4/4) # TODO check this math
            h3 = linear(tf.reshape(h3,[-1, h3_depth]), 1, name='d_h3_lin')

        return tf.nn.sigmoid(h3), h3

    def frame_interpolator(self, image_shape):
        x = tf.placeholder(tf.float32, [image_shape[0], image_shape[1], image_shape[2], 2*image_shape[3]], name='x') # input is two images
        y = tf.placeholder(tf.float32, image_shape, name='y')

        yhat = self.generator(x)

        d, d_logits = self.discriminator(x,y, reuse=False) # discriminator on truth images
        d_, d_logits_ = self.discriminator(x,yhat, reuse=True) # discriminator on generated images

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                        labels=tf.ones_like(d)))

        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                        labels=tf.zeros_like(d_)))

        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                        labels=tf.ones_like(d_)))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # LEARNING CONFIGURATIONS
        learning_rate = 0.002
        beta1 = 0.5

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                    .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                    .minimize(g_loss, var_list=g_vars)

        return {'x':x, 'y':y, 'yhat':yhat, 'd_loss':d_loss, 'g_loss':g_loss,
                'd_vars':d_vars, 'g_vars':g_vars, 'd_optim':d_optim, 'g_optim':g_optim}
