import tensorflow as tf
import numpy as np
import math
import glob
#import msssim
from scipy import misc
import matplotlib.animation as animation

from frame_interpolator import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylab import *


def normalize_frames(frames, medians):
    return frames - medians

def unnormalize_frames(frames, medians):
    return frames + medians

def compute_medians(frames, window_size):
    num_frames = frames.shape[0]
    frame_shape = frames[1,:,:,:].shape
    frame_size = frames[1,:,:,:].size

    # medians = []
    # for i in range(num_frames):
    #     window_inds = range(max(i-window_size//2,0),
    #         min(i+window_size//2 + 1,num_frames))
    #     oned_frames = np.reshape(frames[window_inds,:,:,:],
    #      [-1, frame_size])
    #     median_frame = np.median(oned_frames, axis=0)
    #     median_frame = np.reshape(median_frame, frame_shape)
    #     medians.append(median_frame)

    oned_frames = np.reshape(frames,
         [-1, frame_size])
    median_frame = np.median(oned_frames, axis=0)
    median_frame = np.reshape(median_frame, frame_shape)
    medians = np.tile(median_frame, [num_frames, 1,1,1])

    return np.array(medians)


def compile_input_data(before_frames, after_frames):
    frame_shape = before_frames[0,:,:,:].shape
    frame_size = before_frames[0,:,:,:].size

    medians = compute_medians(before_frames,20)
    before_norm = normalize_frames(before_frames, medians)
    after_norm = normalize_frames(after_frames, medians)

    training_inputs = np.concatenate((before_norm, after_norm), axis=3)

    return (training_inputs, medians)

def load_video(input_video_dir):
    image_paths = glob.glob(input_video_dir + "/*.png")
    image_paths.sort()

    frames = []

    downsample_factor = 1
    # load data into train_inputs/targets
    for i in range(0,len(image_paths)):
        frame = np.array(misc.imread(image_paths[i]))
        frame = frame[::downsample_factor,::downsample_factor,:]

        frames.append(frame)

    frames = np.array(frames)

    return frames

def create_datasets(frames):
    downsampled = frames[::2,:,:,:]

    test_targets = frames[1:-1:2,:,:,:]
    test_before = downsampled[:-1,:,:,:]
    test_after = downsampled[1:,:,:,:]

    training_targets = downsampled[1:-1,:,:,:]
    training_before = downsampled[:-2,:,:,:]
    training_after = downsampled[2:,:,:,:]

    training_inputs,medians = compile_input_data(training_before, training_after)
    training_targets = normalize_frames(training_targets, medians)

    test_inputs,medians = compile_input_data(test_before, test_after)
    test_targets = normalize_frames(test_targets, medians)

    return {"downsampled": downsampled, "training_inputs": training_inputs,
            "training_targets": training_targets, "test_inputs": test_inputs,
            "test_targets": test_targets, "medians":medians}

def train_network(fi, optimizer, training_inputs, training_targets, n_epochs, sess):
    # Fit all the training data
    n_examples = training_inputs.shape[0]
    print(n_examples)
    batch_size = 20
    for epoch_i in range(n_epochs):
        shuffled_inds = np.random.permutation(n_examples)
        for batch_i in range(n_examples // batch_size):
            batch_inds = range(batch_i*batch_size, (batch_i+1)*batch_size)
            batch_inds = shuffled_inds[batch_inds]
            batch_xs = training_inputs[batch_inds,:,:,:]
            batch_ys = training_targets[batch_inds,:,:,:]

            sess.run(optimizer, feed_dict={fi['x']: batch_xs, fi['y']: batch_ys})

        print(epoch_i, sess.run(fi['loss'], feed_dict={fi['x']: training_inputs, fi['y']: training_targets}))

    return

def upsample(saved_frames, trained_net, sess):
    # compute median and missing frames
    (network_inputs, medians) = compile_input_data(saved_frames[:-1,:,:,:],saved_frames[1:,:,:,:])
    network_outputs = sess.run(trained_net['yhat'],
        feed_dict={trained_net['x']: network_inputs})

    output_frames = unnormalize_frames(network_outputs, medians)

    # interleave saved frames with generated frames
    full_recon_vid_shape = list(saved_frames.shape)
    full_recon_vid_shape[0] = full_recon_vid_shape[0]*2 - 1
    full_recon_vid = zeros(full_recon_vid_shape)

    full_recon_vid[::2,:,:,:] = saved_frames
    full_recon_vid[1:-1:2,:,:,:] = output_frames

    return full_recon_vid


def save_vid(vid_frames, filename):
    dpi = 100
    img_width = vid_frames.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(img_width,img_width,3))
    im.set_clim([0,1])
    fig.set_size_inches([img_width/dpi,img_width/dpi])

    tight_layout()

    def update_img(n):
        tmp = np.minimum(vid_frames[n,:,:,:]/255,np.ones(vid_frames[n,:,:,:].shape))
        tmp = np.maximum(tmp, np.zeros(tmp.shape))
        im.set_data(tmp)
        return im

    ani = animation.FuncAnimation(fig, update_img, vid_frames.shape[0], interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save(filename,writer=writer,dpi=dpi)
    return ani

def main(retrain=True):
    print('Loading frame data...')
    frames = load_video('./stefan')
    datasets = create_datasets(frames)

    sess = tf.Session()
    img_width = frames[0,:,:,:].shape[1]
    fi = frame_interpolator([None,img_width,img_width,3])

    learning_rate = 0.01
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(fi['loss'])

    saver = tf.train.Saver()

    if retrain:
        print('Training network...')
        sess.run(tf.global_variables_initializer())
        train_network(fi, optimizer, datasets['training_inputs'],
              datasets['training_targets'], 200, sess)
        save_path = saver.save(sess, "saved_net.ckpt")
        print("Model saved in file: %s" % save_path)
    else:
        print('Loading network from file...')
        saver.restore(sess, "saved_net.ckpt")
        train_network(fi, optimizer, datasets['training_inputs'],
              datasets['training_targets'], 50, sess)
        save_path = saver.save(sess, "saved_net.ckpt")
        print("Model saved in file: %s" % save_path)

    # evaluate performance on test set
    test_loss = sess.run(fi['loss'], feed_dict={fi['x']: datasets['test_inputs'],
                                                fi['y']: datasets['test_targets']})

    print('Test Loss:', test_loss)

    print('Saving video...')
    upsampled_frames = upsample(datasets['downsampled'], fi, sess)
    save_vid(upsampled_frames, "upsampled.mp4")
    compression_frames = upsample(datasets['downsampled'][::2,:,:,:], fi, sess)
    save_vid(compression_frames, "compression.mp4")

    print('Saving datasets as .mat...')
    # save train and test outputs to matlab files
    import scipy.io
    test_outputs = sess.run(fi['yhat'], feed_dict={fi['x']: datasets['test_inputs'],
                                                   fi['y']: datasets['test_targets']})
    train_outputs = sess.run(fi['yhat'], feed_dict={fi['x']: datasets['training_inputs'],
                                                    fi['y']: datasets['training_targets']})
    datasets['train_outputs'] = train_outputs
    datasets['test_outputs'] = test_outputs
    datasets['all_frames'] = frames
    scipy.io.savemat('all_data.mat', mdict=datasets)





if __name__ == '__main__':
    main(True)
