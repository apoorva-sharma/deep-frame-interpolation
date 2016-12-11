import numpy as np
from scipy import misc
import glob
from tensorflow.contrib.learn.python.learn.datasets import base


class DataSet(object):

	def __init__(self, images, labels):
		self._images = images
		self._labels = labels
		self._num_examples = images.shape[0]
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


def read_data_set(downsample_factor=1):
	image_paths = glob.glob("./IMG/*.png")
	image_paths.sort()
	# [print(i) for i in image_paths]

	train_inputs = []
	train_targets = []

	# load data into train_inputs/targets
	for i in range(len(image_paths)-2):
		before_target = 255-np.array(misc.imread(image_paths[i]))
		target = 255-np.array(misc.imread(image_paths[i+1]))
		after_target = 255-np.array(misc.imread(image_paths[i+2]))
		
		if downsample_factor > 1:
			before_target = before_target[::downsample_factor,::downsample_factor,:];
			target = target[::downsample_factor,::downsample_factor,:];
			after_target = after_target[::downsample_factor,::downsample_factor,:];

		x = np.concatenate((before_target,after_target),axis = 2)

		train_inputs.append(x)
		train_targets.append(target)
		
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets)
	print(train_inputs.shape)


	## split into train, test, validation
	dataset_size = len(train_inputs)
	test_size = int(0.15*dataset_size)
	validation_size = test_size

	# shuffle data
	perm = np.arange(dataset_size)
	np.random.shuffle(perm)
	train_inputs = train_inputs[perm]
	train_targets = train_targets[perm]
	

	# split
	validation_inputs = train_inputs[-validation_size:]
	validation_targets = train_targets[-validation_size:]

	test_inputs = train_inputs[-(validation_size+test_size):-validation_size]
	test_targets = train_targets[-(validation_size+test_size):-validation_size]

	train_inputs = train_inputs[:-(validation_size+test_size)]
	train_targets = train_targets[:-(validation_size+test_size)]

	print('Train size:', train_inputs.shape)
	print('Test size:', test_inputs.shape)
	print('Validation size:', validation_inputs.shape)

	# package as tf.Datasets object and return
	train = DataSet(train_inputs, train_targets)
	validation = DataSet(validation_inputs, validation_targets)
	test = DataSet(test_inputs, test_targets)

	return base.Datasets(train=train, validation=validation, test=test)




if __name__ == '__main__':
	read_data_set()





