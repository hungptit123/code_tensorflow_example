import tensorflow as tf 
import numpy as np 
import os
import cv2

def load_dataset(num_outputs):
	cifar10 = tf.keras.datasets.cifar10
	# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

	(x_training, y_train), (x_testing, y_test) = cifar10.load_data()

	x_train = np.zeros((x_training.shape[0], x_training.shape[1], x_training.shape[2]))
	i = 0
	for image in x_training:
		x_train[i] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	i = 0
	x_test = np.zeros((x_testing.shape[0], x_testing.shape[1], x_testing.shape[2]))
	i = 0
	for image in x_testing:
		x_test[i] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# convert tpye data Integer into FLoat
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# normalization
	x_train = x_train/255.0
	x_test = x_test/255.0

	# reshape
	# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
	# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

	# one hot coding
	y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
	y_test = tf.keras.utils.to_categorical(y_test, num_outputs)
	return x_train, y_train, x_test, y_test

# define hyper parameter
num_outputs = 10
n_width = 32
n_height = 32
n_depths = 1
num_inputs = 32*32*1
n_epochs = 10
batch_size = 100
learning_rate = 0.01

x_train, y_train, x_test, y_test = load_dataset(num_outputs)


def buil_network(x):
	# first layer we use 32 kernel with shape of size 4x4 => output is 32x32x32x1
	layer1_w = tf.Variable(tf.random_normal(shape = [4,4,n_depths, 32], stddev = 0.1, name = "L1_W"))
	layer1_b = tf.Variable(tf.random_normal(shape = [32], name = "L1_B"))
	layer1_conv = tf.nn.relu(tf.nn.conv2d(input = x, filter = layer1_w, 
							strides = [1,1,1,1], padding = 'SAME') + layer1_b)
	# using ksize = 2x2x1 region and stride = 2x2x1 
	# thus the region not overlap with each other
	# => ouput is 32*16*16*1 
	layer1_pool = tf.nn.max_pool(value = layer1_conv, 
								ksize = [1,2,2,1], 
								strides = [1,2,2,1], 
								padding = "SAME")


	# define layer 2
	# using 64 kernel
	layer2_w = tf.Variable(tf.random_normal(shape = [4,4,32,64], stddev = 0.1, name = "L2_W"))
	layer2_b = tf.Variable(tf.random_normal(shape = [64], name = "L2_B"))
	layer2_conv = tf.nn.relu(tf.nn.conv2d(input = layer1_pool, filter = layer2_w, 
							strides = [1,1,1,1], padding = 'SAME') + layer2_b)

	# => output layer2_conv: 64x16x16x1
	layer2_pool = tf.nn.max_pool(value = layer2_conv, 
								ksize = [1,2,2,1], 
								strides = [1,2,2,1],
								padding = "SAME")
	# => ouput layer2_pool: 64x8x8x1

	# fully connected layer
	layer3_w = tf.Variable(tf.random_normal(shape = [64*8*8*1, 1024], stddev = 0.1, name = "L3_W"))
	layer3_b = tf.Variable(tf.random_normal(shape = [1024], name = "L3_B"))
	layer3_fc = tf.nn.relu(tf.matmul(tf.reshape(layer2_pool, [-1, 64*8*8*1]), layer3_w) + layer3_b)
	# output layer3_fc: nx1024
	layer4_w = tf.Variable(tf.random_normal(shape = [1024, num_outputs], stddev = 0.1, name = "L4_W"))
	layer4_b = tf.Variable(tf.random_normal(shape = [num_outputs], name = "L4_B"))

	layer4_out = tf.matmul(layer3_fc, layer4_w) + layer4_b
	return layer4_out


def build_model():
	x = tf.placeholder(dtype = tf.float32, name = "x", shape = [None, num_inputs])
	y = tf.placeholder(dtype = tf.float32, name = "y", shape = [None, num_outputs])

	x_ = tf.reshape(x, [-1, n_width, n_height, n_depths])

	model = buil_network(x_)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = y))

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		res = 0.0
		k = int (x_train.shape[0]/batch_size)
		for i in range(n_epochs):
			for j in range(k):
				print (j)
				if j==k-1:
					x_batch = x_train[j*batch_size:, :]
					y_batch = y_train[j*batch_size:, :]
					sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
				else :
					x_batch = x_train[j*batch_size:(j+1)*batch_size, :]
					y_batch = y_train[j*batch_size:(j+1)*batch_size, :]
					sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
					result = sess.run(loss, feed_dict = {x:x_batch, y:y_batch})
					res += result*1.0/k
			print ("i = ", i, " loss = ", res)
			acc = sess.run(accuracy, feed_dict = {x:x_test, y:y_test})
			print ("accuracy: ", acc)

build_model()


