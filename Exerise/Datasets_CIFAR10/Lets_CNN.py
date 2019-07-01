import tensorflow as tf 
import numpy as np 
import os
import cv2

def load_dataset(num_outputs):
	cifar10 = tf.keras.datasets.cifar10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# convert tpye data Integer into FLoat
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# normalization
	x_train = x_train/255.0
	x_test = x_test/255.0

	# reshape
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

	# one hot coding
	y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
	y_test = tf.keras.utils.to_categorical(y_test, num_outputs)
	return x_train, y_train, x_test, y_test

# define hyper parameter
num_outputs = 10
n_width = 32
n_height = 32
n_depth = 3
num_inputs = 32*32*3
learning_rate = 0.01
n_epochs = 10

x_train, y_train, x_test, y_test = load_dataset(num_outputs)

def buil_network(x):
	# first layer CNN
	layer1_w = tf.Variable(tf.random_normal(shape = [4, 4, n_depth, 32], 
											stddev = 0.1, name = "L1_W"))
	layer1_b = tf.Variable(tf.random_normal(shape = [32], name = "L1_B"))
	# Convulation with activation = "relu"
	# => output of layer1_conv2d: 64x32x32x3
	layer1_conv2d = tf.nn.relu(tf.nn.conv2d(input = x, filter = layer1_w, 
											strides = [1,1,1,1], padding = "SAME")+layer1_b)

	# Max pool
	# => output of layer1_pool: 64x16x16x3
	layer1_pool = tf.nn.max_pool(value = layer1_conv2d, ksize = [1,2,2,1], 
								strides = [1,2,2,1], padding = "SAME")

	# hidden layer 2
	layer2_w = tf.Variable(tf.random_normal(shape = [4, 4, 32, 64], 
											stddev = 0.1, name = "L2_W"))
	layer2_b = tf.Variable(tf.random_normal(shape = [64], name = "L2_B"))
	
	# convulation layer 2 with activation = "relu"
	# => output of convulation2: 128x16x16x3
	layer2_conv2d = tf.nn.relu(tf.nn.conv2d(input = layer1_pool, filter = layer2_w, 
											strides = [1,1,1,1], padding = "SAME") + layer2_b)
	# Max pool
	# => output of max_pool: 128x8x8x3
	layer2_pool = tf.nn.max_pool(value = layer2_conv2d, ksize = [1,2,2,1],
								strides = [1,2,2,1], padding = "SAME")

	# fully connected
	layer3_w = tf.Variable(tf.random_normal(shape = [64*8*8, 1024], 
											stddev = 0.1, name = "L3_W"))
	layer3_b = tf.Variable(tf.random_normal(shape = [1024], name = "L3_B"))

	layer3_fc = tf.nn.relu(tf.matmul(tf.reshape(layer2_pool, [-1, 64*8*8]), layer3_w)+layer3_b)

	# # output prediction
	layer4_w = tf.Variable(tf.random_normal(shape = [1024, num_outputs], 
											stddev = 0.1, name = "L4_W"))
	layer4_b = tf.Variable(tf.random_normal(shape = [num_outputs], name = "L4_B"))

	layer4_out = tf.matmul(layer3_fc, layer4_w) + layer4_b

	# init = tf.global_variables_initializer()

	# with tf.Session() as sess:
	# 	sess.run(init)
	# 	x_batch = x_train[:100, :]
	# 	k = sess.run(layer4_out, feed_dict = {x:x_batch})
	# 	print (k.shape)

	return layer4_out


def build_model():
	x = tf.placeholder(dtype = tf.float32, shape = [None, num_inputs])
	y = tf.placeholder(dtype = tf.float32, shape = [None, num_outputs])

	x_ = tf.reshape(x, [-1,n_width, n_height, n_depth])

	model = buil_network(x_)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = y))

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		batch_size = 100
		result_loss = 0.0
		num = int (x_train.shape[0]/batch_size)
		for i in range(n_epochs):
			for j in range(num):
				# print (j)
				if j == num-1:
					x_batch = x_train[j*batch_size:, :]
					y_batch = y_train[j*batch_size:, :]
					sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
					result = sess.run(loss, feed_dict = {x:x_batch, y:y_batch})
					result_loss += 1.0/num*result
				else :
					x_batch = x_train[j*batch_size:(j+1)*batch_size, :]
					y_batch = y_train[j*batch_size:(j+1)*batch_size, :]
					sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})	
					result = sess.run(loss, feed_dict = {x:x_batch, y:y_batch})
					result_loss += 1.0/num*result
			print ("loss: ", result_loss)
			acc = sess.run(accuracy, feed_dict = {x:x_test, y:y_test})
			print ("accuracy: ", acc)

build_model()

