import tensorflow as tf 
import numpy as np 
import cv2

# define parameter
n_weight = 28
n_height = 28
n_depths = 1
num_outputs = 10
num_inputs = n_height*n_depths*n_weight # size is 784
n_epochs = 10
batch_size = 100
learning_rate = 0.01

# config = tf.


def load_dataset():
	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# convert type of train and test into float
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# reshape
	x_train = x_train.reshape(x_train.shape[0], num_inputs)
	x_test = x_test.reshape(x_test.shape[0], num_inputs)

	# normalization
	x_train = x_train/255.0
	x_test = x_test/255.0

	# one hot coding label
	y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
	y_test = tf.keras.utils.to_categorical(y_test, num_outputs)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()

def buil_network(x):
	# first layer we use 32 kernel with shape of size 4x4 => output is 32x28x28x1
	# layer1_w = tf.Variable(tf.random_normal(shape = [2,2,n_depths, 1], stddev = 0.1, name = "L1_W"))
	# layer1_b = tf.Variable(tf.random_normal(shape = [1], name = "L1_B"))
	# layer1_conv = tf.nn.relu(tf.nn.conv2d(input = x, filter = layer1_w, 
	# 						strides = [1,1,1,1], padding = 'SAME') + layer1_b)
	# using ksize = 2x2x1 region and stride = 2x2x1 
	# thus the region not overlap with each other
	# => ouput is 32*14*14*1 
	layer1_pool = tf.nn.max_pool(value = x, 
								ksize = [1,2,2,1], 
								strides = [1,2,2,1], 
								padding = "SAME")


	# define layer 2
	# using 64 kernel
	# layer2_w = tf.Variable(tf.random_normal(shape = [2,2,1,1], stddev = 0.1, name = "L2_W"))
	# layer2_b = tf.Variable(tf.random_normal(shape = [1], name = "L2_B"))
	# layer2_conv = tf.nn.relu(tf.nn.conv2d(input = layer1_pool, filter = layer2_w, 
	# 						strides = [1,1,1,1], padding = 'SAME') + layer2_b)

	# => output layer2_conv: 64x14x14x1
	layer2_pool = tf.nn.max_pool(value = layer1_pool, 
								ksize = [1,2,2,1], 
								strides = [1,2,2,1],
								padding = "SAME")
	return layer2_pool
	# => ouput layer2_pool: 64x7x7x1

	# fully connected layer
	# layer3_w = tf.Variable(tf.random_normal(shape = [64*7*7*1, 1024], stddev = 0.1, name = "L3_W"))
	# layer3_b = tf.Variable(tf.random_normal(shape = [1024], name = "L3_B"))
	# layer3_fc = tf.nn.relu(tf.matmul(tf.reshape(layer2_pool, [-1, 64*7*7*1]), layer3_w) + layer3_b)
	# # output layer3_fc: nx1024
	# layer4_w = tf.Variable(tf.random_normal(shape = [1024, num_outputs], stddev = 0.1, name = "L4_W"))
	# layer4_b = tf.Variable(tf.random_normal(shape = [num_outputs], name = "L4_B"))

	# layer4_out = tf.matmul(layer3_fc, layer4_w) + layer4_b
	# return layer4_out


def display(image1, image2):
	while True:
		cv2.imshow("image source", image1)
		cv2.imshow("image fix", image2)
		if cv2.waitKey(0) == ord('d'):
			break


def build_model():
	x = tf.placeholder(dtype = tf.float32, name = "x", shape = [None, num_inputs])
	y = tf.placeholder(dtype = tf.float32, name = "y", shape = [None, num_outputs])

	x_ = tf.reshape(x, [-1, n_weight, n_height, n_depths])

	model = buil_network(x_)

	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = y))

	# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		k = sess.run(x_, feed_dict = {x:x_train[:1000, :]})
		result = sess.run(model, feed_dict = {x:x_train[:1000, :]})
		print (result.shape)
		for i in range(10):
			display(k[i], result[i])
		# res = 0.0
		# for i in range(n_epochs):
		# 	for j in range(600):
		# 		# print (j)
		# 		if j==600-1:
		# 			x_batch = x_train[j*batch_size:, :]
		# 			y_batch = y_train[j*batch_size:, :]
		# 			sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
		# 		else :
		# 			x_batch = x_train[j*batch_size:(j+1)*batch_size, :]
		# 			y_batch = y_train[j*batch_size:(j+1)*batch_size, :]
		# 			sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
		# 			result = sess.run(loss, feed_dict = {x:x_batch, y:y_batch})
		# 			res += result*1.0/600
		# 	print ("i = ", i, " loss = ", res)
		# 	acc = sess.run(accuracy, feed_dict = {x:x_test, y:y_test})
		# 	print ("accuracy: ", acc)

build_model()


