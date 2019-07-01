import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.fashion_mnist

def load_dataset():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

	# convert integer into float
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	# nomalization data
	x_train = x_train/255.0
	x_test= x_test/255.0

	# one hot coding 
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset() 
num_outputs = 10
num_inputs = 784
num_layes = 1
num_neurons = [256,10]

def mlp(x, num_inputs, num_outputs, num_layes, num_neurons):
	w = []
	b = []
	for i in range(num_layes):
		if i==0:
			w.append(tf.Variable(tf.ones([num_inputs, num_neurons[i]])))
		else:
			w.append(tf.Variable(tf.ones([num_neurons[i-1], num_neurons[i]])))
		b.append(tf.ones([num_neurons[i]]))
	w.append(tf.Variable(tf.ones([num_neurons[num_layes-1], num_neurons[num_layes]])))
	b.append(tf.Variable(tf.ones([num_neurons[num_layes]])))
	layer = x
	for i in range(num_layes):
		layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
		# layer = tf.matmul(layer, w[i]) + b[i]
	# model
	layer = (tf.matmul(layer, w[num_layes]) + b[num_layes])
	return layer

def excute_model():
	learning_rate = 0.01
	x = tf.placeholder(tf.float32, shape = [None, num_inputs])
	y = tf.placeholder(tf.float32, shape = [None, num_outputs])

	w = tf.Variable(tf.ones([784, 10]))
	b = tf.Variable(tf.ones([10]))	

	model = mlp(x, num_inputs, num_outputs, num_layes, num_neurons)
	# costfunction or lossfunction
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))

	# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

	prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy_function = tf.reduce_mean(tf.cast(prediction, tf.float32))
	n_epochs = 30
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		x_batch = np.ones([100, 784])
		y_batch = np.ones([100, 10])
		res = 0.0
		for i in range(n_epochs):
			for j in range(600):
				if j==600-1:
					x_batch = x_train[j*100:, :]
					y_batch = y_train[j*100:, :]
					
				else :
					x_batch = x_train[j*100:j*100+100, :] 
					y_batch = y_train[j*100:j*100+100, :] 
				sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
			result = sess.run(loss, feed_dict = {x:x_train, y:y_train})
			print (result)

		# sess.run(model, feed_dict = (x:))
		# accuracy = sess.run(loss, feed_dict = {x:x_train, y:y_train})
		accuracy = sess.run(accuracy_function, feed_dict = {x:x_test, y:y_test})
		print ("accuracy: ", accuracy)
excute_model()


