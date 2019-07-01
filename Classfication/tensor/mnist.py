import tensorflow as tf
import numpy as np

n_inputs = (28, 28, 1)
n_outputs = 10

def load_data():
	mnist = tf.keras.datasets.fashion_mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	x_train = x_train/255
	x_test = x_test/255

	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)
	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

def build_network(x):
	n_filters = [16, 16, 32, 32]
	layers = x
	i = 0
	for filters in n_filters: 
		layers = tf.layers.conv2d(inputs = layers, filters = filters, kernel_size = (3,3),
								strides = (1,1), padding = "SAME", activation = 'relu')
		if (i+1) == 2:
			layers = tf.layers.max_pooling2d(inputs = layers, pool_size = (2,2), 
											strides = (2,2), activation = 'relu')
	layers = tf.layers.flatten(inputs = layers)

	layers = tf.layers.dense(inputs = layers, units = 128, activation = 'relu')

	model = tf.layers.dense(inputs = layers, units = n_outputs)

	return model

def build_model():
	x = tf.placeholder(shape = (None, 28, 28, 1), dtype = tf.float32)
	y = tf.placeholder(shape = (None, n_outputs), dtype = tf.int8)

	model = build_network(x)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model,
						labels = y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


	batch_size = 100
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(2):
			total = 0.0
			for j in range(int (60000/batch_size)):
				x_batch = x_train[j*100:(j+1)*100, :, :, :]
				# print (x_batch.shape)
				y_batch = y_train[j*100:(j+1)*100, :]
				# print (y_batch.shape)
				sess.run(optimizer, feed_dict = {x:x_batch, y:y_batch})
				loss_ = sess.run(loss, feed_dict = {x:x_batch, y:y_batch})
				total += loss_
				# print (total)
				# break
			# print ("ket thuc")
			print (total/600)
			# break
		predict = tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
		print ("accuracy: ", accuracy.eval(feed_dict = {x:x_test, y:y_test}))
			# if i%100 == 0:
			# 	print (loss_)
		# writer = tf.summary.FileWriter('tflogs', sess.graph)


build_model()