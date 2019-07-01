import tensorflow as tf
import numpy as np
import cv2
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = tf.keras.utils.to_categorical(y_train, 0)
y_test = tf.keras.utils.to_categorical(y_test, 0)

def build_network(x):
	layers = tf.layers.conv2d(inputs = x, filters = 32, kernel_size = (4,4), 
							strides = (1,1), padding = "same", activation = 'relu')
	# layers = tf.layers.conv2d(inputs = layers, filters = 32, kernel_size = (3,3), 
	# 						strides = (1,1), padding = "same", activation = 'relu')
	layers = tf.layers.max_pooling2d(inputs = layers, pool_size = (2,2), 
									strides = (2,2))
	# layers = tf.layers.conv2d(inputs = layers, filters = 64, kernel_size = (3,3), 
	# 						strides = (1,1), padding = "same", activation = 'relu')
	layers = tf.layers.conv2d(inputs = layers, filters = 64, kernel_size = (3,3), 
							strides = (1,1), padding = "same", activation = 'relu')
	layers = tf.layers.max_pooling2d(inputs = layers, pool_size = (2,2), 
									strides = (2,2))
	layers = tf.layers.flatten(inputs = layers)

	layers = tf.layers.dense(inputs = layers, units = 128, activation = 'relu')

	model = tf.layers.dense(inputs = layers, units = 10)
	modle = tf.nn.softmax(model, name = "output")
	return model

def build_model():
	x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
	y = tf.placeholder(shape = [None, 10], dtype = tf.int32)

	model = build_network(x)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model,
																labels = y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

	prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	if not os.path.isdir("save_weight"):
		os.mkdir("save_weight")

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch_size = 100
		for epochs in range(1):
			total_loss = 0.0
			for i in range(int (x_train.shape[0]/batch_size)-1):
				x_batch = x_train[i*100:(i+1)*100]
				y_batch = y_train[i*100:(i+1)*100]
				if i == 500:
					break
				r_loss, _ = sess.run([loss, optimizer], feed_dict = {
					x : x_batch,
					y : y_batch
					})
				total_loss += r_loss/(x_train.shape[0]/batch_size)
				# print (total_loss)
			print (total_loss)
		saver.save(sess, "save_weight/model.ckpt")
		result = sess.run(accuracy, feed_dict = {
			x:x_test,
			y:y_test
			})
		print (result)
build_model()

def restore_model():

	# x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = "input")
	# y = tf.placeholder(shape = [None, 10], dtype = tf.int32, name = "in2")

	x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = "input_x")

	build_network(x)

	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model,
	# 															labels = y))
	# optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

	# prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
	# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	# if not os.path.isdir("save_weight"):
	# 	os.mkdir("save_weight")
	# if not os.path.isdir("output_graph"):
	# 	os.mkdir("output_graph")

	saver = tf.train.Saver()

	with tf.Session() as sess:
		# sess.run(tf.global_variables_initializer())
		saver.restore(sess, "save_weight/model.ckpt")
		frozen_graph_def = graph_util.convert_variables_to_constants(
      					sess, sess.graph_def, ['output'])
		tf.train.write_graph(sess,sess.graph_def, 'output_graph', 'train.pbtxt', False)
		# print (prediction)
		# print (prediction.name)
		# print (x.name)
		# result = sess.run(accuracy, feed_dict = {
		# 	'input:0':x_test[:1000],
		# 	y:y_test[:1000]
		# 	})
		# print (result)
# restore_model()

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# build_model()

def predict():
	load_graph("output_graph/train.pbtxt")
	softmax_tensor = tf.get_default_graph().get_tensor_by_name("output:0")
	print (softmax_tensor)
	sess = tf.Session()
	result = sess.run(softmax_tensor, feed_dict = {
		'input:0':x_test[:100],
		'in2:0':y_test[:100]
		})
	print (result)


# predict()