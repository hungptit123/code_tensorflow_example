import tensorflow as tf 
import numpy as np 
from sklearn import svm
import pickle
import cv2
# from tensorflow.keras.models 

# hyper - parammeter
n_widths = 28
n_heights = 28
n_depths = 1

# def normalize(data, datatest):
# 	y = datatest
# 	x = data
# 	max_x = np.max(x, axis = 0)
# 	min_x = np.min(x, axis = 0)
# 	avagen = np.sum(x, axis = 0) * 1.0 / x.shape[0]
# 	for i in  range(784):
# 		r = max_x[i] - min_x[i]
# 		if r==0:
# 			r = 1
# 		x[:, i:i+1] = (x[: , i:i+1] - avagen[i])/r
# 		y[:, i:i+1] = (y[: , i:i+1] - avagen[i])/r
# 	return x, y

def load_data():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], n_widths*n_heights)
	x_test = x_test.reshape(x_test.shape[0], n_widths*n_heights)

	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	x_train = x_train/255.0
	x_test = x_test/255.0

	# x_train, x_test = normalize(x_train, x_test)

	# x_train = (x_train/255.0 - 0.5) * 2
	# x_test = (x_test/255.0 - 0.5) * 2

	return x_train, y_train, x_test, y_test

def display(image1, image2):
	while True:
		cv2.imshow("image 1", image1)
		cv2.imshow("image 2", image2)
		if cv2.waitKey(1) == 27:
			break

def build_network(x_):
	layer1 = tf.nn.max_pool(value = x_,
							ksize = [1,2,2,1],
							strides = [1,2,2,1],
							padding = "SAME")
	return layer1

def load():
	x_train, y_train, x_test, y_test = load_data()
	x1 = [] 
	y1 = []
	count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for i in range(x_train.shape[0]):
		count[y_train[i]] += 1
		if count[y_train[i]] <= 2000:
			x1.append(x_train[i])
			y1.append(y_train[i])
	x1 = np.asarray(x1)
	y1 = np.asarray(y1)
	print (x1.shape)
	print (y1.shape)
	return x1, y1, x_test, y_test
x_train, y_train, x_test, y_test = load()

def build_model_svm(x_train, y_train, x_test, y_test):
	# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
	# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
	c = 1000.0
	model = svm.SVC(C = c, kernel = 'poly', gamma = 0.01)
	model.fit(x_train, y_train)
	# pickle.dump(model, open("save_model/model_c100", "wb"))

	print ("da save model")

	# load model
	# model = pickle.load(open("save_model/model_c100", "rb"))
	result = model.score(x_test, y_test)
	print (result)
build_model_svm(x_train, y_train, x_test, y_test)
# x = tf.placeholder(tf.float32)
# x_ = tf.reshape(x, [-1, n_widths, n_heights, n_depths])

# model = build_network(x_)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)
# 	layer1 = sess.run(model, feed_dict = {x:x_train})
# 	layer2 = sess.run(model, feed_dict = {x:x_test})
# 	build_model_svm(x_train, y_train, x_test, y_test)
