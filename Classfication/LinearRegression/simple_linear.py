import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets as skds

def visualization(X, y):
	plt.figure(figsize = (14,8))
	plt.plot(X,y, 'g.')
	plt.title("Original Dataset")
	plt.show()

# prepare data set
def prepare_data_set(m, n , m_train):
	X, y = skds.make_regression(n_samples = m,
								n_features = n,
								n_informative = 1,
								n_targets = n,
								noise = 20.0)
	if y.ndim == 1:
		y = np.reshape(y, (len(y), 1))
	X_train = X[:m_train, :]
	y_train = y[:m_train, :]
	X_test = X[m_train:, :]
	y_test = y[m_train:, :]
	return X_train, X_test, y_train, y_test 

def normalization(data, m_train, n):
	x_max = np.max(data, axis = 0)
	x_min = np.min(data, axis = 0)
	x_avange = np.sum(data, axis = 0)*1.0/m_train
	for i in range(n):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		data[:, i:i+1] = (data[:, i:i+1] - x_avange[i])/r

def build_model_linear():
	m = 400
	n = 1
	m_train = 300
	learning_rate = 0.1
	epochs = 10000
	X_train, X_test, y_train, y_test = prepare_data_set(m,n,m_train)
	# normalization(X_train,m_train,n)
	# print (y_test)
	W = tf.Variable(tf.ones([1,n]), dtype = tf.float32)
	b = tf.Variable(tf.ones([m_train,1]), dtype = tf.float32)
	 
	x = tf.placeholder(name = 'x', dtype = tf.float32, shape = [m_train,n])
	y = tf.placeholder(name = 'y', dtype = tf.float32, shape = [m_train, 1])
	# linear regression
	linear = tf.matmul(x, W) + b
	# cost function
	cost = 1.0/(2*m_train)*tf.reduce_sum(tf.square(linear - y))
	# gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(epochs+1):
			sess.run(optimizer, feed_dict = {x:X_train, y:y_train})
			if i%2000 == 0:
				print (sess.run(cost, feed_dict = {x:X_train, y:y_train}))
	# 	# evaluation 
		w_, b_ = sess.run([W,b])
		print (w_.shape)
		print (b_.shape)
		# result = np.matmul(X_test, w_) + b_
		# print (result-y_test)
build_model_linear()