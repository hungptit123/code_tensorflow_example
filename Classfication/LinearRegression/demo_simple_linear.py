import tensorflow as tf 
import csv 
import numpy as np 
import os

dir_data = "/home/hunglv/Documents/tensorflow/data/wisconsin.wdbc.data.csv"

def load_data(dir_data, m ,n ,m_train):
	# if os.path.isfile(dir_data):
		# print (1)
	file = open(dir_data, "r")
	reader = csv.reader(file)
	data = np.zeros((m,n))
	i = 0
	for row in reader:
		for j in range(2,len(row)):
			data[i,j-1] = row[j]
		if row[1] == 'M':
			data[i, 0] = 1
		else :
			data[i, 0] = 0
		# print (data[i, 0])
		# if i >= 500:
		# 	print (row[1])
			# break
		i += 1
			# break
	# np.random.shuffle(data)
	# print (data[m_train:, :1])
	X_train = data[:m_train, 1:]
	X_test = data[m_train:, 1:]
	y_train = data[:m_train, :1]
	y_test = data[m_train:, :1]
	# print (y_test)
	return X_train, y_train, X_test, y_test

def normalization(data, n, m_train):
	x_max = np.max(data, axis = 0)
	x_min = np.min(data, axis = 0)
	x_avange = np.sum(data, axis = 0) * 1.0/m_train 
	for i in range(n-1):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		data[i:i+1, :] = 1.0*(data[i:i+1, :] - x_avange[i])/r
	return data

def linear_regression():
	m = 569
	m_train = 500
	n = 31
	learning_rate = 0.0001
	epochs = 5001
	if os.path.isfile(dir_data) == True:
		X_train, y_train, X_test, y_test = load_data(dir_data, m, n ,m_train)
		X_train = normalization(X_train, n, m_train)
		# print (y_train)
		# print (X_train[0])
		# return 0
		# print (X_train)

		W = tf.Variable(tf.zeros([30,1]))
		# b = tf.Variable(tf.zeros([1]))

		x = tf.placeholder(dtype = tf.float32, shape = [500, 30])
		y = tf.placeholder(dtype = tf.float32)

		model = tf.matmul(x,W)
		
		cost = 1.0/(2*m_train)*tf.reduce_sum(tf.square(model-y))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			for i in range(epochs):
				sess.run(optimizer, feed_dict = {x:X_train, y:y_train})
				if i%1000 == 0:
					print (sess.run(cost, feed_dict = {x:X_train, y:y_train}))
			w_ = sess.run(W)
			result = np.matmul(X_test, w_)
			for i in range(69):
				if result[i] >= 0.5:
					result[i] = 1
				else :
					result[i] = 0
			# print (y_test)
			# print (result)
			# print ("accuracy: ", 100 - 100.0/69*np.sum(np.abs(result-y_test)))
linear_regression()