import tensorflow as tf 
import numpy as np 
import os
import cv2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential

def load_dataset(num_outputs):
	cifar10 = tf.keras.datasets.cifar10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# convert tpye data Integer into FLoat
	# x_train = x_train.astype(np.float32)
	# x_test = x_test.astype(np.float32)

	# normalization
	x_train = (x_train/255.0 - 0.5)*2
	x_test = (x_test/255.0 - 0.5) * 2

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
learning_rate = 0.001
n_epochs = 10
n_filter = [128, 256, 512]

x_train, y_train, x_test, y_test = load_dataset(num_outputs)
# for i in range(len(x_train)):
# 	while (True):
# 		cv2.imshow("frame", x_train[i])
# 		if cv2.waitKey(1) == 27:
# 			break

def build_model():
	model = Sequential()

	model.add(Reshape(target_shape = (n_width, n_height, n_depth), input_shape = (num_inputs, )))

	# using 32 filter and ksize = 2x2 with strides = 1x1
	# => output: 32x32x32x3
	model.add(Conv2D(filters = n_filter[0], kernel_size = (2,2), strides = (1,1), 
					padding = "SAME", activation = 'relu'))

	# Max pool
	# => output: 32x16x16x3 
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

	model.add(Dropout(0.02))

	# using 64 filter and ksize = 2x2 with strides = 1x1
	# => output: 64x16x16x3 
	model.add(Conv2D(filters = n_filter[1], kernel_size = (2,2), strides = (1,1), 
					padding = "SAME", activation = 'relu'))

	# Max pool
	# => output: 64x8x8x3  
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
	model.add(Dropout(0.02))

	# using 128 filter and ksize = 2x2 with strides = 1x1
	# => output: 128x8x8x3 
	model.add(Conv2D(filters = n_filter[2], kernel_size = (2,2), strides = (1,1), 
					padding = "SAME", activation = 'relu'))

	# Max pool
	# => output: 128x4x4x3 
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
	model.add(Dropout(0.02))

	# using flatten to reshape 128x4x4x3 into 128*4*4*3
	# => output: 128*4*4*3
	model.add(Flatten())

	# using fully connected with 1024 neuron 
	# => output: 1024
	model.add(Dense(units = 256, activation = 'relu'))
	model.add(Dropout(0.02))
	model.add(Dense(units = 1024, activation = 'relu'))
	model.add(Dropout(0.02))
	# model.add(Dense(units = 256, activation = 'relu'))
	# model.add(Dropout(0.02))
	# using finally fully connected with 10 neuron 
	# => output: 10
	model.add(Dense(units = 10, activation = 'softmax'))

	model.summary()

	model.compile(loss = "categorical_crossentropy", 
					optimizer = Adam(learning_rate), 
					metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size = 100, epochs = 20)

	result = model.evaluate(x_test, y_test)

	print ("accuracy: ", result)

build_model()