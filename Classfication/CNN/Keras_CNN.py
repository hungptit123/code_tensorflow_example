import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, SGD


# define parameter
n_width = 28
n_height = 28
n_depths = 1
num_outputs = 10
num_inputs = n_height*n_depths*n_width # size is 784
n_epochs = 10
batch_size = 100
learning_rate = 0.01
n_filter = [16,32]


def load_dataset():
	mnist = tf.keras.datasets.fashion_mnist

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

def build_model():
	model = Sequential()

	# reshape input have 784 dimention into: 28x28x1
	model.add(Reshape(target_shape = (n_width, n_height, n_depths), 
						input_shape = (num_inputs, )))

	# convulation layer 1
	# using kernel_size 4x4 and strides = 1
	model.add(Conv2D(filters = n_filter[0], kernel_size = (3,3), strides = (1,1), 
					padding = "SAME", activation = 'relu'))
	model.add(Conv2D(filters = n_filter[0], kernel_size = (3,3), strides = (1,1), 
					padding = "SAME", activation = 'relu'))
	# max pool regoin 2x2 and strides = 2x2
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

	model.add(Conv2D(filters = n_filter[1], kernel_size = (3,3), strides = (1,1), 
					padding = "SAME", activation = 'relu'))
	model.add(Conv2D(filters = n_filter[1], kernel_size = (3,3), strides = (1,1), 
					padding = "SAME", activation = 'relu'))
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

	# flatten convert output MaxPool2D from 64x7x7x1 into 64*7*7*1
	model.add(Flatten())
	# fully connected or Dense
	model.add(Dense(units = 256, activation = 'relu'))
	model.add(Dense(units = num_outputs, activation = 'softmax'))

	model.summary()

	model.compile(loss = "categorical_crossentropy", 
					optimizer = Adam(learning_rate), 
					metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size = 100, epochs = 3)

	result = model.evaluate(x_test, y_test)
	print ("accuracy: ", result)
build_model()



