import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Conv2D, Reshape, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD

# define hyper - parammeter
n_widths = 28
n_heights = 28
n_depths = 1
n_inputs = n_widths*n_heights*n_depths
n_outputs = 10
n_neurons = [64, 128]

def load_dataset():
	mnist = tf.keras.datasets.fashion_mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	x_train = (x_train/255.0 - 0.5) * 2
	x_test = (x_test/255 - 0.5) * 2

	if len(x_train.shape) == 3:
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
	else:
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_train.shape[3])

	y_train = tf.keras.utils.to_categorical(y_train, n_outputs)
	y_test = tf.keras.utils.to_categorical(y_test, n_outputs)
	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()
print (x_test.shape)


def build_network():
	inputs = Input(shape = (n_inputs, ), name = "inputs")

	layer = Reshape(target_shape = (n_widths, n_widths, n_depths), input_shape = (n_inputs, ))(inputs)

	# convulation layer1
	layer = Conv2D(filters = n_neurons[0], kernel_size = (2,2), strides = (1,1)
					, padding = "SAME", activation = 'relu')(layer)

	layer = MaxPool2D(pool_size = (2,2), strides = (2,2))(layer)

	layer = Dropout(0.02)(layer)

	# convulation layer2
	layer = Conv2D(filters = n_neurons[1], kernel_size = (2,2), strides = (1,1)
					, padding = "SAME", activation = 'relu')(layer)

	layer = MaxPool2D(pool_size = (2,2), strides = (2,2))(layer)

	layer = Dropout(0.02)(layer)

	layer = Flatten()(layer)
	# # fully connected - Dense
	layer = Dense(units = 1024, activation = 'relu')(layer)
	layer = Dropout(0.02)(layer)	
	outputs = Dense(units = n_outputs, activation = 'softmax')(layer)
	model = Model(inputs = inputs, outputs = outputs)

	model.summary()

	model.compile(loss = 'categorical_crossentropy', 
					optimizer = Adam(0.001), 
					metrics = ['accuracy'])

	model.fit(x = x_train, y = y_train, batch_size = 100, epochs = 2)

	result = model.evaluate(x_test, y_test)
	print ("accuracy: ", result)

build_network()