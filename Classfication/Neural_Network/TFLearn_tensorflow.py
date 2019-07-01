import tensorflow as tf 
import numpy as np 
import os
import tflearn

# define parameter 
num_inputs = 784
num_outputs = 10
num_neuron = [256]
num_layers = len(num_neuron)

def load_dataset():
	mnist = tf.keras.datasets.fashion_mnist
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

def build_neural_network():
	input_layer = tflearn.input_data(shape = [None, num_inputs])
	dense = tflearn.fully_connected(input_layer, num_neuron[0], activation = 'relu')
	for i in range(1, num_layers):
		dense = tflearn.fully_connected(dense, num_neuron[i], activation = 'relu')
	dense = tflearn.fully_connected(dense, num_outputs, activation = 'softmax')
	return dense

def run_moel():
	softmax = build_neural_network()
	optimizer = tflearn.Adam(0.001)
	net = tflearn.regression(softmax, optimizer = optimizer, metric = tflearn.metrics.Accuracy(), 
								loss = 'categorical_crossentropy')
	model = tflearn.DNN(net)
	model.fit(x_train, y_train, n_epoch = 10, batch_size = 100, show_metric = True)
	result = model.evaluate(x_test, y_test)
	print ("lost: ", "accuracy: ", result)
run_moel()