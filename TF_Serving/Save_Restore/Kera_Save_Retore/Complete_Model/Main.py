import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Input, Dropout

# define hyper-parameter
n_widths = 28
n_height = 28
n_depths = 1
n_inputs = n_widths*n_height*n_depths
n_outpus = 10
n_neurons = [n_inputs,512,256,256,512]
n_layers = len(n_neurons)

def load_dataset():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# reshape dataset
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

	# convert dtype train and test from integer(unit8) into float
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# normalization data
	x_train = x_train/255.0
	x_test = x_test/255.0

	# one hot coding
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()

def build_model_NN():
	inputs = Input(shape = (n_inputs, ), name = "Input")
	layer = inputs

	for i in range(1, n_layers):
		layer = Dense(units = n_neurons[i], activation = 'relu')(layer)
		Dropout(0.02)

	outputs = Dense(units = 10, activation = 'softmax', name = 'outputs')(layer)

	model = Model(inputs=  inputs, outputs = outputs)

	model.summary()

	model.compile(loss = 'categorical_crossentropy', 
					optimizer = Adam(0.01), 
					metrics = ['accuracy'])

	model.fit(x_train, y_train, batch_size = 100, epochs = 1)
	result = model.evaluate(x_test, y_test)
	print ("accuracy: ", result)

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
		
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
	

build_model_NN()
