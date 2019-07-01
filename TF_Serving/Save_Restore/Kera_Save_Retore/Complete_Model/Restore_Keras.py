import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Model, model_from_json
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

def build_model():
	# load_json and create model
	json_file = open("model.json", "r")
	json_file_loaded = json_file.read()
	json_file.close()
	# load model
	model = model_from_json(json_file_loaded)
	# load weight into model
	model.load_weights("model.h5")
	print ("loaded model from disk")

	model.compile(loss = 'categorical_crossentropy', 
					optimizer = Adam(0.01),
					metrics = ['accuracy'])

	result = model.evaluate(x_test, y_test)
	print ("accuracy: ", result)

build_model()