import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Lambda, Input, Dropout
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt

# define hypter - parameters
learning_rate = 0.001
n_widths = 28
n_heights = 28
n_depths = 1
n_inputs = n_widths*n_heights*n_depths
n_outputs = n_inputs
# neuron in each layer
n_neurons = [512,256]
n_layers = len(n_neurons)
# the dimension of flatten variables
n_neurons_z = 128

def load_dataset():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# reshape 
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

	# convert type data from integer into float
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# normalization
	x_train = x_train/255.0
	x_test = x_test/255.0

	# one hot coding
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()

def add_noise(X):
	return X + 0.5 * np.random.randn(X.shape[0], X.shape[1])

x_train_noise = add_noise(x_train)

# define a function that calculates the sum of reconstruction and regularization loss

def display(images, labels, count = 0, one_hot = False):
	images = images.reshape([-1, n_widths, n_heights])
	if count == 0:
		count = images.shape[0]
	# idx_list = random.sample(range(len(labels)), count)
	for i in range(count):
		plt.subplot(4,4, i+1)
		plt.title(labels[i])
		plt.imshow(images[i], cmap = 'gray')
		plt.axis('off')
	plt.tight_layout()
	plt.show()


def build_model_functional():
	# build the input
	x = Input(shape = (n_inputs, ), name = "inputs")

	# build the encoder layer
	layer = x
	for i in range(n_layers):
		layer = Dense(units = n_neurons[i], activation = 'relu', 
					name = 'enc_{0}'.format(i)) (layer)
		Dropout(0.02)

	z_mean = Dense(units = n_neurons_z, name = 'z_mean')(layer)
	Dropout(0.02)
	z_log_v = Dense(units = n_neurons_z, name = 'z_log_v')(layer)
	Dropout(0.02)

	# create the noise and posterior distribution
	# the first noisy distribution
	epsilon = backend.random_normal(shape = backend.shape(z_log_v), mean = 0, stddev = 1.0)

	# the second posterior distribution
	z = Lambda(lambda zargs: zargs[0] + backend.exp(zargs[1]*0.5)*epsilon, name = 'z')([z_mean, z_log_v])

	# add decoder layer
	layer = z

	for i in range(n_layers-1, -1, -1):
		layer = Dense(units = n_neurons[i], activation = 'relu', 
						name = 'dec_{0}'.format(i))(layer)
		Dropout(0.02)

	# # define the final output layer
	y_hat = Dense(units = n_outputs, activation = 'sigmoid', name = "output")(layer)

	# define the model
	model = Model(inputs = x, outputs = y_hat)
	print (model.summary())

	def vae_loss(y, y_hat):
		rec_loss = - backend.sum(y * backend.log(1e-10 + y_hat) + (1-y)*backend.log(1e-10 + 1 - y_hat), axis = -1)

		reg_loss = -0.5 * backend.sum(1 + z_log_v - backend.square(z_mean) - backend.exp(z_log_v), axis = -1)

		loss = backend.mean(rec_loss + reg_loss)

		return loss

	model.compile(loss = vae_loss, 
					optimizer = Adam(0.001))

	# model.fit(x = x_train, y = x_train, batch_size = 100, epochs = 2)

	# train_images = x_train[10:15, :]
	# train_labels = y_train[10:15, :]
	# y = np.argmax(train_labels, 1)

	# test_images = x_test[5:10, :]
	# test_labels = y_test[5:10, :]
	# y_t = np.argmax(test_labels, 1)

	# t = model.predict(train_images)

	# k = model.predict(test_images)
	# display(t, y)
	# display(k, y_t)

build_model_functional()