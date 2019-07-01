import tensorflow as tf 
import numpy as np 
import random
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import _thread
import cv2
import time

# define parameter
n_widths = 28
n_heights = 28
n_depths = 1
n_inputs = n_widths*n_heights*n_depths
n_outputs = n_inputs
n_neurons = [512,256,256,512]
n_layers = len(n_neurons)
learning_rate = 0.001
batch_size = 100
n_eopchs = 10

def load_dataset():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	x_train = x_train/255.0
	x_test = x_test/255.0

	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()

def add_noise(X):
	return X + 0.5 * np.random.randn(X.shape[0], X.shape[1])

# x_train = add_noise(x_train)
# x_test = add_noise(x_test)

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

train_images = x_train[10:15, :]
train_labels = y_train[10:15, :]
y = np.argmax(train_labels, 1)

test_images = x_test[5:10, :]
test_labels = y_test[5:10, :]
y_test = np.argmax(test_labels, 1)

# t = model.predict(train_images)

# k = model.predict(test_images)
display(train_images, y)
display(test_images, y_test)