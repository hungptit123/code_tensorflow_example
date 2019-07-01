import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Dense, Input, MaxPool2D, Conv2D, Reshape, LeakyReLU, UpSampling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

num_inputs = 28*28
num_outputs = 10

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
	# x_train = x_train/255.0
	# x_test = x_test/255.0

	# one hot coding label
	y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
	y_test = tf.keras.utils.to_categorical(y_test, num_outputs)

	return x_train[:100], y_train[:100]

x_train, y_train = load_dataset()

# print (x_train[0])

g_learning_rate = 0.00001
d_learning_rate = 0.01
n_x = 784
n_z = 256
g_n_layers = 3
d_n_layers = 1
g_n_neuron = [256, 512, 1024]
d_n_neuron = [256]

z_test = np.random.uniform(-1.0, 1.0, size = [8, n_z])
def normalization(X):
	return (X - 0.5)/0.5

def display_images(images):
	for i in range(images.shape[0]):
		plt.subplot(1,8, i+1)
		plt.imshow(images[0], cmap = 'gray')
		plt.axis('off')
	plt.tight_layout()
	plt.show()

# Generator
g_model = Sequential()
g_model.add(Dense(units = 3200, input_shape = (n_z, ), name = 'g_in'))
g_model.add(LeakyReLU())
g_model.add(Reshape(target_shape = (5, 5, 128)))
g_model.add(UpSampling2D(size = (2,2)))
g_model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = (1,1), 
					padding = 'SAME'))
g_model.add(LeakyReLU())
g_model.add(UpSampling2D(size = (2,2)))
g_model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = (1,1), 
					padding = 'SAME'))
g_model.add(LeakyReLU())
g_model.add(UpSampling2D(size = (2,2)))
g_model.add(Conv2D(filters = 16, kernel_size = (5,5), strides = (1,1), 
					padding = 'SAME'))
g_model.add(LeakyReLU())
g_model.add(Flatten())
g_model.add(Dense(units = n_x, activation = 'tanh'))

# g_model.summary()

# Discriminate
d_model = Sequential()
d_model.add(Reshape(target_shape = (28, 28, 1), input_shape = (n_x, )))
d_model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = (1,1), 
					padding = 'SAME'))
d_model.add(LeakyReLU())
d_model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
d_model.add(Flatten())
d_model.add(Dense(units = 1, activation = 'sigmoid'))
# d_model.summary()

d_model.compile(loss = 'binary_crossentropy', 
				optimizer = SGD(d_learning_rate))

d_model.trainable = False

# GAN
z_in = Input(shape = (n_z, ), name = 'g_in')
x_in = g_model(z_in)
gan_out = d_model(x_in)

gan_model = Model(inputs = z_in, outputs = gan_out)
print ("GAN:")
gan_model.summary()
gan_model.compile(loss = 'binary_crossentropy', 
				optimizer = Adam(g_learning_rate))

# n_epochs = 400
# batch_size = 100
# n_epochs_print = 40
# for epoch in range(1, n_epochs):
# 	epochs_d_loss = 0.0
# 	epochs_g_loss = 0.0
# 	print ("the epochs: ", epoch)
# 	for batch in range(10):
# 		x_batch = x_train
# 		z_batch = np.random.uniform(-1.0, 1.0, size = [batch_size, n_z])
# 		g_batch = g_model.predict(z_batch)
		
# 		x_in = np.concatenate([x_batch, g_batch])
# 		y_out = np.ones(batch_size*2)
# 		y_out[:batch_size] = 0.9
# 		y_out[batch_size:] = 0.1
# 		d_model.trainable = True
# 		batch_d_loss = d_model.train_on_batch(x_in, y_out)

# 		z_batch = np.random.uniform(-1.0, 1.0, size = [batch_size, n_z])
# 		x_in = z_batch
# 		y_out = np.ones(batch_size)
# 		d_model.trainable = False
# 		batch_g_loss = gan_model.train_on_batch(x_in, y_out)
# 		epochs_d_loss += batch_d_loss
# 		epochs_g_loss += batch_g_loss

# 	average_d_loss = epochs_d_loss/10
# 	average_g_loss = epochs_g_loss/10
# 	# print ('epoch: {0:0.4d}	average_d_loss: {1:0.6} 	average_g_loss: {2:0.8}'.format(epoch, average_d_loss, average_g_loss))
# 	print ("epochs: ", epoch)
# 	print ("average_d_loss: ", average_d_loss)
# 	print ("average_g_loss: ", average_g_loss)
# 	if epoch % 20 == 0:
# 		x_pre = g_model.predict(z_test)
# 		display_images(x_pre.reshape(-1, 28, 28))

