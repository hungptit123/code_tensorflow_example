import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# define hyper - parammeter
g_learning_rate = 0.00001
d_learning_rate = 0.01
n_x = 784
n_z = 256
g_n_layers = 3
d_n_layers = 1
g_n_neuron = [256, 512, 1024]
d_n_neuron = [256]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# x_train = mnist.train.images
# y_train = mnist.train.labels
# x_test = mnist.test.images
# y_test = mnist.test.labels
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

# define the generator network
g_model = Sequential()
g_model.add(Dense(units = g_n_neuron[0], name = 'g_0', input_shape = (n_z, )))
g_model.add(LeakyReLU(0.2))
for i in range(1, g_n_layers):
	g_model.add(Dense(units = g_n_neuron[i], name = 'g{}'.format(i)))
	g_model.add(LeakyReLU(0.2))

g_model.add(Dense(units = n_x, name = 'g_out', activation = 'tanh'))
# print ("Generator:")
# g_model.summary()

# define discriminator 
d_model = Sequential()
d_model.add(Dense(units = d_n_neuron[0], name = 'd_0', input_shape = (n_x, )))
d_model.add(LeakyReLU(0.2))
d_model.add(Dropout(0.3))
for i in range(1, d_n_layers):
	d_model.add(Dense(units = d_n_neuron[i], name = 'd_{}'.format(i)))
	d_model.add(LeakyReLU(0.2))
	d_model.add(Dropout(0.3))

d_model.add(Dense(units = 1, name = 'd_out', activation = 'sigmoid'))
# print ("Discriminator: ")
# d_model.summary()
d_model.compile(loss = 'binary_crossentropy', 
				optimizer = SGD(lr = d_learning_rate))

# define GAN network
d_model.trainable = False
z_in = Input(shape = (n_z, ), name = 'z_in')
x_in = g_model(z_in)
gan_out = d_model(x_in)

gan_model = Model(inputs = z_in, outputs = gan_out)
print ("GAN:")
gan_model.summary()
gan_model.compile(loss = 'binary_crossentropy', 
				optimizer = Adam(g_learning_rate))

# n_epochs = 400
# batch_size = 100
# n_batches = int(mnist.train.num_examples/batch_size)
# n_epochs_print = 50
# for epoch in range(1, n_epochs):
# 	epochs_d_loss = 0.0
# 	epochs_g_loss = 0.0
# 	print ("the epochs: ", epoch)
# 	for batch in range(n_batches):
# 		x_batch, _ = mnist.train.next_batch(batch_size)
# 		x_batch = normalization(x_batch)
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

# 	if epoch % 10 == 0:
# 		average_d_loss = epochs_d_loss/n_batches
# 		average_g_loss = epochs_g_loss/n_batches
# 		# print ('epoch: {0:0.4d}	average_d_loss: {1:0.6} 	average_g_loss: {2:0.8}'.format(epoch, average_d_loss, average_g_loss))
# 		print ("epochs: ", epoch)
# 		print ("average_d_loss: ", average_d_loss)
# 		print ("average_g_loss: ", average_g_loss)
# 		x_pre = g_model.predict(z_test)
# 		display_images(x_pre.reshape(-1, 28, 28))

