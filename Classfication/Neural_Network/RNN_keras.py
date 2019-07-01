import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential

# defind parameter
num_outputs = 10
num_inputs = (28,28)

def load_dataset():
	mnist = tf.keras.datasets.fashion_mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# covert from integer into float
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)

	# normalizations
	x_train = x_train/255.0
	x_test = x_test/255.0

	# x_train = x_train.reshape(-1,28,28)

	# one hot coding lable
	y_test = tf.keras.utils.to_categorical(y_test, num_outputs)
	y_train = tf.keras.utils.to_categorical(y_train, num_outputs)

	return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = load_dataset()
# print (x_train.shape)
def build_model_RNN():
	model = Sequential()

	model.add(tf.keras.layers.SimpleRNN(units = 16, activation = 'relu', input_shape = (28,28)))
	model.add(Dense(units = num_outputs, activation = 'softmax'))
	# model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy', 
					optimizer = Adam(), 
					metrics = ['accuracy'])

	model.summary()

	model.fit(x_train, y_train, batch_size = 1000, epochs = 5)

	result = model.evaluate(x_test, y_test)

	print ("accuracy: ", result)

build_model_RNN()
