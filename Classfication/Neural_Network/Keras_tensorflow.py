import tensorflow as tf 
import numpy as np 
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import TensorBoard

NAME = "logs"
tensorboard = TensorBoard(log_dir = NAME)

num_outputs = 10
num_layers = 2
num_neurons = [256, 256]
num_epochs = 50
batch_sizes = 100
num_inputs = 784
learning_rate = 0.01

# load dataset
mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape from 28x28 into 784
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

# tranform one hot codeing
y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
y_test = tf.keras.utils.to_categorical(y_test, num_outputs)

# convert interger into float
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# normalization
X_train = X_train/255.0
X_test = X_test/255.0

# start build model neural network

# create a sequantial model
model = Sequential()

# build model neural network
model.add(Dense(units = num_neurons[0], activation = "relu", 
				input_shape = [num_inputs])) # add first hidden layer
# next hidden layer
for i in range(1, num_layers):
	model.add(Dense(units = num_neurons[i], activation = 'relu'))

# finall full connect -> output (using softmax)
model.add(Dense(units = num_outputs, activation = 'softmax'))

# print the model details
model.summary()

# compile the model with Adam
model.compile(loss = 'categorical_crossentropy', 
				optimizer = Adam(),
				metrics = ['accuracy'])

# train the model
model.fit(X_train, y_train, epochs = 2, batch_size = 100, callbacks = [tensorboard])

# evaluate the model 
score = model.evaluate(X_test, y_test)
print ("loss: ", score[0])
print ("accuracy: ", score[1])
