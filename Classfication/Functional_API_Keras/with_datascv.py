import tensorflow as tf 
import numpy as np 
import csv
import os
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam, SGD

def readData(m):
	file = open("/home/hunglv/Documents/tensorflow/data/wisconsin.wdbc.data.csv","r")
	reader = csv.reader(file)
	i = 0
	data = np.zeros((569, 32))
	i = 0
	for row in reader:
		for j in range(2,len(row)):
			data[i,j-1] = row[j]
		if row[1] == 'M':
			data[i,1] = 1
			data[i,0] = 0
		else :
			data[i,1] = 0
			data[i,0] = 1
		i += 1
	np.random.shuffle(data) 
	x_train = data[:m, 2:]
	y_train = data[:m, :2]
	x_test = data[m:, 2:]
	y_test = data[m:, :2]
	return x_train, y_train, x_test, y_test

def normalization(data1, data2, n, m_train):
	x_max = np.max(data1, axis = 0)
	x_min = np.min(data1, axis = 0)
	x_avange = np.sum(data1, axis = 0) * 1.0/m_train 
	for i in range(n-1):
		r = x_max[i] - x_min[i]
		if r==0:
			r = 1
		data1[i:i+1, :] = 1.0*(data1[i:i+1, :] - x_avange[i])/r
		data2[i:i+1, :] = 1.0*(data2[i:i+1, :] - x_avange[i])/r
	return data1, data2
x_train, y_train, x_test, y_test = readData(500)
x_train, x_test = normalization(x_train, x_test, 31, 500)
# print (y_train[0])

n_inputs = 30
n_neurons = [64,64]
n_outputs = 2

def build_the_model():
	inputs = Input(shape = (n_inputs, ), name = "input")

	shared_layer = Dense(units = n_neurons[0], activation = 'sigmoid')

	x = shared_layer(inputs)

	outputs = Dense(units = n_outputs, activation = 'softmax', name = "outputs")(inputs)

	model = Model(inputs = inputs, outputs = outputs)

	model.compile(loss = 'mse', 
					optimizer = Adam(0.01),
					metrics = ['accuracy'])

	model.fit(x_train, y_train, batch_size = 500, epochs = 5000)

	result = model.evaluate(x_test, y_test)

	print ("accuracy: ", result)

build_the_model()

