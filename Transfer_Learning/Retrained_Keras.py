import tensorflow as tf 
import numpy as np 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Reshape, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD

# define hyper - parammeters 
n_widths = 224
n_heights = 224
n_depths = 3
n_inputs = n_widths*n_heights*n_depths
n_outputs = 8

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (n_widths, n_heights, n_depths))

for layer in base_model.layers[:15]:
	layer.trainable = False

inputs = base_model.output
layer = inputs

layer = Flatten()(inputs)
layer = Dense(units = 256, activation = 'relu')(layer)
layer = Dropout(0.02)(layer)
outputs = Dense(units = n_outputs, activation = 'softmax')(layer)

# model = Model(inputs = inputs, outputs = outputs)
model = Model(inputs = base_model.inputs, outputs = outputs)

model.compile(loss = 'categorical_crossentropy', 
				optimizer = SGD(0.001, momentum = 0.9), 
				metrics = ['accuracy'])

model.summary()
	