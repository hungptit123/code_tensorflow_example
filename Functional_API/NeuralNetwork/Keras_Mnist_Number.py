import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model 

# this returns a tensor
inputs = Input(shape = (784, ), name = "input")

# build model layers
layer1 = Dense(units = 64, name = "layer1", activation = 'relu')(inputs)
layer2 = Dense(units = 64, name = "layer2", activation = 'relu')(layer1)

outputs = Dense(units = 32, name = 'output', activation = 'softmax')

model = Model(inputs = inputs, outputs = outputs)
model.summary() 