import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model 

inputs = Input(shape = (64, ))

# a layer instance is callable on a tensor and return a tensor 
# layer_shared = Dense(units = 64, activation = 'relu')(inputs)
# x = Dense(units = 64, activation = 'relu')(layer_shared)
layer_shared = Dense(units = 128, activation = 'relu')
# layer_shared_128 = Dense(units = 128, activation = 'relu')

x = layer_shared(inputs)
# x = layer_shared_128(x)
# x = layer_shared(x)

prediction = Dense(units = 10, name = 'outputs', activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = prediction)

model.summary()