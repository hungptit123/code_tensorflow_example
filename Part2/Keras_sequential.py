import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import SGD, Adam

# defind parameter
n_classes = 10
n_inputs = 28*28
n_epochs = 4
batch_size = 2

# get data
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the tow dimentional 28*28 into one dimentional 784
# print (x_train.shape)
x_train = np.reshape(x_train, (60000, n_inputs))
x_test = np.reshape(x_test, (10000, n_inputs))

# convert the input value into float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# normalize the value of image vector to fit under 1
x_train = x_train/255
x_test = x_test/255

# convert data into one hot encodes format
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# build a sequential model
model = tf.keras.models.Sequential()
# the first layer has to specify dimensions of the input vestor
model.add(Dense(units = 256, activation = 'sigmoid', input_shape = (n_inputs,)))
# add dropout layer for preventing overfitting
model.add(Dropout(0.1))

model.add(Dense(units = 256, activation = 'sigmoid'))
model.add(Dropout(0.1))
# output layer only have the neuron equal to number of class
model.add(Dense(units = n_classes, activation = 'softmax'))

# print the summary of ourmodel
model.summary()

# complie the model
model.compile(loss = 'categorical_crossentropy',
				optimizer = Adam(),
				metrics = ['accuracy'])
# trainning the model
model.fit(x_train, y_train, batch_size = batch_size, epochs = n_epochs)
# evaluation the model
predict = model.predict(x_test)
print (predict[0])
# result = model.evaluate(x_test, y_test)
# print ("loss: ", result)
# print ("accuracy: ", result[1])
