import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import SGD, Adam

num_outputs = 10 # 0-9 digits
num_inputs = 784 # total pixels
learning_rate = 0.001
num_epochs = 1
batch_size = 100

# get data
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the tow dimentional 28*28 into one dimentional 784
# print (x_train.shape)
x_train = np.reshape(x_train, (60000, num_inputs))
x_test = np.reshape(x_test, (10000, num_inputs))

# convert the input value into float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# normalize the value of image vector to fit under 1
x_train = x_train/255
x_test = x_test/255

# convert data into one hot encodes format
y_train = tf.keras.utils.to_categorical(y_train, num_outputs)
y_test = tf.keras.utils.to_categorical(y_test, num_outputs)

# input images
x = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs], name="x")
# output labels
y = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name="y")
# model paramteres
w = tf.Variable(tf.zeros([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")
model = tf.nn.softmax(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for epoch in range(num_epochs):
		sess.run(optimizer, feed_dict = {x:x_train, y:y_train})
		# predictions_check = tf.equal(tf.argmax(model, 1), tf.argmax (y,1))
		# accuracy function = ts.reduce mean (tf. cast (predictions_check, ts. float 32))
