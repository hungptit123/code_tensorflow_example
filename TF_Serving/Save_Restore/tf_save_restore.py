import tensorflow as tf 
import numpy as np 

def save_all_the_variables():
	inputs = np.array([[1.0,2.0,3.0,4.0]])
	# define parameter
	W = tf.Variable(tf.random_normal(shape = (4,1), stddev = 0.1, dtype = tf.float32))
	b = tf.Variable(tf.random_normal(shape = (1,1), dtype = tf.float32))

	x = tf.placeholder(shape = (None, 4), name = 'x', dtype = tf.float32)

	y = tf.matmul(x,W) + b

	# create object saver
	# if you want to select variables to save
	# you could save the following ways: saver = tf.train.Saver({'key': value})
	saver = tf.train.Saver()

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		outputs = sess.run(y, feed_dict = {x:inputs})
		# save the model
		saved_the_model = saver.save(sess, "full_save_graph/model.ckpt")
		print ("Model saved in {}".format(saved_the_model))
		print ("Value of Variables W {} , b {}".format(W.eval(),b.eval()))
		print ("ouputs: {}".format(outputs))

# print ("save all the variables......")
# save_all_the_variables()
# print ("............................")

def restore_the_variables():
	inputs = np.array([[1.0,2.0,3.0,4.0]])
	# define parameter
	W = tf.Variable(tf.random_normal(shape = (4,1), stddev = 0.1, dtype = tf.float32))
	b = tf.Variable(tf.random_normal(shape = (1,1), dtype = tf.float32))

	x = tf.placeholder(shape = (None, 4), name = 'x', dtype = tf.float32)

	y = tf.matmul(x,W) + b

	# create the model saver
	# if you want to select to restore
	# you could retore the following ways: saver = tf.train.Saver({'key': value})
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# save the model
		restore_the_model = saver.restore(sess, "full_save_graph/model.ckpt")
		outputs = sess.run(y, feed_dict = {x:inputs})
		print ("Value of Variables W {} , b {}".format(W.eval(),b.eval()))
		print ("ouputs: {}".format(outputs))

# print ("restore the variables.......")
# restore_the_variables()
# print ("............................")


