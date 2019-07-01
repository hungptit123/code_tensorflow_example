import tensorflow as tf
import numpy as np

c = np.ones((4,2))
d = np.ones((2,2))

def save_graph():
	x = tf.placeholder(tf.float64, shape = [None, 2], name = "ab")
	e = tf.matmul(x, d, name='example')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		test = sess.run(e, feed_dict = {
				x:c,
			})
		# print (c)
		# tf.train.write_graph(sess.graph_def, 'output_graph', 'a.pbtxt', False)
		# print (test)
save_graph() 

def load_graph():
	filename = "output_graph/a.pbtxt"
	with tf.gfile.FastGFile(filename, 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    tf.import_graph_def(graph_def, name='')
	    test = tf.get_default_graph().get_tensor_by_name("example:0")
	    with tf.Session() as sess:
		    a = sess.run(test, feed_dict = {
		    	"ab:0":c
		    	})
		    print (a)
load_graph()