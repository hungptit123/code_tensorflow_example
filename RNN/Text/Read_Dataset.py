import pickle
import Configuration as CF
import os 
import tensorflow as tf
import zipfile
import numpy as np

def read_data(filename):
	"""Extract the first file enclosed in a zip file as a list of words."""
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	data = np.asarray(data)
	x = data[:128*2500]
	pickle.dump(x, open(CF.DIR_DATA, 'wb'))
	# print (data[:50])
	# return data

read_data(CF.DIR)