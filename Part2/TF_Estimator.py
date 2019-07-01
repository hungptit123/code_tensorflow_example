import tensorflow as tf 
import numpy as np 

def loadData():
	mnist = tf.keras.datasets.fashion_mnist
	(train_image, train_label), (test_image, test_label) = mnist.load_data()
	return train_image, train_label, test_label, test_label
# load data 
train_image, train_label, test_label, test_label = loadData()
# the number of class
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
