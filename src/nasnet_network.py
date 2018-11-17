from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.applications.nasnet import NASNetLarge
from keras.layers import Dense, Dropout, Flatten,Input, Reshape
import numpy as np

class NASNET_Network(Model):
	def __init__(self,
				 input_dim, 
				 output_classes,
				 last_freez_layers = None,
				 top_dense_units = 16,
				 dropout = 0.5):

		self.dim = input_dim
		self.classes = output_classes
		self.dropout = dropout

		X = Input(shape=input_dim, name='X')
		inputs = [X]

		print('Download_nasnet')
		nas_net = NASNetLarge(weights='imagenet',include_top=False,input_shape=input_dim)
		print('Freeze first layers')
		if last_freez_layers is None:
			last_freez_layers = len(nas_net.layers)
		for layer in nas_net.layers[:last_freez_layers]:
			layer.trainable = False

		X = nas_net(X)
		X = Flatten()(X)
		X = Dense(top_dense_units, activation='relu')(X)
		X = Dropout(dropout)(X)
		y = Dense(output_classes, activation='softmax')(X)
		outputs = [y]
		super(NASNET_Network, self).__init__(inputs=inputs, outputs=outputs)

	def say_name(self):
		pass






