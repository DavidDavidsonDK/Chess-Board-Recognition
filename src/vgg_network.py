from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten,Input, Reshape
import numpy as np

class VGG_Network(Model):
    def __init__(self,
                 input_dim, 
                 output_classes,
                 last_freez_layers = -4,
                 dropout = 0.5):

    	self.dim = input_dim
    	self.classes = output_classes
    	self.dropout = dropout

        X = Input(shape=input_dim, name='X')
        inputs = [X]

        print('Download_vgg')
        vgg_conv = VGG16(weights='imagenet',include_top=False,input_shape=input_dim)
        print('Pop last two dense layers')

        print('Freeze first layers')
        for layer in vgg_conv.layers[:last_freez_layers]:
            layer.trainable = False

        X = vgg_conv(X)
        X = Flatten()(X)
        X = Dense(512, activation='relu')(X)
        X = Dropout(dropout)(X)
        y = Dense(output_classes, activation='softmax')(X)
        outputs = [y]
        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        pass