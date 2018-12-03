import os
import numpy as np
from preprocess import create_data
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from vgg_network import VGG_Network
import keras




if __name__ == '__main__':
	print('create test and dev data')
	#os.chdir('./Chess-Board-Recognition/src')

	train_path = '../data/raw/Chess ID Public Data/output_train/'
	test_path = '../data/raw/Chess ID Public Data/output_test/'

	train_data, train_lbl = create_data(train_path,train=True)
	test_data, test_lbl = create_data(test_path,train=False)

	x = np.concatenate([train_data,test_data],axis = 0)
	y = np.concatenate([train_lbl, test_lbl],axis = 0)
	train_data, test_data, train_lbl, test_lbl = train_test_split(x, y, test_size=0.22, random_state=42)

	train_data = preprocess_input(train_data,mode='tf')
	test_data = preprocess_input(test_data,mode='tf')
	input_shape = (227, 227, 3)
	print('Define model')
	model = VGG_Network(input_dim=input_shape,output_classes=13,last_freez_layers = None ,dropout = 0.15)
	print(model.summary())

	print('start training')
	save_best = keras.callbacks.ModelCheckpoint(filepath = '../models/model.{epoch:02d}-{val_loss:.2f}.hdf5' ,
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            save_weights_only=False, 
                                            mode='auto', period=1)

	cl_weights = dict(zip(np.arange(13),[3,10,1,1,1,10,1,10,1,1,1,10,0.01]))

	model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


	model.fit(train_data, train_lbl,
                        batch_size=128,
                        epochs=10,
                        verbose=1,
                        validation_data=(test_data, test_lbl),
                        callbacks=[save_best],
                        class_weight = cl_weights
                      )
	print('end training')


