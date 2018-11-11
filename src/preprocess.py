import os
import numpy as np
import cv2
import pickle

#train_path = '../data/raw/Chess ID Public Data/output_train/'
#test_path = '../data/raw/Chess ID Public Data/output_test/'

label_names = ['bb','bk','bn','bp','bq','br','wb','wk','wn','wp','wq','wr','empty']
label_to_index_map = dict(zip(label_names,np.arange(len(label_names))))


def create_data(path_of_images, img_shape = (227,227,3),train=True):

    maxlen = len([1 for d in label_names for img_name in os.listdir(path_of_images + d) if img_name.endswith('.jpg')])
    data = np.zeros(shape = (maxlen, np.product(img_shape)),dtype=np.uint8)
    labels = np.zeros(maxlen,dtype=np.uint8)
    i = 0
    print("Reading images and flat them ...")
    for d in label_names:
        d_images = os.listdir(path_of_images + d)
        for img_name in d_images:
            if img_name.endswith('.jpg'):
                img_arr  = cv2.imread(path_of_images + d + '/'+img_name).reshape(-1)
                data[i,:] = img_arr[:]
                labels[i] = label_to_index_map[d]
                i+=1
    print("Save image arrays to 'data/processed ...'")
    if train:
        with open('../data/processed/train_x.npy','wb') as file:
            pickle.dump(data,file)
            file.close()
        with open('../data/processed/train_y.npy','wb') as file:
            pickle.dump(labels,file)
            file.close()
    else:
        with open('../data/processed/test_x.npy','wb') as file:
            pickle.dump(data,file)
            file.close()
        with open('../data/processed/test_y.npy','wb') as file:
            pickle.dump(labels,file)
            file.close()
    return data[:], labels[:]