import numpy as np
import torch
from sklearn.model_selection import train_test_split

def data_proc(data_path, label_path):
	data = np.load(data_path)
	labels = np.load(label_path)
	labels = labels.astype(np.float64)
	pieces = ['k','q','r','b','n','p','K','Q','R','B','N','P']
	data_sparse = np.zeros(shape=(data.shape[0],12,8,8))
	for ind_board, board in enumerate(data):
	    for ind,em in enumerate(pieces):
	        data_sparse[ind_board,ind,:,:] = np.int0(board == em)

	X_train, X_val, y_train, y_val = train_test_split(data_sparse, labels, test_size=0.2)
	X_train = torch.tensor(X_train).float()
	X_val = torch.tensor(X_val).float()
	y_train = torch.tensor(y_train).float()
	y_val = torch.tensor(y_val).float()

	return X_train, X_val, y_train, y_val