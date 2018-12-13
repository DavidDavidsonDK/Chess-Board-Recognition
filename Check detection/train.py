import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from data_proc import data_proc
from chess_net_baseline import chess_net

data_path = '../../data.npy'
labels_path = '../../labels.npy'

X_train, X_val, y_train, y_val = data_proc(data_path, labels_path)

model = chess_net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

BATCH_SIZE = 32
EPOCH = 20

TRAIN_SIZE = X_train.shape[0]
VAL_SIZE = X_val.shape[0]

train_batch_count = X_train.shape[0]//32
val_batch_count = X_val.shape[0]//32
val_loss_min = np.Inf

for epoch in range(EPOCH):
    shuffle_index = np.random.permutation(TRAIN_SIZE)
    X_train = X_train[shuffle_index]
    y_train = y_train[shuffle_index]
    
    train_loss = 0
    train_acc = 0
    for i in range(train_batch_count):
        model.train()
        x_batch = X_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
        y_batch = y_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
        
        optimizer.zero_grad()
        prob = model(x_batch)
        loss = criterion(prob, y_batch)
        loss.backward()
        optimizer.step()
        
        pred = (prob + 0.5).floor()
        train_acc += torch.sum(pred.view(pred.shape[0]) == y_batch).item()
        train_loss += loss.item()
            
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        model.eval()
        
        for i in range(val_batch_count):
            x_batch = X_val[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
            y_batch = y_val[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
            
            prob = model(x_batch)
            loss = criterion(prob, y_batch)
            
            pred = (prob + 0.5).floor()
            val_acc += torch.sum(pred.view(pred.shape[0]) == y_batch).item()
            val_loss += loss.item()
            
    train_loss = train_loss/train_batch_count
    val_loss = val_loss/val_batch_count
    
    print('Epoch :',epoch+1)     
    print(f'\tTraining loss : {train_loss}')
    print(f'\tValidation loss : {val_loss}')
    
    print(f'\tTraining accuracy : {100*train_acc/TRAIN_SIZE}')
    print(f'\tValidation accuracy : {100*val_acc/VAL_SIZE}')

    if val_loss <= val_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ... '.format(val_loss_min, val_loss))
        torch.save(model.state_dict(), 'model_chess.pt')
        val_loss_min = val_loss
    print('\n')