from sys import path
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralNetworks import EncodingNN

BASE = '../'
PATH = BASE+'data/'


filehandler = open(PATH+"Audio_X.pkl","rb")
X_ = pkl.load(filehandler)
filehandler.close()

filehandler = open(PATH+"Audio_Y.pkl","rb")
Y_ = pkl.load(filehandler)
filehandler.close()

print('Loaded Training Set')

X_ = StandardScaler().fit_transform(X_)

print(X_.mean(axis=0))
print(X_.std(axis=0))

Y = torch.from_numpy(Y_.flatten())
X = torch.from_numpy(X_).float()

audio_model = EncodingNN()


#Defining criterion
num_epochs = 1200
learning_rate = 1e-4

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(audio_model.parameters(), lr=learning_rate)

# Train the model
print('Start training: ')
i = 0
for epoch in range(num_epochs):
    # Forward pass
    encoded, decoded = audio_model(X)
    loss = criterion(decoded, X)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    i = i+1
    if (i % 100 == 0):
        print ('Epoch [{}/{}], Loss: {:.4f}'
           .format(epoch+1, num_epochs, loss.item()))

torch.save(audio_model, BASE+'saved_models/audio_model')
print('Saved model as audio_model')
