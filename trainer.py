# ==============================================================================
# File: trainer.py
# Project: The-Office
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 17th March 2023 2:33:55 pm
# Modified By: Dillon Koch
# -----
#
# -----
# training a model
# ==============================================================================


import sys
from os.path import abspath, dirname

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OfficeDataset
from model_rnn import RNNModel

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


# training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 10
emb_size = 1000
input_size = 1000
hidden_size = 1000
num_layers = 3
epochs = 10000
learning_rate = 0.001

office_dataset = OfficeDataset('train', sequence_length)
dict_size = len(office_dataset.vocab)

dataloader = DataLoader(office_dataset, batch_size=32, shuffle=True)
model = RNNModel(input_size, hidden_size, num_layers, dict_size, emb_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(i, end='\r')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if epoch == 0 and i == 0:
            print('initial loss', loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += sum(preds == labels)
        total += len(inputs)

    print(f'{correct} correct, {total} total')
    print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, running_loss / len(dataloader)))
