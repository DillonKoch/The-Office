# ==============================================================================
# File: model_rnn.py
# Project: The-Office
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 15th March 2023 1:11:17 pm
# Modified By: Dillon Koch
# -----
#
# -----
# RNN model for classifying lines
# ==============================================================================

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dict_size, emb_size):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dict_size = dict_size
        self.emb_size = emb_size

        self.embedding = nn.Embedding(dict_size, emb_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 21)
        self.softmax = nn.Softmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):  # Run
        x = self.embedding(x)
        # x.size(0) is batch_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
