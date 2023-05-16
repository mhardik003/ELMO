import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, embedding_matrix):
        super(ELMo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding = embedding_matrix
        self.embedding.weight = nn.Parameter(
            self.embedding.weight, requires_grad=True)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim,
                             batch_first=True, bidirectional=True)
        self.linear_out = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, back_data):
        back_embed = self.embedding(back_data)
        back_lstm1, _ = self.lstm1(back_embed)
        back_lstm2, _ = self.lstm2(back_lstm1)
        linear_out = self.linear_out(back_lstm2)
        return linear_out


class scoreClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, elmoEmbedding, elmo_lstm1, elmo_lstm2, num_classes):
        super(scoreClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(
            elmoEmbedding, padding_idx=0)
        self.embeddings.weight = nn.Parameter(
            self.embeddings.weight, requires_grad=False)
        self.weights = nn.Parameter(torch.tensor(
            [0.33, 0.33, 0.33]), requires_grad=False)
        self.lstm1 = elmo_lstm1
        self.lstm2 = elmo_lstm2
        self.linear1 = nn.Linear(embedding_dim, hidden_dim*2)
        self.linear_out = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_data):
        embed = self.embeddings(input_data)
        embedChange = self.linear1(embed)
        lstm1, _ = self.lstm1(embed)
        lstm2, _ = self.lstm2(lstm1)
        elmo_out = (self.weights[0]*lstm1 + self.weights[1]*lstm2 + self.weights[2]
                    * embedChange)/(self.weights[0]+self.weights[1]+self.weights[2])
        elmo_max = torch.max(elmo_out, dim=1)[0]
        elmo_max = self.dropout(elmo_max)
        linear_out = self.linear_out(elmo_max)

        l2_reg = torch.tensor(0.).to(device)
        for param in self.linear1.parameters():
            l2_reg += torch.norm(param, 2)
        for param in self.linear_out.parameters():
            l2_reg += torch.norm(param, 2)

        # scores = self.relu(linear_out)
        return linear_out, l2_reg
