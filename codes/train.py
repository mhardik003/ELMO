from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import tqdm
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

from codes.dataPreparation import PreProcessing
from codes.model import ELMo, scoreClassifier
from codes.dataPreparation import GetDatasets


# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the data
preProcessing = PreProcessing('../Data/sst_train_label.csv', '../Data/sst_train_label.csv', 4)
train_data = GetDatasets('../Data/sst_train_label.csv',
                         preProcessing.vocab, preProcessing.word2idx, True)
valid_data = GetDatasets('../Data/sst_valid_label.csv',
                         preProcessing.vocab, preProcessing.word2idx, True)
test_data = GetDatasets('../Data/sst_test_label.csv',
                        preProcessing.vocab, preProcessing.word2idx, True)

# global variables
VOCAB_SIZE = preProcessing.vocabSize
BATCH_SIZE = 32
EMBEDDING_DIM = 300
HIDDEN_DIM = 100


# get the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = torch.utils.data.DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False)


# load the glove embeddings ( embeddings will be saved in separate files )
glove_file = './glove.6B/glove.6B.300d.txt'
glove_dict = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split(' ')
        word = tokens[0]
        embedding = np.array([float(val) for val in tokens[1:]])
        glove_dict[word] = embedding

# UNK and PAD embedding
UNK_emb = np.mean(list(glove_dict.values()), axis=0)
PAD_emb = np.zeros(300)

# print(len(glove_dict))
vocab = preProcessing.vocab
embeddings = []
for word in vocab:
    if word == '<UNK>':
        embeddings.append(UNK_emb)
    elif word == '<PAD>':
        embeddings.append(PAD_emb)
    elif word in glove_dict:
        embeddings.append(glove_dict[word])
    else:
        emb = np.random.uniform(-0.25, 0.25, 300)
        embeddings.append(emb)

embeddings = torch.tensor(embeddings, dtype=torch.float)
# embeddings = torch.stack(embeddings)
embedding = nn.Embedding.from_pretrained(
    embeddings, freeze=False, padding_idx=0)

print(embedding.weight.shape)
# save the embedding
torch.save(embedding, 'embeddings2.pt')

# create the model
elmo = ELMo(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, embeddings)
# print(elmo)
elmo.to(device)

#training

def train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    iter = 0
    for (tokens, label, forward, backward) in tqdm(train_dataloader, desc='Training'):
        forward = forward.to(device)
        backward = backward.to(device)
        optimizer.zero_grad()
        output = model(backward)
        loss = criterion(output.view(-1, VOCAB_SIZE), forward.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iter += 1
        if iter % 1000 == 0:
            print('Iteration: ', iter, 'Train Loss: ', total_loss/iter)
    return total_loss/len(train_dataloader)


def validate_epoch(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    iter = 0
    with torch.no_grad():
        for (tokens, label, forward, backward) in tqdm(valid_dataloader, desc='Validating'):
            forward = forward.to(device)
            backward = backward.to(device)
            output = model(backward)
            loss = criterion(output.view(-1, VOCAB_SIZE), forward.view(-1))
            total_loss += loss.item()
            iter += 1
            if iter % 500 == 0:
                print('Iteration: ', iter, 'Validation Loss: ', total_loss/iter)
    return total_loss/len(valid_dataloader)


def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, epochs):
    best_valid_loss = float('inf')
    losses = {'epoch': [], 'train_loss': [], 'valid_loss': []}
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        valid_loss = validate_epoch(model, valid_dataloader, criterion)
        print('Train Loss: ', train_loss, 'Valid Loss: ', valid_loss)
        losses['epoch'].append(epoch)
        losses['train_loss'].append(train_loss)
        losses['valid_loss'].append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'elmo_model1.pt')
            # print('Model saved')
    return losses


optimizer = optim.Adam(elmo.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
losses = train_model(elmo, train_dataloader,
                     valid_dataloader, optimizer, criterion, 20)

elmo.load_state_dict(torch.load('elmo_model1.pt'))
# Parameters
for name, param in elmo.named_parameters():
    if param.requires_grad:
        print(name, param.data, param.shape)
elmo_embeddings = list(elmo.parameters())[0].cpu().detach().numpy()
torch.save(elmo_embeddings, 'elmo_embeddings1.pt')

elmo_lstm1 = elmo.lstm1
print(elmo_lstm1.parameters())
elmo_lstm2 = elmo.lstm2
print(elmo_lstm2.parameters())
elmo_embeddings = list(elmo.parameters())[0].to(device)


# use classifier 
classifier = scoreClassifier(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, elmo_embeddings, elmo_lstm1, elmo_lstm2, 2)
print(classifier)
classifier.to(device)
#Initializing optimizer
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
#Initializing loss function
criterion = nn.CrossEntropyLoss()

def train_classifier_epoch(model, train_dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    iter = 0
    total_acc = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        input_data, labels, _, _ = batch
        logits, l2_reg = model(input_data)
        loss = criterion(logits, labels) + 0.001*l2_reg
        loss.backward()
        optimizer.step()
        _, preds = torch.max(logits, dim=1)
        train_acc = accuracy_score(
            labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
        total_acc += train_acc
        train_loss += loss.item()
        iter += 1
        if iter % 1000 == 0:
            print("Iteration: {}, Train Loss: {}".format(iter, loss.item()))

    return train_loss/len(train_dataloader), total_acc/len(train_dataloader)


def eval_classifier_epoch(model, val_dataloader, criterion, device):
    model.eval()
    losses = 0
    iter = 0
    total_acc = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = tuple(t.to(device) for t in batch)
            input_data, labels, _, _ = batch
            logits, _ = model(input_data)

            _, preds = torch.max(logits, dim=1)
            val_acc = accuracy_score(
                labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
            total_acc += val_acc
            loss = criterion(logits, labels)
            losses += loss.item()
            iter += 1
            if iter % 500 == 0:
                print("Iteration: {}, Validation Loss: {}".format(
                    iter, loss.item()))
    return losses/len(val_dataloader), total_acc/len(val_dataloader)


def train_classifier(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=10):

    losses = {'epoch': [], 'train_loss': [], 'valid_loss': []}
    acc = {'epoch': [], 'train_acc': [], 'valid_acc': []}
    min_valid_loss = float('inf')
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))

        train_loss, train_acc = train_classifier_epoch(
            model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = eval_classifier_epoch(
            model, val_dataloader, criterion, device)
        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            print("Saving model...")
            # torch.save(model.state_dict(), 'elmo2_classifier.pt')

        print("Train Loss: {}, Train Accuracy: {}".format(train_loss, train_acc))
        print("Validation Loss: {}, Validation Accuracy: {}".format(val_loss, val_acc))
        losses['epoch'].append(epoch)
        losses['train_loss'].append(train_loss)
        losses['valid_loss'].append(val_loss)
        acc['epoch'].append(epoch)
        acc['train_acc'].append(train_acc)
        acc['valid_acc'].append(val_acc)

    return losses, acc


train_losses = train_classifier(
    classifier, train_dataloader, valid_dataloader, optimizer, criterion, device, epochs=25)


classifier.load_state_dict(torch.load('elmo2_classifier.pt'))
classifier.eval()
y_true = []
y_pred = []
confusion_matrix = np.zeros((3, 3))

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = tuple(t.to(device) for t in batch)
        input_data, labels, _, _ = batch
        logits, _ = classifier(input_data)
        _, preds = torch.max(logits, dim=1)
        y_true.extend(labels.cpu().detach().numpy())
        y_pred.extend(preds.cpu().detach().numpy())
        for i in range(len(labels)):
            confusion_matrix[labels[i]][preds[i]] += 1

print(classification_report(y_true, y_pred, target_names=[
      'negative', 'neutral', 'positive'], zero_division=1))
print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
print("F1 Score: {}".format(f1_score(y_true, y_pred), average='micro'))
print("Precision: {}".format(precision_score(y_true, y_pred), average='micro'))
print("Recall: {}".format(recall_score(y_true, y_pred), average='micro'))
print("Confusion Matrix: ")
print(confusion_matrix)
