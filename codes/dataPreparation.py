import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PreProcessing():
    def __init__(self, filePath, savePath, minFreq):
        self.filePath = filePath
        self.savePath = savePath
        self.minFreq = minFreq
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.vocab = []
        self.vocabSize = 0
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<UNK>'
        self.vocabSize = 2
        self.word2count['<PAD>'] = 0
        self.word2count['<UNK>'] = 0
        self.vocab.append('<PAD>')
        self.vocab.append('<UNK>')
        self.tokens = []
        self.label = []
        self.removeWords()

    def removeWords(self):

        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)
        ps = PorterStemmer()
        data = pd.read_csv(self.filePath)

        preProcessedData = []
        for sentence in tqdm(data['sentence'], desc='Preprocessing'):
            tokens = word_tokenize(sentence)
            # stemming 
            filtered_tokens  = [ps.stem(token) for token in tokens]
            # remove stop words
            filtered_tokens = [token for token in filtered_tokens if token not in stop_words]
            # remove punctuations
            filtered_tokens = [token for token in filtered_tokens if token not in punctuations]
            preProcessed_sentence = ' '.join(filtered_tokens)
            preProcessedData.append(preProcessed_sentence)

        word_counts = {}
        for sentence in preProcessedData:
            tokens = word_tokenize(sentence)
            for token in tokens:
                if token not in word_counts:
                    word_counts[token] = 1
                else:
                    word_counts[token] += 1

        for word, count in word_counts.items():
            if count >= self.minFreq:
                self.word2idx[word] = self.vocabSize
                self.word2count[word] = count
                self.idx2word[self.vocabSize] = word
                self.vocabSize += 1
                self.vocab.append(word)


class GetDatasets(Dataset):
    def __init__(self, filePath, vocab, word2idx, pad):
        self.filePath = filePath
        self.vocab = vocab
        self.word2idx = word2idx
        self.tokens = []
        self.label = []
        self.pad = pad
        self.forward_data = []
        self.backward_data = []

        data = pd.read_csv(self.filePath)
        maxLen = 0
        print(data.shape)
        for sentence in data['sentence']:
            tokens = word_tokenize(sentence)
            if len(tokens) > maxLen:
                maxLen = len(tokens)
        for i in tqdm(range(data.shape[0]), desc='Tokenising and Padding'):
            sentence = data['sentence'][i]
            tokens = word_tokenize(sentence)
            for j in range(len(tokens)):
                if tokens[j] in self.word2idx:
                    tokens[j] = self.word2idx[tokens[j]]
                else:
                    tokens[j] = self.word2idx['<UNK>']

            if self.pad:
                tokens.extend([self.word2idx['<PAD>']]*(maxLen-len(tokens)))

            data['sentence'][i] = tokens
            self.tokens.append(tokens)
            self.label.append(data['label'][i])
            self.forward_data.append(tokens[1:])
            self.backward_data.append(tokens[:-1])

        self.label = torch.tensor(self.label)
        self.tokens = torch.tensor(self.tokens)
        self.forward_data = torch.tensor(self.forward_data)
        self.backward_data = torch.tensor(self.backward_data)

    def __getitem__(self, index):
        return self.tokens[index], self.label[index], self.forward_data[index], self.backward_data[index]

    def __len__(self):
        return self.tokens.shape[0]


preProcessing = PreProcessing('sst_train_label.csv', 'sst_train_label.csv', 4)
train_data = GetDatasets('sst_train_label.csv',
                         preProcessing.vocab, preProcessing.word2idx, True)
valid_data = GetDatasets('sst_valid_label.csv',
                         preProcessing.vocab, preProcessing.word2idx, True)
test_data = GetDatasets('sst_test_label.csv',
                        preProcessing.vocab, preProcessing.word2idx, True)

# do for multi_nli dataset
# preProcessing = PreProcessing('multi_nli_train.csv','multi_nli_train.csv',4)
# train_data = GetDatasets('multi_nli_train.csv',preProcessing.vocab,preProcessing.word2idx,True)
# valid_data = GetDatasets('multi_nli_valid.csv',preProcessing.vocab,preProcessing.word2idx,True)
# test_data = GetDatasets('multi_nli_test.csv',
#                         preProcessing.vocab, preProcessing.word2idx, True)
