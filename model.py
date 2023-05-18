import torch
import torch.nn as nn
from torch.utils.data import Dataset
from nltk import word_tokenize
import torch.nn.functional as F


class NewsDataset(Dataset):

    def __init__(self, dataset, w2v, embedding_size, max_length):
        super().__init__()

        self.dataset = dataset

        self.dataset['News'] = self.dataset['News'].str.lower()

        self.dataset['Sentiment'] = dataset['Sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})

        self.embedding = w2v

        self.embedding_size = embedding_size
        
        self.max_length = max_length

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        tokenized_sentence = word_tokenize(self.dataset.loc[item, 'News'])
        try:
            sentence = [self.embedding[token] for token in tokenized_sentence]
        except:
            sentence = [self.embedding.get_word_vector(token) for token in tokenized_sentence]
            
        n = len(sentence)

        y = torch.zeros(3)
        y[self.dataset.loc[item, 'Sentiment']] = 1
        y = y.unsqueeze(0)

        x = torch.FloatTensor(sentence).reshape((n, self.embedding_size))
        x = F.pad(x, (0, 0, self.max_length - n, 0), 'constant', 0)
        return x, y


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 4), 3)
        )

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)  # x corresponds to the hidden state ath the end of the sequence

        x = self.classifier(x)

        return F.softmax(x, dim=-1), hidden


if __name__ == "__main__":
    pass
