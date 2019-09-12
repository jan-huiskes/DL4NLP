import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn
import torch
import torch.optim as optim
sys.path.append('../utils')

from data_loader import Dataset


def split_dataset(dataset, test_percentage=0.1):
    """
    Split a dataset in a train and test set.

    Parameters
    ----------
    dataset : dataset.Data
        Custom dataset object.
    test_percentage : float, optional
        Percentage of the data to be assigned to the test set.
    """
    test_size = round(len(dataset) * test_percentage)
    train_size = len(dataset) - test_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

class CNN(nn.Module):


    def __init__(self, vocab_size, embedding_dim=128, n_filters=50, filter_sizes=[2,3,4], output_dim=3,
                  pad_idx=None, embedding=None):
        super().__init__()

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        else:
            self.embedding = embedding
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]

        text = text.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)

        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = torch.cat(pooled, dim = 1)
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)


def main():
    #some params
    test_percentage = 0.15
    val_percentage = 0.15
    batch_size= 10
    num_epochs = 100
    embedding_dim=300
    # load data
    dataset = Dataset("../data/cleaned_tweets_orig.csv", use_embedding="Random", embedd_dim=embedding_dim)
    train_data, test_data = split_dataset(dataset, test_percentage + val_percentage )
    test_data, val_data = split_dataset(test_data, 0.5 )

    def my_collate(batch):
        #y to be returned.
        new_batch = {"y": torch.Tensor().new_empty((0, 5))}
        #get labels and max length
        max_length = 0
        for x, y in batch:

            max_length = max(max_length, len(x))
            new_batch['y'] = torch.cat((new_batch['y'], y.view(-1,5)), 0)

        #get x tensors with the same length with shape [len, batch_size]
        new_batch["x"]= torch.Tensor().new_empty((max_length, 0))
        for x,y in batch:

            x = x.to(torch.float).permute(1,0)
            x = nn.functional.pad(x, (0, max_length - x.shape[1]), mode='constant', value=0).permute(1,0)
            new_batch['x'] = torch.cat((new_batch['x'], x), 1)

        new_batch['x'] = new_batch['x'].to(dtype=torch.long)

        return new_batch['x'] , new_batch['y']

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size , collate_fn= my_collate)
    vocab_size = len(dataset.vocab)
    model = CNN(vocab_size, embedding_dim)
    model.embedding.weight.data.copy_(dataset.vocab.vectors)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    def train_epoch(model, loader, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for batch in loader:
            x, y = batch[0], batch[1].to(torch.long)
            optimizer.zero_grad()
            predictions = model(x).squeeze(1)

            loss = criterion(predictions, y[:, 0])

            acc = criterion(predictions, batch.label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(loader), epoch_acc / len(loader)

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        break






"""
    conf_matrix = confusion_matrix(y_test, scores)
    class_report = classification_report(y_test, scores)

    print('\nCONFUSION MATRIX\n----------------\n')
    print(conf_matrix)

    print('\nCLASSSIFICATION REPORT\n----------------------\n')
    print(class_report)
"""

if __name__ == '__main__':

    main()
