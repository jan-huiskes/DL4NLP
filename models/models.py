
import torch.nn as nn
import torch

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



class LSTM(nn.Module):


    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, batch_size = 10,
                 lstm_num_layers=1, pad_idx=None, embedding=None):
        super().__init__()

        self.batch_size = batch_size
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        else:
            self.embedding = embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)





    def forward(self, text, hidden_tuple = None):

        #text = [sent len, batch size]
        text = text.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        out, (hidden_out, cell_out) = self.lstm(embedded, hidden_tuple)
        out = self.fc(hidden_out[-1])
        #out = [batch size, out_dim]
        return out
