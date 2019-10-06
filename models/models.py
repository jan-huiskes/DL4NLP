
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNN(nn.Module):


    def __init__(self, vocab_size, embedding_dim=300, n_filters=50, in_channels=1, filters=[2,3,4], output_dim=3,
                 pad_idx=None, embedding=None, combine = False):
        super().__init__()

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        else:
            self.embedding = embedding
        self.combine = combine
        if combine:
            in_channels = 2
            self.embedding_glove = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=n_filters,
                      kernel_size=(filter_size, embedding_dim))
            for filter_size in filters
        ])
        self.fc = nn.Linear(len(filters) * n_filters, output_dim)



    def forward(self, text, z = None):

        text = text.permute(1, 0)
        mask = (text!=1)
        embedded = self.embedding(text)

        embedded = embedded.unsqueeze(1)
        if self.combine:
            gloved = self.embedding_glove(text).unsqueeze(1)
            embedded = torch.cat((embedded, gloved), dim = 1)
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(1).unsqueeze(-1)

            embedded = embedded * z_mask

        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = torch.cat(pooled, dim = 1)

        return self.fc(cat)



class LSTM(nn.Module):


    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, output_dim=3, batch_size = 10,
                 lstm_num_layers=1, pad_idx=None, embedding=None, combine = False, dropout=0):
        super().__init__()

        self.combine = combine
        self.batch_size = batch_size
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        else:
            self.embedding = embedding

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_num_layers, batch_first=True, dropout=dropout)
        if combine:
            self.combine = True
            self.embedding_glove = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
            self.lstm = nn.LSTM(embedding_dim*2, hidden_dim, num_layers=lstm_num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, text, hidden_tuple = None, z=None):
        text = text.permute(1, 0)
        mask = (text != 1)
        embedded = self.embedding(text)
        if self.combine:
            gloved =self.embedding_glove(text)
            embedded = torch.cat((embedded, gloved), dim = 2)
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)
            embedded = embedded * z_mask
        out, (hidden_out, cell_out) = self.lstm(embedded, hidden_tuple)
        out = self.fc(hidden_out[-1])

        return out


class BernoulliGate(nn.Module):

    def __init__(self, in_features, out_features=1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True)
        )

    def forward(self, x):
        logits = self.layer(x)
        dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
        return dist

class ZEncoder(nn.Module):
    def __init__(self,embedding =None, emb_size = 300, hidden = 200):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(emb_size, hidden, batch_first=True, bidirectional=True)

    def forward(self, x, pads, lengths):
        x = x.permute(1,0)
        emb = self.embedding(x)
        emb = emb.to(dtype=torch.float)
        #packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hx, cx) = self.lstm(emb)
        #outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, 1

class ZGenerator(nn.Module):
    def __init__(self, embedding = None, hidden = 200):
        super().__init__()
        emb_size = embedding.weight.shape[1]
        enc_size = hidden * 2

        self.encoder = ZEncoder( embedding , emb_size, hidden)
        self.bernoulli = BernoulliGate(enc_size)
        self.z_dist = None
        self.z=0
    def forward(self, x, no_pads):
        lengths = no_pads.long().sum(0)
        h, _ = self.encoder(x, no_pads, lengths)
        z_dist = self.bernoulli(h)
        self.z_dist = z_dist
        z = z_dist.sample()
        z = z.squeeze(-1)
        z = torch.where(no_pads.t(), z, z.new_zeros(z.size()))
        self.z=z
        return z



class Rationalisation_model(nn.Module):

    def __init__(self, vocab_size, embedding_dim=300, model = "CNN",hidden_dim=128, output_dim=3, n_filters = 50,
                 filters = [2,3,4], lstm_num_layers = 2,  batch_size = 10, dropout = 0, pad_idx=None, embedding=None, combine = False, lambda_1 = 0.0001, lambda_2 =0.0001, criterion = None ):
        super().__init__()

        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        if model == "CNN":
            self.encoder = CNN( embedding_dim = embedding_dim, vocab_size=vocab_size, n_filters=n_filters, output_dim=output_dim, in_channels=1,
                               filters = filters, embedding=self.embedding, combine = combine, pad_idx=pad_idx)
        elif model == "LSTM":
            self.encoder =  LSTM(vocab_size=vocab_size, embedding=self.embedding, hidden_dim=hidden_dim, output_dim = output_dim,
                               batch_size=batch_size, lstm_num_layers = lstm_num_layers, combine = combine, dropout=dropout, pad_idx=pad_idx)
        elif model == "Bert":
            raise NotImplementedError
        self.generator = ZGenerator(embedding=self.embedding, hidden=hidden_dim)
        self.criterion = criterion

    def forward(self, x, sample = False):
        mask = (x != 1)

        z = self.generator(x,mask)
        y = self.encoder(x, z=z)

        #for print/sampling purposes
        z = z.to(dtype=torch.uint8, device= "cuda:0").t()
        z_mask = (mask* z)
        masked_x = torch.where(z_mask, x, torch.ones(x.size(), device = "cuda:0").long())
        if not sample:
            return y
        else:
            return z_mask, masked_x, y, z

    def get_loss(self, y_pred, y_true, mask = None, soft = False, weights = None, device = None):
        """
        Get loss of this NN
        """
        if soft:
            loss_vec = self.criterion(y_pred, y_true, weights, device)
        else:
            loss_vec = self.criterion(y_pred, y_true)
        loss = loss_vec.mean()

        z = self.generator.z.squeeze()

        logp_z0 = self.generator.z_dist.log_prob(0.).squeeze(2)
        logp_z1 = self.generator.z_dist.log_prob(1.).squeeze(2)

        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask.t(), logpz, logpz.new_zeros([1]))

        zsum = z.sum(1)
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)

        cost_vec =  zsum * self.lambda_1 + zdiff * self.lambda_2 #+loss_vec.detach()
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)
        total_loss = cost_logpz #+loss
        return total_loss