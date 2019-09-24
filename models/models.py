
import torch.nn as nn
import torch


# https://github.com/bastings/interpretable_predictions/tree/master/latent_rationale
# HERE IS SOME CODE FROM THAT LINK
class BernoulliGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1):
        super(BernoulliGate, self).__init__()

        self.layer = Sequential(
            Linear(in_features, out_features, bias=True)
        )

    def forward(self, x):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        dist = Bernoulli(logits=logits)
        return dist

    def get_loss(dist, sparsity = 0.0003, coherent_factor = 2.):
        """
        Get loss of this NN
        """

        z = dist.sample()
        z = z.squeeze()

        logp_z0 = dist.log_prob(0.).squeeze(2)  # [B,T], log P(z = 0 | x)
        logp_z1 = dist.log_prob(1.).squeeze(2)  # [B,T], log P(z = 1 | x)

        logpz = torch.where(z == 0, logp_z0, logp_z1)

                # sparsity regularization
        zsum = z.sum(1)  # [B]
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)  # [B]


        cost_vec =  zsum * sparsity + zdiff * coherent_factor
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward
        return cost_logpz


class CNN(nn.Module):


    def __init__(self, vocab_size, embedding_dim=128, n_filters=50, in_channels=1, filter_sizes=[2,3,4], output_dim=3,
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
        if self.combine:
            gloved = self.embedding_glove(text).unsqueeze(1)

            embedded = torch.cat((embedded, gloved), dim = 1)
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
                 lstm_num_layers=1, pad_idx=None, embedding=None, combine = False):
        super().__init__()

        self.batch_size = batch_size
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        else:
            self.embedding = embedding

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_num_layers, batch_first=True)
        if combine:
            self.combine = True
            self.embedding_glove = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
            self.lstm = nn.LSTM(embedding_dim*2, hidden_dim, num_layers=lstm_num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)





    def forward(self, text, hidden_tuple = None):

        #text = [sent len, batch size]
        text = text.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)

        if self.combine:
            gloved =self.embedding_glove(text)
            embedded = torch.cat((embedded, gloved), dim = 2)

        #embedded = [batch size, sent len, 2xemb dim]
        out, (hidden_out, cell_out) = self.lstm(embedded, hidden_tuple)
        out = self.fc(hidden_out[-1])
        #out = [batch size, out_dim]
        return out
