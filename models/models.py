
import torch.nn as nn
import torch


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



    def forward(self, text):

        text = text.permute(1, 0)
        embedded = self.embedding(text)

        embedded = embedded.unsqueeze(1)
        if self.combine:
            gloved = self.embedding_glove(text).unsqueeze(1)
            embedded = torch.cat((embedded, gloved), dim = 1)
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

class Encoder(nn.Module):
    def __init__(self, emb_size = 300, hidden = 200):
        super().__init__()


        self.lstm = nn.LSTM(emb_size, hidden, batch_first=False, bidirectional=True)

    def forward(self, x_emb, pads, z = None):
        x = x_emb.to(dtype=torch.float)
        if z is not None:
            z_mask = pads.float() * z.unsqueeze(-1).float()

            x = x * z_mask.unsqueeze(-1)
        #packed_sequence = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, (hx, cx) = self.lstm(x)
        #outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        final = torch.cat([hx[-2], hx[-1]], dim=-1)
        return outputs, final

class ZGenerator(nn.Module):
    def __init__(self, embedding = None, hidden = 200):
        super().__init__()
        self.embedding = embedding
        emb_size = embedding.weight.shape[1]
        enc_size = hidden * 2

        self.encoder = Encoder( emb_size, hidden)
        self.z_layer = BernoulliGate(enc_size)
        self.z_dist = None

    def forward(self, x, pads):
        lengths = pads.long().sum(1)
        emb = self.embedding(x)  # [B, T, E]
        h, _ = self.encoder(emb, pads, lengths)

        # compute parameters for Bernoulli p(z|x)
        z_dist = self.z_layer(h)
        self.z_dist = z_dist
        if self.training:  # sample
            z = z_dist.sample()  # [B, T, 1]
        else:  # deterministic
            z = (z_dist.probs >= 0.5).float()   # [B, T, 1]

        z = z.squeeze(-1)  # [B, T, 1]  -> [B, T]
        z = torch.where(pads, z, z.new_zeros([1]))

        #self.z = z
        #self.z_dists = [z_dist]

        return z



class Rationalisation_model(nn.Module):

    def __init__(self, vocab_size, embedding_dim=300, model = "CNN",hidden_dim=128, output_dim=3, n_filters = 50,
                 filters = [2,3,4], lstm_num_layers = 1,  batch_size = 10, dropout = 0, pad_idx=1, embedding=None, combine = False, lambda_1 = 0.0003, lambda_2 =2, criterion = None ):
        super().__init__()

        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        if model == "CNN":
            self.encoder = CNN( embedding_dim = embedding_dim, vocab_size=vocab_size, n_filters=n_filters, output_dim=output_dim, in_channels=1,
                               filters = filters, embedding=self.embedding, combine = combine)
        elif model == "LSTM":
            self.encoder =  LSTM(vocab_size=vocab_size, embedding=self.embedding, hidden_dim=hidden_dim, output_dim = output_dim,
                               batch_size=batch_size, lstm_num_layers = lstm_num_layers, combine = combine, dropout=dropout)
        elif model == "Bert":
            raise NotImplementedError
        self.generator = ZGenerator(embedding=self.embedding, hidden=hidden_dim,)
        self.criterion = criterion
        self.z = 0
    def forward(self, x):


        mask = (x != 1)  # [B,T]

        z = self.generator(x, mask)
        self.z = z
        z_mask = (mask.float() * z).to(dtype=torch.long) # [B, T]
        masked_x = x * z_mask
        print(masked_x.size())
        y = self.encoder(masked_x)

        return y

    def get_loss(self, y_pred, y_true, mask = None, soft = False, weights = None, device = None):
        """
        Get loss of this NN
        """
        if soft:
            loss_vec = self.criterion(y_pred, y_true, weights, device)
        else:
            loss_vec = self.criterion(y_pred, y_true)
        loss = loss_vec.mean()

        z = self.z.squeeze()

        logp_z0 = self.generator.z_dist.log_prob(0.).squeeze(2)
        logp_z1 = self.generator.z_dist.log_prob(1.).squeeze(2)

        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        zsum = z.sum(1)
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)



        cost_vec = loss_vec.detach() + zsum * self.lambda_1 + zdiff * self.lambda_2
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward


        # pred diff doesn't do anything if only 1 aspect being trained
        pred_diff = (y_pred.max(dim=1)[0] - y_pred.min(dim=1)[0])
        pred_diff = pred_diff.mean()

        # generator cost
        cost_g = cost_logpz

        # encoder cost
        cost_e = loss

        main_loss = cost_e + cost_g
        return main_loss