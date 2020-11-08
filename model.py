import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
print(device)

# construct the VAE model, the model has three parts:
# 1. encoder: a LSTM, and two linear layer to get the mean and the variance
# 2. decoder: a LSTM to treat the mean and variance to get the words
# 3. A VAE connect the two model


class Encoder(nn.Module):
    """ To encode the information of all the sentence.
    It contains a embedding layer, LSTM and two linear layer to get the mean and the variance
    """

    def __init__(self, vocab_size, embed_size, hid_dim, num_layers, dropout, linear_dim, batch_first=True,
                 bidirectional=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.linear_dim = linear_dim
        print(self.linear_dim)
        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        print(self.linear_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embed_size, self.hid_dim, self.num_layers, bidirectional=bidirectional)
        self.l1 = nn.Linear(2 * self.hid_dim, self.linear_dim)
        self.l2 = nn.Linear(2 * self.hid_dim, self.linear_dim)

    def forward(self, input):
        """
        input.shape: (batch_size, seq_length)
        """
        embed = self.embedding(input)  # (batch_size, seq_length, embed_size)
        output, (hn, cn) = self.lstm(self.dropout(embed))  # output: (batch_size, seq_len, hid_dim * 2)  hn: (batch, num_layers * 2, hid_dim) cn: (batch, num_layers * 2, hid_dim)
        sentence_embedding = torch.mean(output, 1)  # sentence_embedding: (batch_size, hid_dim * 2)
        mean = self.l1(sentence_embedding)  # (batch_size, linear_dim)
        variance = self.l2(sentence_embedding)  # (batch_size, linear_dim)
        return mean, variance


class Decoder(nn.Module):
    """
    用一个LSTM作为解码器，使之能够推测
    """

    def __init__(self, vocab_size, embed_dim, hid_dim, context_dim, num_layers, dropout, batch_first=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim + self.context_dim, self.hid_dim, self.num_layers)
        self.linear = nn.Linear(self.hid_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, input):
        """
        context: (batch_size, hid_dim)
        input: (batch_size, seq_len)
        """
        embed = self.embedding(input)  # (batch_size, seq_length, embed_dim)
        seq_len = embed.shape[1]
        context = torch.unsqueeze(context, 1)  # (batch_size, 1, hid_dim)
        context = context.repeat(1, seq_len, 1)
        embed = torch.cat([embed, context], dim=2)
        output, (h_n, c_n) = self.lstm(self.dropout(embed))  # (batch_size, seq_len, hid_dim)
        output = self.linear(output)  # (batch_size, seq_len, vocab_size)

        return output


class VAE(nn.Module):
    """
    Construct a complete VAE.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        mean, variance = self.encoder(input)
        context = torch.normal(mean, variance)  # (batch_size, hid_dim)
        output = self.decoder(context, input)
        return output


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4, 5], [2, 1, 3, 4, 2]])
    a = torch.from_numpy(a)
    encoder = Encoder(100, 200, 200, 1, 0.6, 40)
    decoder = Decoder(100, 200, 200, 40, 1, 0.6)
    vae = VAE(encoder, decoder)
    output = vae(a)
    print(output)
    print(output.shape)
