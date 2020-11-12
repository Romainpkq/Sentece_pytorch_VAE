import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
import random
from processing import OurDataset, infinite_iter

# use the gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.set_device(0)

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
        """
        :param vocab_size: the size of the vocabulary, int
        :param embed_dim: the dimension of the embedding. int
        :param hid_dim: the hidden dimension of the LSTM
        :param num_layers: the number of layers, int
        :param dropout: the value of dropout. (0, 1)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.linear_dim = linear_dim
        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        # print(self.linear_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embed_size, self.hid_dim, self.num_layers, batch_first=batch_first,
                            bidirectional=bidirectional)
        if bidirectional:
            self.l1 = nn.Linear(2 * self.hid_dim, self.linear_dim)
            self.l2 = nn.Linear(2 * self.hid_dim, self.linear_dim)
        else:
            self.l1 = nn.Linear(self.hid_dim, self.linear_dim)
            self.l2 = nn.Linear(self.hid_dim, self.linear_dim)

    def forward(self, input):
        """
        input.shape: (batch_size, seq_length)
        """
        embed = self.embedding(input)  # (batch_size, seq_length, embed_size)
        output, (hn, cn) = self.lstm(self.dropout(embed))  # output: (batch_size, seq_len, hid_dim * bidirectional)  hn: (batch, num_layers * 2, hid_dim) cn: (batch, num_layers * 2, hid_dim)
        # print("encoder output.shape:", output.shape)
        sentence_embedding = torch.mean(output, 1)  # sentence_embedding: (batch_size, hid_dim * 2)
        mean = self.l1(sentence_embedding)  # (batch_size, linear_dim)
        variance = self.l2(sentence_embedding)  # (batch_size, linear_dim)
        return mean, variance


class Decoder(nn.Module):
    """
    用一个LSTM作为解码器，使之能够推测
    """

    def __init__(self, vocab_size, embed_dim, hid_dim, context_dim, num_layers, dropout, batch_first=True):
        """
        :param vocab_size: the size of the vocabulary, int
        :param embed_dim: the dimension of the embedding. int
        :param hid_dim: the hidden dimension of the LSTM
        :param context_dim: the sample result from the Gaussian distribution
        :param num_layers: the number of layers, int
        :param dropout: the value of dropout. (0, 1)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim + self.context_dim, self.hid_dim, self.num_layers, batch_first=batch_first)
        self.linear = nn.Linear(self.hid_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, input, h_0, c_0):
        """
        context: (batch_size, 1, linear_dim)
        input: (batch_size)
        """
        # batch_size = input.shape[0]
        input = input.unsqueeze(1)
        # seq_length = 1

        embed = self.embedding(input)  # (batch_size, 1, embed_dim)

        embed = torch.cat([embed, context], dim=2)  # (batch_size, 1, hid_dim + context_dim)
        output, (h_n, c_n) = self.lstm(self.dropout(embed), (h_0, c_0))  # (batch_size, 1, hid_dim)
        # print("output.shape:", output.shape)
        output = self.linear(output)  # (batch_size, 1, vocab_size)

        return output, h_n, c_n


class VAE(nn.Module):
    """
    Construct a complete VAE.
    """

    def __init__(self, encoder, decoder):
        """
        encoder: Encoder()
        decoder: Decoder()
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_force_ratio):
        """
        input: shape: (batch_size, seq_len)
        target: shape； (batch_size, seq_len)
        """
        mean, variance = self.encoder(input)  # mean: (batch_size, linear_dim)  variance: (batch_size, linear_dim)
        std = torch.exp(0.5 * variance)
        batch_size, target_len, vocab_size = target.shape[0], target.shape[1], self.decoder.vocab_size

        context = mean + torch.rand_like(mean) * std  # (batch_size, linear_dim)
        context = context.unsqueeze(dim=1)  # (batch_size, 1, linear_dim)

        # 准备一个存储空间来存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size)

        # 取<BOS> token
        input1 = target[:, 0]  # input1: (batch_size)

        preds = []
        h_0 = torch.zeros((self.decoder.num_layers, batch_size, self.decoder.hid_dim)).to(device)
        c_0 = torch.zeros((self.decoder.num_layers, batch_size, self.decoder.hid_dim)).to(device)
        for i in range(1, target_len):
            output, h_0, c_0 = self.decoder(context, input1, h_0, c_0)

            output1 = output.squeeze(1)
            outputs[:, i] = output1  # (batch_size, 1, vocab_size)
            # 决定是否使用正确的答案来训练
            teacher_force = random.random() <= teacher_force_ratio
            # 取出几率最大的单词
            top1 = output.argmax(2)  # (batch_size)
            # print('top1:', top1)

            if teacher_force:
                input1 = target[:, i]
                # input1 = input1.unsqueeze(1)
                # print("input1.shape:", input1.shape)
            else:
                input1 = top1.squeeze(1)

            preds.append(top1)  # (batch_size, 1)

        preds = torch.cat(preds, 1)

        return mean, variance, outputs, preds

    def inference(self, input, target):
        """
        input: shape: (batch_size, seq_len)
        target: shape； (batch_size, seq_len)
        """
        mean, variance = self.encoder(input)
        # mean: (batch_size, linear_dim)  variance: (batch_size, linear_dim)
        std = torch.exp(0.5 * variance)
        batch_size, target_len, vocab_size = target.shape[0], target.shape[1], self.decoder.vocab_size

        context = mean + torch.rand_like(mean) * std  # (batch_size, linear_dim)
        context = context.unsqueeze(dim=1)  # (batch_size, 1, linear_dim)

        # 准备一个存储空间来存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size)

        # 取<BOS> token
        input1 = target[:, 0]  # input1: (batch_size)
        preds = []
        h_0 = torch.zeros((self.decoder.num_layers, batch_size, self.decoder.hid_dim))
        c_0 = torch.zeros((self.decoder.num_layers, batch_size, self.decoder.hid_dim))
        for i in range(1, target_len):
            # h_0, c_0 = h_0.permute(1, 0, 2), c_0.permute(1, 0, 2)
            output, h_0, c_0 = self.decoder(context, input1, h_0, c_0)
            output1 = output.squeeze(1)
            outputs[:, i] = output1  # (batch_size, 1, vocab_size)
            top1 = output.argmax(2)  # (batch_size)
            input1 = top1.squeeze(1)
            # print("top1.shape:", top1.shape)

            preds.append(top1)  # (batch_size, 1)

        preds = torch.cat(preds, 1)

        return outputs, preds


def loss_function(output, input, mean, variance):
  output = output.cuda()
  kl_loss = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
  batch_size = input.shape[0]
  reconstruction_loss = 0
  for i in range(batch_size):
    reconstruction_loss += nn.NLLLoss()(output[i, :, :], input[i, :])

  return (kl_loss + reconstruction_loss) / 2


if __name__ == '__main__':
    path1 = "./1.txt"
    data1 = OurDataset(path1, 7, 1)
    dataloader = data.DataLoader(data1, 2, shuffle=False)
    train_iter = infinite_iter(dataloader)

    encoder = Encoder(100, 200, 200, 1, 0.6, 40)
    decoder = Decoder(100, 200, 200, 40, 1, 0.6)
    vae = VAE(encoder, decoder)
    train_data = next(train_iter)
    output, preds = vae.inference(train_data, train_data)
    print(preds)

