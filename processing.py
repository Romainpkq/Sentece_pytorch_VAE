# read the data, the format of the data is .txt
import torch
import torch.nn as nn
import numpy as np


def read_dataset(path):
    """
    Return a list of list which contains the words, mainly is a .txt file
    """
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.lower().strip().split()
            sentences.append(line)

    return sentences


# construct a dataset
class OurDataset(nn.Module):
    """
    Construct our dataset.
    """

    def __init__(self, path, max_output_len, min_words):
        """
        :path: the dir of the dataset
        :max_output_len: the max length of the example
        :min_words: the min_words that count the data.
        """
        self.sentences = read_dataset(path)
        self.word2int = self._get_word2int(self.sentences, min_words)
        self.max_output_len = max_output_len

    def _get_word2int(self, sentences, mins):
        word2int1 = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        vocab = {}

        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        for word, counts in vocab.items():
            if counts >= mins:
                word2int1[word] = len(word2int1)

        return word2int1

    def get_vocab_size(self):
        return len(self.word2int)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        example = self.sentences[index]
        example = ["<BOS>"] + example

        train_data = [self.word2int.get(word, 1) for word in example]

        if len(train_data) > self.max_output_len - 1:
            train_data = train_data[:(self.max_output_len - 1)]
            train_data.append(3)
            # print('train-data:', len(train_data))
        else:
            train_data = train_data + [0 for i in range(self.max_output_len - 1 - len(train_data))]
            train_data.append(3)

        # print(train_data)
        assert len(train_data) == self.max_output_len
        train_data = np.asarray(train_data)
        train_data = torch.LongTensor(train_data)

        return train_data


# get the data from the dataset loader
def infinite_iter(data_loader):
  it = iter(data_loader)
  while True:
    try:
      ret = next(it)
      yield ret
    except StopIteration:
      it = iter(data_loader)

