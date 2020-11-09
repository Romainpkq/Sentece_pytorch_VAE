# utils: 储存模型，载入模型，建构模型，将一连串的数字还原成句子
import torch
from model import *


def save_model(model, store_model_path, step):
  torch.save(model.state_dict(), f'{sotre_model_path}/model_{step}.ckpt')
  return


def load_model(model, load_model_path):
  print(f"Load model from {load_model_path}")
  model.load_state_dict(torch.load(f'{load_model_path.ckpt}'))
  return model


def build_model(config, vocab_size):
  encoder = Encoder(vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.encoder_linear_dim)
  decoder = Decoder(vocab_size, config.emb_dim, config.hid_dim, config.linear_dim, config.decoder_num_layers, config.dropout)
  model = VAE(encoder, decoder)
  print(model)
  # 构建optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  print(optimizer)
  if config.load_model:
    model = load_model(model, config.load_model_path)
  model = model.to(device)

  return model, optimizer


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if token == '<EOS>':
                break
            sentences.append(sentence)

    return sentences
