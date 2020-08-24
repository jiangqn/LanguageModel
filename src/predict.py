import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
import pandas as pd
from src.constants import SOS, EOS
from src.eval import get_ppl

def predict(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'language_model.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')
    annotated_save_path = os.path.join(base_path, 'annotated_train.tsv')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    train_data = TabularDataset(path=os.path.join(base_path, 'train.tsv'),
                                format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    train_iter = Iterator(train_data, batch_size=config['batch_size'], shuffle=False, device=device)

    model = torch.load(save_path)

    ppl = get_ppl(model, train_iter)

    df = pd.read_csv(os.path.join(base_path, 'train.tsv'), delimiter='\t')
    df['ppl'] = ppl

    df.to_csv(annotated_save_path, sep='\t')