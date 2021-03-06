import torch
from torch import nn
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
from src.eval import eval_language_model
from src.constants import SOS, EOS, PAD_INDEX

def test_language_model(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'language_model.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]

    test_data = TabularDataset(path=os.path.join(base_path, 'test.tsv'),
                                format='tsv', skip_header=True, fields=fields)
    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)
    TEXT.vocab = vocab

    device = torch.device('cuda:0')
    test_iter = Iterator(test_data, batch_size=config['batch_size'], shuffle=False, device=device)

    model = torch.load(save_path)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    test_loss = eval_language_model(model, test_iter, criterion)
    print('test_loss: %.4f\ttest_ppl: %.4f' % (test_loss, 2 ** test_loss))