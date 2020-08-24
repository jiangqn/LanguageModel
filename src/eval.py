import torch
from torch import nn
from torchtext.data import Iterator
from typing import List
from src.constants import PAD_INDEX

def get_ppl(model: nn.Module, data_iter: Iterator) -> List[float]:

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='none')

    ppl = []

    model.eval()
    with torch.no_grad():

        for i, batch in enumerate(data_iter):

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            output_sentence = output_sentence.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, output_sentence)
            loss = loss.reshape(batch_size, -1)
            output_sentence = output_sentence.reshape(batch_size, -1)
            mask = (output_sentence != PAD_INDEX).float()
            output_sentence_lens = mask.sum(dim=1, keepdim=False)
            batch_ppl = torch.pow(2, (loss * mask).sum(dim=1, keepdim=False) / output_sentence_lens)
            ppl.extend(batch_ppl.tolist())

            if i % 100 == 0:
                print(i)

    return ppl

def eval_language_model(model, data_iter, criterion):

    total_tokens = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            output_sentence = output_sentence.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, output_sentence)

            mask = (output_sentence != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()

    loss = total_loss / total_tokens
    return loss