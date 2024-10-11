import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from transformerX import TransformerX
from utils import create_mask, subsequent_mask, LabelSmoothing, compute_learing_rate
    

def data_gen(vocab_size, batch_size, num_batch, padding_idx=0):
    
    for i in range(num_batch):

        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        x_mask, y_mask, target, label = create_mask(src, tgt, padding_idx)

        batch_dict = {
            "x": src, "y": target, "x_mask": x_mask, "y_mask": y_mask, "label": label
        }
        
        yield batch_dict


def decode(model, src, src_mask, max_len):
        
        enc = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).type_as(src)

        for _ in range(max_len - 1):
            dec_mask = subsequent_mask(ys.size(1)).type_as(src)
            # print(dec_mask)
            dec = model.decode(ys, enc, src_mask, dec_mask)
            logits = model.generator(dec[:, -1])
            _, next_word = torch.max(logits, dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=-1)

        print("Example Untrained Model Prediction:", ys)

        return ys


def train():

    num_eps = 200
    embed_dim = 512
    vocab_size = 10
    padding_idx = 0
    model = TransformerX(num_layer=6, num_head=8, emd_dim=embed_dim, vocab_size=vocab_size)

    optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothing(vocab_size, padding_idx, smoothing=0.1)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: compute_learing_rate(step, embed_dim, factor=1.0),
    )

    for ep in range(num_eps):
        model.train()
        total_loss = 0.0
        num_steps = 0

        train_dataloader = data_gen(vocab_size, batch_size=8, num_batch=100)
        eval_dataloader = data_gen(vocab_size, batch_size=8, num_batch=10)

        for i, batch_dict in enumerate(train_dataloader):
            optimizer.zero_grad()
    
            pred = model(batch_dict['x'], batch_dict['y'], batch_dict['x_mask'], batch_dict['y_mask'])
            logits = model.generator(pred)
            loss = criterion(logits.contiguous().view(-1, vocab_size), batch_dict['label'].contiguous().view(-1))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f"Epoch: {ep:2d} | Step: {i:6d} | Loss: {loss:6.2f} ")

            total_loss += loss.item()
            num_steps += 1

        print(f"Epoch: {ep:2d} | Train Loss: {(total_loss / float(num_steps)):6.2f} ")

        if ep % 10 == 0:
            model.eval()
            eval_loss = 0.0
            num_eval_steps = 0

            for i, batch_dict in enumerate(eval_dataloader): 

                pred = model(batch_dict['x'], batch_dict['y'], batch_dict['x_mask'], batch_dict['y_mask'])
                logits = model.generator(pred)
                loss = criterion(logits.contiguous().view(-1, vocab_size), batch_dict['label'].contiguous().view(-1))

                eval_loss += loss.item()
                num_eval_steps += 1

            print(f"Epoch: {ep:2d} | Test Loss: {(eval_loss / float(num_eval_steps)):6.2f} ")

    
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src_mask = torch.ones(1, 1, 10)
    pred = decode(model, src, src_mask, 10)


import os
import spacy
import torchtext.data as data
import torchtext.datasets as datasets
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocab():

    spacy_en, spacy_de = load_tokenizers()

    tokenize_en = lambda x: tokenize(x, spacy_en)
    tokenize_de = lambda x: tokenize(x, spacy_de)

    SRC = data.Field(
        tokenize = tokenize_de,
        lower= True,
        init_token = "<sos>",
        eos_token = "<eos>"
    )
    TRG = data.Field(
        tokenize = tokenize_en,
        lower= True,
        init_token = "<sos>",
        eos_token = "<eos>"
    )

    train, val, test = datasets.Multi30k.splits(
        exts=('.de', '.en'),
        fields = (SRC, TRG)
    )
    vocab = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2, 
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

if __name__ == "__main__":

    # train()
    build_vocab()