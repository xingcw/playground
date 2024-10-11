import os
import spacy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from transformerX import TransformerX
from utils import compute_learing_rate, LabelSmoothing, create_mask


def tokenize_data(example, tokenizer_en, tokenizer_de, bos_token, eos_token, max_length=10000):
    
    en_tokens = [bos_token] + [token.text for token in tokenizer_en.tokenizer(example['en'])][:max_length] + [eos_token]
    de_tokens = [bos_token] + [token.text for token in tokenizer_de.tokenizer(example['de'])][:max_length] + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}


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


def tokenize(example, vocab_en, vocab_de):

    return {
        "en_ids": [vocab_en[x] for x in example["en_tokens"]],
        "de_ids": [vocab_de[x] for x in example["de_tokens"]],
    }


def get_collate_fn(pad_idx=0, max_len=128):

    def collate_fn(batch_dict):

        padded_batch_en_ids = [x['en_ids'] + [pad_idx] * (max_len - len(x['en_ids'])) for x in batch_dict]
        padded_batch_de_ids = [x['de_ids'] + [pad_idx] * (max_len - len(x['de_ids'])) for x in batch_dict]

        return {
            "en_ids": torch.LongTensor(padded_batch_en_ids),
            "de_ids": torch.LongTensor(padded_batch_de_ids)
        }
    return collate_fn


def get_dataloader(dataset, batch_size, max_len=128, shuffle=False, pad_idx=0):

    collate_fn = get_collate_fn(pad_idx, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    return dataloader


def train():

    dataset = load_dataset("bentrevett/multi30k")

    tokenizer_en, tokenizer_de = load_tokenizers()

    max_length = 100000
    bos_token = "<bos>"
    eos_token = "<eos>"

    fn_kwargs = {
        "tokenizer_en": tokenizer_en,
        "tokenizer_de": tokenizer_de,
        "max_length": max_length,
        "bos_token": bos_token,
        "eos_token": eos_token,
    }

    train_dataset = dataset['train'].map(tokenize_data, fn_kwargs=fn_kwargs)
    valid_dataset = dataset['validation'].map(tokenize_data, fn_kwargs=fn_kwargs)
    test_dataset = dataset['test'].map(tokenize_data, fn_kwargs=fn_kwargs)

    vocab_en = build_vocab_from_iterator(train_dataset["en_tokens"] + valid_dataset["en_tokens"] + test_dataset["en_tokens"])
    vocab_de = build_vocab_from_iterator(train_dataset["de_tokens"] + valid_dataset["de_tokens"] + test_dataset["de_tokens"])

    fn_kwargs = {"vocab_en": vocab_en, "vocab_de": vocab_de}

    train_data = train_dataset.map(tokenize, fn_kwargs=fn_kwargs)
    valid_data = valid_dataset.map(tokenize, fn_kwargs=fn_kwargs)
    test_data = test_dataset.map(tokenize, fn_kwargs=fn_kwargs)

    max_train_len = max([max(len(x['en_ids']), len(x['de_ids'])) for x in train_data])
    max_eval_len = max([max(len(x['en_ids']), len(x['de_ids'])) for x in valid_data])
    
    train_dataloader = get_dataloader(train_data, batch_size=8, max_len=max_train_len, shuffle=True)
    eval_dataloader = get_dataloader(valid_data, batch_size=8, max_len=max_eval_len)

    num_eps = 200
    embed_dim = 512
    padding_idx = 0
    vocab_dec_size = len(vocab_de)
    model = TransformerX(
        num_layer=6, 
        num_head=8, 
        emd_dim=embed_dim, 
        vocab_enc_size=len(vocab_en),
        vocab_dec_size=len(vocab_de)
    )

    optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothing(vocab_dec_size, padding_idx, smoothing=0.1)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: compute_learing_rate(step, embed_dim, factor=1.0),
    )

    for ep in range(num_eps):
        model.train()
        total_loss = 0.0
        num_steps = 0

        for i, batch_dict in enumerate(train_dataloader):
            optimizer.zero_grad()

            x_mask, y_mask, target, label = create_mask(batch_dict['en_ids'], batch_dict['de_ids'], padding_idx)
            pred = model(batch_dict['en_ids'], target, x_mask, y_mask)
            logits = model.generator(pred)
            loss = criterion(logits.contiguous().view(-1, vocab_dec_size), label.contiguous().view(-1))

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
                loss = criterion(logits.contiguous().view(-1, vocab_dec_size), batch_dict['label'].contiguous().view(-1))

                eval_loss += loss.item()
                num_eval_steps += 1

            print(f"Epoch: {ep:2d} | Test Loss: {(eval_loss / float(num_eval_steps)):6.2f} ")


if __name__ == "__main__":

    train()