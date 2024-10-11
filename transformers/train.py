import os
import spacy
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator


def yield_tokens(datasets, tokenizer):

    data_all = []
    for d in datasets:
        data_all += d
    
    for d in data_all:
        yield tokenizer[d]


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


def train():

    dataset = load_dataset("bentrevett/multi30k")

    data_en = [dataset[k]['en'] for k in ("train", "validation", "test")]
    data_de = [dataset[k]['de'] for k in ("train", "validation", "test")]

    tokenizer_en, tokenizer_de = load_tokenizers()

    vocab_en = build_vocab_from_iterator(
        yield_tokens(data_en, tokenizer_en),
        min_freq=2, 
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_de = build_vocab_from_iterator(
        yield_tokens(data_de, tokenizer_de),
        min_freq=2, 
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_de.set_default_index(vocab_de["<unk>"])

    