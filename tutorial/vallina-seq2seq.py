"""
The program is split into multiple blocks,
you can execute the program at once, or execute
it block by block if you are already installed
jupyter-related package.

The purpose of this program is to demostrate
how to build a vallina sequence to sequence model
from scratch to solve translation problem
from German to English.

The program is based on the paper,
Sequence to Sequence Learning with Neural Networks
(https://arxiv.org/abs/1409.3215), and the implementation
is based on this tutorial (https://github.com/bentrevett/pytorch-seq2seq).
"""


#%%
import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field
import spacy


# %%
"""
load spacy model according to the language 
used in dataset (English and German)
"""
spacy_de = spacy.load("de_core_news_sm")
space_en = spacy.load("en_core_web_sm")


# %%
"""
tokenize sentence into list of tokens
"""

def tokenize_de(text: str) -> list[str]:
    """
    why reverse the order of input text,
    you may refer to this article (https://stackoverflow.com/questions/51003992/why-do-we-reverse-input-when-feeding-in-seq2seq-model-in-tensorflow-tf-reverse).
    """
    tokens = spacy_de.tokenizer(text)
    return [tok.text for tok in tokens][::-1]

def tokenize_en(text: str) -> list[str]:
    tokens = space_en.tokenizer(text)
    return [tok.text for tok in tokens]


# %%
"""
pytoch's Field is like a text processor,
we create two Field for source text and target text.

With torchtext's Field, we convert a sentence into
a list of tokens, insert <sos> token at the beginning of list,
and append <eos> at the end.

Finally, we convert all the tokens in list to lowercase.
"""

src_field = Field(
    tokenize = tokenize_de,
    init_token = '<sos>',
    eos_token = '<eos>',
    lower = True
)

trg_field = Field(
    tokenize = tokenize_en,
    init_token = '<sos>',
    eos_token = '<eos>',
    lower = True
)


# %%
"""
The dataset we use is Multi30k which is
imported from torchtext datasets.

Use exts = (src, trg) to specify source and target data,
and fields to process dataset.
"""
train_data, valid_data, test_data = Multi30k.splits(
    exts = ('.de', '.en'),
    fields = (src_field, trg_field)
)

print(f'number of training example: {len(train_data)}')
print((f'number of validation example: {len(valid_data)}'))
print((f'number of testing example: {len(test_data)}'))

print(f'\n#1 example in traning examples: ')
print(f'src: {train_data.examples[0].src}')
print(f'trg: {train_data.examples[0].trg}')


# %%
"""
Instead of using word as token, we have to build vocabulary
table to convert word token into integer token.

Source language and target language have their own vocabulary
table, and we only allow the word tokens appear at least 2 times
to allow in vocabulary table.
"""

src_field.build_vocab(train_data, min_freq = 2)
trg_field.build_vocab(train_data, min_freq = 2)

print(f'number of unique tokens in source (de) language: {len(src_field.vocab)}')
print(f'number of unique tokens in target (en) language: {len(trg_field.vocab)}')


# %%

