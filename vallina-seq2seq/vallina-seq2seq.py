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
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
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
"""
At the final step of processing data,
we have to convert the dataset into batches of exampl.

We will create an Iterator which can be iterated on to 
return a batch of data. 

Additionally, we have make sure that all the sentences
are padded to the same length in on batch. (torchtext's Iterator
will handle this for us)

Instead of using default torchtext's Iterator, we use
"BucketIterator" (https://torchtext.readthedocs.io/en/latest/data.html#bucketiterator) 
which batches examples of similar length together
to minimize amount of padding needed in one batch.

In order to train model with gpu, we can move data (tensor) to gpu.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = 128,
    device = device
)


# %%
"""
Seq2Seq model consist of encoder and decoder.
Hence, we will create three model encdoer, decoder and seq2seq
model which encapsulates encdoer and decoder.
"""

"""
Encoder consists of following layers:
- embedding
- lstm_1
- lstm_2

We will use the final hidden state and final cell state
of each layer lstm to make context vector which will fed into decoder
to generate prediction.
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, debug=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.debug = debug

        """
        Input a batch of examples with shape, (batch_size, sequence_length),
        embedding layer gives each token a dense embedding vector, and output
        a tensor with shape, (batch_size, sequence_length, embedding_dim).

        input_dim: dimension of one-hot encoding of each token in source language
        emb_dim: dimension of embedding of each token
        """
        self.embedding = nn.Embedding(input_dim, emb_dim)

        """
        RNN consists of two-layers LSTM. In each time step, we feed a vector
        into LSTM with its previous cell state, it outputs two tensor, 
        hidden state and cell state.

        emb_dim: dimension of each token
        hid_dim: dimension of hidden state and cell state in LSTM
        n_layers: number of LSTM which will be stacked on top of each other

        In multi-layer LSTM, in every time step, hidden state output from 
        layer(i) LSTM is the input of layer(i+1) LSTM. Hence, the argument, 
        dropout, specifies how similar the hidden state output from layer(i) LSTM 
        and the input of layer(i+1) LSTM.
        """
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch):
        if self.debug:
            print(f'(in encoder\'s forward) input_batch shape = {input_batch.shape}')
        
        embeded = self.embedding(input_batch)

        if self.debug:
            print(f'(in encoder\'s forward) embeded shape = {embeded.shape}')

        embeded = self.dropout(embeded)

        """
        outputs: hidden state of top layer lstm in all time steps
        hidden_state: final hidden state of each layer lstm
        cell state: final cell state of each layer lstm
        """
        outputs, (hidden_state, cell_state) = self.rnn(embeded)

        if self.debug:
            print(f'(in encoder\'s forward) rnn output shape:')
            print(f'\toutputs = {outputs.shape}')
            print(f'\thidden_state = {hidden_state.shape}')
            print(f'\tcell_state = {cell_state.shape}')
        
        return hidden_state, cell_state


# %%
"""
Decoder consists of following layers:
- embedding
- lstm_1
- lstm_2

The composition and operation of decoder are similar to them
in encdoer. However, we use the final hidden state and cell state of
each layer lstm in encoder to initialize each layer lstm in decoder.

The hidden states of the top layer lstm in decoder in all time steps
are fed into a linear function to predict the new token.
"""
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, debug=False):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.debug = debug
        

        """
        output_dim: dimension of one-hot encoding of each token in target language
        emb_dim: dimension of embedding of each token
        """
        self.embedding = nn.Embedding(output_dim, emb_dim)

        """
        emb_dim: dimension of each token
        hid_dim: dimension of hidden state and cell state in LSTM
        n_layers: number of LSTM which will be stacked on top of another one
        """
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout = dropout)

        """
        feed hidden states of top layer lstm to linear function to prediction
        next token
        """
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_batch, initial_hidden, initial_cell):
        if self.debug:
            print(f'(in decoder\'s forward) input_batch shape = {input_batch.shape}')
        
        input_batch = input_batch.unsqueeze(0)

        if self.debug:
            print(f'(in decoder\'s forward) input_batch (after unsqueeze) shape = {input_batch.shape}')

        embeded = self.embedding(input_batch)
        embeded = self.dropout(embeded)

        if self.debug:
            print(f'(in decoder\'s forward) embeded shape = {embeded.shape}')

        
        outputs, (hidden_state, cell_state) = self.rnn(embeded, (initial_hidden, initial_cell))

        if self.debug:
            print(f'(in decoder\'s forward) rnn output shape:')
            print(f'\toutputs = {outputs.shape}')
            print(f'\thidden_state = {hidden_state.shape}')
            print(f'\tcell_state = {cell_state.shape}')

        outputs = outputs.squeeze(0)

        if self.debug:
            print(f'(in decoder\'s forward) outputs (after squeeze) shape = {outputs.shape}')
        
        prediction = self.fc_out(outputs)

        if self.debug:
            print(f'(in decoder\'s forward) prediction shape = {prediction.shape}')
        
        return prediction, hidden_state, cell_state


#%%
"""
try to feed some data to encdoer and decoder
"""
for batch in train_iterator:
    first_batch = batch
    break

first_batch_src = first_batch.src
first_batch_trg = first_batch.trg
print(f'first_batch_src shape = {first_batch_src.shape}')
print(f'first_batch_trg shape = {first_batch_trg.shape}')
print()

enc = Encoder(
    input_dim = len(src_field.vocab), 
    emb_dim = 256,
    hidden_dim = 512,
    n_layers = 2, 
    dropout = 0.5,
    debug = True).to(device)


enc_hidden, enc_cell = enc(first_batch_src)
print()
print(f'enc_hidden shape = {enc_hidden.shape}')
print(f'enc_cell shape = {enc_cell.shape}')
print()

dec = Decoder(
    output_dim = len(trg_field.vocab), 
    emb_dim = 256, 
    hidden_dim = 512, 
    n_layers = 2, 
    dropout = 0.5,
    debug = True).to(device)

dec_input = first_batch_trg[0, :]
prediction, dec_hidden, dec_cell = dec(dec_input, enc_hidden, enc_cell)
print()
print(f'prediction shape = {prediction.shape}')
print(f'dec_hidden shape = {dec_hidden.shape}')
print((f'dec_cell shape = {dec_cell.shape}'))


# %%
"""
After building encoder and decoder, we want to create a seq2seq model
which encapsulates these two models.

The seq2seq model works as the following steps (in high level):
1. input data fed into encoder
2. encoder generates context vector
3. decoder generates predicted data

Note1: 
Because we want to initialize the hidden state and cell state of lstm in decoder
with the context vector produced by encoder, (context vector is just the final hidden state
and cell state of each layer lstm in encoder) we must ensure the number of lstm layers and 
dimesion in encoder are same as ones in decoder.

Note2:
In order to train our decoder quickly, we will make use of 'teacher-forcing' technique.
Therefore, we should have a 'teacher forcing ratio'. With the probability equal to the teacher
forcing ratio, we will use the actual ground-truth next token as the input of decoder. With the 
probability equal to (1 - teacher forcing ratio), we will use the decoder's prediction as the next-step input.


"""