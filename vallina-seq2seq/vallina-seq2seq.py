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
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
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

def tokenize_de(text):
    """
    why reverse the order of input text,
    you may refer to this article (https://stackoverflow.com/questions/51003992/why-do-we-reverse-input-when-feeding-in-seq2seq-model-in-tensorflow-tf-reverse).
    """
    tokens = spacy_de.tokenizer(text)
    return [tok.text for tok in tokens][::-1]

def tokenize_en(text):
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
we have to convert the dataset into batches of examples.

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

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
        self.n_layers = n_layers
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

Note3:
We will use a loop to repeatly feed token to decoder, and store its predition in a big tensor.
The first token fed to decoder is <sos>, decoder predicts the first output token y1. Until decoder predicts <eos>
token, we ends the loop (we never feed <eos> token to decoder). Therefore, the target sequence is [<sos>, y1, y2, y3, <eos>] 
but the predicted sequence is [0, y1, y2, y3, <eos>]. 
"""

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimension of LSTM in encoder and decoder should be equal"
        assert encoder.n_layers == decoder.n_layers, "Number of LSTM in encoder and decoder should be equal"

    def forward(self, source_batch, target_batch, teacher_forcing_ratio = 0.5):
        
        batch_size = target_batch.shape[1]
        target_length = target_batch.shape[0]
        target_vocab_size = self.decoder.output_dim

        # print(f"[seq2seq forward] source_batch shape: {source_batch.shape}")
        # print(f"[seq2seq forward] target_batch shape: {target_batch.shape}")

        # a big tensor to store decoder's output
        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)
        # print(f"[seq2seq forward] outputs shape: {outputs.shape}")

        # context vector of encoder to initialize decoder
        hidden_state, cell_state = self.encoder(source_batch)
        # print(f"[seq2seq forward] hidden_state shape: {hidden_state.shape}")
        # print(f"[seq2seq forward] cell_state shape: {cell_state.shape}")

        # first input token to decoder (<sos>)
        input = source_batch[0, :]

        for t in range(1, target_length):
            # print(f"[seq2seq forward] t: {t}")

            # print(f"[seq2seq forward] input shape: {input.shape}")
            output, hidden_state, cell_state = self.decoder(input, hidden_state, cell_state)

            # print(f"[seq2seq forward] output shape: {output.shape}")
            # print(f"[seq2seq forward] hidden_state shape: {hidden_state.shape}")
            # print(f"[seq2seq forward] cell_state shape: {cell_state.shape}")

            outputs[t] = output
            
            if random.random() < teacher_forcing_ratio:
                input = target_batch[t, :]
            else:
                input = output.argmax(1)
        
        # print(f"[seq2seq forward] outputs shape: {outputs.shape}")
        return outputs

        

# %%
"""
After declaring the class of seq2seq model, we want to initialize it. We have to initialze encdoer and decoder,
and feed them to seq2seq model. 
"""

INPUT_DIM = len(src_field.vocab)
OUTPUT_DIM = len(trg_field.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYER = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(
    input_dim = INPUT_DIM, 
    emb_dim = ENC_EMB_DIM,
    hidden_dim = LSTM_HIDDEN_DIM,
    n_layers = LSTM_NUM_LAYER, 
    dropout = ENC_DROPOUT)

dec = Decoder(
    output_dim = OUTPUT_DIM, 
    emb_dim = DEC_EMB_DIM, 
    hidden_dim = LSTM_HIDDEN_DIM, 
    n_layers = LSTM_NUM_LAYER, 
    dropout = DEC_DROPOUT)

model = Seq2Seq(
    encoder=enc,
    decoder=dec,
    device=device
).to(device)


#%%
"""
In the paper, they initialize the all weights of model from a uniform distribution between -0.08 and +0.08.
We apply model on a function to initialize the weights. Every module in model will apply on this function.
"""

def init_weights(mod):
    print(f"module: {mod}")
    for name, param in mod.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


# %%
"""
We can also calculate the number of trainable paramaters in model.
"""

def count_parameters(model):
    return sum([param.numel() for param in model.parameters() if param.requires_grad])

print(f"Number of trainable parameters: {count_parameters(model)}")



#%%
"""
We use Adam as optimizer to optimize parameters in model
"""
optimizer = optim.Adam(model.parameters())


#%%
"""
Because the decoder's prediction is a classification problem, we use CrossEntropy as the loss function
to calculate loss. However, if current groundtruth token is <pad>, we should not calculate the loss of this prediction.
"""

target_pad_token = trg_field.pad_token
target_pad_token_idx = trg_field.vocab.stoi[target_pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=target_pad_token_idx)


#%%
"""
Finally, we want to define a train() function containing a loop to train our model. In the loop: 

1. We will feed a input batch (shape = [num_token, batch_size]) to model, and get output batch (shape = [num_token, batch_size, output dim]). 
We should calculate loss between output batch (shape = [num_token, batch_size, output dim]) and target batch (shape = [num_token, batch_size]).

2. As metioned above, if we focus on a example instead of batch of examples, the model's output is [0, y1, y2, y3, <eos>], but the target is [<sos>, y1, y2, y3, <eos>].
Therefore, we want to slice out the first token in model's output and target.

3. However, the loss function in pytorch accepts 2-dimension input and 1-dimension taregt, we reshape the tensor with view() method.

4. After getting loss of this batch, we calculate gradient of loss with respect to all parameters.

5. In order to prevent gradient explode / vanish in rnn, we clip the gradient.

6. Update parameters with gradients
"""

def train(model, iterator, optimizer, criterion, clip):

    # change model to train mode
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
    
        src_batch = batch.src # shape = [num_token, batch_size]
        trg_batch = batch.trg # shape = [num_token, batch_size]

        # reset gradient
        optimizer.zero_grad()

        output = model(src_batch, trg_batch)
        # output's shape = [num_token, batch_size, output_dim]

        # slice out first token in model's output and target
        output = output[1:]
        trg_batch = trg_batch[1:]

        # reshape model's output to 2-dimension, target to 1-dimension
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg_batch = trg_batch.view(-1)

        # calculate loss
        loss = criterion(output, trg_batch)

        # calculate gradient of loss with respect to all parameters
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update weights
        optimizer.step()

        epoch_loss += loss.item()

        print(f"Batch: {i+1:03}/{len(iterator)}, loss: {loss.item()}")

    
    return epoch_loss / len(iterator)


#%%
"""
In addition to training model, we also evaluate model. The evaluate() function is similar to train(), except for the key difference:
not updating parameters. Therefore, we will not use optimizer in evaluate() function.
"""

def evaluate(model, iterator, criterion):

    # change model to evaluation mode
    model.eval()

    epoch_loss = 0

    # because we do not update parameters in model, using torch.no_grad() will speed up execution.
    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src_batch = batch.src # shape = [num_token, batch_size]
            trg_batch = batch.trg # shape = [num_token, batch_size]

            # reset gradient
            # optimizer.zero_grad()

            output = model(src_batch, trg_batch, 0) # turn off teacher forcing
            # output's shape = [num_token, batch_size, output_dim]

            # slice out first token in model's output and target
            output = output[1:]
            trg_batch = trg_batch[1:]

            # reshape model's output to 2-dimension, target to 1-dimension
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg_batch = trg_batch.view(-1)

            # calculate loss
            loss = criterion(output, trg_batch)

            # calculate gradient of loss with respect to all parameters
            # loss.backward()

            # clip gradient
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # update weights
            # optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# %%
"""
We use a function to estimate the time of epoch.
"""
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#%%
"""
We start training seq2seq model! In each epoch, we check the loss of validation, if it is the best one so far, 
we will save the current model.
"""

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    t1 = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    t2 = time.time()

    epoch_mins, epoch_secs = epoch_time(t1, t2)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "vallina-seq2seq.pt")
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



#%%
model.load_state_dict(torch.load('vallina-seq2seq.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')# %%
