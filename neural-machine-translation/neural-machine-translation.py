import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchtext.data import BucketIterator
from torch.utils.data import Dataset

import spacy
import numpy as np

import random
import math
import time

import pandas as pd
from tqdm import tqdm

class CNNDailyDataset(Dataset):

    def __init__(self, path, transforms, vocabs):

        self.transform_src = transforms[0]
        self.transform_trg = transforms[1]

        self.srcs = []
        self.trgs = []

        print(f"Load dataset: {path}")
        df = pd.read_csv(path)
        print(f"Transform dataset")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            article = self.transform_src(row['article'])
            article = article[:256] + article[-256:]
            self.srcs.append(article)
            self.trgs.append(self.transform_trg(row['highlights']))
        
        if vocabs != None:
            self.vocab_stoi = vocabs[0]
            self.vocab_itos = vocabs[1]
        else:
            self.vocab_stoi = {}
            self.vocab_itos = {}
            print(f"Build vocab table")
            self.build_vocab()

    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, index):

        src = []
        for token in self.srcs[index]:
            if token in self.vocab_stoi:
                src.append(self.vocab_stoi[token])
            else:
                src.append(self.vocab_stoi["<unk>"])
        
        trg = []
        for token in self.trgs[index]:
            if token in self.vocab_stoi:
                trg.append(self.vocab_stoi[token])
            else:
                trg.append(self.vocab_stoi["<unk>"])

        return {'src': src, 'trg': trg}

    def build_vocab(self):
        dict = {}
        for sent in self.srcs:
            for token in sent:
                if token not in dict:
                    dict[token] = 1
                else:
                    dict[token] += 1
        for sent in self.trgs:
            for token in sent:
                if token not in dict:
                    dict[token] = 1
                else:
                    dict[token] += 1
        count = 0
        for key, value in dict.items():
            if value > 100:
                self.vocab_stoi[key] = count
                self.vocab_itos[count] = key
                count += 1
        self.vocab_stoi["<unk>"] = count
        self.vocab_itos[count] = "<unk>"
        count += 1
        self.vocab_stoi["<pad>"] = count
        self.vocab_itos[count] = "<pad>"
        
    
    def pad_batch(self, batch):
        src_tensors = [torch.tensor(emp['src']) for emp in batch]
        trg_tensors = [torch.tensor(emp['trg']) for emp in batch]
        return pad_sequence(src_tensors, padding_value=self.vocab_stoi["<pad>"]), pad_sequence(trg_tensors, padding_value=self.vocab_stoi["<pad>"])


spacy_en = spacy.load("en_core_web_sm")

def transform_src(text):
    tokens = spacy_en.tokenizer(text.lower())
    tokens = [tok.text for tok in tokens]
    tokens = ['<sos>'] + tokens + ['<eos>']
    return tokens

def transform_trg(text):
    tokens = spacy_en.tokenizer(text.lower())
    tokens = [tok.text for tok in tokens]
    tokens = ['<sos>'] + tokens + ['<eos>']
    return tokens

train_dataset = CNNDailyDataset(
    path="../cnn_daily_ds/train.csv",
    transforms=(transform_src, transform_trg),
    vocabs=None
)

valid_dataset = CNNDailyDataset(
    path="../cnn_daily_ds/valid.csv",
    transforms=(transform_src, transform_trg),
    vocabs=(train_dataset.vocab_stoi, train_dataset.vocab_itos)
)

"""Define the device."""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

"""Create the iterators."""

BATCH_SIZE = 32

train_iterator = BucketIterator(
    train_dataset,
    batch_size = BATCH_SIZE,
    device = device,
    sort_key=lambda x: len(x['src']),
    repeat=True,
    sort=False,
    shuffle=True,
    sort_within_batch=True
)

valid_iterator = BucketIterator(
    valid_dataset,
    batch_size = BATCH_SIZE,
    device = device,
    sort_key=lambda x: len(x['src']),
    repeat=True,
    sort=False,
    shuffle=True,
    sort_within_batch=True
)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        tmp = torch.cat((hidden, encoder_outputs), dim = 2)
        # print(f"<debug> tmp shape = {tmp.shape}")
        energy = torch.tanh(self.attn(tmp)) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(train_dataset.vocab_stoi)
OUTPUT_DIM = len(train_dataset.vocab_stoi)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

"""We use a simplified version of the weight initialization scheme used in the paper. Here, we will initialize all biases to zero and all weights from $\mathcal{N}(0, 0.01)$."""

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

"""Calculate the number of parameters. We get an increase of almost 50% in the amount of parameters from the last model. """

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

"""We create an optimizer."""

optimizer = optim.Adam(model.parameters())

"""We initialize the loss function."""
target_pad_token_idx =  train_dataset.vocab_stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=target_pad_token_idx)

"""We then create the training loop..."""

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0

    iterator.create_batches()
    i = 0
    for batch in iterator.batches:

        src_batch, trg_batch = train_dataset.pad_batch(batch)
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        
        optimizer.zero_grad()
        
        output = model(src_batch, trg_batch)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg_batch = trg_batch[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg_batch)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        print(f"Batch: {i+1:03}/{len(iterator)}, loss: {loss.item()}")
        i += 1

    return epoch_loss / len(iterator)

"""...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        iterator.create_batches()
        i = 1
        for batch in iterator.batches:

            src_batch, trg_batch = valid_dataset.pad_batch(batch)
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            output = model(src_batch, trg_batch, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg_batch = trg_batch[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg_batch)

            epoch_loss += loss.item()

            if (i+1)%10 == 0:
                print(f"Batch: {i+1:03}/{len(iterator)}, loss: {loss.item()}")
            i += 1
        
    return epoch_loss / len(iterator)

"""Finally, define a timing function."""

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""Then, we train our model, saving the parameters that give us the best validation loss."""

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'nmt.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')