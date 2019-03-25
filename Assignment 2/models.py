#!/usr/bin/env python
# coding: utf-8

# In[20]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In[21]:


import argparse
import math, copy, time
import collections
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda as cuda
import numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
np = numpy


# In[14]:


# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")
    
    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
           
        super(RNN, self).__init__()
        
        self.emb_size     = emb_size
        self.hidden_size  = hidden_size
        self.seq_len      = seq_len
        self.vocab_size   = vocab_size
        self.num_layers   = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.batch_size   = batch_size
      
        #embedding layer
        self.emb_layer = nn.Embedding(self.vocab_size, self.emb_size)

        #fully connected layers
        self.fc_layers = nn.ModuleList()
        
        for i in range(num_layers):
            #first layer
            if i == 0:
                self.fc_layers.append(nn.Linear(self.emb_size, self.hidden_size))
            #all other layers
            else:
                self.fc_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        
        #output layer
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

        #dropout layers
        self.dropout_layer = nn.Dropout(p = 1-self.dp_keep_prob)

        #recurrent layers
        self.recurrent_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers)

        #tanh function
        self.tanh = nn.Tanh()
        
        #Softmax function
        self.softmax = torch.nn.Softmax()
        
        #default hidden state
        self.default_hidden = torch.zeros([num_layers, batch_size, hidden_size], dtype=torch.float)
    
    def init_weights_uniform(self):
        #initialization of the weights of the first fc layer
        nn.init.uniform_(self.output_layer.weight, a = -0.1, b = 0.1)
        self.output_layer.bias.data.fill_(0)
        
        for layer in range(self.num_layers):
            if layer < self.num_layers:
                nn.init.uniform_(self.fc_layers[layer], a = -0.1, b = 0.1)
                self.fc_layers[layer].bias.data.fill_(0)
            
            nn.init.uniform_(self.recurrent_layers[layer], a = -0.1, b = 0.1)
            self.recurrent_layers[layer].bias.data.fill_(0)
        

    def init_hidden(self):

        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float)

    
    def forward(self, inputs, hidden=None):
        
#     Arguments:
#         - inputs: A mini-batch of input sequences, composed of integers that 
#                     represent the index of the current token(s) in the vocabulary.
#                         shape: (seq_len, batch_size)
#         - hidden: The initial hidden states for every layer of the stacked RNN.
#                         shape: (num_layers, batch_size, hidden_size)
    
#     Returns:
#         - Logits for the softmax over output tokens at every time-step.
#               **Do NOT apply softmax to the outputs!**
#               Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
#               this computation implicitly.
#                     shape: (seq_len, batch_size, vocab_size)
#         - The final hidden states for every layer of the stacked RNN.
#               These will be used as the initial hidden states for all the 
#               mini-batches in an epoch, except for the first, where the return 
#               value of self.init_hidden will be used.
#               See the repackage_hiddens function in ptb-lm.py for more details, 
#               if you are curious.
#                     shape: (num_layers, batch_size, hidden_size)
        hidden = hidden if hidden is not None else torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float, requires_grad=True) 
        hidden = hidden.type(torch.float)
        
        mini_batch = self.emb_layer(inputs)
        
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size])
        
        for timestep in range(self.seq_len):
            x = mini_batch[timestep, :, :]
            
            
            for layer in range(self.num_layers):
                x = self.fc_layers[layer](x)
                x = self.tanh(x + self.recurrent_layers[layer](hidden[layer].clone()))
                x = self.dropout_layer(x)
                
                hidden[layer, :, :] = x

            logits[timestep, :, :] = self.output_layer(x)     
        
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden
        
    def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.
        
        hidden = hidden if hidden is not None else torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float, requires_grad=True) 
        hidden = hidden.type(torch.float)
        
        mini_batch = self.emb_layer(inputs)
        
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size])
        
        for timestep in range(self.seq_len):
            x = mini_batch[timestep, :, :]
            
            for layer in range(self.num_layers):
                x = self.fc_layers[layer](x)
                x = self.tanh(x + self.recurrent_layers[layer](hidden[layer].clone()))
                x = self.dropout_layer(x)
                
                hidden[layer, :, :] = x

            logits[timestep, :, :] = self.output_layer(x)        
            samples = torch.max(self.softmax(logits), dim=2)
   
        return samples

# # Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
#   """
#   Follow the same instructions as for RNN (above), but use the equations for 
#   GRU, not Vanilla RNN.
#   """
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        self.emb_size     = emb_size
        self.hidden_size  = hidden_size
        self.seq_len      = seq_len
        self.vocab_size   = vocab_size
        self.num_layers   = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.batch_size   = batch_size

        #embedding layer
        self.emb_layer = nn.Embedding(self.vocab_size, self.emb_size)

        #W_r layers
        self.W_r_layers = nn.ModuleList()
        
        for i in range(num_layers):
            #first W_r layer
            if i == 0:
                self.W_r_layers.append(nn.Linear(self.emb_size, self.hidden_size))
            #all other layers
            else:
                self.W_r_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        
        #W_z layers
        self.W_z_layers = nn.ModuleList()

        for i in range(num_layers):
            #first W_z layer
            if i == 0:
                self.W_z_layers.append(nn.Linear(self.emb_size, self.hidden_size))
            #all other layers
            else:
                self.W_z_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        #W_h layers
        self.W_h_layers = nn.ModuleList()

        for i in range(num_layers):
            #first W_h layer
            if i == 0:
                self.W_h_layers.append(nn.Linear(self.emb_size, self.hidden_size))
            #all other layers
            else:
                self.W_h_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        #W_y layer
        self.W_y = nn.Linear(self.hidden_size, self.vocab_size)

        #U_r layers
        self.U_r_layers = nn.ModuleList()

        for i in range(num_layers):
            #first U_r layer
            if i == 0:
                self.U_r_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            #all other layers
            else:
                self.U_r_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        #U_z layers
        self.U_z_layers = nn.ModuleList()
        
        for i in range(num_layers):
            #first U_z layer
            if i == 0:
                self.U_z_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            #all other layers
            else:
                self.U_z_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        #U_h layers
        self.U_h_layers = nn.ModuleList()

        for i in range(num_layers):
        #first U_h layer
            if i == 0:
                self.U_h_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            #all other layers
            else:
                self.U_h_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        #dropout layers
        self.dropout_layer = nn.Dropout(p = 1-self.dp_keep_prob)

        #tanh function
        self.tanh = nn.Tanh()
        
        #sigmoid function
        self.sigmoid = nn.Sigmoid()

        def init_weights_uniform(self):
            nn.init.uniform_(self.W_y, a = -0.1, b = 0.1)
            self.W_y.bias.data.fill_(0)
            
            for layer in range(self.num_layers):
                nn.init.uniform_(self.W_r[layer], a = -0.1, b = 0.1)
                self.W_r[layer].bias.data.fill_(0)
                nn.init.uniform_(self.W_z[layer], a = -0.1, b = 0.1)
                self.W_z[layer].bias.data.fill_(0)
                nn.init.uniform_(self.W_h[layer], a = -0.1, b = 0.1)
                self.W_h[layer].bias.data.fill_(0)
                nn.init.uniform_(self.U_r[layer], a = -0.1, b = 0.1)
                self.U_r[layer].bias.data.fill_(0)
                nn.init.uniform_(self.U_z[layer], a = -0.1, b = 0.1)
                self.U_z[layer].bias.data.fill_(0)
                nn.init.uniform_(self.U_h[layer], a = -0.1, b = 0.1)
                self.U_h[layer].bias.data.fill_(0)

    def init_hidden(self):
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float)

    def forward(self, inputs, hidden=None):

        hidden = hidden if hidden is not None else torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float, requires_grad=True) 
        hidden = hidden.type(torch.float)

        mini_batch = self.emb_layer(inputs)

        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size])

        for timestep in range(self.seq_len):
            x = mini_batch[timestep, :, :]
            
            for layer in range(self.num_layers):
                
                r_t = self.sigmoid(self.W_r_layers[layer](x.clone()))# + self.U_r_layers[layer](hidden[layer].clone()))
                z_t = self.sigmoid(self.W_z_layers[layer](x.clone()) + self.U_z_layers[layer](hidden[layer].clone()))
                h_tilde = self.tanh(self.W_h_layers[layer](x.clone()) + self.U_h_layers[layer](r_t*hidden[layer].clone()))
                h_t = (1 - z_t).clone() * h_tilde.clone() + z_t.clone() * hidden[layer].clone()
                x = h_t
                hidden[layer, :, :] = h_t.clone()

                if layer == self.num_layers-1:
                    logits[timestep, :, :] = self.W_y(h_t)

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
        hidden = hidden if hidden is not None else torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float, requires_grad=True) 
        hidden = hidden.type(torch.float)

        mini_batch = self.emb_layer(inputs)

        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size])

        for timestep in range(self.seq_len):
            x = mini_batch[timestep, :, :]
            
            for layer in range(self.num_layers):
                
                r_t = self.sigmoid(self.W_r_layers[layer](x.clone()))# + self.U_r_layers[layer](hidden[layer].clone()))
                z_t = self.sigmoid(self.W_z_layers[layer](x.clone()) + self.U_z_layers[layer](hidden[layer].clone()))
                h_tilde = self.tanh(self.W_h_layers[layer](x.clone()) + self.U_h_layers[layer](r_t*hidden[layer].clone()))
                h_t = (1 - z_t).clone() * h_tilde.clone() + z_t.clone() * hidden[layer].clone()
                x = h_t
                hidden[layer, :, :] = h_t.clone()

                if layer == self.num_layers-1:
                    logits[timestep, :, :] = self.W_y(h_t)

        return samples


# # Problem 3
# ##############################################################################
# #
# # Code for the Transformer model
# #
# ##############################################################################

# """
# Implement the MultiHeadedAttention module of the transformer architecture.
# All other necessary modules have already been implemented for you.

# We're building a transfomer architecture for next-step prediction tasks, and 
# applying it to sequential language modelling. We use a binary "mask" to specify 
# which time-steps the model can use for the current prediction.
# This ensures that the model only attends to previous time-steps.

# The model first encodes inputs using the concatenation of a learned WordEmbedding 
# and a (in our case, hard-coded) PositionalEncoding.
# The word embedding maps a word's one-hot encoding into a dense real vector.
# The positional encoding 'tags' each element of an input sequence with a code that 
# identifies it's position (i.e. time-step).

# These encodings of the inputs are then transformed repeatedly using multiple
# copies of a TransformerBlock.
# This block consists of an application of MultiHeadedAttention, followed by a 
# standard MLP; the MLP applies *the same* mapping at every position.
# Both the attention and the MLP are applied with Resnet-style skip connections, 
# and layer normalization.

# The complete model consists of the embeddings, the stacked transformer blocks, 
# and a linear layer followed by a softmax.
# """

# #This code has been modified from an open-source project, by David Krueger.
# #The original license is included below:
# #MIT License
# #
# #Copyright (c) 2018 Alexander Rush
# #
# #Permission is hereby granted, free of charge, to any person obtaining a copy
# #of this software and associated documentation files (the "Software"), to deal
# #in the Software without restriction, including without limitation the rights
# #to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# #copies of the Software, and to permit persons to whom the Software is
# #furnished to do so, subject to the following conditions:
# #
# #The above copyright notice and this permission notice shall be included in all
# #copies or substantial portions of the Software.
# #
# #THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# #IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# #FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# #AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# #LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# #OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# #SOFTWARE.



# #----------------------------------------------------------------------------------

# # TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units

        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill
        
        self.n_heads = n_heads
        self.query_linear = clones(nn.Linear(n_units,self.d_k), n_heads)
        self.key_linear = clones(nn.Linear(n_units,self.d_k), n_heads)
        self.value_linear = clones(nn.Linear(n_units,self.d_k), n_heads)
        self.output_linear = nn.Linear(n_units,n_units)
        self.dropout = nn.Dropout(dropout)
        
        self.first_A = [None]*n_heads
        self.second_A = [None]*n_heads
        self.A = [None]*n_heads
        self.H = [None]*n_heads
        
        
    def forward(self, query, key, value, mask=torch.ones([2,3,3])):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
#         device = torch.device(“cuda:0” if torch.cuda.is_available() else “cpu”)


        key = key.cuda()
        query = query.cuda()
        value = value.cuda()
        
        for i in range(self.n_heads):
            self.first_A[i] = torch.matmul(self.query_linear[i](query),self.key_linear[i](key).transpose(1,2))/np.sqrt(self.d_k) 
            self.second_A[i] = self.first_A[i].masked_fill(mask==0, 1e-9)
            self.A[i] = self.dropout(F.softmax(self.second_A[i],dim = -1))
            self.H[i] = torch.matmul(self.A[i], self.value_linear[i](value)) 
        
        A = self.output_linear(torch.cat(self.H,2))

        return A 

#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False).cuda()
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
