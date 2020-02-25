import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import init
import os

#doing batch_first=True implementation first
#then modify later to per sample training
class loader(Dataset):
    def __init__(self, txt_file):
        content = open(txt_file, 'r').read()

        self.data = []

        while content:
            start = content.find('<start>')
            end = content.find('<end>')+5
            x_i = content[start:end+1]

            pad_len = 100 - len(x_i) % 100 + 1
            x_i = x_i + 'λ' * pad_len

            self.data.append(x_i)
            content = content[end+1:]

    def __len__(self):
        return len(self.data)
        

    #return the idx-th chunk and target
    def __getitem__(self, idx):
        
        return self.data[idx]
        
#         i = idx*100
#         j = (idx+1)*100
#         chunk = self.content[i:j]
#         target = self.content[i+1:j+1]
        
#         return chunk, target #chunk, chunk+1


class RNnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight, num_layers=1):
        super(RNnet, self).__init__()
        
        self.hidden_size = hidden_size
        self.char_embeddings = nn.Embedding.from_pretrained(weight)
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity = "relu", num_layers = 1)
        self.hidden2out = nn.Linear(hidden_size, output_size)
               
 
    def forward(self, sequence, states):
        
        #convert sequence to LongTensor before passing in
#         inp = torch.LongTensor(sequence)
        embeds = self.char_embeddings(sequence)
#         print(embeds.shape)
        embeds = torch.transpose(embeds, 0, 1)
#         print(torch.transpose(embeds, 0, 1).shape)
    
        rnnout, states = self.rnn(embeds, states)
#         print(lstmout[0], lstmout.shape)
#         print(states[0].shape)
        
        output = self.hidden2out(rnnout)
        output = output.view(-1, 94)
#         output = torch.squeeze(output)
        
#         print(output, output.shape)
        
        return output
    