'''
author: meng-zha
data: 2020/05/28
'''
# attention is modified from 
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_size, point_size, device):
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.point_size = point_size

        self.fc = []
        self.relu = nn.ReLU(inplace=True)
        for i in range(point_size[0]):
            self.fc.append(
                nn.Sequential(nn.Linear(point_size[1], 16), nn.Dropout(0.5),
                              nn.ReLU(True), nn.Linear(16, 8), nn.Dropout(0.5),
                              nn.ReLU(True)))
        self.fc = nn.ModuleList(self.fc)
        self.linear = nn.Linear(point_size[0] * 8, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        output = torch.zeros(batch_size,
                             seq_length,
                             self.point_size[0],
                             8,
                             device=self.device)
        for i in range(input.shape[1]):
            for j in range(self.point_size[0]):
                output[:, i, j, :] = self.fc[j](input[:, i, j, :])
        output = output.view(batch_size, seq_length, -1)
        output = self.relu(self.linear(output))
        hidden = self.initHidden(batch_size)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, point_size, past, forward, device):
        super(AttnDecoder, self).__init__()
        self.device = device
        self.past = past
        self.forw = forward
        self.hidden_size = hidden_size
        self.point_size = point_size

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                 nn.ReLU(True),
                                nn.Linear(hidden_size//2, point_size[0]))
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # attention
        self.attn = nn.Linear(point_size[0]+hidden_size,self.past)
        self.attn_combine = nn.Linear(point_size[0]+hidden_size,hidden_size)

    def forward(self, input,hidden, encoder_outputs):
        batch_size = input.shape[0]
        output = torch.zeros(batch_size,self.forw,self.point_size[0],device=self.device)

        for i in range(self.forw):
            attn_weights = F.softmax(self.relu(self.attn(torch.cat((input,hidden.squeeze(0)),dim=1))),dim=1) 
            attn_applied = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)
            attn_combine = self.relu(self.attn_combine(torch.cat((input.unsqueeze(1),attn_applied),dim=2)))
            out,hidden = self.gru(attn_combine,hidden)
            out = self.fc(out.squeeze(1))

            output[:,i,:] = out
            input = out

        return output
