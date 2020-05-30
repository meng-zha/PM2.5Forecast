
'''
author: meng-zha
data: 2020/05/28
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
from datetime import datetime
import os

from dataloader import AirDataset
from model import Encoder,AttnDecoder
from test import evaluate,mean,var

def parse_args():
    parser = argparse.ArgumentParser(description='PM2.5 prediction')
    parser.add_argument("--name",default='',type=str)
    parser.add_argument("--epochs",default=100,type=float)
    parser.add_argument("--lr",default=1e-3,type=float)
    parser.add_argument("--batch",default=256,type=int)
    parser.add_argument("--hidden",default=8,type=int)

    return parser.parse_args()

def train():
    args = parse_args()
    with open('./spot.txt','r') as f:
        spots = f.readline().split()

    tbd = os.path.join('./logger',datetime.now().strftime("%Y%m%d%H%M%S")[:12]+args.name)
    writer = SummaryWriter(tbd)

    check = os.path.join('./checkpoints',datetime.now().strftime("%Y%m%d%H%M%S")[:12]+args.name)
    if not os.path.exists(check):
        os.mkdir(check)

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    train_dataset = AirDataset('./air_quality',mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch,shuffle=True,num_workers=8)
    # [mean,var] = [3.6336625600012824,1.1400852379547814]

    encoder_model = Encoder(35*args.hidden,(35,7),device).cuda()
    decoder_model = AttnDecoder(35*args.hidden,(35,7),24,6,device).cuda()

    # fake_encoder = torch.rand(8,24,35,7,device=device)
    # writer.add_graph(encoder_model,fake_encoder)
    # fake_decoder = (torch.rand(8,35,device=device),torch.rand(1,8,280,device=device),torch.rand(8,24,280,device=device))
    # writer.add_graph(decoder_model,fake_decoder)

    optimizer_encoder = optim.Adam(encoder_model.parameters(),lr=args.lr,betas=(0.9,0.99))
    optimizer_decoder = optim.Adam(decoder_model.parameters(),lr=args.lr,betas=(0.9,0.99))
    criterion = torch.nn.MSELoss()
    
    for epoch in range(args.epochs):
        encoder_model.train()
        decoder_model.train()
        for i, data in enumerate(train_dataloader, 0):
            # ['hours' 'SO2' 'NO2' 'O3' 'CO' 'PM2.5' 'AQI'] 
            input, label = data
            input, label = input.to(dtype=torch.float,device=device), label.to(dtype=torch.float,device=device)[...,5]
            output,hidden = encoder_model(input)

            decoder_input = input[:,-1,:,5]
            output = decoder_model(decoder_input,hidden,output)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            real = label*var+mean
            real = torch.exp(real)-1
            predict = output*var+mean
            predict = torch.exp(predict)-1

            weight = torch.clamp(real,0,500)/500.
            weight = 2-(1-weight**2)**0.5

            loss = criterion(weight*output,weight*label)
            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder.step()
            writer.add_scalar('train/loss',loss,epoch*len(train_dataset)+i)
            print(f'epoch:{epoch} iter:{i}/{len(train_dataloader)} loss:{loss.cpu()}')

        writer.add_histogram(f'real/hist_{epoch}',real.view(-1),epoch)
        writer.add_histogram(f'predict/hist_{epoch}',predict.view(-1),epoch)
        writer.add_histogram(f'error/hist_{epoch}',(real-predict).view(-1),epoch)

        err,acc = evaluate(encoder_model,decoder_model,'val')
        print(f'epoch:{epoch} error:{err.mean()},acc:{acc.mean()}')
        for j,spot in enumerate(spots):
            writer.add_scalar(f'{spot}/err',err.mean(0)[j],epoch)
            writer.add_scalar(f'{spot}/acc',acc.mean(0)[j],epoch)

        for j in range(6):
            writer.add_scalar(f'{j}-hour/err',err.mean(1)[j])
            writer.add_scalar(f'{j}-hour/acc',acc.mean(1)[j])

        writer.add_scalar(f'00-avg/err',err.mean())
        writer.add_scalar(f'00-avg/acc',acc.mean())

        torch.save(encoder_model.state_dict(),os.path.join(check,f'encoder_{epoch}.pth'))
        torch.save(decoder_model.state_dict(),os.path.join(check,f'decoder_{epoch}.pth'))
        

if __name__ == "__main__":
    train()