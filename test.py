'''
author: meng-zha
data: 2020/05/28
'''
import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataloader import AirDataset
from model import Encoder,AttnDecoder

[mean,var] = [3.652727377067978,1.0758863206940295]

def evaluate(encoder,decoder,mode):
    encoder.eval()
    decoder.eval()
    val_dataset = AirDataset('./air_quality',mode=mode)
    val_dataloader = DataLoader(val_dataset,batch_size=512,shuffle=False,num_workers=8)

    loss = torch.zeros(6,35,device='cuda:0')
    acc = torch.zeros(6,35,device='cuda:0')
    
    mutation = 0.
    mutation_acc = torch.zeros(6,35,device='cuda:0')
    for i, data in enumerate(val_dataloader, 0):
        with torch.no_grad():
            input, label = data
            input, label = input.to(dtype=torch.float,device='cuda:0'), label.to(dtype=torch.float,device='cuda:0')[...,5]
            output,hidden = encoder(input)

            decoder_input = input[:,-1,:,5]
            output = decoder(decoder_input,hidden,output)

            real = label*var+mean
            real = torch.exp(real)-1
            predict = output*var+mean
            predict = torch.exp(predict)-1
            loss += (torch.abs(real-predict)).sum(0)

            # mutation
            past = input[:,18:,5]*var+mean
            past = torch.exp(past)-1
            minus = past.reshape(input.shape[0],-1).mean(1) - real.reshape(input.shape[0],-1).mean(1)
            minus = torch.abs(minus)
            outbreak = torch.where(minus>200)

            real=numer2aqi(real)
            predict=numer2aqi(predict)
            acc += (real==predict).sum(0)
            mutation_acc += (real[outbreak]==predict[outbreak]).sum(0)
            mutation += outbreak[0].shape[0]
    return loss/len(val_dataset),acc/len(val_dataset),mutation_acc/mutation

def numer2aqi(data):
    data[data<=35] = 0
    data[(data>35)*(data<=75)] = 1
    data[(data>75)*(data<=115)] = 2
    data[(data>115)*(data<=150)] = 3
    data[(data>150)*(data<=250)] = 4
    data[data>250] = 5
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='PM2.5 test')
    parser.add_argument("--mode",default='val',type=str)
    parser.add_argument("--checkpoints_path",default='./checkpoints',type=str)
    parser.add_argument("--model",default=10,type=int)
    parser.add_argument("--hidden",default=16,type=int)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    encoder_path = os.path.join(args.checkpoints_path,f'encoder_{args.model}.pth')
    decoder_path = os.path.join(args.checkpoints_path,f'decoder_{args.model}.pth')
    encoder_model = Encoder(35*args.hidden,(35,7),device).cuda()
    decoder_model = AttnDecoder(35*args.hidden,(35,7),24,6,device).cuda()

    encoder_model.load_state_dict(torch.load(encoder_path))
    decoder_model.load_state_dict(torch.load(decoder_path))

    err,acc,mutation = evaluate(encoder_model,decoder_model,args.mode)
    print(err.mean(1),acc.mean(1),mutation.mean(1))
    print(err.mean(),acc.mean(),mutation.mean())

if __name__ == "__main__":
    main()