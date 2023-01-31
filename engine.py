import os, random
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import conf as cfg




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_setp(net: nn.Module, data:DataLoader, opt:Optimizer, criterion:nn.Module):
    epochloss = 0.0
    numbatchs = len(data)
    net.train()
    for (X1, X2) in data:
        X1 = X1.squeeze(dim=0).to(dev)
        X2 = X2.squeeze(dim=0).to(dev)
        (_, _), (res1, res2) = net(X1, X2)
        loss = criterion(res1, res2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epochloss+=loss.item()

    return epochloss/numbatchs


def val_setp(net: nn.Module, data:DataLoader, opt:Optimizer, criterion:nn.Module):
    epochloss = 0.0
    numbatchs = len(data)
    net.eval()
    with torch.no_grad():
        for (X1, X2) in data:
            X1 = X1.squeeze(dim=0).to(dev)
            X2 = X2.squeeze(dim=0).to(dev)
            (_, _), (res1, res2) = net(X1, X2)
            loss = criterion(res1, res2)
            epochloss+=loss.item()

    return epochloss/numbatchs




def main():
    pass




if __name__ == '__main__':
    main()