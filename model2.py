import os
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import conf as cfg



class VideoPrint(nn.Module):

    def __init__(self, inch=3, depth: int=20) -> None:
        super().__init__()
        self.depth = depth
        self.inch = inch
        self.noisext = self.blks()

    def blks(self):
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU())
        lastlayer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')

        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=64), nn.ReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x1, x2):
        out1 = self.noisext(x1)
        out2 = self.noisext(x2)
        res1, res2 = x1[:, 0:1, :, :]-out1, x2[:, 0:1, :, :]-out2
        return (res1, res2)




def main():
    x = torch.randn(size=(10, 3, 64, 64))
    # model = VideoPrint(inch=1, depth=20)
    # summary(model, input_size=[[10, 1, 64, 64], [10, 1, 64, 64]])
    # # model = torch.load(os.path.join(cfg.paths['model'], 'dncnn_15.pth'))
    # # summary(model, input_size=[10, 1, 48, 48])
    # x1 = torch.randn(size=(5, 3, 3, 3))
    # x2 = torch.randn(size=(5, 1, 3, 3))
    # diff = x1[:, 0:1, :, :] - x2
    # print(diff.shape)
    model = VideoPrint()
    res1, res2 = model(x, x)
    print(res1.shape, res2.shape)



if __name__ == '__main__':
    main()