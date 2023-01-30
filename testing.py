import os, random
import conf as cfg
import torch
from torch import nn as nn
from matplotlib import pyplot as plt
import utils
import datasetup as dst
import model as m
import cv2


dev = torch.device('cpu')
testdata = cfg.paths['val']

def add_noise(img, sigma=1, mean=0, h1=100, h2=200, w1=100, w2=200):
    """
    h1<h2, w1<w2
    """
    imgt = torch.from_numpy(img).float()
    noisypart = imgt[h1:h2, w1:w2, :]
    filter = torch.randn_like(noisypart)*sigma + mean
    imgt[h1:h2, w1:w2, :] = filter
    return imgt.permute(2, 0, 1)

def copy_move(img, h1=100, h2=200, w1=100, w2=200, d=100):
    """
    d<100
    """
    imgt = torch.from_numpy(img).float()
    copypart = imgt[h1:h2, w1:w2, :]
    imgt[h1+d:h2+d, w1+d:w2+d, :] = copypart
    return imgt.permute(2, 0, 1)


def splicing(img, splicimg, h1=100, h2=200, w1=100, w2=200):
    """
    h2>h1, w2>w1
    """
    imgt = torch.from_numpy(img).float()
    splicet = torch.from_numpy(splicimg)
    splicepart = splicet[h1:h2, w1:w2, :]
    imgt[h1:h2, w1:w2, :] = splicepart
    return imgt.permute(2, 0, 1)




kk = 0    

def imgman(imgpath):
    imglist = os.listdir(imgpath)
    imgs = random.sample(imglist, 2)
    img1 = cv2.imread(os.path.join(imgpath, imgs[0]))
    img2 = cv2.imread(os.path.join(imgpath, imgs[1]))

    firsttuple = [torch.from_numpy(img1).float(), add_noise(img1), copy_move(img1), splicing(img1, img2)]
    secondtuple = [torch.from_numpy(img2).float(), add_noise(img2), copy_move(img2), splicing(img2, img1)]
    return firsttuple, secondtuple

def visulize(list1img, list2img):
    l = len(list1img)
    axs, fig = plt.subplots(nrows=2, ncols=l)
    for i in range(l):
        axs[0, i].imshow(list1img[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(list2img[i], cmap='gray')
        axs[1, i].axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(cfg.paths['model'], f'{kk}.png'))


def result(modelpath, modelname):
    kt = utils.KeepTrack(path=modelpath)
    state = kt.load_ckp(fname=modelname)
    # model="", opt="", epoch=1, trainloss=0.1, valloss=0.1
    model_state_dict = state['model']
    net = m.VideoPrint(inch=1, depth=20)
    net.load_state_dict(model_state_dict)
    imgset1 , imgset2 = imgman(imgpath=testdata)

    noiseset1 = []
    noiseset2 = []
    l = len(imgset1)
    net.eval()
    with torch.no_grad():
        for i in range(l):
            noise1, noise2 = net(imgset1[i][1:2, :, :].unsqueeze(dim=0), imgset2[i][1:2, :, :].unsqueeze(dim=0))
            noiseset1.append(noise1)
            noiseset2.append(noise2)

    list1 = [img.detach().squeeze().numpy() for img in noiseset1]
    list1.append(imgset1[0].detach().squeeze().permute(1, 2, 0).numpy())
    list2 = [img.detach().squeeze().numpy() for img in noiseset2]
    list2.append(imgset2[0].detach().squeeze().permute(1, 2, 0).numpy())

    visulize(list1img=list1, list2img=list2)


    

    




def main():
    models = [f'singlecamfingerprint_{i}.pt' for i in range(39)]
    # mn = 'singlecamfingerprint_38.pt'
    modelpath = cfg.paths['model']
    for i in range(39):
        mn = models[i]
        kk=i
        result(modelpath=modelpath, modelname=mn)
        # kk = i




if __name__ == '__main__':
    main()

