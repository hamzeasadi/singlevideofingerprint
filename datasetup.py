import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



loss = dict(
    s0=(0, [list(range(1, 100)), list(range(102, 200))], 0), s1=(0, [list(range(2, 100)),[100], list(range(102, 200))], 100), 
    s2=(0, [list(range(2, 100)), list(range(101, 200))], 101)
)



def main():
    x = torch.randn(size=(8, 1, 2, 2))
    # for i in range(x.size()[0]):
    #     row = x[i]
    #     left = x[0:i] 
    #     right = x[i+4:]
    # x1 = torch.tensor([1,1,2,2,3,3,4,4])
    # x2 = torch.tensor([1,1,2,2,3,3,4,4])
    # for i in range(0, x1.size()[0], 2):
    #     xi0 = x1[i]
    #     xi1 = x1[i+1]

    # x1 = x[1]
    # dist = torch.square(torch.linalg.matrix_norm(torch.subtract(x1, x)))
    # print(dist.squeeze())
    bs = 4
    g = 2
    clss = torch.tensor(list(range(1, bs//2 +1) ) )  
    samples = torch.tensor(2*[i//2 for i in range(2, bs+2)])
    indesx = torch.tensor(list(range(2*bs)))

    batch_train = dict()
    k = 0
    for i in range(2*bs):
        if i<bs:
            if i%2 == 0:
                sub1 = torch.cat((indesx[:i], indesx[i+1:bs+i], indesx[bs+i+2:]))
                lbl = (sub1 == i+1).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, i+1, lbl, sub1)
                k+=1

                sub2 = torch.cat((indesx[:i], indesx[i+2:bs+i+1], indesx[bs+i+2:]))
                lbl = (sub2 == bs+i).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, bs+i, lbl, sub2)
                k+=1

                sub3 = torch.cat((indesx[:i], indesx[i+2:bs+i], indesx[bs+i+1:]))
                lbl = (sub3 == bs+i+1).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, bs+i+1, lbl, sub3)
                k+=1

            else:
                sub1 = torch.cat((indesx[:i-1], indesx[i+1:bs+i], indesx[bs+i+1:]))
                lbl = (sub1 == bs+i-1).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, bs+i-1, lbl, sub1)
                k+=1

                sub2 = torch.cat((indesx[:i-1], indesx[i+1:bs+i-1], indesx[bs+i:]))
                lbl = (sub2 == bs+i).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, bs+i, lbl, sub2)
                k+=1

        else:
            if i%2 == 0:
                sub1 = torch.cat((indesx[:i-bs], indesx[i-bs+2:i], indesx[i+1:]))
                lbl = (sub1 == i+1).nonzero(as_tuple=True)[0][0]
                batch_train[f's{k}'] = (i, i+1, lbl, sub1)
                k+=1
      


    # for k, val in batch_train.items():
    #     print(len(val[-1]))
    # x = torch.tensor([1, 2, 3,4])
    # idx = (x == 2).nonzero(as_tuple=True)[0][0]
    # print(x[idx])
    print(batch_train)
        
    

        






        
    





  


if __name__ == '__main__':
    main()