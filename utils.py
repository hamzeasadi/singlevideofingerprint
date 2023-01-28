import numpy as np
import os
import torch



def losstemp(bs=4, g=2):
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

    return batch_train





def main():
    pass


if __name__ == '__main__':
    main()