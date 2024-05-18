import sys
sys.path.append('lib')
import torch
import torch.nn as nn
class SVM(torch.nn.Module):
    def __init__(self,  xdim1,xdim2, nclass=2)  :
        super(SVM, self).__init__()
        self.fc1 = nn.Linear((xdim1[1]+xdim2[1])*xdim1[2], nclass).cuda()
        self.BN = nn.BatchNorm1d(xdim1[2], eps=0.01).cuda()
       
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.BN(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        output = self.fc1(x)
        return output, None