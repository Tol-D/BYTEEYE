import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = layers.GraphConvolution(nfeat, nhid)
        self.gc2 = layers.GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # relu(axw)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.max(x, dim=1)[0].squeeze()
        return x
        #return F.log_softmax(x, dim=1)
