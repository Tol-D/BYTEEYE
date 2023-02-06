#GAT model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from . import layers

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [layers.GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            #add_module是Module类的成员函数，输入参数为Module.add_module(name: str, module: Module)。功能为，为Module添加一个子module，对应名字为name
            #add_module()函数也可以在GAT.init(self)以外定义A的子模块

        self.out_att = layers.GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #输出层的输入张量的shape之所以为(nhid * nheads, nclass)是因为在forward函数中多个注意力机制在同一个节点上得到的多个不同特征被拼接成了一个长的特征

    #SMVulDetector
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        #x = x.unsqueeze(0)  #for TMD
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        #x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #x = F.elu(self.out_att(x[0], adj))
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        # x = Variable(x, requires_grad=True)
        #x = F.log_softmax(x[0],dim=1)
        #x = F.log_softmax(x, dim=1)
        #x = Variable(x, requires_grad=True)
        return x

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         # 多个图注意力层
#         self.attentions = [layers.GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         # 输出层
#         self.out_att = layers.GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)
