import os
from sklearn.metrics import f1_score
from argparse import ArgumentParser
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning)
from torch_geometric.nn import GINConv, GCNConv, GraphConv, GATConv, SAGEConv, WLConv, NNConv, SplineConv, PDNConv, PANConv
from torch_geometric.nn import SGConv, TransformerConv, TAGConv, SSGConv, MFConv

from numbers import Number

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type="gcn", filter_size=4):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        if gnn_type == 'gcn':
            self.conv_layers = nn.ModuleList([
                GCNConv(self.in_channels, self.hidden_channels),
                GCNConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'gat':
            self.conv_layers = nn.ModuleList([
                GATConv(self.in_channels, self.hidden_channels),
                GATConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'GraphConv':
            self.conv_layers = nn.ModuleList([
                GraphConv(self.in_channels, self.hidden_channels),
                GraphConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'SGConv':
            self.conv_layers = nn.ModuleList([
                SGConv(self.in_channels, self.hidden_channels),
                SGConv(self.hidden_channels, self.out_channels) 
            ])
        elif gnn_type == 'TAGConv':
            self.conv_layers = nn.ModuleList([
                TAGConv(self.in_channels, self.hidden_channels),
                TAGConv(self.hidden_channels, self.out_channels) 
            ])     
        elif gnn_type == 'MFConv':
            self.conv_layers = nn.ModuleList([
                MFConv(self.in_channels, self.hidden_channels),
                MFConv(self.hidden_channels, self.out_channels) 
            ])                 
        elif gnn_type == 'pan':
            self.conv_layers = nn.ModuleList([
                PANConv(self.in_channels, self.hidden_channels, filter_size=filter_size),
                PANConv(self.hidden_channels, self.out_channels, filter_size=filter_size) 
            ])
        #self.prelu = nn.PReLU(self.hidden_channels)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv_layer in enumerate(self.conv_layers):
            if self.gnn_type == 'pan':
                x = conv_layer(x, edge_index)[0]
            else:
                x = conv_layer(x, edge_index, edge_weight)
            
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
                
        return x


class GIB(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super(GIB, self).__init__()

        self.out_channels = out_channels
    
    def forward(self, encoded_output, reparam=True, num_sample=1):
        mu = encoded_output[:,:self.out_channels]
        std = F.softplus(encoded_output[:, self.out_channels:self.out_channels*2]-5, beta=1)

        if reparam:
            encoding = self.reparametrize_n(mu, std, num_sample)
        else:
            encoding = mu

        return (mu, std), encoding

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps =torch.Tensor(std.size()).normal_().to(std.device)

        return mu + eps * std
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * 0.05
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)