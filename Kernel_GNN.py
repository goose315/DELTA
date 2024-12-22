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
from torch_geometric.nn import GINConv, GCNConv, GraphConv, GATConv, SAGEConv, WLConv, NNConv, SplineConv, WLConvContinuous, PDNConv
from numbers import Number

class KGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type="gcn"):
        super(KGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        if gnn_type == 'wl':
            self.conv_layers = nn.ModuleList([
                WLConvContinuous(),
                WLConvContinuous() 
            ])
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, edge_index, edge_weight):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, edge_weight)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class KGNN(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, hidden_channels, gnn_type="gcn"):
        super(KGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        if gnn_type == 'pdn':
            self.conv_layers = nn.ModuleList([
                PDNConv(self.in_channels, self.hidden_channels, edge_dim, hidden_channels),
                PDNConv(self.hidden_channels, self.out_channels, edge_dim, hidden_channels)
            ])
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, edge_index, edge_attr=None):
        for i, conv_layer in enumerate(self.conv_layers):
            if isinstance(conv_layer, PDNConv):
                x = conv_layer(x, edge_index, edge_attr)
            else:
                x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x