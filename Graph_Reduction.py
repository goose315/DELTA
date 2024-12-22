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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from numbers import Number

class MaskFeature(nn.Module):
    def __init__(self, feat_dim, device):
        super(MaskFeature, self).__init__()
        self.s_mask_x = nn.Parameter(self.construct_feat_mask(feat_dim))
        self.t_mask_x = nn.Parameter(self.construct_feat_mask(feat_dim))
    
    def forward(self, x, domain, use_sigmoid=True, reparam=True):
        mask = self.s_mask_x if domain =='source' else self.t_mask_x
        mask = torch.sigmoid(mask) if use_sigmoid else mask
        if reparam:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2.0
            mean_tensor = torch.zeros_like(x, dtype=torch.float) -x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(x.device)
            x = x * mask + z * (1 - mask)
        else:
            x = x * mask
        return x
    
    def construct_feat_mask(self, feat_dim, init_strategy="ones"):
        mask = torch.ones(feat_dim)
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

class DropEdge(nn.Module):
    def __init__(self, source_edge_num, target_edge_num, device):
        super(DropEdge, self).__init__()
        self.s_edge_prob = self.construct_edge_prob(source_edge_num)
        self.t_edge_prob = self.construct_edge_prob(target_edge_num)

    def forward(self, prob, device, reparam=True):
        temperature = 1
        if reparam:
            eps = torch.rand(prob.size())
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(device)
            gate_inputs = (prob + gate_inputs) / temperature
            edge_weight = torch.sigmoid(gate_inputs)
        else:
            edge_weight = torch.sigmoid(prob)
        return edge_weight
   
    def construct_edge_prob(self, edge_num, init_strategy="ones"):
        prob = nn.Parameter(torch.ones(edge_num)*100)  #make initial weight close to 1
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                prob.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(prob, 0.0)
        return prob