import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
import sys
import argparse
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pytorch_metric_learning import losses
from GNN_model import GNN, GIB, GRL
from torch_geometric.nn import GINConv, GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

sys.path.append("/data/zeyu/active_learning_domain/deep-active-learning")
sys.path.append("/data/zeyu/active_learning_domain/GIFI")
sys.path.append("/data/zeyu/active_learning_domain/")

from gnn.dataset.DomainData import DomainData
from GNN_model import GNN
from Graph_Reduction import MaskFeature, DropEdge
from utils import get_node_central_weight
from Loss_functions import Semi_loss
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import torch
from torch_geometric.utils import degree, add_self_loops, k_hop_subgraph
import torch.nn.functional as F
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Run the GNN model with specific parameters.')
parser.add_argument('--seed', type=int, default=666, help='Random seed for reproducibility')
parser.add_argument('--source', type=str, required=True, help='Source dataset name')
parser.add_argument('--target', type=str, required=True, help='Target dataset name')
parser.add_argument('--label_rate', type=float, default=0.01, help='Label rate for training data')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--n_clusters', type=int, default=50, help='Number of Kmeans clusters')
parser.add_argument('--k', type=int, default=3, help='Number of K')
parser.add_argument('--threhold', type=float, default=0.5, help='Distance threhold')
parser.add_argument('--backboneGNN', type=str, help='backboneGNN')
parser.add_argument('--backboneGNN2', type=str, help='backboneGNN2')
args = parser.parse_args()

set_seed(args.seed)
budget = args.n_clusters
label_rate = args.label_rate
source = args.source
target = args.target
epochs = 200

# Load Dataset
dataset = DomainData("/data/zeyu/active_learning_domain/GIFI/data/{}".format(source), name=source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
#print(source_data)

dataset = DomainData("/data/zeyu/active_learning_domain/GIFI/data/{}".format(target), name=target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
#print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

source_train_size = int(source_data.size(0) * label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


"""
#############################################################################################################
###################################################Edge_subnetwork#################################################
#############################################################################################################
"""

save_dir = '/data1/active_learning_domain/saved_model_latest'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

hidden_channels = 512
out_channels = 256
filter_size = 3
temperature = 0.5
mask_feature = MaskFeature(source_data.x.size(1), device).to(device)
drop_edge = DropEdge(source_data.edge_index.size(1), target_data.edge_index.size(1), device).to(device)
if args.backboneGNN == 'pan':
    GNN_encoder = GNN(source_data.x.size(1), hidden_channels, hidden_channels, gnn_type=args.backboneGNN, filter_size=filter_size).to(device)
else:
    GNN_encoder = GNN(source_data.x.size(1), hidden_channels, hidden_channels, gnn_type=args.backboneGNN).to(device)
cls_model = nn.Sequential(nn.Linear(out_channels, dataset.num_classes)).to(device)
gib_layer = GIB(hidden_channels, out_channels).to(device)
domain_model = nn.Sequential(GRL(), nn.Linear(out_channels, 64), nn.ReLU(), nn.Dropout(1e-1), nn.Linear(64, 2)).to(device)

models = [mask_feature, drop_edge, GNN_encoder, gib_layer, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
SupConLoss = losses.SupConLoss(temperature=temperature)

all_target_adloss_means = []
max_target_adloss_mean = float('-inf')
min_target_adloss_mean = float('inf')
corresponding_max_target_adloss = None
corresponding_min_target_adloss = None

def train_Backbone_1(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, args.label_rate)

    new_s_x = mask_feature(source_data.x, 'source')
    new_s_edge_weight = drop_edge(drop_edge.s_edge_prob, device)
    new_t_x = mask_feature(target_data.x, 'target')
    new_t_edge_weight = drop_edge(drop_edge.t_edge_prob, device)

    encoded_source = GNN_encoder(new_s_x, source_data.edge_index, new_s_edge_weight)
    encoded_target = GNN_encoder(new_t_x, target_data.edge_index, new_t_edge_weight)

    (s_mu, s_std), encoded_source = gib_layer(encoded_source)
    (t_mu, t_std), encoded_target = gib_layer(encoded_target)
    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)
    # Classifier Loss
    if args.backboneGNN == 'pan':
        criterion = SupConLoss
        cls_loss = criterion(source_logits[label_mask], source_data.y[label_mask])
    else:
        criterion = nn.CrossEntropyLoss()
        cls_loss = criterion(source_logits[label_mask], source_data.y[label_mask])


    # DA loss
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)

    if args.backboneGNN == 'pan':
        criterion = SupConLoss
        source_domain_cls_loss = criterion(source_domain_preds, torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device))
        target_domain_cls_loss = criterion(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        source_domain_cls_loss = criterion(source_domain_preds, torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device))
        target_domain_cls_loss = criterion(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))
    
    criterionad = nn.CrossEntropyLoss(reduction='none')

    target_adloss = criterionad(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))
    loss_grl = source_domain_cls_loss + target_domain_cls_loss

    target_adloss_mean = target_adloss.mean()

    global all_target_adloss_means, max_target_adloss_mean, min_target_adloss_mean
    global corresponding_max_target_adloss, corresponding_min_target_adloss
    
    all_target_adloss_means.append(target_adloss_mean.item())
    if target_adloss_mean.item() > max_target_adloss_mean:
        max_target_adloss_mean = target_adloss_mean.item()
        corresponding_max_target_adloss = target_adloss.clone().detach()
    if target_adloss_mean.item() < min_target_adloss_mean:
        min_target_adloss_mean = target_adloss_mean.item()
        corresponding_min_target_adloss = target_adloss.clone().detach()

    # IB loss
    info_loss = -0.5 * (1 + 2 * s_std.log() - s_mu.pow(2) - s_std.pow(2)).sum(1).mean().div(math.log(2))
    info_loss += -0.5 * (1 + 2 * t_std.log() - t_mu.pow(2) - t_std.pow(2)).sum(1).mean().div(math.log(2))

    source_softmax_out = nn.Softmax(dim=1)(source_logits)
    target_softmax_out = nn.Softmax(dim=1)(target_logits)
    source_pseudo_label = source_softmax_out.argmax(dim=1)
    target_pseudo_label = target_softmax_out.argmax(dim=1)
    source_pseudo_label[label_mask] = source_data.y[label_mask]

    mu = 0.5 - math.cos(min(math.pi, (2 * math.pi * float(epoch) / epoch))) / 2
    ks = int(source_data.y[label_mask].size(0) * mu) * 3
    kt = int(source_data.y[label_mask].size(0) * mu) * 3
    s_rn_weight, s_indices = get_node_central_weight(source_data, new_s_edge_weight, source_pseudo_label, ks, device)
    t_rn_weight, t_indices = get_node_central_weight(target_data, new_t_edge_weight, target_pseudo_label, kt, device)

    # inner
    source_softmax_out[label_mask] = F.one_hot(source_data.y, source_softmax_out.size(1)).float()[label_mask]
    inner_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(ks,)).to(device)
    inner_labeled = torch.index_select(source_data.y[label_mask], 0, inner_index)
    inner_labeled = F.one_hot(inner_labeled, source_softmax_out.size(1))
    inner_encoded = torch.index_select(encoded_source[label_mask], 0, inner_index)

    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((ks,)).unsqueeze(1).to(device)
    uns_encoded_source = alpha * encoded_source[s_indices] + (1 - alpha) * inner_encoded
    yhat_source = alpha * source_softmax_out[s_indices] + (1 - alpha) * inner_labeled
    un_source_logits = cls_model(uns_encoded_source)
    semi_loss = Semi_loss(un_source_logits, yhat_source, s_rn_weight)

    # outer
    outer_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(kt,)).to(device)
    outer_labeled = torch.index_select(source_data.y[label_mask], 0, outer_index)
    outer_labeled = F.one_hot(outer_labeled, source_softmax_out.size(1))
    outer_encoded = torch.index_select(encoded_source[label_mask], 0, outer_index)

    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((kt,)).unsqueeze(1).to(device)
    uns_encoded_target = alpha * encoded_target[t_indices] + (1 - alpha) * outer_encoded
    yhat_target = alpha * target_softmax_out[t_indices] + (1 - alpha) * outer_labeled
    un_target_logits = cls_model(uns_encoded_target)
    semi_loss = Semi_loss(un_target_logits, yhat_target, t_rn_weight)

    total_loss = cls_loss + loss_grl + 1e-3 * info_loss + 5 * semi_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=False)
    optimizer.step()
    #print(f"Total Loss: {total_loss.item():.6f}, GRL Loss: {loss_grl.item():.6f}, Info Loss: {info_loss.item():.6f}, Semi Loss: {semi_loss.item():.6f}")

    return target_adloss, new_s_edge_weight, new_t_edge_weight, new_s_x, new_t_x, encoded_source, encoded_target

def test_Backbone_1(data, domain='source', mask=None, reparam=False):
    for model in models:
        model.eval()
    if domain == 'source':
        new_x = mask_feature(data.x, 'source', reparam)
        new_edge_weight = drop_edge(drop_edge.s_edge_prob, device, reparam)
    elif domain == 'target':
        new_x = mask_feature(data.x, 'target', reparam)
        new_edge_weight = drop_edge(drop_edge.t_edge_prob, device, reparam)

    encoded_output = GNN_encoder(new_x, data.edge_index, new_edge_weight)

    (_, _), encoded_output = gib_layer(encoded_output, reparam)

    if mask is not None:
        encoded_output = encoded_output[mask]
    logits = cls_model(encoded_output)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return encoded_output, logits, preds, accuracy, macro_f1, micro_f1

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

for epoch in range(1, epochs):
    pair_data_1 = train_Backbone_1(epoch)
    source_encoded_output_Backbone_1, source_logits_Backbone_1, source_preds_Backbone_1, source_correct, _, _ = test_Backbone_1(source_data, 'source', source_data.test_mask)
    target_encoded_output_Backbone_1, target_logits_Backbone_1, target_preds_Backbone_1, target_correct, macro_f1, micro_f1 = test_Backbone_1(target_data, 'target')
    #print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct, target_correct, macro_f1, micro_f1))
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        
        torch.save(mask_feature.state_dict(), os.path.join(save_dir, 'mask_feature.pth'))
        torch.save(drop_edge.state_dict(), os.path.join(save_dir, 'drop_edge.pth'))
        torch.save(GNN_encoder.state_dict(), os.path.join(save_dir, 'GNN_encoder.pth'))
        torch.save(gib_layer.state_dict(), os.path.join(save_dir, 'gib_layer.pth'))
        torch.save(cls_model.state_dict(), os.path.join(save_dir, 'cls_model.pth'))
        torch.save(domain_model.state_dict(), os.path.join(save_dir, 'domain_model.pth'))


best_source_acc_percent = best_source_acc * 100
best_target_acc_percent = best_target_acc * 100
best_macro_f1_percent = best_macro_f1 * 100
best_micro_f1_percent = best_micro_f1 * 100

line = "{}\n -Backbone_1= Epoch: {}, best_source_acc: {:.4f}, best_target_acc: {:.4f}, best_macro_f1: {:.4f}, best_micro_f1: {:.4f}" \
    .format(args.seed, best_epoch, best_source_acc_percent, best_target_acc_percent, best_macro_f1_percent, best_micro_f1_percent)

print(line)
source_logits_branch1 = source_logits_Backbone_1
source_encoded_output_branch1 = source_encoded_output_Backbone_1
target_encoded_output_branch1 = target_encoded_output_Backbone_1
target_logits_branch1 = target_logits_Backbone_1
target_preds_branch1 = target_preds_Backbone_1
"""
#############################################################################################################
###################################################Path_subnetwork#################################################
#############################################################################################################
"""
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import math

from GNN_model import GNN, GIB, GRL
from Graph_Reduction import MaskFeature, DropEdge
from utils import get_node_central_weight
from Loss_functions import Semi_loss
from pytorch_metric_learning import losses

hidden_channels = 512
out_channels = 256
filter_size = 3
temperature = 0.05

mask_feature = MaskFeature(source_data.x.size(1), device).to(device)
drop_edge = DropEdge(source_data.edge_index.size(1),target_data.edge_index.size(1), device).to(device)
if args.backboneGNN2 == 'pan':
    GNN_encoder2 = GNN(source_data.x.size(1), hidden_channels, hidden_channels, gnn_type=args.backboneGNN2, filter_size=filter_size).to(device)
else:
    GNN_encoder2 = GNN(source_data.x.size(1), hidden_channels, hidden_channels, gnn_type=args.backboneGNN2).to(device)
cls_model = nn.Sequential(nn.Linear(out_channels, dataset.num_classes),).to(device)
gib_layer = GIB(hidden_channels, out_channels).to(device)
domain_model = nn.Sequential(GRL(), nn.Linear(out_channels, 64), nn.ReLU(), nn.Dropout(1e-1), nn.Linear(64, 2),).to(device)

models = [mask_feature, drop_edge, GNN_encoder2, gib_layer, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape [num_samples, feature_dim]
            labels: Tensor of shape [num_samples]
        Returns:
            loss: SupConLoss
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        num_samples = features.shape[0]

        # Normalize the feature vectors
        features = F.normalize(features, dim=1)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute the contrastive loss
        # To prevent numerical issues with logsumexp trick
        exp_similarity_matrix = torch.exp(similarity_matrix)
        exp_similarity_matrix = exp_similarity_matrix * (1 - torch.eye(num_samples).to(device))

        # Compute the log-probabilities for each sample
        log_prob = similarity_matrix - torch.log(exp_similarity_matrix.sum(dim=1, keepdim=True) + 1e-12)

        # Compute the mean log-probabilities for positive samples
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # Loss is the negative mean of these mean log-probabilities
        loss = -mean_log_prob_pos.mean()
        
        return loss
SupConLoss = SupConLoss()


def train_Backbone_PAN(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, args.label_rate)

    new_s_x = mask_feature(source_data.x, 'source')
    new_s_edge_weight = drop_edge(drop_edge.s_edge_prob, device)
    new_t_x = mask_feature(target_data.x, 'target')
    new_t_edge_weight = drop_edge(drop_edge.t_edge_prob, device)

    encoded_source = GNN_encoder2(new_s_x, source_data.edge_index)
    encoded_target = GNN_encoder2(new_t_x, target_data.edge_index)

    (s_mu, s_std), encoded_source = gib_layer(encoded_source)
    (t_mu, t_std), encoded_target = gib_layer(encoded_target)
    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)
    # Classifier Loss

    if args.backboneGNN2 == 'pan':
        criterion = SupConLoss
        cls_loss = criterion(source_logits[label_mask], source_data.y[label_mask])
    else:
        criterion = nn.CrossEntropyLoss()
        cls_loss = criterion(source_logits[label_mask], source_data.y[label_mask])

    # DA loss
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)

    if args.backboneGNN2 == 'pan':
        criterion = SupConLoss
        source_domain_cls_loss = criterion(source_domain_preds, torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device))
        target_domain_cls_loss = criterion(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        source_domain_cls_loss = criterion(source_domain_preds, torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device))
        target_domain_cls_loss = criterion(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))
    criterionad = nn.CrossEntropyLoss(reduction='none')

    target_adloss = criterionad(target_domain_preds, torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device))

    loss_grl = source_domain_cls_loss + target_domain_cls_loss


    # IB loss
    info_loss = -0.5*(1+2*s_std.log()-s_mu.pow(2)-s_std.pow(2)).sum(1).mean().div(math.log(2))
    info_loss += -0.5*(1+2*t_std.log()-t_mu.pow(2)-t_std.pow(2)).sum(1).mean().div(math.log(2))

    source_softmax_out = nn.Softmax(dim=1)(source_logits)
    target_softmax_out = nn.Softmax(dim=1)(target_logits)
    source_pseudo_label = source_softmax_out.argmax(dim=1)
    target_pseudo_label = target_softmax_out.argmax(dim=1)
    source_pseudo_label[label_mask] = source_data.y[label_mask]

    
    mu = 0.5 - math.cos(min(math.pi,(2*math.pi*float(epoch) / epoch)))/2
    ks = int(source_data.y[label_mask].size(0)*mu)*3
    kt = int(source_data.y[label_mask].size(0)*mu)*3
    s_rn_weight, s_indices = get_node_central_weight(source_data, new_s_edge_weight, source_pseudo_label, ks, device)
    t_rn_weight, t_indices = get_node_central_weight(target_data, new_t_edge_weight, target_pseudo_label, kt, device)

    ##  inner
    source_softmax_out[label_mask] = F.one_hot(source_data.y,source_softmax_out.size(1)).float()[label_mask]
    inner_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(ks,)).to(device)
    inner_labeled = torch.index_select(source_data.y[label_mask], 0, inner_index)
    inner_labeled = F.one_hot(inner_labeled,source_softmax_out.size(1))
    inner_encoded = torch.index_select(encoded_source[label_mask], 0, inner_index)
    
    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((ks,)).unsqueeze(1).to(device)
    uns_encoded_source = alpha * encoded_source[s_indices] + (1-alpha) * inner_encoded
    yhat_source = alpha * source_softmax_out[s_indices] + (1-alpha) * inner_labeled
    un_source_logits = cls_model(uns_encoded_source)
    semi_loss = Semi_loss(un_source_logits, yhat_source, s_rn_weight)

    ##  outter 
    outer_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(kt,)).to(device)
    outer_labeled = torch.index_select(source_data.y[label_mask], 0, outer_index)
    outer_labeled = F.one_hot(outer_labeled,source_softmax_out.size(1))
    outer_encoded = torch.index_select(encoded_source[label_mask], 0, outer_index)

    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((kt,)).unsqueeze(1).to(device)
    uns_encoded_target = alpha * encoded_target[t_indices]+ (1-alpha) * outer_encoded
    yhat_target = alpha * target_softmax_out[t_indices] + (1-alpha) * outer_labeled
    un_target_logits = cls_model(uns_encoded_target)
    semi_loss = Semi_loss(un_target_logits, yhat_target, t_rn_weight)

    total_loss = cls_loss + loss_grl + 1e-3*info_loss + 5*semi_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=False)
    optimizer.step()
    #print(f"Total Loss: {total_loss.item():.6f}, CLS Loss: {cls_loss.item():.6f}, GRL Loss: {loss_grl.item():.6f}, Info Loss: {info_loss.item():.6f}, Semi Loss: {semi_loss.item():.6f}")

    return target_adloss, new_s_edge_weight, new_t_edge_weight, new_s_x, new_t_x, encoded_source, encoded_target

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0

def test_Backbone_PAN(data, domain = 'source',mask=None, reparam=False):
    for model in models:
        model.eval()
    if domain == 'source':
        new_x = mask_feature(data.x, 'source', reparam)
        new_edge_weight = drop_edge(drop_edge.s_edge_prob, device, reparam)
    elif domain == 'target':
        new_x = mask_feature(data.x, 'target', reparam)
        new_edge_weight = drop_edge(drop_edge.t_edge_prob, device, reparam)

    encoded_output = GNN_encoder2(new_x, data.edge_index)

    (_, _), encoded_output = gib_layer(encoded_output, reparam)

    if mask is not None:
        encoded_output = encoded_output[mask]
    logits = cls_model(encoded_output)
    #print("logits:", logits)
    preds = logits.argmax(dim=1)
    #print("preds:", preds)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return encoded_output, logits, preds, accuracy, macro_f1, micro_f1

for epoch in range(1, epochs):
    pair_data_2 = train_Backbone_PAN(epoch)
    source_encoded_output_Backbone_PAN, source_logits_Backbone_PAN, source_preds_Backbone_PAN, source_correct, _, _ = test_Backbone_PAN(source_data, 'source', source_data.test_mask)
    target_encoded_output_Backbone_PAN, target_logits_Backbone_PAN, target_preds_Backbone_PAN, target_correct, macro_f1, micro_f1 = test_Backbone_PAN(target_data, 'target')
    #print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct, target_correct, macro_f1, micro_f1))
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch


best_source_acc_percent = best_source_acc * 100
best_target_acc_percent = best_target_acc * 100
best_macro_f1_percent = best_macro_f1 * 100
best_micro_f1_percent = best_micro_f1 * 100

line = "{}\n -Backbone_2= Epoch: {}, best_source_acc: {:.4f}, best_target_acc: {:.4f}, best_macro_f1: {:.4f}, best_micro_f1: {:.4f}" \
    .format(args.seed, best_epoch, best_source_acc_percent, best_target_acc_percent, best_macro_f1_percent, best_micro_f1_percent)

print(line)
source_logits_branch2 = source_logits_Backbone_PAN
source_encoded_output_branch2 = source_encoded_output_Backbone_PAN
target_encoded_output_branch2 = target_encoded_output_Backbone_PAN
target_logits_branch2 = target_logits_Backbone_PAN
target_preds_branch2 = target_preds_Backbone_PAN


import torch.nn.functional as F
# Identify different indices
target_probs_branch1 = F.softmax(target_logits_Backbone_1, dim=1)
target_probs_branch2 = F.softmax(target_logits_Backbone_PAN, dim=1)
cosine_similarity = F.cosine_similarity(target_probs_branch1, target_probs_branch2, dim=1)
cosine_distance = 1 - cosine_similarity

epochs = args.epochs




"""
#############################################################################################################
###########################Proposed###########################################################
#############################################################################################################
"""

#print(normalized_similarities_embedding)
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-12)

import torch
import numpy as np
from torch_geometric.data import Data

marked_features = source_data.x[label_mask]
degrees = degree(source_data.edge_index[0], source_data.x.size(0)).to(device)
marked_degrees = degrees[label_mask]
weighted_mean_marked_features = torch.sum(marked_features * marked_degrees.view(-1, 1), dim=0) / marked_degrees.sum()
#mean_marked_features = marked_features.mean(dim=0)

distances = torch.norm(target_data.x - weighted_mean_marked_features, p=2, dim=1)

sorted_distances, sorted_indices = torch.sort(distances, descending=True)
#distances = min_max_normalize(distances)
distances = distances.detach().cpu().numpy()
distances = distances.reshape(-1, 1)

normalized_similarities_embedding = distances

import numpy as np
import torch
from torch_geometric.utils import degree, add_self_loops, k_hop_subgraph
import torch.nn.functional as F

def compute_khop(data, k):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    all_subgraph_nodes = []

    for node in range(num_nodes):
        subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        all_subgraph_nodes.append((subgraph_nodes.tolist(), subgraph_edge_index))

    return all_subgraph_nodes

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-12)

def calculate_khop_entropy_and_kl(logits, subgraph_logits):
    logits_probs = F.softmax(logits, dim=1)
    subgraph_probs = F.softmax(subgraph_logits, dim=1)
    khop_entropy = -torch.sum(subgraph_probs * torch.log(subgraph_probs + 1e-12), dim=1)
    khop_entropy = min_max_normalize(khop_entropy)
    node_entropy = -torch.sum(logits_probs * torch.log(logits_probs + 1e-12), dim=1)
    node_entropy = min_max_normalize(node_entropy)

    kl_divergence = torch.sum(logits_probs * (torch.log(logits_probs + 1e-12) - torch.log(subgraph_probs + 1e-12)), dim=1)
    kl_divergence = min_max_normalize(kl_divergence)
    khop_entropy_kl_divergence = khop_entropy + kl_divergence
    node_entropy_kl_divergence = node_entropy + kl_divergence
    return khop_entropy, node_entropy, kl_divergence, khop_entropy_kl_divergence, node_entropy_kl_divergence

def calculate_weightedkhop_subgraph_logits(data, logits, k):
    all_subgraph_info = compute_khop(data, k)
    subgraph_logits = []

    for subgraph_nodes, subgraph_edge_index in all_subgraph_info:
        try:
            subgraph_edge_index, _ = add_self_loops(subgraph_edge_index)
            
            subgraph_node_degrees = degree(subgraph_edge_index[0], len(subgraph_nodes))
            
            subgraph_node_weights = 1.0 / subgraph_node_degrees
            
            subgraph_logits_weighted = logits[subgraph_nodes] * subgraph_node_weights.view(-1, 1)
            subgraph_logit = subgraph_logits_weighted.mean(dim=0)
            subgraph_logits.append(subgraph_logit)
        except IndexError as e:
            print(f"IndexError: {e} with nodes: {subgraph_nodes}")
            continue

    return torch.stack(subgraph_logits)

# Calculate k-hop subgraph logits
weightedsubgraph_logits1 = calculate_weightedkhop_subgraph_logits(target_data, target_logits_Backbone_1, k=args.k)
weightedsubgraph_logits2 = calculate_weightedkhop_subgraph_logits(target_data, target_logits_Backbone_PAN, k=args.k)

# Calculate KL
weightedkhop_entropy_scores1, node_entropy_scores1, kl_divergence_scores1, weightedkhop_entropy_kl_divergence_scores1, node_entropy_kl_divergence_scores1 = calculate_khop_entropy_and_kl(target_logits_Backbone_1, weightedsubgraph_logits1)
weightedkhop_entropy_scores2, node_entropy_scores2, kl_divergence_scores2, weightedkhop_entropy_kl_divergence_scores2, node_entropy_kl_divergence_scores2 = calculate_khop_entropy_and_kl(target_logits_Backbone_PAN, weightedsubgraph_logits2)
weightedkhop_entropy_scores = weightedkhop_entropy_scores1 + weightedkhop_entropy_scores2

# Convert numpy arrays to PyTorch tensors
target_preds_branch1 = torch.tensor(target_logits_Backbone_1).to('cuda')
target_preds_branch2 = torch.tensor(target_logits_Backbone_PAN).to('cuda')
target_encoded_output_branch1 = torch.tensor(target_encoded_output_Backbone_1).to('cuda')
target_logits_branch1 = torch.tensor(target_logits_Backbone_1).to('cuda')

weightedkhop_entropy_scores_reshaped = weightedkhop_entropy_scores.reshape(-1, 1)
weightedkhop_entropy_scores_cpu = weightedkhop_entropy_scores_reshaped.detach().cpu().numpy()
DELTA_Score = weightedkhop_entropy_scores_cpu + normalized_similarities_embedding
#print("All scores:", all_scores)

# Identify different indices
different_indices_weighted_page = (cosine_distance > args.threhold).nonzero(as_tuple=True)[0].cpu()
#different_indices_weighted_KL = (target_preds_branch1 != target_preds_branch2).nonzero(as_tuple=True)[0].cpu()
different_logits_branch1 = target_logits_branch1[different_indices_weighted_page]
different_encoded_output_branch1 = target_encoded_output_branch1[different_indices_weighted_page]


import numpy as np
import torch
from torch_geometric.utils import degree, add_self_loops, k_hop_subgraph
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Identify different indices
different_indices_weighted_KL = (cosine_distance > args.threhold).nonzero(as_tuple=True)[0].cpu().numpy()

# Extract scores for different indices
different_scores = DELTA_Score[different_indices_weighted_KL]

# Get the top `args.n_clusters` indices based on the scores
top_indices_in_different = np.argsort(different_scores, axis=0)[-args.n_clusters:][::-1]
top_indices = different_indices_weighted_KL[top_indices_in_different.flatten()]

# Assign pseudo-labels and update masks
label_rate_target = 0.0
target_train_size = int(target_data.size(0) * label_rate_target)
label_mask_weightedkhop_entropy_KL = np.array([1] * target_train_size + [0] * (target_data.size(0) - target_train_size)).astype(bool)
label_mask_weightedkhop_entropy_KL[top_indices] = True
label_mask_weightedkhop_entropy_KL = torch.tensor(label_mask_weightedkhop_entropy_KL).to(device)

# Optional: Print selected indices for verification
print("Top indices based on scores:", top_indices)

mask_feature.load_state_dict(torch.load(os.path.join(save_dir, 'mask_feature.pth')))
drop_edge.load_state_dict(torch.load(os.path.join(save_dir, 'drop_edge.pth')))
GNN_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'GNN_encoder.pth')))
gib_layer.load_state_dict(torch.load(os.path.join(save_dir, 'gib_layer.pth')))
cls_model.load_state_dict(torch.load(os.path.join(save_dir, 'cls_model.pth')))
domain_model.load_state_dict(torch.load(os.path.join(save_dir, 'domain_model.pth')))

models = [mask_feature, drop_edge, GNN_encoder, gib_layer, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
SupConLoss = losses.SupConLoss(temperature=temperature)


def train_DELTA(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, args.label_rate)

    new_s_x = mask_feature(source_data.x, 'source')
    new_s_edge_weight = drop_edge(drop_edge.s_edge_prob, device)
    new_t_x = mask_feature(target_data.x, 'target')
    new_t_edge_weight = drop_edge(drop_edge.t_edge_prob, device)

    encoded_source = GNN_encoder(new_s_x, source_data.edge_index, new_s_edge_weight)
    encoded_target = GNN_encoder(new_t_x, target_data.edge_index, new_t_edge_weight)

    (s_mu, s_std), encoded_source = gib_layer(encoded_source)
    (t_mu, t_std), encoded_target = gib_layer(encoded_target)
    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)
    # Classifier Loss
    cls_loss1 = CrossEntropyLoss(source_logits[label_mask], source_data.y[label_mask])
    cls_loss2 = CrossEntropyLoss(target_logits[label_mask_weightedkhop_entropy_KL], target_data.y[label_mask_weightedkhop_entropy_KL])

    # DA loss
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)

    source_domain_cls_loss = CrossEntropyLoss(
        source_domain_preds,
        torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    target_domain_cls_loss = CrossEntropyLoss(
        target_domain_preds,
        torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    loss_grl = source_domain_cls_loss + target_domain_cls_loss


    # IB loss
    info_loss = -0.5*(1+2*s_std.log()-s_mu.pow(2)-s_std.pow(2)).sum(1).mean().div(math.log(2))
    info_loss += -0.5*(1+2*t_std.log()-t_mu.pow(2)-t_std.pow(2)).sum(1).mean().div(math.log(2))

    source_softmax_out = nn.Softmax(dim=1)(source_logits)
    target_softmax_out = nn.Softmax(dim=1)(target_logits)
    source_pseudo_label = source_softmax_out.argmax(dim=1)
    target_pseudo_label = target_softmax_out.argmax(dim=1)
    source_pseudo_label[label_mask] = source_data.y[label_mask]
    target_pseudo_label[label_mask_weightedkhop_entropy_KL] = target_data.y[label_mask_weightedkhop_entropy_KL]
    
    mu = 0.5 - math.cos(min(math.pi,(2*math.pi*float(epoch) / epoch)))/2
    ks = int(source_data.y[label_mask].size(0)*mu)*3
    kt = int(source_data.y[label_mask].size(0)*mu)*3
    s_rn_weight, s_indices = get_node_central_weight(source_data, new_s_edge_weight, source_pseudo_label, ks, device)
    t_rn_weight, t_indices = get_node_central_weight(target_data, new_t_edge_weight, target_pseudo_label, kt, device)

    ##  inner
    source_softmax_out[label_mask] = F.one_hot(source_data.y,source_softmax_out.size(1)).float()[label_mask]
    inner_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(ks,)).to(device)
    inner_labeled = torch.index_select(source_data.y[label_mask], 0, inner_index)
    inner_labeled = F.one_hot(inner_labeled,source_softmax_out.size(1))
    inner_encoded = torch.index_select(encoded_source[label_mask], 0, inner_index)
    
    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((ks,)).unsqueeze(1).to(device)
    uns_encoded_source = alpha * encoded_source[s_indices] + (1-alpha) * inner_encoded
    yhat_source = alpha * source_softmax_out[s_indices] + (1-alpha) * inner_labeled
    un_source_logits = cls_model(uns_encoded_source)
    semi_loss = Semi_loss(un_source_logits, yhat_source, s_rn_weight)

    ##  outter 
    outer_index = torch.randint(low=0, high=source_logits[label_mask].size(0), size=(kt,)).to(device)
    outer_labeled = torch.index_select(source_data.y[label_mask], 0, outer_index)
    outer_labeled = F.one_hot(outer_labeled,source_softmax_out.size(1))
    outer_encoded = torch.index_select(encoded_source[label_mask], 0, outer_index)

    beta = 1
    alpha = torch.distributions.Beta(beta, beta).sample((kt,)).unsqueeze(1).to(device)
    uns_encoded_target = alpha * encoded_target[t_indices]+ (1-alpha) * outer_encoded
    yhat_target = alpha * target_softmax_out[t_indices] + (1-alpha) * outer_labeled
    un_target_logits = cls_model(uns_encoded_target)
    semi_loss = Semi_loss(un_target_logits, yhat_target, t_rn_weight)

    total_loss = cls_loss1 + cls_loss2 + loss_grl + 1e-3*info_loss + 5*semi_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=False)
    optimizer.step()
    #print(f"Total Loss: {total_loss.item():.6f}, GRL Loss: {loss_grl.item():.6f}, Info Loss: {info_loss.item():.6f}, Semi Loss: {semi_loss.item():.6f}")

    return (new_s_edge_weight, new_t_edge_weight, new_s_x, new_t_x, encoded_source, encoded_target)

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0

def test_DELTA(data, domain = 'source',mask=None, reparam=False):
    for model in models:
        model.eval()
    if domain == 'source':
        new_x = mask_feature(data.x, 'source', reparam)
        new_edge_weight = drop_edge(drop_edge.s_edge_prob, device, reparam)
    elif domain == 'target':
        new_x = mask_feature(data.x, 'target', reparam)
        new_edge_weight = drop_edge(drop_edge.t_edge_prob, device, reparam)

    encoded_output = GNN_encoder(new_x, data.edge_index)

    (_, _), encoded_output = gib_layer(encoded_output, reparam)

    if mask is not None:
        encoded_output = encoded_output[mask]
    logits = cls_model(encoded_output)
    #print("logits:", logits)
    preds = logits.argmax(dim=1)
    #print("preds:", preds)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return encoded_output, logits, preds, accuracy, macro_f1, micro_f1

for epoch in range(1, epochs):
    pair_data = train_DELTA(epoch)
    source_encoded_output, source_logits, source_preds, source_correct, _, _ = test_DELTA(source_data, 'source', source_data.test_mask)
    target_encoded_output, target_logits, target_preds, target_correct, macro_f1, micro_f1 = test_DELTA(target_data, 'target')
    #print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct, target_correct, macro_f1, micro_f1))
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch


best_source_acc_percent = best_source_acc * 100
best_target_acc_percent = best_target_acc * 100
best_macro_f1_percent = best_macro_f1 * 100
best_micro_f1_percent = best_micro_f1 * 100

line = "{}\n -DELTA= Epoch: {}, best_source_acc: {:.4f}, best_target_acc: {:.4f}, best_macro_f1: {:.4f}, best_micro_f1: {:.4f}" \
    .format(args.seed, best_epoch, best_source_acc_percent, best_target_acc_percent, best_macro_f1_percent, best_micro_f1_percent)

print(line)
