import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.sage_conv import SAGEConv

from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class LowLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LowLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class GNNLayer(torch.nn.Module):
    def __init__(self, sageMode, in_feats, out_feats, h_feats, linear, layer_size=2):
        super(GNNLayer, self).__init__()
        self.layer_size = layer_size
        self.sageMode = sageMode
        self.linear = linear
        if self.sageMode == "GraphSAGE":
            self.sage1 = SAGEConv(in_feats, h_feats)
            self.sage2 = SAGEConv(h_feats, out_feats)
            self.sagex = [SAGEConv(h_feats, h_feats)
                          for layer in range(layer_size - 2)]
        elif self.sageMode == "GAT":
            self.sage1 = GATConv(in_feats, h_feats, dropout=0.5)
            self.sage2 = GATConv(h_feats, out_feats, dropout=0.5)
            self.sagex = [GATConv(h_feats, h_feats, dropout=0.5)
                          for layer in range(layer_size - 2)]
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        if not self.linear:
            x = F.relu(x)
        for layer in range(self.layer_size - 2):
            x = self.sagex[layer](x, edge_index)
            if not self.linear:
                x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        return x


class Classification(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(Classification, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feats))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.weight.mm(embeds.t()).t(), 1)
        return logists


class FedGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(FedGCN, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, out_feats)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x, adj):
        x = self.linear1(x)
        x = adj.mm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = adj.mm(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x
