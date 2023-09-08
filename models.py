import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.sage_conv import SAGEConv
from torch_geometric.data import Data

from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import numpy as np


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

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)

        return inputs + rand.to(inputs.device)


class FeatGenerator(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(FeatGenerator, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.dropout = dropout
        self.sample = Sampling()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))

        return x


class NumPredictor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


# Mend the graph via NeighGen
class MendGraph(nn.Module):
    def __init__(self, num_pred):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph(self, x, edge_index, pred_degree, gen_feats):
        device = gen_feats.device
        num_node, num_feature = x.shape
        new_edges = []
        gen_feats = gen_feats.view(-1, self.num_pred, num_feature)

        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        x = x.detach()
        fill_feats = torch.vstack((x, gen_feats.view(-1, num_feature)))

        for i in range(num_node):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                new_edges.append(
                    np.asarray([i, num_node + i * self.num_pred + j]))

        new_edges = torch.tensor(np.asarray(new_edges).reshape((-1, 2)),
                                 dtype=torch.int64).T
        new_edges = new_edges.to(device)
        if len(new_edges) > 0:
            fill_edges = torch.hstack((edge_index, new_edges))
        else:
            fill_edges = torch.clone(edge_index)
        return fill_feats, fill_edges

    def forward(self, x, edge_index, pred_missing, gen_feats):
        fill_feats, fill_edges = self.mend_graph(x, edge_index, pred_missing,
                                                 gen_feats)

        return fill_feats, fill_edges


class LocalSage_Plus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 gen_hidden,
                 linear,
                 dropout=0.5,
                 num_pred=5):
        super(LocalSage_Plus, self).__init__()

        self.encoder_model = GNNLayer(sageMode="GraphSAGE",
                                      in_feats=in_channels,
                                      out_feats=gen_hidden,
                                      h_feats=hidden,
                                      linear=linear,
                                      layer_size=2)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = FeatGenerator(latent_dim=gen_hidden,
                                 dropout=dropout,
                                 num_pred=num_pred,
                                 feat_shape=in_channels)
        self.mend_graph = MendGraph(num_pred)

        self.classifier = GNNLayer(sageMode="GraphSAGE",
                                      in_feats=in_channels,
                                      out_feats=out_channels,
                                      h_feats=hidden,
                                      linear=linear,
                                      layer_size=2)

    def forward(self, data):
        x = self.encoder_model(data.x, data.edge_index)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(mend_feats, mend_edge_index)
        return degree, gen_feat, nc_pred[:data.num_nodes]

    def inference(self, impared_data, raw_data):
        x = self.encoder_model(impared_data.x, impared_data.edge_index)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(raw_data.x,
                                                      raw_data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(mend_feats, mend_edge_index)
        return degree, gen_feat, nc_pred[:raw_data.num_nodes]


class FedSage_Plus(nn.Module):
    def __init__(self, local_graph: LocalSage_Plus):
        super(FedSage_Plus, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, data):
        x = self.encoder_model(data.x, data.edge_index)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(mend_feats, mend_edge_index)
        return degree, gen_feat, nc_pred[:data.num_nodes]