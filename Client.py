from torch_geometric.data.data import Data
from collections import defaultdict
from torch_geometric.loader import NeighborLoader
from models import *
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import *

class Client(nn.Module):
    def __init__(self, data, mode, sageMode, in_feats, h_feats, num_classes, test_num, lr, dropout,
                 device, mixup):
        super(Client, self).__init__()
        self.data = data
        self.mode = mode
        self.sageMode = sageMode
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.test_num = test_num
        self.node_num = self.data.x.shape[0]
        self.device = device
        self.prox_miu = 1.0
        self.mixup = mixup

        self.lowLayer = LowLayer(in_feats, h_feats).to(self.device)
        self.gnnLayer = GNNLayer(
            sageMode, h_feats, h_feats, h_feats, layer_size=2).to(self.device)
        self.classification = Classification(
            h_feats, num_classes).to(self.device)
        if self.mode == "fedego":
            self.dropout = nn.Dropout(p=dropout)

        if self.mode in ["fedgcn"]:
            self.fedgcn = FedGCN(
                in_feats, h_feats, num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.init()

    @torch.no_grad()
    def init(self):
        """calculate the nodes of a specific label"""
        self.cnt = torch.zeros(self.num_classes)
        for node in range(self.node_num):
            self.cnt[self.data.y[node]] += 1

    @torch.no_grad()
    def getMask(self, upload=False):
        if self.mode == "graphfl":
            val_size = 80
            test_size = self.test_num
            train_size = 80
        else:
            val_size = self.node_num // 5
            test_size = self.test_num
            if upload:
                train_size = self.node_num - val_size - test_size
            else:
                train_size = self.node_num // 5
        rand_indices = np.random.permutation(
            self.node_num - test_size - val_size)
        self.train_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.train_mask[rand_indices[:train_size]] = True
        self.val_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.val_mask[rand_indices[train_size:min(train_size + val_size, self.node_num -
                                                  test_size)]] = True
        self.test_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.test_mask[self.node_num - test_size:] = True

    @torch.no_grad()
    def getLoader(self, batch_size, mode="train", node=None, num_workers=0):
        if mode == "train":
            data = self.data
            input_nodes = self.train_mask
        elif mode == "val":
            data = self.data
            input_nodes = self.val_mask
        elif mode == "share":
            data = self.lowEmbed
            input_nodes = self.train_mask
        elif mode == "test":
            data = self.data
            input_nodes = self.test_mask
        loader = NeighborLoader(data,
                                num_neighbors=[6] * 2,
                                batch_size=batch_size,
                                input_nodes=input_nodes,
                                replace=True,
                                directed=True,
                                num_workers=num_workers)
        return loader

    @torch.no_grad()
    def getGradient(self):
        gradient_dict = {}
        for name, param in self.named_parameters():
            gradient_dict.update({name: param.grad.detach()})
        return gradient_dict

    def getLoss(self, out, y):
        x = F.softmax(out, dim=1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss = y_one_hot * torch.log(x + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

    def getProxLoss(self):
        loss = []
        for param in self.state_dict().keys():
            loss.append(
                torch.norm(self.state_dict()[param] - self.init_model[param].detach(),
                           p='fro') / torch.numel(self.state_dict()[param]))
        return sum(loss)

    @torch.no_grad()
    def initShareBatch(self, layer=3):
        self.share_x = []
        self.share_edge_index_u = []
        self.share_edge_index_v = []
        self.share_edge_index = []
        self.share_y = []
        self.share_mask = []
        self.share_node_idx = []
        self.share_node_cnt = 0
        self.map_layer = [[] for l in range(layer)]
        node_cnt = 0
        for l in range(layer):
            self.map_layer[l] = [
                node for node in range(node_cnt, node_cnt + 6**l)]
            node_cnt += 6**l
        self.add_u = []
        self.add_v = []

        # from u to v
        for v in range(6 + 1):
            for u in range(6 * v + 1, 6 * v + 7):
                self.add_u.append(u)
                self.add_v.append(v)

    @torch.no_grad()
    def updateShareBatch(self, batch, layer=3):
        b_size = batch.batch_size
        adj_lists = defaultdict(list)
        edge_index = batch.edge_index.cpu()
        edge_cnt = edge_index.shape[1]
        x = batch.x.cpu()
        y = batch.y
        y = F.one_hot(y, num_classes=self.num_classes).float().cpu()

        # reconstruct the adjacency list
        for i in range(edge_cnt):
            adj_lists[int(edge_index[1][i])].append(int(edge_index[0][i]))
        # nodes in each layer
        node_layer = [[] for l in range(layer)]
        # node 0, use node 0 to be uploaded if Mixup is not applied
        if self.mixup:
            node_layer[0] = [node for node in range(b_size)]
        else:
            node_layer[0] = [0 for node in range(b_size)]

        for l in range(1, layer):
            for node in node_layer[l - 1]:
                node_layer[l] += adj_lists[node]

        share_x = torch.zeros((43, self.h_feats))
        share_y = torch.zeros((43, self.num_classes))

        # a new ego-graph
        self.share_edge_index_u += [u +
                                    self.share_node_cnt for u in self.add_u]
        self.share_edge_index_v += [v +
                                    self.share_node_cnt for v in self.add_v]

        # Mixup
        for l in range(layer):
            layer_length = len(self.map_layer[l])
            for i in range(len(node_layer[l])):
                share_x[self.map_layer[l][i %
                                          layer_length]] += x[node_layer[l][i]]
                share_y[self.map_layer[l][i %
                                          layer_length]] += y[node_layer[l][i]]
        share_x /= b_size
        share_y /= b_size
        self.share_x.append(share_x)
        self.share_y.append(share_y)
        self.share_node_idx.append(self.share_node_cnt)
        self.share_node_cnt += 43

    @torch.no_grad()
    def setLamb(self, global_p, lamb_c, lamb_fixed=False):
        self.p = torch.sum(
            F.one_hot(self.data.y, num_classes=self.num_classes).float(), axis=0)
        self.p /= self.data.y.shape[0]
        self.emd = torch.sum(abs(self.p-global_p))
        # self.lamb = 1 - (7.5/(self.emd+3)-1.5).item()\
        if lamb_fixed == True:
            self.lamb = lamb_c
        else:
            self.lamb = pow(self.emd/2, lamb_c)

    @torch.no_grad()
    def getLowEmbed(self):
        x = self.data.x.to(self.device)
        x = self.lowLayer(x)
        self.lowEmbed = Data(
            x=x, edge_index=self.data.edge_index, y=self.data.y)

    @torch.no_grad()
    def getShareBatch(self, batch_size):
        self.eval()

        with torch.no_grad():
            self.getLowEmbed()
            self.getMask(upload=True)

        train_loader = self.getLoader(batch_size, mode="share")

        for batch in train_loader:
            self.updateShareBatch(batch)

        self.share_x = torch.cat(self.share_x, 0).detach()
        self.share_y = torch.cat(self.share_y, 0).detach()
        self.share_edge_index = torch.stack([
            torch.tensor(self.share_edge_index_u),
            torch.tensor(self.share_edge_index_v)
        ], 0).detach()
        self.share_mask = torch.zeros(
            self.share_node_cnt, dtype=torch.bool).detach()
        self.share_mask[self.share_node_idx] = True
        return self.share_x, self.share_edge_index, self.share_y, self.share_mask

    def supervisedTrain(self, batch_size):
        self.train()

        with torch.no_grad():
            self.getMask()

        train_loader = self.getLoader(batch_size, mode="train")
        total_examples = total_loss = 0
        
        if self.mode == "fedego":
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size
                x = batch.x
                y = batch.y
                edge_index = batch.edge_index

                x = self.lowLayer(x)
                x = self.dropout(x)
                x = self.gnnLayer(x, edge_index)
                x = self.dropout(x)
                out = self.classification(x)[:b_size]
                loss = self.getLoss(out, y[:b_size])

                loss.backward()
                self.optimizer.step()

                total_examples += b_size
                total_loss += float(loss) * b_size
        else:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size
                x = batch.x
                y = batch.y
                edge_index = batch.edge_index

                x = self.lowLayer(x)
                x = self.gnnLayer(x, edge_index)
                out = self.classification(x)[:b_size]
                loss = F.cross_entropy(out, y[:b_size])

                if self.mode == "fedprox":
                    prox_loss = self.getProxLoss()
                    if loss > prox_loss * 5 * self.prox_miu and self.prox_miu <= 1:
                        self.prox_miu *= 10
                    elif loss < prox_loss * self.prox_miu / 2 and self.prox_miu >= 0.0001:
                        self.prox_miu /= 10
                    loss += self.prox_miu * prox_loss

                loss.backward()
                self.optimizer.step()

                total_examples += b_size
                total_loss += float(loss) * b_size

        return total_loss / total_examples

    def graphflSupport(self, batch_size):
        self.train()
        train_loader = self.getLoader(batch_size, mode="train")
        total_examples = total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            b_size = batch.batch_size
            x = batch.x
            y = batch.y
            edge_index = batch.edge_index

            x = self.lowLayer(x)
            x = self.gnnLayer(x, edge_index)
            out = self.classification(x)[:b_size]
            loss = F.cross_entropy(out, y[:b_size])
            loss.backward()
            self.optimizer.step()

            total_examples += b_size
            total_loss += float(loss) * b_size
        return total_loss / total_examples

    def graphflQuery(self, batch_size):
        self.train()
        train_loader = self.getLoader(batch_size, mode="val")
        total_examples = total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            b_size = batch.batch_size
            x = batch.x
            y = batch.y
            edge_index = batch.edge_index

            x = self.lowLayer(x)
            x = self.gnnLayer(x, edge_index)
            out = self.classification(x)[:b_size]
            loss = F.cross_entropy(out, y[:b_size])
            loss.backward()

            total_examples += b_size
            total_loss += float(loss) * b_size
        return total_loss / total_examples

    @torch.no_grad()
    def evaluate(self, batch_size):
        self.eval()

        val_loader = self.getLoader(batch_size, mode="test")

        pred_all = []
        y_all = []
        total_examples = total_loss = 0

        for batch in val_loader:
            batch = batch.to(self.device)
            b_size = batch.batch_size
            x = batch.x
            edge_index = batch.edge_index
            y = batch.y

            x = self.lowLayer(x)
            x = self.gnnLayer(x, edge_index)
            out = self.classification(x)[:b_size]
            pred = out.argmax(dim=-1)

            pred_all += pred.tolist()
            y_all += y[:b_size].tolist()

            loss = self.getLoss(out, y[:b_size])

            total_examples += b_size
            total_loss += float(loss) * b_size

        micro_f1 = f1_score(y_all, pred_all, average='micro')
        return micro_f1, total_loss/total_examples

    @torch.no_grad()
    def test(self, test_data, batch_size, print_y=False):
        self.eval()
        test_loader = NeighborLoader(test_data,
                                     num_neighbors=[6] * 2,
                                     batch_size=batch_size,
                                     input_nodes=None,
                                     replace=True,
                                     directed=True,
                                     num_workers=0)

        pred_all = []
        y_all = []

        for batch in test_loader:
            batch = batch.to(self.device)
            b_size = batch.batch_size
            x = batch.x
            edge_index = batch.edge_index
            y = batch.y

            x = self.lowLayer(x)
            x = self.gnnLayer(x, edge_index)
            out = self.classification(x)[:b_size]
            pred = out.argmax(dim=-1)

            pred_all += pred.tolist()
            y_all += y[:b_size].tolist()

        if print_y == True:
            distibution = np.zeros(self.num_classes)
            for node in pred_all:
                distibution[node] += 1
            print("distribution")
            print((np.array(distibution)/len(pred_all)).tolist())
        micro_f1 = f1_score(y_all, pred_all, average='micro')
        return micro_f1

    @torch.no_grad()
    def getWD(self, global_dict):
        self.wd = 0
        local_dict = self.state_dict()
        for key in global_dict.keys():
            self.wd += np.linalg.norm(global_dict[key].cpu().numpy().flatten() -
                                      local_dict[key].cpu().numpy().flatten(), ord=2)

    def getMaskLoss(self, out, y, mask):
        x = F.softmax(out, dim=1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss = y_one_hot * torch.log(x + 0.0000001)
        loss = torch.sum(loss, dim=1).mul(mask)
        loss = -torch.mean(loss, dim=0)
        return loss

    def fedgcn_train(self):
        self.train()

        with torch.no_grad():
            self.getMask()

        self.optimizer.zero_grad()
        x = self.data.x.to(self.device)
        y = self.data.y.to(self.device)
        adj = self.adj.to(self.device)
        mask = self.train_mask.to(self.device)

        # x = self.fedgcn(x, adj)
        # out = self.classification(x)
        out = self.fedgcn(x, adj)
        loss = self.getMaskLoss(out, y, mask)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def fedgcn_evaluate(self):
        self.eval()

        with torch.no_grad():
            self.getMask()

        pred_all = []
        y_all = []

        x = self.data.x.to(self.device)
        y = self.data.y.to(self.device)
        adj = self.adj.to(self.device)
        mask = self.test_mask.to(self.device)

        # x = self.fedgcn(x, adj)
        # out = self.classification(x)
        # pred = out.argmax(dim=-1)

        out = self.fedgcn(x, adj)
        pred = out.argmax(dim=-1)

        pred_all += pred.tolist()
        y_all += y.tolist()

        loss = self.getMaskLoss(out, y, mask)

        micro_f1 = f1_score(y_all, pred_all, average='micro')
        return micro_f1, loss.item()

    @torch.no_grad()
    def fedgcn_test(self, test_data, test_adj, print_y=False):
        self.eval()

        pred_all = []
        y_all = []

        x = test_data.x.to(self.device)
        y = test_data.y.to(self.device)
        test_adj = test_adj.to(self.device)

        # x = self.fedgcn(x, test_adj)
        # out = self.classification(x)
        # pred = out.argmax(dim=-1)

        out = self.fedgcn(x, test_adj)
        pred = out.argmax(dim=-1)

        pred_all += pred.tolist()
        y_all += y.tolist()

        if print_y == True:
            distibution = np.zeros(self.num_classes)
            for node in pred_all:
                distibution[node] += 1
            print("distribution")
            print((np.array(distibution)/len(pred_all)).tolist())
        micro_f1 = f1_score(y_all, pred_all, average='micro')
        return micro_f1
