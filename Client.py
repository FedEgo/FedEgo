from torch_geometric.data.data import Data
from collections import defaultdict
from torch_geometric.loader import NeighborLoader, DataLoader
from models import *
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import *
from utils import HideGraph, GreedyLoss, GraphMender
import torch.nn.functional as F
import copy


class Client(nn.Module):
    def __init__(self, data, mode, sageMode, in_feats, h_feats, num_classes, test_num, lr, dropout,
                 device, mixup, linear, sigma=0, gen_hidden=64, hide_portion=0.5):
        super(Client, self).__init__()
        self.data = data
        self.mode = mode
        self.sageMode = sageMode
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.test_num = test_num
        self.lr = lr
        self.node_num = self.data.x.shape[0]
        self.device = device
        self.prox_miu = 1.0
        self.mixup = mixup
        self.linear = linear
        self.sigma = sigma

        if self.mode in ["fedego_ne"]:
            self.lowLayer = LowLayer(h_feats, h_feats).to(self.device)
            self.gnnLayer = GNNLayer(
                sageMode, in_feats, h_feats, h_feats, linear, layer_size=2).to(self.device)
        else:
            if self.mode not in ["fedsage", "fedsageplus"]:
                self.lowLayer = LowLayer(in_feats, h_feats).to(self.device)
                self.gnnLayer = GNNLayer(
                    sageMode, h_feats, h_feats, h_feats, linear, layer_size=2).to(self.device)
            else:
                self.gnnLayer = GNNLayer(
                    sageMode, in_feats, h_feats, h_feats, linear, layer_size=2).to(self.device)
        self.classification = Classification(
            h_feats, num_classes).to(self.device)

        self.dropout = nn.Dropout(p=dropout)

        if self.mode in ["fedgcn"]:
            self.fedgcn = FedGCN(
                in_feats, h_feats, num_classes).to(self.device)
        
        if self.mode in ["fedsageplus"]:
            self.num_pred = 5
            self.a = 1.0
            self.b = 1.0
            self.c = 1.0
            self.getMask()
            self.data.train_mask = self.train_mask
            self.data.val_mask = self.val_mask
            self.data.test_mask = self.test_mask
            self.hide_data = HideGraph(hide_portion, self.num_pred)(self.data)
            self.local_gen = LocalSage_Plus(self.data.x.shape[-1],
                                      num_classes,
                                      h_feats,
                                      gen_hidden,
                                      linear,
                                      dropout,
                                      num_pred=self.num_pred).to(self.device)
            self.criterion_num = F.smooth_l1_loss
            self.criterion_feat = GreedyLoss

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
        elif mode in ["share"]:
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
            node_layer[0] = np.arange(b_size, dtype=np.int64).tolist()
        else:
            node_layer[0] = np.zeros(b_size, dtype=np.int64).tolist()

        for l in range(1, layer):
            for node in node_layer[l - 1]:
                node_layer[l] += adj_lists[node]

        for l in range(layer):
            node_layer[l] = np.array(node_layer[l], dtype=np.int64)

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
            batch_layer_length = len(node_layer[l])
            pos = [[] for _ in range(layer_length)]
            for i in range(batch_layer_length):
                pos[i % layer_length].append(node_layer[l][i])
            pos = torch.tensor(pos)
            for i in range(layer_length):
                share_x[self.map_layer[l][i]] = torch.sum(x[pos[i]], dim=0)
                share_y[self.map_layer[l][i]] = torch.sum(y[pos[i]], dim=0)
        share_x /= b_size
        share_y /= b_size
        share_x += self.sigma * torch.randn((43, self.h_feats))
            
        self.share_x.append(share_x)
        self.share_y.append(share_y)
        self.share_node_idx.append(self.share_node_cnt)
        self.share_node_cnt += 43

    @torch.no_grad()
    def updateShareBatchNe(self, batch, layer=3):
        b_size = batch.batch_size
        x = batch.x.cpu()
        y = batch.y
        y = F.one_hot(y, num_classes=self.num_classes).float().cpu()

        share_x = torch.zeros((43, self.h_feats))
        share_y = torch.zeros((43, self.num_classes))

        self.share_edge_index_u += [self.share_node_cnt]
        self.share_edge_index_v += [self.share_node_cnt]

        # Mixup
        share_x = torch.sum(x[:b_size], dim=0)
        share_y = torch.sum(y[:b_size], dim=0)
        share_x /= b_size
        share_y /= b_size
        share_x = share_x.view([1, -1])
        share_y = share_y.view([1, -1])
        self.share_x.append(share_x)
        self.share_y.append(share_y)
        self.share_node_idx.append(self.share_node_cnt)
        self.share_node_cnt += 1

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
        edge_index = self.data.edge_index.to(self.device)
        if self.mode in ["fedego_ne"]:
            x = self.gnnLayer(x, edge_index)
        else:
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
            if self.mode in ["fedego_ne"]:
                self.updateShareBatchNe(batch)
            else:
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

        if self.mode in ["fedego", "fedego_np", "fedego_nr"]:
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
        elif self.mode in ["fedego_ne"]:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size
                x = batch.x
                y = batch.y
                edge_index = batch.edge_index

                x = self.gnnLayer(x, edge_index)
                x = self.dropout(x)
                x = self.lowLayer(x)
                x = self.dropout(x)
                out = self.classification(x)[:b_size]
                loss = self.getLoss(out, y[:b_size])

                loss.backward()
                self.optimizer.step()

                total_examples += b_size
                total_loss += float(loss) * b_size
        elif self.mode in ["fedsage", "fedsageplus"]:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size
                x = batch.x
                y = batch.y
                edge_index = batch.edge_index

                x = self.gnnLayer(x, edge_index)
                out = self.classification(x)[:b_size]
                loss = F.cross_entropy(out, y[:b_size])

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

            if self.mode in ["fedego_ne"]:
                x = self.gnnLayer(x, edge_index)
                x = self.lowLayer(x)
                out = self.classification(x)[:b_size]
            elif self.mode in ["fedsage", "fedsageplus"]:
                x = self.gnnLayer(x, edge_index)
                out = self.classification(x)[:b_size]
            else:
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

            if self.mode in ["fedego_ne"]:
                x = self.gnnLayer(x, edge_index)
                x = self.lowLayer(x)
                out = self.classification(x)[:b_size]
            elif self.mode in ["fedsage", "fedsageplus"]:
                x = self.gnnLayer(x, edge_index)
                out = self.classification(x)[:b_size]
            else:
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

    def local_pre_train(self):
        self.train()
        data = self.hide_data.to(self.device)
        self.criterion_num = F.smooth_l1_loss
        self.criterion_feat = GreedyLoss

        self.optimizer.zero_grad()
        pred_missing, pred_feat, nc_pred = self.local_gen(data)
        pred_missing, pred_feat, nc_pred = pred_missing[data.train_mask], pred_feat[data.train_mask], nc_pred[data.train_mask]
        loss_num = self.criterion_num(pred_missing, data.num_missing[data.train_mask])
        loss_feat = self.criterion_feat(
            pred_feats=pred_feat,
            true_feats=data.x_missing[data.train_mask],
            pred_missing=pred_missing,
            true_missing=data.num_missing[data.train_mask],
            num_pred=self.num_pred
        ).requires_grad_()
        loss_clf = F.cross_entropy(nc_pred, data.y[data.train_mask])
        loss = (self.a * loss_num + self.b * loss_feat + self.c * loss_clf).float()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def build_fedgen(self):
        self.fed_gen = FedSage_Plus(self.local_gen).to(self.device)
        return self.fed_gen.state_dict()

    def fed_gen_train(self, client_num):
        self.train()
        data = self.hide_data.to(self.device)

        self.optimizer.zero_grad()
        pred_missing, pred_feat, nc_pred = self.local_gen(data)
        pred_missing, pred_feat, nc_pred = pred_missing[data.train_mask], pred_feat[data.train_mask], nc_pred[data.train_mask]
        loss_num = self.criterion_num(pred_missing, data.num_missing[data.train_mask])
        loss_feat = self.criterion_feat(
            pred_feats=pred_feat,
            true_feats=data.x_missing[data.train_mask],
            pred_missing=pred_missing,
            true_missing=data.num_missing[data.train_mask],
            num_pred=self.num_pred
        ).requires_grad_()
        loss_clf = F.cross_entropy(nc_pred, data.y[data.train_mask])
        loss = (self.a * loss_num + self.b * loss_feat + self.c * loss_clf).float() / client_num
        loss.backward()
        self.optimizer.step()
    
    def embedding(self):
        data = self.hide_data.to(self.device)
        return self.fed_gen.encoder_model(data.x, data.edge_index).to('cpu')
    
    def update_by_grad(self, grads):
        """
        Arguments:
            grads: grads of other clients to optimize the local model
        :returns:
            state_dict of generation model
        """
        for key in grads.keys():
            if isinstance(grads[key], list):
                grads[key] = torch.FloatTensor(grads[key]).to(self.device)
            elif isinstance(grads[key], torch.Tensor):
                grads[key] = grads[key].to(self.device)
        
        for key, value in self.fed_gen.named_parameters():
            value.grad += grads[key]
        
        self.optimizer.step()
        return self.fed_gen.state_dict()
    
    def cal_grad(self, model_para, embedding, true_missing, client_num):
        """
        Arguments:
            model_para: model parameters
            embedding: output embeddings after local encoder
            true_missing: number of missing node
        :returns:
            grads: grads to optimize the model of other clients
        """
        para_backup = copy.deepcopy(self.fed_gen.state_dict())
        
        for key in model_para.keys():
            if isinstance(model_para[key], list):
                model_para[key] = torch.FloatTensor(model_para[key])
        self.fed_gen.load_state_dict(model_para)
        self.fed_gen.train()

        raw_data = self.data.to(self.device)
        embedding = torch.FloatTensor(embedding).to(self.device)
        true_missing = true_missing.long().to(self.device)
        pred_missing = self.fed_gen.reg_model(embedding)
        pred_feat = self.fed_gen.gen(embedding)

        # Random pick node and compare its neighbors with predicted nodes
        choice = np.random.choice(raw_data.num_nodes, embedding.shape[0])
        global_target_feat = []
        for c_i in choice:
            neighbors_ids = raw_data.edge_index[1][torch.where(
                raw_data.edge_index[0] == c_i)[0]]
            while len(neighbors_ids) == 0:
                id_i = np.random.choice(raw_data.num_nodes, 1)[0]
                neighbors_ids = raw_data.edge_index[1][torch.where(
                    raw_data.edge_index[0] == id_i)[0]]
            choice_i = np.random.choice(neighbors_ids.detach().cpu().numpy(),
                                        self.num_pred)
            for ch_i in choice_i:
                global_target_feat.append(
                    raw_data.x[ch_i].detach().cpu().numpy())
        global_target_feat = np.asarray(global_target_feat).reshape(
            (embedding.shape[0], self.num_pred,
             raw_data.num_node_features))
        loss_feat = self.criterion_feat(pred_feats=pred_feat,
                                        true_feats=global_target_feat,
                                        pred_missing=pred_missing,
                                        true_missing=true_missing,
                                        num_pred=self.num_pred)
        loss = self.b * loss_feat
        loss = (1.0 / client_num * loss).requires_grad_()
        loss.backward()
        grads = {
            key: value.grad
            for key, value in self.fed_gen.named_parameters()
        }
        # Rollback
        self.fed_gen.load_state_dict(para_backup)
        return grads

    def setup_fedsage(self):
        self.filled_data = GraphMender(
            model=self.fed_gen,
            impaired_data = self.hide_data.cpu(),
            original_data = self.data,
            num_pred = self.num_pred
        )
        self.data = self.filled_data
        self.optimizer = torch.optim.Adam([
            {'params': self.gnnLayer.parameters()},
            {'params': self.classification.parameters()}
        ], lr=self.lr)