from torch._C import dtype
from torch_geometric.data.data import Data
from models import *
from torch_geometric.loader import NeighborLoader
import numpy as np
from torch.optim import *


class Server(nn.Module):
    def __init__(self, mode, sageMode, h_feats, num_classes, lr, dropout, device, linear):
        super(Server, self).__init__()
        self.mode = mode
        self.sageMode = sageMode
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.device = device
        self.linear = linear

        if self.mode in ["fedego_ne"]:
            self.lowLayer = LowLayer(h_feats, h_feats).to(self.device)
        else:
            self.gnnLayer = GNNLayer(
                sageMode, h_feats, h_feats, h_feats, linear, 2).to(self.device)
        self.classification = Classification(
            h_feats, num_classes).to(self.device)
        self.dropout = nn.Dropout(p=dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.print = True

    @torch.no_grad()
    def loadData(self, share_x, share_edge_index, share_y, share_mask):
        self.data = Data(x=share_x, edge_index=share_edge_index, y=share_y)
        self.share_mask = share_mask
        self.share_nodes = torch.arange(len(self.share_mask))[self.share_mask]
        self.node_num = len(self.share_mask)
        self.share_node_num = len(self.share_nodes)
        if self.print:
            print("total uploaded training nodes:", self.share_node_num)
            self.print = False

    @torch.no_grad()
    def getLoader(self, batch_size, num_workers=0):
        loader = NeighborLoader(self.data,
                                num_neighbors=[6] * 2,
                                batch_size=batch_size,
                                input_nodes=self.train_mask,
                                replace=True,
                                directed=True,
                                num_workers=num_workers)
        return loader

    @torch.no_grad()
    def getTrainMask(self):
        rand_indices = np.random.permutation(self.share_node_num)
        train_size = self.share_node_num // 5
        self.train_nodes = self.share_nodes[rand_indices[:train_size]]
        self.train_mask = torch.zeros(self.node_num, dtype=torch.bool)
        self.train_mask[self.train_nodes] = True

    def getLoss(self, out, y_one_hot):
        """传入的已经是one_hot编码"""
        x = F.softmax(out, dim=1)
        loss = y_one_hot * torch.log(x + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

    def supervisedTrain(self, batch_size):
        self.train()
        self.getTrainMask()

        train_loader = self.getLoader(batch_size)

        total_examples = total_loss = 0

        if self.mode in ["fedego", "fedego_nr"]:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size

                x = self.gnnLayer(batch.x, batch.edge_index)
                x = self.dropout(x)
                out = self.classification(x)[:b_size]
                loss = self.getLoss(out, batch.y[:b_size])

                loss.backward()
                self.optimizer.step()

                total_examples += b_size
                total_loss += float(loss) * b_size
        elif self.mode in ["fedego_ne"]:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size
                x = self.lowLayer(batch.x)
                x = self.dropout(x)
                out = self.classification(x)[:b_size]
                loss = self.getLoss(out, batch.y[:b_size])

                loss.backward()
                self.optimizer.step()

                total_examples += b_size
                total_loss += float(loss) * b_size
        else:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                b_size = batch.batch_size

                x = self.gnnLayer(batch.x, batch.edge_index)
                out = self.classification(x)[:b_size]
                loss = F.cross_entropy(out, batch.y[:b_size])

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

    @torch.no_grad()
    def getDistribution(self):
        self.p = torch.sum(self.data.y, axis=0)
        self.p /= self.data.y.shape[0]
        return self.p
