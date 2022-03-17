from collections import defaultdict
import random
import numpy as np
import torch
from torch_geometric.data.data import Data
from torch_geometric.datasets import CitationFull, Amazon, Planetoid, WikiCS, Reddit2, Coauthor
from torch_geometric.datasets import CoraFull
from utils import *


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config, client_num, device):
        super(DataCenter, self).__init__()
        self.config = config
        self.client_num = client_num
        self.device = device

    def load_dataSet(self,
                     sample_rate,
                     test_sample_rate,
                     major_rate,
                     test_num=200,
                     major_label=2,
                     dataSet='cora',
                     split_mode="label"):
        if dataSet in [
                "cora", "pubmed", "citeseer", "wiki", "corafull",
        ]:
            if dataSet in ["cora", "pubmed", "citeseer"]:
                data = Planetoid(
                    name=dataSet, root=self.config['file_path.' + dataSet])
            elif dataSet == "wiki":
                data = WikiCS(root=self.config['file_path.wiki'])
            elif dataSet == "corafull":
                data = CoraFull(root=self.config['file_path.corafull'])
            data = data[0]
            x = data.x.numpy()
            y = torch.flatten(data.y).numpy()
            edge_index = data.edge_index.numpy()
            edge_cnt = edge_index.shape[1]
            # print("node", x.shape)
            # print("label:", len(set(y)))
            # print("edge cnt", edge_cnt/2)
            # exit(0)
            """adj_lists split the dataset"""
            adj_lists = defaultdict(set)
            node_map = {}
            for i in range(edge_cnt):
                paper1 = edge_index[0][i]
                paper2 = edge_index[1][i]
                if not paper1 in node_map:
                    node_map[paper1] = len(node_map)
                if not paper2 in node_map:
                    node_map[paper2] = len(node_map)
                adj_lists[node_map[paper1]].add(node_map[paper2])
                adj_lists[node_map[paper2]].add(node_map[paper1])

            x = x[list(node_map)]
            y = y[list(node_map)]
            for i in range(edge_cnt):
                edge_index[0][i] = node_map[edge_index[0][i]]
                edge_index[1][i] = node_map[edge_index[1][i]]
            assert len(x) == len(y) == len(adj_lists)
        num_classes = len(set(y))
        node_num = x.shape[0]
        in_feats = x.shape[1]

        print("num_labels", num_classes)
        print("total node num", node_num)
        print("feature dimension", x.shape[1])
        print("finish file reading", flush=True)
        print("total edges", edge_cnt // 2)

        setattr(self, dataSet + '_num_classes', num_classes)
        setattr(self, dataSet + '_in_feats', in_feats)

        sampling_node_nums = int(node_num * test_sample_rate)
        print("node in test", sampling_node_nums, flush=True)

        test_index = list(np.random.permutation(
            np.arange(node_num))[:sampling_node_nums])

        distibution = np.zeros(num_classes)
        for node in y[test_index]:
            distibution[node] += 1
        print("test distribution")
        print((np.array(distibution)/len(test_index)).tolist(), flush=True)

        pos = {}
        for i, node in enumerate(test_index):
            pos[node] = i

        test_edge_index_u = []
        test_edge_index_v = []
        for u in test_index:
            for v in test_index:
                if (v in adj_lists[u]):
                    """undirect edge"""
                    test_edge_index_u.append(pos[u])
                    test_edge_index_u.append(pos[v])
                    test_edge_index_v.append(pos[v])
                    test_edge_index_v.append(pos[u])
        assert len(test_edge_index_u) % 2 == 0 and len(
            test_edge_index_v) % 2 == 0
        test_edge_index = torch.stack(
            [torch.tensor(test_edge_index_u),
             torch.tensor(test_edge_index_v)], 0)
        test_x = torch.tensor(x[test_index])
        test_y = torch.tensor(y[test_index])
        test_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)
        setattr(self, dataSet + '_test_data', test_data)
        setattr(self, dataSet + '_test_index', test_index)

        # sort node by label
        delete_index = list(test_index)
        sampling_node_num = int(node_num * sample_rate)
        print("node in each client", sampling_node_num)

        # clients_data = loadWork(self.client_num, x, y, adj_lists, node_by_label,
        #                         client_nodes, num_classes, sampling_node_num, major_rate)
        clients_data = []
        index_list = []
        for cid in range(self.client_num):
            """sort nodes"""
            client_nodes = np.delete(np.arange(node_num), delete_index)
            node_by_label = defaultdict(list)
            for node in client_nodes:
                node_by_label[y[node]].append(node)
            """major label nodes"""
            holding_label = np.random.permutation(
                np.arange(num_classes))[:major_label]
            holding_label_index = []
            print("Major label of", "client", cid, ":", holding_label)
            for label in holding_label:
                holding_label_index += node_by_label[label]
            major_num = int(sampling_node_num * major_rate)
            if (major_num > len(holding_label_index)):
                print("Major label not enough nodes")
            major_num = min(major_num, len(holding_label_index))
            major_index = list(np.random.permutation(
                holding_label_index)[:major_num])
            major_pos = []
            for pos, node in enumerate(client_nodes):
                if node in major_index:
                    major_pos.append(pos)
            major_pos = np.array(major_pos, dtype=np.int)

            # other label
            rest_num = sampling_node_num - major_num
            rest_index = np.delete(client_nodes, major_pos)
            other_index = list(np.random.permutation(rest_index)[:rest_num])
            index = major_index + other_index

            connect = set()
            for u in index:
                for v in index:
                    if (u in adj_lists[v]):
                        connect.add(u)
                        connect.add(v)
            index = list(connect)
            print("node num in client", cid + 1, ":", len(index))
            """random shuffle"""
            random.shuffle(index)
            """delete the local test nodes"""
            delete_index += index[len(index) - test_num:]

            pos = {}  # mapping from the original node to the new node
            for i, node in enumerate(index):
                pos[node] = i
            index_list += [index]

            client_edge_index_u = []
            client_edge_index_v = []
            for u in index:
                for v in index:
                    if (u in adj_lists[v]):
                        """双向边"""
                        client_edge_index_u.append(pos[u])
                        client_edge_index_u.append(pos[v])
                        client_edge_index_v.append(pos[v])
                        client_edge_index_v.append(pos[u])
            assert len(client_edge_index_u) % 2 == 0 and len(
                client_edge_index_v) % 2 == 0
            client_edge_index = torch.stack(
                [torch.tensor(client_edge_index_u),
                 torch.tensor(client_edge_index_v)], 0)
            client_x = torch.tensor(x[index])
            client_y = torch.tensor(y[index])
            client_data = Data(
                x=client_x, edge_index=client_edge_index, y=client_y)
            clients_data.append(client_data)
            print(f"Client {cid+1} finish loading data", flush=True)
            
            """print the distibution"""
            distibution = np.zeros(num_classes)
            for node in client_y:
                distibution[node] += 1
            print("distribution")
            print((np.array(distibution)/len(client_y)).tolist(), flush=True)

        setattr(self, dataSet + '_data', clients_data)
        setattr(self, dataSet + '_index_list', index_list)
        setattr(self, dataSet + '_total_data', data)
