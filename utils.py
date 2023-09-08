from queue import Queue
import copy
from scipy.sparse import data
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch.nn.functional as F


@torch.no_grad()
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def fedavgJob(client, client_train_epoch, batch_size, qloss):
    loss = []
    for i in range(client_train_epoch):
        loss.append(client.supervisedTrain(batch_size))
        qloss.put(loss)


def fedavgWork(clients, client_num, client_train_epoch, batch_size):
    loss_list = []
    qloss = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=fedavgJob,
                             args=(clients[cid], client_train_epoch, batch_size, qloss))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    for _ in range(client_num):
        loss_list.append(qloss.get())
    return loss_list


def SumUpGradient(gradient_dict):
    gen_grads = {}
    for cid, grads_list in gradient_dict.items():
        for grads in grads_list:
            if cid not in gen_grads:
                gen_grads[cid] = dict()
                for key in grads.keys():
                    gen_grads[cid][key] = torch.FloatTensor(grads[key].cpu())
            else:
                for key in grads.keys():
                    gen_grads[cid][key] += torch.FloatTensor(grads[key].cpu())
    return gen_grads


def LocalGenJob(client, gen_train_epochs, qcontent, cid):
    # Local pre-train
    for _ in range(gen_train_epochs):
        l = client.local_pre_train()
        print("-----client{} -- local_epoch {} -- gen_train_loss {:.4f}".format(cid, _, l), flush=True)
    # Build fedgen base on locgen
    gen_para = client.build_fedgen()
    embedding = client.embedding()
    num_missing = client.hide_data.num_missing
    qcontent.put([gen_para, embedding, num_missing, cid])
    
 
def LocalGenWork(clients, client_num, gen_train_epochs):
    content_list = []
    qcontent = Queue()

    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=LocalGenJob,
                             args=(clients[cid], gen_train_epochs, qcontent, cid))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    for _ in range(client_num):
        content_list.append(qcontent.get())
    return content_list


def FedGenJob(client, contents, qcontent, cid):
    grads = {}
    for i in range(len(contents)):
        if i != cid:
            grads[i] = client.cal_grad(*contents[i], len(contents))
    qcontent.put(grads)


def FedGenWork(clients, client_num, contents):
    gradient_dict = {}
    qcontent = Queue()

    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=FedGenJob,
                             args=(clients[cid], contents, qcontent, cid))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    for _ in range(client_num):
        grads = qcontent.get()
        for key, values in grads.items():
            if key not in gradient_dict:
                gradient_dict[key] = []
            gradient_dict[key].append(values)
    return gradient_dict


def UpdateGenJob(client, grads, client_num, qcontent, cid):
    client.fed_gen_train(client_num)
    gen_para = client.update_by_grad(grads)
    embedding = client.embedding()
    num_missing = client.hide_data.num_missing
    qcontent.put([gen_para, embedding, num_missing, cid])


def UpdateGenWork(clients, client_num, gen_grads):
    content_list = []
    qcontent = Queue()

    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=UpdateGenJob,
                             args=(clients[cid], gen_grads[cid], client_num, qcontent, cid))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    for _ in range(client_num):
        content_list.append(qcontent.get())
    return content_list


def fedegoJob(client, client_train_epoch, batch_size, qloss, qx, qy, qedge, qmask):
    loss = []
    with torch.no_grad():
        client.initShareBatch()
    for i in range(client_train_epoch):
        loss.append(client.supervisedTrain(batch_size))
    with torch.no_grad():
        share_x, share_edge_index, share_y, share_mask = client.getShareBatch(
            batch_size)
        qloss.put(loss)
        qx.put(share_x)
        qy.put(share_y)
        qedge.put(share_edge_index)
        qmask.put(share_mask)


def fedegoWork(clients, client_num, client_train_epoch, batch_size):
    clients_share_x = []
    clients_share_y = []
    clients_share_edge_index = []
    clients_share_mask = []
    loss_list = []
    qloss = Queue()
    qx = Queue()
    qy = Queue()
    qedge = Queue()
    qmask = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=fedegoJob,
                             args=(clients[cid], client_train_epoch, batch_size, qloss,
                                   qx, qy, qedge, qmask))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    with torch.no_grad():
        for _ in range(client_num):
            clients_share_x.append(qx.get())
            clients_share_y.append(qy.get())
            clients_share_edge_index.append(qedge.get())
            clients_share_mask.append(qmask.get())
            loss_list.append(qloss.get())
        clients_share_x = torch.cat(clients_share_x, 0).cpu()
        clients_share_y = torch.cat(clients_share_y, 0).cpu()
        clients_share_mask = torch.cat(clients_share_mask, 0).cpu()
        clients_share_edge_index = torch.cat(clients_share_edge_index, 1).cpu()

    return clients_share_x, clients_share_y, clients_share_mask, clients_share_edge_index, loss_list


def graphflJob(client, client_train_epoch, batch_size, qsloss, qvloss, qgradient, stage):
    support_loss = []
    for i in range(client_train_epoch):
        support_loss.append(client.graphflSupport(batch_size))
    if stage == "MAML":
        val_loss = client.graphflQuery(batch_size)
    with torch.no_grad():
        qsloss.put(support_loss)
        if stage == "MAML":
            qvloss.put(val_loss)
            gradient_dict = client.getGradient()
            qgradient.put(gradient_dict)


def graphflWork(clients, client_num, client_train_epoch, batch_size):
    maml_sloss_list = []
    maml_vloss_list = []
    fl_loss_list = []
    gradient_dict_list = []
    qsloss = Queue()
    qvloss = Queue()
    qgradient = Queue()
    threads = []
    """stageL: MAML"""
    """load the initial weights"""
    with torch.no_grad():
        w_avg = FedAvg([clients[i].state_dict() for i in range(client_num)])
        for cid in range(client_num):
            clients[cid].load_state_dict(w_avg)
    """client update"""
    for cid in range(client_num):
        t = threading.Thread(target=graphflJob,
                             args=(clients[cid], client_train_epoch, batch_size, qsloss,
                                   qvloss, qgradient, "MAML"))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    with torch.no_grad():
        for _ in range(client_num):
            maml_sloss_list.append(qsloss.get())
            maml_vloss_list.append(qvloss.get())
            gradient_dict_list.append(qgradient.get())
        """server update"""
        gradient_dict = FedAvg(gradient_dict_list)
        for key in gradient_dict.keys():
            w_avg[key] -= gradient_dict[key]
        for cid in range(client_num):
            clients[cid].load_state_dict(w_avg)
    """clear"""
    with torch.no_grad():
        while not qsloss.empty():
            qsloss.get()
        while not qvloss.empty():
            qvloss.get()
        while not qgradient.empty():
            qgradient.get()
        threads = []
        gradient_dict_list = []
    """stage: FL"""
    """client finetuning"""
    for cid in range(client_num):
        t = threading.Thread(target=graphflJob,
                             args=(clients[cid], client_train_epoch, batch_size, qsloss,
                                   qvloss, qgradient, "FL"))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    with torch.no_grad():
        for _ in range(client_num):
            fl_loss_list.append(qsloss.get())
        """server aggreation"""
        w_avg = FedAvg([clients[i].state_dict() for i in range(client_num)])
        for cid in range(client_num):
            clients[cid].load_state_dict(w_avg)
    return maml_sloss_list, maml_vloss_list, fl_loss_list


def fedgcnJob(client, client_train_epoch, batch_size, qloss):
    loss = []
    for i in range(client_train_epoch):
        loss.append(client.fedgcn_train())
        qloss.put(loss)


def fedgcnWork(clients, client_num, client_train_epoch, batch_size):
    loss_list = []
    qloss = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=fedgcnJob,
                             args=(clients[cid], client_train_epoch, batch_size, qloss))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    for _ in range(client_num):
        loss_list.append(qloss.get())
    return loss_list


"""normalize the adj"""


def normalize(A):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # degree of the nodes
    d = A.sum(1)
    D = torch.diag(torch.pow(d, -1))
    return D.mm(A)


def printWD(clients, client_num, image_url, r1=3, r2=2, r3=1):
    pass


def fedgcnInit(data,clients, client_num, index_list, test_data):
    # init the adj
    test_node_num = test_data.x.shape[0]
    test_adj = torch.zeros(test_node_num, test_node_num)
    for edge in test_data.edge_index:
        test_adj[edge[0]][edge[1]] = 1
        test_adj[edge[1]][edge[0]] = 1
    test_adj = normalize(test_adj)

    # calculate the sum of the feature
    total_communicate = torch.zeros(data.x.shape)
    cnt = torch.zeros(data.x.shape[0])
    for cid in range(client_num):
        node_num = clients[cid].data.x.shape[0]
        index = index_list[cid]
        clients[cid].adj = torch.zeros(node_num, node_num)
        for edge in clients[cid].data.edge_index:
            clients[cid].adj[edge[0]][edge[1]] = 1
            clients[cid].adj[edge[1]][edge[0]] = 1
        clients[cid].adj = normalize(clients[cid].adj)
        clients[cid].communicate = clients[cid].adj.mm(clients[cid].data.x)
        for node in range(node_num):
            total_communicate[index[node]
                              ] += clients[cid].communicate[node]
            cnt[index[node]] += 1

    for node in range(data.x.shape[0]):
        if cnt[node] == 0:
            continue
        total_communicate[node] /= cnt[node]
    
    return total_communicate,test_adj

def fedegoNeJob(client, client_train_epoch, batch_size, qloss, qx, qy, qedge, qmask):
    loss = []
    with torch.no_grad():
        client.initShareBatch()
    for i in range(client_train_epoch):
        loss.append(client.supervisedTrain(batch_size))
    with torch.no_grad():
        share_x, share_edge_index, share_y, share_mask = client.getShareBatch(
            batch_size)
        qloss.put(loss)
        qx.put(share_x)
        qy.put(share_y)
        qedge.put(share_edge_index)
        qmask.put(share_mask)


def fedegoNeWork(clients, client_num, client_train_epoch, batch_size):
    clients_share_x = []
    clients_share_y = []
    clients_share_edge_index = []
    clients_share_mask = []
    loss_list = []
    qloss = Queue()
    qx = Queue()
    qy = Queue()
    qedge = Queue()
    qmask = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=fedegoNeJob,
                             args=(clients[cid], client_train_epoch, batch_size, qloss,
                                   qx, qy, qedge, qmask))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    with torch.no_grad():
        for _ in range(client_num):
            clients_share_x.append(qx.get())
            clients_share_y.append(qy.get())
            clients_share_edge_index.append(qedge.get())
            clients_share_mask.append(qmask.get())
            loss_list.append(qloss.get())
        clients_share_x = torch.cat(clients_share_x, 0).cpu()
        clients_share_y = torch.cat(clients_share_y, 0).cpu()
        clients_share_mask = torch.cat(clients_share_mask, 0).cpu()
        clients_share_edge_index = torch.cat(clients_share_edge_index, 1).cpu()

    return clients_share_x, clients_share_y, clients_share_mask, clients_share_edge_index, loss_list


'''for fedsageplus'''

class HideGraph(BaseTransform):
    r"""
    Generate impaired graph with labels and features to train NeighGen,
    hide Node from validation set from raw graph.

    Arguments:
        hidden_portion (int): hidden_portion of validation set.
        num_pred (int): hyperparameters which limit
            the maximum value of the prediction

    :returns:
        filled_data : impaired graph with attribute "num_missing"
    :rtype:
        nx.Graph
    """
    def __init__(self, hidden_portion=0.5, num_pred=5):
        self.hidden_portion = hidden_portion
        self.num_pred = num_pred

    def __call__(self, data):

        val_ids = torch.where(data.val_mask == True)[0]
        hide_ids = np.random.choice(val_ids,
                                    int(len(val_ids) * self.hidden_portion),
                                    replace=False)
        remaining_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        remaining_mask[hide_ids] = False
        remaining_nodes = torch.where(remaining_mask == True)[0].numpy()

        data.ids_missing = [[] for _ in range(data.num_nodes)]

        G = to_networkx(data,
                        node_attrs=[
                            'x', 'y', 'train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'ids_missing'
                        ],
                        to_undirected=True)

        for missing_node in hide_ids:
            neighbors = G.neighbors(missing_node)
            for i in neighbors:
                G.nodes[i]['ids_missing'].append(missing_node)
        for i in G.nodes:
            ids_missing = G.nodes[i]['ids_missing']
            del G.nodes[i]['ids_missing']
            G.nodes[i]['num_missing'] = np.array([len(ids_missing)],
                                                 dtype=np.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_pred:
                    G.nodes[i]['x_missing'] = np.vstack(
                        (data.x[ids_missing].cpu().numpy(),
                         np.zeros((self.num_pred - len(ids_missing),
                                   data.x.shape[1]))))
                else:
                    G.nodes[i]['x_missing'] = data.x[
                        ids_missing[:self.num_pred]].cpu().numpy()
            else:
                G.nodes[i]['x_missing'] = np.zeros(
                    (self.num_pred, data.x.shape[1]))

        return from_networkx(nx.subgraph(G, remaining_nodes))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_portion})'


def FillGraph(impaired_data, original_data, pred_missing, pred_feats,
              num_pred):
    # Mend the original data
    original_data = original_data.detach().cpu()
    new_features = original_data.x
    new_edge_index = original_data.edge_index.T
    pred_missing = pred_missing.detach().cpu().numpy()
    pred_feats = pred_feats.detach().cpu().reshape(
        (-1, num_pred, original_data.num_node_features))

    start_id = original_data.num_nodes
    for node in range(len(pred_missing)):
        num_fill_node = np.around(pred_missing[node]).astype(np.int32).item()
        if num_fill_node > 0:
            new_ids_i = np.arange(start_id,
                                  start_id + min(num_pred, num_fill_node))
            org_id = impaired_data.index_orig[node]
            org_node = torch.where(
                original_data.index_orig == org_id)[0].item()
            new_edges = torch.tensor([[org_node, fill_id]
                                      for fill_id in new_ids_i],
                                     dtype=torch.int64)
            new_features = torch.vstack(
                (new_features, pred_feats[node][:num_fill_node]))
            new_edge_index = torch.vstack((new_edge_index, new_edges))
            start_id = start_id + min(num_pred, num_fill_node)
    new_y = torch.zeros(new_features.shape[0], dtype=torch.int64)
    new_y[:original_data.num_nodes] = original_data.y
    filled_data = Data(
        x=new_features,
        edge_index=new_edge_index.T,
        train_idx=torch.where(original_data.train_mask == True)[0],
        valid_idx=torch.where(original_data.val_mask == True)[0],
        test_idx=torch.where(original_data.test_mask == True)[0],
        y=new_y,
    )
    return filled_data


@torch.no_grad()
def GraphMender(model, impaired_data, original_data, num_pred):
    r"""Mend the graph with generation model
    Arguments:
        model (torch.nn.module): trained generation model
        impaired_data (PyG.Data): impaired graph
        original_data (PyG.Data): raw graph
    :returns:
        filled_data : Graph after Data Enhancement
    :rtype:
        PyG.data
    """
    device = impaired_data.x.device
    model = model.to(device)
    pred_missing, pred_feats, _ = model(impaired_data)

    return FillGraph(impaired_data, original_data, pred_missing, pred_feats, num_pred)


def GreedyLoss(pred_feats, true_feats, pred_missing, true_missing, num_pred):
    r"""Greedy loss is a loss function of cacluating the MSE loss for the feature.
    https://proceedings.neurips.cc//paper/2021/file/ \
    34adeb8e3242824038aa65460a47c29e-Paper.pdf
    Fedsageplus models from the "Subgraph Federated Learning with Missing
    Neighbor Generation" (FedSage+) paper, in NeurIPS'21
    Source: https://github.com/zkhku/fedsage

    Arguments:
        pred_feats (torch.Tensor): generated missing features
        true_feats (torch.Tensor): real missing features
        pred_missing (torch.Tensor): number of predicted missing node
        true_missing (torch.Tensor): number of missing node
        num_pred (int): hyperparameters which limit the maximum value of the \
        prediction
    :returns:
        loss : the Greedy Loss
    :rtype:
        torch.FloatTensor
    """
    CUDA, device = (pred_feats.device.type != 'cpu'), pred_feats.device
    if CUDA:
        true_missing = true_missing.cpu()
        pred_missing = pred_missing.cpu()
    loss = torch.zeros(pred_feats.shape)
    if CUDA:
        loss = loss.to(device)
    pred_len = len(pred_feats)
    pred_missing_np = np.round(
        pred_missing.detach().numpy()).reshape(-1).astype(np.int32)
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(
        np.int32)
    true_missing_np = np.clip(true_missing_np, 0, num_pred)
    pred_missing_np = np.clip(pred_missing_np, 0, num_pred)
    for i in range(pred_len):
        for pred_j in range(min(num_pred, pred_missing_np[i])):
            if true_missing_np[i] > 0:
                if isinstance(true_feats[i][true_missing_np[i] - 1],
                              np.ndarray):
                    true_feats_tensor = torch.tensor(
                        true_feats[i][true_missing_np[i] - 1])
                    if CUDA:
                        true_feats_tensor = true_feats_tensor.to(device)
                else:
                    true_feats_tensor = true_feats[i][true_missing_np[i] - 1]
                loss[i][pred_j] += F.mse_loss(
                    pred_feats[i][pred_j].unsqueeze(0).float(),
                    true_feats_tensor.unsqueeze(0).float()).squeeze(0)

                for true_k in range(min(num_pred, true_missing_np[i])):
                    if isinstance(true_feats[i][true_k], np.ndarray):
                        true_feats_tensor = torch.tensor(true_feats[i][true_k])
                        if CUDA:
                            true_feats_tensor = true_feats_tensor.to(device)
                    else:
                        true_feats_tensor = true_feats[i][true_k]

                    loss_ijk = F.mse_loss(
                        pred_feats[i][pred_j].unsqueeze(0).float(),
                        true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if torch.sum(loss_ijk) < torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j] = loss_ijk
            else:
                continue
    return loss.unsqueeze(0).mean().float()