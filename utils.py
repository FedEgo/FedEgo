from queue import Queue
import copy
from scipy.sparse import data
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
import numpy as np


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