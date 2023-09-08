import sys
import torch
import pyhocon
import random

from torch_geometric import datasets
from dataCenter import *
from utils import *
from models import *
from options import args_parser
from Client import *
from Server import *
import time

args = args_parser()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("using device", device_id, torch.cuda.get_device_name(device_id))

# device = torch.device("cuda" if args.cuda else "cpu")
device = "cuda:3"
print("DEVICE:", device, flush=True)

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    args.learning_rate = 3e-4
    lamb_fixed = args.lamb_fixed

    if args.dataSet in ["cora", "citeseer"]:
        args.client_num = 5
        args.sample_rate = 0.5
        args.global_sample_rate = 0.3
        args.learning_rate = 3e-4
        args.test_num = 200
        args.h_feats = 512
        args.epochs = 200
    elif args.dataSet in ["wiki"]:
        args.client_num = 10
        args.sample_rate = 0.2
        args.global_sample_rate = 0.3
        args.h_feats = 512
        if args.mode in ["fedego"]:
            args.epochs = 300
            args.lr = 5e-4
    elif args.dataSet in ["corafull"]:
        args.client_num = 10
        args.learning_rate = 3e-4
        args.sample_rate = 0.3
        args.global_sample_rate = 0.3
        args.h_feats = 512
        if args.mode in ["fedego"]:
            args.lr = 5e-4
    elif args.dataSet in ["FedDBLP"]:
        args.client_num = 7

    client_train_epoch = args.client_train_epoch
    server_train_epoch = args.server_train_epoch
    client_num = args.client_num
    lamb_c = args.lamb_c
    major_rate = args.major_rate
    mode = args.mode
    lr = args.lr
    h_feats = args.h_feats
    batch_size = args.batch_size
    dropout = args.dropout
    test_num = args.test_num
    major_label = args.major_label
    sageMode = args.sageMode
    mixup = args.mixup
    timing = args.timing
    early_stopping = args.early_stopping
    early_stopping_bound = 1.3
    cal_cost = args.cal_cost
    linear = args.linear
    sigma = args.sigma

    logUrl = "./log/" + args.dataSet + "_" + mode
    image_url = "./images/" + args.dataSet + "/" + mode
    logUrl += args.logsuffix + "_seed_" + \
        str(args.seed)+"_lambc_"+str(args.lamb_c)
    image_url += args.logsuffix+"_seed_" + \
        str(args.seed)+"_lambc_"+str(args.lamb_c)+".png"

    print("mode: ", mode)
    print("major_rate: ", major_rate)
    print("seed: ", args.seed)
    print("lamb_c: ", lamb_c)
    print("log addr: ", logUrl)
    print("image addr: ", image_url)
    print("learning rate: ", args.lr)
    print("batch size: ", batch_size)

    writer = SummaryWriter(logUrl)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config, client_num, device)
    dataCenter.load_dataSet(args.sample_rate, args.global_sample_rate, major_rate,
                            test_num, major_label, ds, args.split_mode)
    print("load data finished ", flush=True)

    data = getattr(dataCenter, ds + "_data")
    test_data = getattr(dataCenter, ds + "_test_data")
    in_feats = getattr(dataCenter, ds + "_in_feats")
    num_classes = getattr(dataCenter, ds + "_num_classes")

    # init clients
    clients = [
        Client(data[i], mode, sageMode, in_feats, h_feats, num_classes, test_num, lr, dropout,
               device, mixup, linear, sigma).to(device) for i in range(client_num)
    ]

    print("clients init finished ", flush=True)

    # init server
    if mode in ["fedego", "fedego_nr", "fedego_ne"]:
        server = Server(mode, sageMode, h_feats,
                        num_classes, lr, dropout, device, linear)
        server_max_vali_f1 = 0

    local_f1 = [0 for i in range(client_num)]
    local_avg_f1 = []
    local_loss = [0 for i in range(client_num)]
    local_avg_loss = []
    test_f1 = [0 for i in range(client_num)]
    test_avg_f1 = []
    max_local_f1 = [0 for i in range(client_num)]
    max_test_f1 = [0 for i in range(client_num)]
    min_local_loss = 1e5
    max_local_avg_f1 = 0
    max_test_avg_f1 = 0

    if timing:
        start_time = time.time()

    if cal_cost:
        weight_cost = 0
        ego_cost = 0

    if mode in ["fedavg", "fedprox"]:
        for epoch in range(1, args.epochs + 1):
            with torch.no_grad():
                w_avg = FedAvg([clients[i].state_dict()
                               for i in range(client_num)])
                for cid in range(client_num):
                    clients[cid].load_state_dict(w_avg)
                    clients[cid].init_model = w_avg

            loss_list = fedavgWork(clients, client_num,
                                   client_train_epoch, batch_size)

            with torch.no_grad():
                """load the averaged model of clients"""
                with torch.no_grad():
                    w_avg = FedAvg([clients[i].state_dict()
                                   for i in range(client_num)])
                for cid in range(client_num):
                    local_f1[cid], local_loss[cid] = clients[cid].evaluate(
                        batch_size)
                    test_f1[cid] = clients[cid].test(test_data, batch_size)
                    max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                    max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                local_avg_f1.append(np.mean(local_f1))
                max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                local_avg_loss.append(np.mean(local_loss))
                min_local_loss = min(min_local_loss, local_avg_loss[-1])
                test_avg_f1.append(np.mean(test_f1))
                max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

            if cal_cost:
                for key in clients[0].state_dict().keys():
                    weight_cost += sys.getsizeof(clients[0].state_dict()
                                                 [key].storage())/1024/1024*client_num
                print("upload weight cost:", weight_cost)

            print("-----epoch",
                  epoch,
                  "test f1:",
                  test_avg_f1[-1],
                  "| local f1:",
                  local_avg_f1[-1],
                  " -----",
                  flush=True)

            if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                print("Early stopping...")
                break

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)
    elif mode == "graphfl":
        """support set and the query set for GraphFl"""
        for cid in range(client_num):
            clients[cid].getMask()

        for epoch in range(1, args.epochs + 1):
            maml_sloss_list, maml_vloss_list, fl_loss_list = graphflWork(
                clients, client_num, client_train_epoch, batch_size)

            with torch.no_grad():
                for cid in range(client_num):
                    local_f1[cid], local_loss[cid] = clients[cid].evaluate(
                        batch_size)
                    test_f1[cid] = clients[cid].test(test_data, batch_size)
                    max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                    max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                local_avg_f1.append(np.mean(local_f1))
                max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                local_avg_loss.append(np.mean(local_loss))
                min_local_loss = min(min_local_loss, local_avg_loss[-1])
                test_avg_f1.append(np.mean(test_f1))
                max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

            writer.add_scalar(f"local avg F1:",
                              local_avg_f1[-1], global_step=epoch)
            writer.add_scalar(
                f"test avg F1:", test_avg_f1[-1], global_step=epoch)
            print("-----epoch",
                  epoch,
                  "test f1:",
                  test_avg_f1[-1],
                  "| local f1:",
                  local_avg_f1[-1],
                  " -----",
                  flush=True)

            if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                print("Early stopping...")
                break

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)
    elif mode in ["fedego", "fedego_nr", "fedego_np", "fedego_ne", "local"]:
        shallowKeys = []
        deepKeys = []
        for epoch in range(1, args.epochs + 1):
            with torch.no_grad():
                if mode in ["fedego", "fedego_np"]:
                    """reduction layers"""
                    w_avg = FedAvg([clients[i].state_dict()
                                   for i in range(client_num)])

                    if epoch == 1:
                        for key in w_avg.keys():
                            if mode in ["fedego_ne"]:
                                if (key.startswith("gnnLayer")):
                                    shallowKeys.append(key)
                                else:
                                    deepKeys.append(key)
                            else:
                                if (key.startswith("lowLayer")):
                                    shallowKeys.append(key)
                                else:
                                    deepKeys.append(key)

                    shallow_dict = {k: w_avg[k] for k in shallowKeys}
                    for cid in range(client_num):
                        clients[cid].load_state_dict(
                            shallow_dict, strict=False)
            """train"""
            clients_share_x, clients_share_y, clients_share_mask, clients_share_edge_index, loss_list = fedegoWork(
                clients, client_num, client_train_epoch, batch_size)
            if cal_cost:
                ego_cost += sys.getsizeof(clients_share_x.storage())/1024/1024
                ego_cost += sys.getsizeof(clients_share_y.storage())/1024/1024
                ego_cost += sys.getsizeof(clients_share_mask.storage()
                                          )/1024/1024
                ego_cost += sys.getsizeof(clients_share_edge_index.storage()
                                          )/1024/1024
                for key in clients[0].state_dict().keys():
                    if key in deepKeys:
                        continue
                    weight_cost += sys.getsizeof(clients[0].state_dict()
                                                 [key].storage())/1024/1024*client_num
                print("upload ego cost:", ego_cost, "weight cost:",
                      weight_cost, "total cost:", ego_cost+weight_cost, flush=True)

            if mode in ["fedego", "fedego_nr", "fedego_ne"]:
                with torch.no_grad():
                    server.loadData(clients_share_x, clients_share_edge_index,
                                    clients_share_y, clients_share_mask)

                for i in range(server_train_epoch):
                    server.supervisedTrain(batch_size)

                """obtain lambda"""
                global_p = server.getDistribution()
                for cid in range(client_num):
                    clients[cid].setLamb(
                        global_p=global_p, lamb_c=lamb_c, lamb_fixed=lamb_fixed)
                if epoch <= 1:
                    print("EMD:", flush=True)
                    for cid in range(client_num):
                        print(f"{clients[cid].emd.item():.3f}",
                              end=" ", flush=True)
                    print("\nlamb:", flush=True)
                    for cid in range(client_num):
                        print(f"{clients[cid].lamb:.3f}", end=" ", flush=True)
                    print()

            if mode in ["fedego", "fedego_nr", "fedego_ne"]:
                with torch.no_grad():
                    server_dict = server.state_dict()
                    """update in the personalization layers"""
                    for cid in range(client_num):
                        origin_dict = clients[cid].state_dict()
                        deep_dict = {
                            k: clients[cid].lamb * server_dict[k] +
                            (1 - clients[cid].lamb) * origin_dict[k]
                            for k in deepKeys
                        }
                        clients[cid].load_state_dict(deep_dict, strict=False)

            with torch.no_grad():
                for cid in range(client_num):
                    local_f1[cid], local_loss[cid] = clients[cid].evaluate(
                        batch_size)
                    test_f1[cid] = clients[cid].test(test_data, batch_size)
                    max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                    max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                    writer.add_scalar(f"client{cid} local f1",
                                      local_f1[cid], global_step=epoch)
                    writer.add_scalar(f"client{cid} test f1:",
                                      test_f1[cid], global_step=epoch)
                local_avg_f1.append(np.mean(local_f1))
                max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                local_avg_loss.append(np.mean(local_loss))
                min_local_loss = min(min_local_loss, local_avg_loss[-1])
                test_avg_f1.append(np.mean(test_f1))
                max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

            writer.add_scalar(f"local avg f1:",
                              local_avg_f1[-1], global_step=epoch)
            writer.add_scalar(
                f"test avg f1:", test_avg_f1[-1], global_step=epoch)
            print("-----epoch", epoch, "test f1:",
                  test_avg_f1[-1], "| local f1:", local_avg_f1[-1], " -----", flush=True)

            if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                print("Early stopping...")
                break

        if mode in ["fedego"]:
            for cid in range(client_num):
                clients[cid].getWD(server.state_dict())
            server_norm = 0
            for key in server.state_dict().keys():
                server_norm += np.linalg.norm(server.state_dict()
                                              [key].cpu().numpy().flatten(), ord=2)
            wd = [clients[cid].wd/server_norm for cid in range(client_num)]
            print("WD:", wd)

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)
    elif mode in ["dfedgnn"]:
        for epoch in range(1, args.epochs + 1):
            loss_list = fedavgWork(clients, client_num,
                                   client_train_epoch, batch_size)

            with torch.no_grad():
                # load the model
                w_last = [clients[i].state_dict()for i in range(client_num)]
                w_load = []
                for cid in range(client_num):
                    cur_load = copy.deepcopy(w_last[cid])
                    for k in cur_load.keys():
                        for i in range(-1, 2):
                            if i == 0:
                                continue
                            cur_load[k] += w_last[(cid+i+client_num) %
                                                  client_num][k]
                        cur_load[k] = torch.div(cur_load[k], 3)
                    w_load += [cur_load]

                for cid in range(client_num):
                    clients[cid].load_state_dict(w_load[cid])
                    clients[cid].init_model = w_load[cid]

                for cid in range(client_num):
                    local_f1[cid], local_loss[cid] = clients[cid].evaluate(
                        batch_size)
                    test_f1[cid] = clients[cid].test(test_data, batch_size)
                    max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                    max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                local_avg_f1.append(np.mean(local_f1))
                max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                local_avg_loss.append(np.mean(local_loss))
                min_local_loss = min(min_local_loss, local_avg_loss[-1])
                test_avg_f1.append(np.mean(test_f1))
                max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

            print("-----epoch", epoch, "test f1:", test_avg_f1[-1],
                  "| local f1:", local_avg_f1[-1], " -----", flush=True)

            if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                print("Early stopping...")
                break

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)
    elif mode in ["fedgcn"]:
        index_list = getattr(dataCenter, ds + "_index_list")
        data = getattr(dataCenter, ds + "_total_data")
        test_index = getattr(dataCenter, ds + "_test_index")

        total_communicate, test_adj = fedgcnInit(data,
                                                 clients, client_num, index_list, test_data)

        # load the aggregated feature
        for cid in range(client_num):
            node_num = clients[cid].data.x.shape[0]
            index = index_list[cid]
            for node in range(node_num):
                clients[cid].data.x[node] = total_communicate[index[node]]

        for epoch in range(1, args.epochs + 1):
            loss_list = fedgcnWork(clients, client_num,
                                   client_train_epoch, batch_size)

            with torch.no_grad():
                """load the averaged model"""
                with torch.no_grad():
                    w_avg = FedAvg([clients[i].state_dict()
                                   for i in range(client_num)])
                for cid in range(client_num):
                    local_f1[cid], local_loss[cid] = clients[cid].fedgcn_evaluate()
                    test_f1[cid] = clients[cid].fedgcn_test(
                        test_data, test_adj)
                    max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                    max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                local_avg_f1.append(np.mean(local_f1))
                max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                local_avg_loss.append(np.mean(local_loss))
                min_local_loss = min(min_local_loss, local_avg_loss[-1])
                test_avg_f1.append(np.mean(test_f1))
                max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

            print("-----epoch", epoch, "test f1:", test_avg_f1[-1],
                  "| local f1:", local_avg_f1[-1], " -----", flush=True)

            if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                print("Early stopping...")
                break

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)
    
    elif mode in ["fedsage", "fedsageplus"]:
        # Code adapted from FederatedScope
        # https://github.com/alibaba/FederatedScope
        r"""
        FedSage+ consists of three of training stages.
        Stage1: 0, local pre-train for generator.
        Stage2: -> 2 * fedgen_epoch, federated training for generator.
        Stage3: -> 2 * fedgen_epoch + total_round_num: federated training
        for GraphSAGE Classifier
        """
        total_round_num = args.epochs
        fedgen_epoch = args.fedgen_epoch * 2
        total_round_num = total_round_num + fedgen_epoch

        if mode in ["fedsage"]:
            start_epoch = fedgen_epoch + 1
        elif mode in ["fedsageplus"]:
            start_epoch = 1
            
        if mode in ["fedsageplus"]:
            # step1: local pre-train for generator
            content_list = LocalGenWork(clients, client_num, args.gen_train_epochs)
            contents = [[] for _ in range(client_num)]
            for con in content_list:
                contents[con[-1]] = con[:-1]

        for epoch in range(start_epoch, total_round_num + 1):

            if mode in ["fedsageplus"]:
                # step2: federated training for generator
                if epoch <= fedgen_epoch and epoch % 2 == 1:
                    gradient_dict = FedGenWork(clients, client_num, contents)
                    gen_grads = SumUpGradient(gradient_dict)
                if epoch <= fedgen_epoch and epoch % 2 == 0:
                    content_list = UpdateGenWork(clients, client_num, gen_grads)
                    contents = [[] for _ in range(client_num)]
                    for con in content_list:
                        contents[con[-1]] = con[:-1]
                # step3: federated training
                if epoch == fedgen_epoch + 1:
                    for cid in range(client_num):
                        clients[cid].setup_fedsage()
            if epoch > fedgen_epoch:
                with torch.no_grad():
                    w_avg = FedAvg([clients[i].state_dict()
                                for i in range(client_num)])
                    for cid in range(client_num):
                        clients[cid].load_state_dict(w_avg)
                        clients[cid].init_model = w_avg

                loss_list = fedavgWork(clients, client_num,
                                    client_train_epoch, batch_size)
                
                with torch.no_grad():
                    """load the averaged model of clients"""
                    with torch.no_grad():
                        w_avg = FedAvg([clients[i].state_dict()
                                    for i in range(client_num)])
                    for cid in range(client_num):
                        local_f1[cid], local_loss[cid] = clients[cid].evaluate(
                            batch_size)
                        test_f1[cid] = clients[cid].test(test_data, batch_size)
                        max_local_f1[cid] = max(max_local_f1[cid], local_f1[cid])
                        max_test_f1[cid] = max(max_test_f1[cid], test_f1[cid])
                    local_avg_f1.append(np.mean(local_f1))
                    max_local_avg_f1 = max(max_local_avg_f1, local_avg_f1[-1])
                    local_avg_loss.append(np.mean(local_loss))
                    min_local_loss = min(min_local_loss, local_avg_loss[-1])
                    test_avg_f1.append(np.mean(test_f1))
                    max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])
                
                if cal_cost:
                    for key in clients[0].state_dict().keys():
                        weight_cost += sys.getsizeof(clients[0].state_dict()
                                                    [key].storage())/1024/1024*client_num
                    print("upload weight cost:", weight_cost)

                print("-----epoch",
                    epoch,
                    "test f1:",
                    test_avg_f1[-1],
                    "| local f1:",
                    local_avg_f1[-1],
                    " -----",
                    flush=True)

                if epoch > early_stopping and local_avg_loss[-1] / min_local_loss > early_stopping_bound:
                    print("Early stopping...")
                    break

        print("max local f1:", max_local_f1)
        print("max test f1:", max_test_f1)

        print("max local avg f1:", max_local_avg_f1)
        print("max global avg f1:", max_test_avg_f1)

    """print model prediction"""
    for cid in range(client_num):
        clients[cid].test(test_data, batch_size, print_y=True)
    if timing:
        end_time = time.time()
        print("Time:")
        print((end_time-start_time)/60, "min or", (end_time-start_time), "s")
        print((end_time-start_time)//60, "min +",
              (end_time-start_time) % 60, "s")
    print("add local", np.mean(max_local_f1))
    print("add test", np.mean(max_test_f1))
    if cal_cost:
        print("upload ego cost:", ego_cost, "weight cost:",
              weight_cost, "total cost:", ego_cost+weight_cost, flush=True)
