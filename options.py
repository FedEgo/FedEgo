#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataSet", type=str, default="wiki",
                        help="cora, citeseer, wiki, corafull, FedDBLP")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1027)
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--max_vali_f1", type=float, default=0)
    parser.add_argument("--config", type=str, default="experiments.conf")
    parser.add_argument("--client_train_epoch", type=int, default=5)
    parser.add_argument("--server_train_epoch", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=300)
    parser.add_argument("--early_stopping", type=int, default=100)
    parser.add_argument("--cal_cost", action="store_true", help="calculate communication cost")
    parser.add_argument("--linear", action="store_true", help="linear personalization layers")
    parser.add_argument("--sigma", type=float, default=0, help="[0, 0.05, 0.075, 0.15, 0.3]")
    

    parser.add_argument("--mode",
                        type=str,
                        default="fedsage",
                        help="fedavg, local, fedego_np, fedego_nr, fedego, fedprox, dfedgnn, fedgcn, fedsage, fedsageplus, fedego_ne")

    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--split_mode", type=str, default="label",
                        help="label, louvain")
    parser.add_argument("--sample_rate", type=float, default=0.3)
    parser.add_argument("--global_sample_rate", type=float, default=0.3)
    parser.add_argument("--major_rate", type=float, default=0.8)
    parser.add_argument("--share_node_num", type=int, default=500)
    parser.add_argument("--major_label", type=int, default=3)
    parser.add_argument("--lamb_c", type=float, default=0.5)
    parser.add_argument("--lamb_fixed", type=int, default=0)
    parser.add_argument("--mixup", type=int, default=1)
    parser.add_argument("--sageMode", type=str,
                        default="GraphSAGE", help="GraphSAGE, GAT")
    parser.add_argument("--timing", type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--h_feats', type=int, default=64,
                        help='hidden features')

    parser.add_argument('--gen_train_epochs', type=int, default=1)
    parser.add_argument('--fedgen_epoch', type=int, default=20)

    parser.add_argument('--logsuffix', type=str, default="", help='logsuffix')
    args = parser.parse_args()
    return args
