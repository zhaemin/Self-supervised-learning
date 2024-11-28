import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim

from warmup import WarmupCosineAnnealingScheduler

import models.SSL as ssl

def parsing_argument():
    parser = argparse.ArgumentParser(description="argparse_test")
    
    parser.add_argument('-e', '--epochs', metavar='int', type=int, help='epochs', default=2)
    parser.add_argument('-lr', '--learningrate', metavar='float', type=float, help='lr', default=0.0001)
    parser.add_argument('-d', '--dataset', metavar='str', type=str, help='dataset [miniimagenet, cifarfs]', default='miniimagenet')
    parser.add_argument('-opt', '--optimizer', metavar='str', type=str, help='optimizer [adam, sgd]', default='sgd')
    parser.add_argument('-crt', '--criterion', metavar='str', type=str, help='criterion [ce, mse]', default='ce')
    parser.add_argument('-tr', '--train', help='train', action='store_true')
    parser.add_argument('-tc', '--test', help='test', action='store_true')
    parser.add_argument('-val', '--val', help='validation', action='store_true')
    parser.add_argument('-m', '--model', metavar='str', type=str, help='models [protonet, feat, relationnet]', default='moco')
    parser.add_argument('-bs', '--batch_size', metavar='int', type=int, help='batchsize', default=256)
    
    return parser.parse_args()

def set_parameters(args, net):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.learningrate)
        scheduler = None
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80, 100, 120], gamma=0.5)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=net.parameters(), lr=args.learningrate, momentum=0.9, nesterov=True)
        #scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, base_lr=args.learningrate, T_max=100, eta_min=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-3)

    return optimizer,scheduler

def load_model(args):
    if args.model == 'moco':
        net = ssl.MoCo(q_size=4096, momentum=0.999)
    elif args.model == 'simclr':
        net = ssl.SimCLR()
    elif args.model == 'swav':
        net = ssl.SwAV()
    return net
