import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim

from warmup import WarmupCosineAnnealingScheduler

def parsing_argument():
    parser = argparse.ArgumentParser(description="argparse_test")
    
    parser.add_argument('-e', '--epochs', metavar='int', type=int, help='epochs', default=2)
    parser.add_argument('-lr', '--learningrate', metavar='float', type=float, help='lr', default=0.01)
    parser.add_argument('-d', '--dataset', metavar='str', type=str, help='dataset [miniimagenet, cifarfs]', default='miniimagenet')
    parser.add_argument('-opt', '--optimizer', metavar='str', type=str, help='optimizer [adam, sgd]', default='sgd')
    parser.add_argument('-crt', '--criterion', metavar='str', type=str, help='criterion [ce, mse]', default='ce')
    parser.add_argument('-tr', '--train', help='train', action='store_true')
    parser.add_argument('-val', '--val', help='validation', action='store_true')
    parser.add_argument('-ad', '--adaptation', help='adaptation', action='store_true')
    parser.add_argument('-m', '--model', metavar='str', type=str, help='models [protonet, feat, relationnet]', default='moco')
    parser.add_argument('-bs', '--batch_size', metavar='int', type=int, help='batchsize', default=256)
    parser.add_argument('-tc', '--test', metavar='str', type=str, help='knn, fewshot, cross-domain', default='knn')
    parser.add_argument('-b', '--backbone', metavar='str', type=str, help='conv5, resnet10|18', default='resnet10')
    parser.add_argument('-mixup', '--mixup', help='mixup in psco', action='store_true')
    
    parser.add_argument('-tr_ways', '--train_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-ts_ways', '--test_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-shots', '--num_shots', metavar='int', type=int, help='shots', default=5)
    parser.add_argument('-tasks', '--num_tasks', metavar='int', type=int, help='tasks', default=1)
    parser.add_argument('-q', '--num_queries', metavar='int', type=int, help='queries', default=15)
    parser.add_argument('-ep', '--episodes', metavar='int', type=int, help='episodes', default=100)
    
    return parser.parse_args()

def set_parameters(args, net, len_trainloader):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.learningrate)
        scheduler = None
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80, 100, 120], gamma=0.5)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=net.parameters(), lr=args.learningrate, weight_decay=0.0005, momentum=0.9)
        #scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, base_lr=args.learningrate, T_max=100, eta_min=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len_trainloader)

    return optimizer,scheduler



def split_support_query_set(x, y, device, num_class=5, num_shots=5, num_queries=15, num_tasks=1, training=False):
    
    x_list = torch.chunk(x, num_tasks)
    y_list = torch.chunk(y, num_tasks)
    tasks = []
    
    for i in range(num_tasks):
        x, y = x_list[i], y_list[i]
        num_sample_support = num_class * num_shots
        x_support, x_query = x[:num_sample_support], x[num_sample_support:]
        y_support, y_query = y[:num_sample_support], y[num_sample_support:]
        
        _classes = torch.unique(y_support)
        support_idx = torch.stack(list(map(lambda c: y_support.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xs = torch.cat([x_support[idx_list] for idx_list in support_idx])
        
        query_idx = torch.stack(list(map(lambda c: y_query.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xq = torch.cat([x_query[idx_list] for idx_list in query_idx])
        
        ys = torch.arange(0, len(_classes), 1 / num_shots).long().to(device)
        yq = torch.arange(0, len(_classes), 1 / num_queries).long().to(device)
        
        tasks.append([xs, xq, ys, yq])
        
    return tasks

def mixup(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample([input.shape[0]]).to(device=input.device) # beta distribution에서 random sampling [0,1]
    lam = torch.max(lam, 1. - lam) # 항상 lam 값이 0.5 이상이 되게함
    lam_expanded = lam.view([-1] + [1]*(input.dim()-1)) # lam -> b, 1, 1, 1
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam

import models.SSL as ssl
import models.psco as psco
import models.vicreg as vicreg
import models.fewshot_models as fewshot_models

def load_model(args):
    if args.model == 'moco':
        net = ssl.MoCo(args.backbone, q_size=16384, momentum=0.99)
    elif args.model == 'simclr':
        net = ssl.SimCLR(args.backbone)
    elif args.model == 'swav':
        net = ssl.SwAV(args.backbone)
    elif args.model == 'psco':
        net = psco.PsCo(args.backbone, mixup=args.mixup)
    elif args.model == 'protonet':
        net = fewshot_models.ProtoNet()
    elif args.model == 'vicreg':
        net = vicreg.VICReg(args.backbone, mixup=args.mixup)
    return net
