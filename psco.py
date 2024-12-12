import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.SSL import MoCo, sinkhorn
from utils import split_support_query_set

class PsCo(MoCo):
    def __init__(self, q_size=1024, momentum=0.99):
        super(PsCo, self).__init__(q_size, momentum)
        self.projector = self.make_mlp(512, True)
        self.predictor = self.make_mlp(128, False)
        self.shots = 4
    
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        x_q = inputs[0]
        x_k = inputs[1]
        
        #momentum update
        with torch.no_grad():
            for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
            
            for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
        
        q = F.normalize(self.predictor(self.projector(self.encoder(x_q))))
        q_moco = q.clone()
        k = F.normalize(self.projector_k(self.encoder_k(x_k))).detach()
        
        #psco loss
        with torch.no_grad():
            sim = torch.einsum('bd, qd -> bq', k, self.queue.clone().detach())
            labels_tilde = sinkhorn(sim, device)
            support_set, labels = self.select_topk(self.queue, labels_tilde)
            labels = labels.to(device)
        
        logits = torch.einsum('bd, sd -> bs', q, support_set)
        loss_psco = logits.logsumexp(dim=1) - (torch.sum(logits * labels, dim=1) / self.shots)
        loss_psco = loss_psco.mean()
        
        # moco loss
        logits_moco_positive = torch.einsum('bd, bd -> b', q_moco, k).unsqueeze(1) # b 1
        logits_moco_negative = torch.einsum('bd, qd -> bq', q_moco, self.queue.clone().detach())
        logits_moco = torch.cat((logits_moco_positive, logits_moco_negative), dim=1) # b 1+q
        
        labels_moco = torch.zeros(batch_size).long().to(device)
        loss_moco = F.cross_entropy(logits_moco / 0.2, labels_moco)
        
        #enqueue
        self.enqueue(batch_size, k)
        
        return loss_psco + loss_moco
    
    def cross_domain_adapatation(self, inputs, device):
        for param in self.encoder.parameters:
            param.requires_grad = False
        
        batch_size = inputs[0].size(0)
        
        x_q = inputs[0]
        x_k = inputs[1]
        
        q = F.normalize(self.predictor(self.projector(self.encoder(x_q))))
        k = F.normalize(self.projector(self.encoder(x_k))).detach()
        
        # moco loss
        logits_moco_positive = torch.einsum('bd, bd -> b', q, k).unsqueeze(1) # b 1
        logits_moco_negative = torch.einsum('bd, qd -> bq', q, self.queue.clone().detach())
        logits_moco = torch.cat((logits_moco_positive, logits_moco_negative), dim=1) # b 1+q
        
        labels_moco = torch.zeros(batch_size).long().to(device)
        loss_moco = F.cross_entropy(logits_moco / 0.2, labels_moco)
        
        return loss_moco
    
    def select_topk(self, queue, labels_tilde):
        _, indicies = torch.topk(labels_tilde, k=self.shots, dim=1)
        b, k = indicies.shape
        support_set = queue[indicies].clone().detach().view(b * k, -1) # b k d -> (bk) d
        
        labels = torch.zeros([b, b * k])
        
        for i in range(b):
            labels[i, k * i : k * i + k] = 1
        
        return support_set, labels
    
    def fewshot_acc(self, args, inputs, labels, num_ways, device):
        x = self.projector(self.encoder(inputs))
        tasks = split_support_query_set(x, labels, device, num_tasks=4, num_shots=args.num_shots)
        
        correct = 0
        total = 0
        
        for x_support, x_query, y_support, y_query in tasks:
            x_support = F.normalize(x_support)
            x_query = F.normalize(self.predictor(x_query)) # q d
            prototypes = F.normalize(torch.sum(x_support.view(5, args.shots, -1), dim=1), dim=1) # 5 d
            
            logits = torch.einsum('qd, wd -> qw', x_query, prototypes)
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == y_query).sum().item()
            total += y_query.size(0)
        
        acc = 100 * correct / total
        return acc
