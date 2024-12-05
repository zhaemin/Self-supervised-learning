import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.SSL import MoCo, sinkhorn

class PsCo(MoCo):
    def __init__(self, q_size=1024, momentum=0.99):
        super(PsCo, self).__init__(q_size, momentum)
        self.predictor = self.make_mlp(128)
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
        
        q_moco = self.projector(self.encoder(x_q))
        q = F.normalize(self.predictor(q_moco.clone()))
        q_moco = F.normalize(q_moco)
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
    
    def select_topk(self, queue, labels_tilde):
        _, indicies = torch.topk(labels_tilde, k=self.shots, dim=1)
        b, k = indicies.shape
        support_set = queue[indicies].clone().detach().view(b * k, -1) # b k d -> (bk) d
        
        labels = torch.zeros([b, b * k])
        
        for i in range(b):
            labels[i, k * i : k * i + k] = 1
        
        return support_set, labels
