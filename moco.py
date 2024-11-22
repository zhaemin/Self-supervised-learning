import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MoCo(nn.Module):
    def __init__(self, q_size, momentum):
        super(MoCo, self).__init__()
        self.dim = 128
        
        self.register_buffer('queue', F.normalize(torch.randn(q_size, self.dim)))
        self.idx = 0
        self.max_queue_size = q_size
        
        self.momentum = momentum
        
        self.f_q = self.make_encoder()
        self.f_k = self.make_encoder()
        
        for k_param, q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
        
    def make_encoder(self):
        f = models.resnet18()
        f.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        f.maxpool = nn.Identity()
        f.fc = nn.Sequential(
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        
        return f
    
    def encoding(self, x):
        return F.normalize(self.f_q(x), p=2)
        
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        x_q = inputs[0]
        x_k = inputs[1]
        
        #momentum update
        with torch.no_grad():
            for k_param, q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
        
        q = F.normalize(self.f_q(x_q), p=2)
        k = F.normalize(self.f_k(x_k), p=2)
        k = k.detach()
        
        queue = self.queue.clone().detach()
        logits_positive = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(2)
        logits_negative = torch.mm(q, queue.transpose(0, 1))
        
        logits = torch.cat((logits_positive, logits_negative), dim=1)
        
        sim_labels = torch.zeros(batch_size).long().to(device)
        loss = F.cross_entropy(logits / 0.2, sim_labels)
        
        #enqueue
        with torch.no_grad():
            remain = self.idx + batch_size - self.max_queue_size
            if remain > 0:
                self.queue[self.idx : self.idx + batch_size - remain] = k[: batch_size - remain]
                self.queue[: remain] = k[batch_size - remain :]
                self.idx = remain
            else:
                self.queue[self.idx : self.idx + batch_size] = k
                self.idx = (self.idx + batch_size) % self.max_queue_size
            
        return loss