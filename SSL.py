import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SSLFramework(nn.Module):
    def __init__(self):
        super(SSLFramework, self).__init__()
        self.encoder= self.make_encoder()
        self.projector = self.make_mlp()
    
    def make_mlp(self):
        return nn.Sequential(
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
    
    def make_encoder(self):
        encoder = models.resnet18()
        encoder.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity()
        
        return encoder
    
    def encoding(self, x):
        return F.normalize(self.encoder(x), p=2)

class SimCLR(SSLFramework):
    def __init__(self):
        super(SimCLR, self).__init__()
        
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        z1 = self.projector(self.encoder(inputs[0]))
        z2 = self.projector(self.encoder(inputs[1]))
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        z = torch.cat((z1,z2), dim=1).view(batch_size*2, -1)
        logits = torch.einsum('ad, bd -> ab', z, z)
        
        for i in range(batch_size*2):
            logits[i][i] = 0
        
        tmp_labels1 = torch.arange(1, 2*batch_size, step=2)
        tmp_labels2 = torch.arange(0, 2*batch_size, step=2)
        
        labels = []
        for l1, l2 in zip(tmp_labels1, tmp_labels2):
            labels.append(l1)
            labels.append(l2)
        labels = torch.stack(labels).to(device)
        
        loss = F.cross_entropy(logits / 0.5, labels)
        
        return loss

class MoCo(SSLFramework):
    def __init__(self, q_size, momentum):
        super(MoCo, self).__init__()
        self.dim = 128
        
        self.register_buffer('queue', F.normalize(torch.randn(q_size, self.dim), p=2))
        self.register_buffer('idx', torch.zeros(1, dtype=torch.int64))
        self.max_queue_size = q_size
        self.momentum = momentum
        
        self.projector_k = self.make_mlp()
        self.encoder_k = self.make_encoder()
        
        for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
            
        for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
        
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
        
        q = F.normalize(self.projector(self.encoder(x_q)))
        k = F.normalize(self.projector_k(self.encoder_k(x_k)))
        k = k.detach() 
        
        queue = self.queue.clone().detach()
        logits_positive = torch.einsum('bd, bd -> b', q, k).unsqueeze(1)
        logits_negative = torch.einsum('bd, kd -> bk', q, queue)
        
        logits = torch.cat((logits_positive, logits_negative), dim=1)
        
        sim_labels = torch.zeros(batch_size).long().to(device)
        loss = F.cross_entropy(logits / 0.2, sim_labels)
        
        #enqueue
        with torch.no_grad():
            idx = int(self.idx)
            remain = idx + batch_size - self.max_queue_size
            if remain > 0:
                self.queue[idx : idx + batch_size - remain] = k[: batch_size - remain]
                self.queue[: remain] = k[batch_size - remain :]
                self.idx[0] = remain
            else:
                self.queue[idx : idx + batch_size] = k
                self.idx[0] = (idx + batch_size) % self.max_queue_size
        return loss

class SwAV(SSLFramework):
    def __init__(self):
        super(SwAV, self).__init__()
        self.temperature = 0.1
        self.dim = 128
        self.prototypes = nn.Parameter(torch.randn(3000, self.dim)) # 3000 * 128
        self.max_queue_size = 256*15 #3840
        self.register_buffer('queue', torch.randn(2, self.max_queue_size, self.dim))
        self.register_buffer('idx', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('start_queue', torch.zeros(1, dtype=torch.int64))
        
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)
        
        z1 = self.projector(self.encoder(inputs[0])) # batchsize * 128
        z2 = self.projector(self.encoder(inputs[1]))
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        if self.start_queue < 15:
            scores_1 = torch.einsum('nd, kd -> nk', z1, self.prototypes)
            scores_2 = torch.einsum('nd, kd -> nk', z2, self.prototypes)
            self.start_queue[0] += 1
        else:
            z1_aug = torch.cat((z1, self.queue[0]))
            z2_aug = torch.cat((z2, self.queue[1]))
            
            scores_1 = torch.einsum('nd, kd -> nk', z1_aug, self.prototypes)
            scores_2 = torch.einsum('nd, kd -> nk', z2_aug, self.prototypes)
        
        with torch.no_grad():
            code_1 = self.sinkhorn(scores_1, device)
            code_2 = self.sinkhorn(scores_2, device)
        
        prob_1 = F.softmax(scores_1 / self.temperature, dim=1)
        prob_2 = F.softmax(scores_2 / self.temperature, dim=1)
        
        loss = -0.5 * torch.mean(torch.sum(code_1  * torch.log(prob_2), dim=1) + torch.sum(code_2 * torch.log(prob_1), dim=1))
        
        #enqueue
        with torch.no_grad():
            idx = int(self.idx)
            remain = idx + batch_size - self.max_queue_size
            if remain > 0:
                self.queue[0][idx : idx + batch_size - remain] = z1[: batch_size - remain]
                self.queue[0][: remain] = z1[batch_size - remain :]
                self.queue[1][idx : idx + batch_size - remain] = z2[: batch_size - remain]
                self.queue[1][: remain] = z2[batch_size - remain :]
                self.idx[0] = remain
            else:
                self.queue[0][idx : idx + batch_size] = z1
                self.queue[1][idx : idx + batch_size] = z2
                self.idx[0] = (idx + batch_size) % self.max_queue_size
                
        return loss
            
    def sinkhorn(self, scores, device, eps=0.05, niters=3):
        code = torch.transpose(torch.exp(scores / eps), 0, 1)
        code /= torch.sum(code)
        k, b = code.shape
        u, r, c = torch.zeros(k), torch.ones(k) / k, torch.ones(b) / b
        u, r, c = u.to(device), r.to(device), c.to(device)
        
        for _ in range(niters):
            u = torch.sum(code, dim=1)
            code *= (r / u).unsqueeze(1) # k * b
            code *= (c / torch.sum(code, dim=0)).unsqueeze(0) # k * b
        
        return torch.transpose((code / torch.sum(code, dim=0, keepdim=True)), 0, 1) # b * k