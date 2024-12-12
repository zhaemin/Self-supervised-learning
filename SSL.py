import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SSLFramework(nn.Module):
    def __init__(self):
        super(SSLFramework, self).__init__()
        self.encoder= self.make_encoder()
        self.projector = self.make_mlp(512)
    
    def make_mlp(self, input_dim, last_bn=False):
        mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128))
        if last_bn:
            mlp.append(nn.BatchNorm1d(128))
        
        return mlp
    
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
        
        z = torch.cat((z1,z2), dim=0)
        logits = torch.einsum('ad, bd -> ab', z, z)
        logits.fill_diagonal_(float('-inf'))
        
        tmp_labels1 = torch.arange(batch_size, 2*batch_size)
        tmp_labels2 = torch.arange(0, batch_size)
        labels = torch.cat((tmp_labels1, tmp_labels2)).to(device)
        
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
        
        self.projector_k = self.make_mlp(512)
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
        
        self.enqueue(batch_size, k)
            
        return loss
    
    def enqueue(self, batch_size, k):
        with torch.no_grad():
            idx = int(self.idx)
            self.queue[idx : idx + batch_size] = k
            self.idx[0] = (idx + batch_size) % self.max_queue_size

class SwAV(SSLFramework):
    def __init__(self):
        super(SwAV, self).__init__()
        self.temperature = 0.1
        self.dim = 128
        self.mm_prototypes = nn.Linear(128, 3000)
        
    def forward(self, inputs, device):
        with torch.no_grad():
            w = self.mm_prototypes.weight.data.clone()
            w = F.normalize(w, dim=1)
            self.mm_prototypes.weight.copy_(w)
        
        z1 = F.normalize(self.projector(self.encoder(inputs[0]))) # batchsize * 128
        z2 = F.normalize(self.projector(self.encoder(inputs[1])))
        
        scores_1 = self.mm_prototypes(z1)
        scores_2 = self.mm_prototypes(z2)
        
        with torch.no_grad():
            scores_1_tmp = scores_1.detach()
            scores_2_tmp = scores_2.detach()
            
            code_1 = sinkhorn(scores_1_tmp, device)
            code_2 = sinkhorn(scores_2_tmp, device)
            
        prob_1 = F.log_softmax(scores_1 / self.temperature, dim=1)
        prob_2 = F.log_softmax(scores_2 / self.temperature, dim=1)
        
        loss = -0.5 * torch.mean(torch.sum(code_1  * prob_2, dim=1) + torch.sum(code_2 * prob_1, dim=1))
                
        return loss
    
def sinkhorn(scores, device, eps=0.05, niters=3):
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
