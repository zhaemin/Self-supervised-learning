import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        self.f= self.make_encoder()
        self.g = nn.Sequential(
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
    
    def make_encoder(self):
        f = models.resnet18()
        f.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        f.maxpool = nn.Identity()
        f.fc = nn.Identity()
        
        return f
    
    def encoding(self, x):
        return F.normalize(self.f(x), dim=-1)
    
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        x = torch.cat((inputs[0], inputs[1]))
        z = self.g(self.f(x))
        z1, z2 = torch.chunk(z, 2, dim=0)
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        logits1 = torch.einsum('ad, bd -> ab', z1, z2)
        logits2 = torch.einsum('ad, bd -> ab', z2, z1)
        
        sim_labels = torch.arange(0, batch_size).to(device)
        loss = (F.cross_entropy(logits1 / 0.5, sim_labels) + F.cross_entropy(logits2 / 0.5, sim_labels)) / 2
        
        return loss