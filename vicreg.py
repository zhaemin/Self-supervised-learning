import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
from tqdm import tqdm
import numpy as np

import models.encoders as encoders
from models.SSL import SSLFramework

from utils import split_support_query_set, mixup

'''
class VICReg(SSLFramework):
    def __init__(self, backbone, mixup):
        super(VICReg, self).__init__(backbone)
        self.projector =  self.make_mlp(self.outdim, hidden_dim=2048, num_layers=3, out_dim=2048, last_bn=False)
        self.num_features = 2048
        self.mixup = mixup
        
    def forward(self, inputs, device):
        x_a = inputs[0]
        x_b = inputs[1]
        if self.mixup:
            x_mixup, mixup_ind, lam = mixup(inputs[2], alpha=1.)
            z_mixup = self.projector(self.encoder(x_mixup))
        
        batchsize = x_a.size(0)
        
        z_a = self.projector(self.encoder(x_a))
        z_b = self.projector(self.encoder(x_b))
        
        if self.mixup:
            z = torch.sum(z_a, z_b).div(2)
            lam_expanded = lam.view([-1] + [1]*(z.dim()-1)) # lam -> b, 1, 1, 1
            mixup_feature = lam_expanded * z + (1. - lam_expanded) * z[mixup_ind]
            # mixup loss
            mixup_loss = F.mse_loss(z_mixup, mixup_feature)
        
        #invariance loss
        sim_loss = F.mse_loss(z_a, z_b)
        
        #variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 0.0001) # 2048, feature 별 variance
        std_z_b = torch.sqrt(z_b.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z_a)) / 2 + torch.mean(F.relu(1 - std_z_b)) / 2
        
        #covariance loss
        z_a = z_a - z_a.mean(dim=0) # batchsize 2048
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = torch.einsum('am, bn -> mn', z_a, z_a) / (batchsize - 1)
        cov_z_b = torch.einsum('am, bn -> mn', z_b, z_b) / (batchsize - 1)
        cov_z_a = cov_z_a.flatten()[:-1].view(self.num_features - 1, self.num_features + 1)[:, 1:].flatten() # off diagonal -> 41922256
        cov_z_b = cov_z_b.flatten()[:-1].view(self.num_features - 1, self.num_features + 1)[:, 1:].flatten()
        cov_loss = cov_z_a.pow_(2).sum().div(self.num_features)+ cov_z_b.pow_(2).sum().div(self.num_features)
        
        #loss
        loss = 25 * sim_loss + 25 * std_loss + 1 * cov_loss
        if self.mixup:
            loss += mixup_loss
        
        return loss
'''

class VICReg(SSLFramework):
    def __init__(self, backbone, mixup):
        super(VICReg, self).__init__(backbone)
        self.projector =  self.make_mlp(self.outdim, hidden_dim=2048, num_layers=3, out_dim=2048, last_bn=False)
        self.num_features = 2048
        self.mixup = mixup

    def forward(self, inputs, device):
        x = inputs[0]
        y = inputs[1]
        batch_size = x.size(0)
        
        x = self.projector(self.encoder(x))
        y = self.projector(self.encoder(y))
        if self.mixup:
            z_mixup, mixup_ind, lam = mixup(inputs[2], alpha=1.)
            z_mixup = self.projector(self.encoder(z_mixup))
            
            feature_mixup = x + y / 2
            lam_expanded = lam.view([-1] + [1]*(feature_mixup.dim()-1)) # lam -> b, 1
            mixup_feature = lam_expanded * feature_mixup + (1. - lam_expanded) * feature_mixup[mixup_ind]
            # mixup loss
            mixup_loss = F.mse_loss(z_mixup, mixup_feature)
        
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        
        loss = (25 * repr_loss + 25 * std_loss + 1 * cov_loss)
        if self.mixup:
            loss += mixup_loss
        
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()