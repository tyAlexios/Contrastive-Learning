import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjHead(nn.Module):
    def __init__(self, indim=2048, outdim=128):
        super(ProjHead, self).__init__()
        
        self.hidden = nn.Linear(indim, outdim)
        self.out = nn.Linear(outdim, outdim)
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

