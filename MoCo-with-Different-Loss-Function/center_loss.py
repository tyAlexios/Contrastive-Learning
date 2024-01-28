import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        self.lamda = 1
        
        if use_gpu:
            self.fc = nn.Linear(feat_dim, num_classes).cuda()
        else:
            self.fc = nn.Linear(feat_dim, num_classes)
            
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        
        out = self.fc(x)
        ceLoss = F.cross_entropy(out, labels)
        
        
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        centerLoss = dist.clamp(min=1e-12, max=1e+12).sum() * (1. / 2. / batch_size)
        
        
        loss = ceLoss + self.lamda * centerLoss
        
        return out, loss, ceLoss, centerLoss

