import torch

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk+1, 1, False, True)
        pred = pred // 4
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct[1:].float()

        res = []
        for k in topk:
            correct_k = correct[:k].sum(0).contiguous().view(-1)
            rank_k = torch.where(correct_k>0, 1., 0.).sum(0, keepdim=True)
            res.append(rank_k.mul_(100.0 / (batch_size-1)))
            
        return res