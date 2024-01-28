import torch
import torch.nn.functional as F
import os
import copy

def takeSecond(elem):
    return elem[1]

def estimate(centers, data, labels, num_pos, save_path=None, use_gpu=True):
    
    correct = 0.0
    count = 0.0
    data = F.normalize(data, dim=1)
    
    centers = F.normalize(centers, dim=1)
    if use_gpu:
        data = data.cuda()
        centers = centers.cuda()
        labels = labels.cuda()
    
    if save_path is not None:
        file = open(save_path, "a")
    
    for i, (feature, label) in enumerate(zip(data, labels)):
        
        if save_path is not None and i%num_pos == 0:
            file.write("\n\n"+50*'='+"  CLASS:{}  ".format(int(label))+50*'='+'\n')

        cos_sim_label = F.cosine_similarity(feature, centers[int(label)], dim=0)
        
        rank = []
            
        for j, c in enumerate(centers):
            cos = F.cosine_similarity(feature, c, dim=0)
            rank.append((j, cos))
        rank.sort(key=takeSecond, reverse=True)
        
        if int(rank[0][0]) == int(label):
            correct += 1
        count += 1
        
        if save_path is not None:
            file.write("\n\nView: {0}    Label_Center: {1}    Cos_sim_label: {2:7.4f}\nTop5([center]cos_sim):    ".format(i%num_pos, int(label), cos_sim_label))
            for k in range(5):
                file.write("[{0:3}]{1:7.4f}  ".format(rank[k][0], rank[k][1]))
        
    if save_path is not None:
        file.close()
    
    return correct, count
