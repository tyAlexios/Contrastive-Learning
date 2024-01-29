# Contrastive-Learning-Exporation
Codes for my own implements and experiments on Contrastive Learning

# Testing Result for *Center Loss Standard*

## Objective
This part is exploring the characteristics of different loss functions (CrossEntropyLoss &	CenterLoss) on finetuning the projection head of MoCo. The result indicates that the result of CenterLoss is better than CrossEntropyLoss when the dataset is not large enougLR_model_1e-3_LR_center_1e-2_lamda1

## Training information
1. ResNet is frozen and only projection head is trained
2. learning-rate of model(projection head): 1e-3
3. loss-function: Center Loss
4. learning-rate of Center Loss: 1e-2
5. Center Loss lamda: 1

## Training and testing log
```
Epoch 300/300
Epoch: [300][ 1/28]	CrossEntropyLoss 0.0045 (0.0045)	CenterLoss 0.2327 (0.2327)	Loss 0.0278 (0.0278)	ceAcc@1 100.00 (100.00)	ceAcc@5 100.00 (100.00)	ceAcc@10 100.00 (100.00)
Epoch: [300][ 7/28]	CrossEntropyLoss 0.0031 (0.0054)	CenterLoss 0.2973 (0.2932)	Loss 0.0329 (0.0347)	ceAcc@1 100.00 (100.00)	ceAcc@5 100.00 (100.00)	ceAcc@10 100.00 (100.00)
Epoch: [300][13/28]	CrossEntropyLoss 0.0034 (0.0042)	CenterLoss 0.3808 (0.3324)	Loss 0.0415 (0.0375)	ceAcc@1 100.00 (100.00)	ceAcc@5 100.00 (100.00)	ceAcc@10 100.00 (100.00)
Epoch: [300][19/28]	CrossEntropyLoss 0.0038 (0.0046)	CenterLoss 0.3898 (0.3485)	Loss 0.0427 (0.0395)	ceAcc@1 100.00 ( 99.96)	ceAcc@5 100.00 (100.00)	ceAcc@10 100.00 (100.00)
Epoch: [300][25/28]	CrossEntropyLoss 0.0053 (0.0047)	CenterLoss 0.4251 (0.3505)	Loss 0.0478 (0.0397)	ceAcc@1 100.00 ( 99.97)	ceAcc@5 100.00 (100.00)	ceAcc@10 100.00 (100.00)

Test (Rank):
ACC@1:    69.1729%    ACC@5:    86.9674%    ACC@10:    92.4812%
```
