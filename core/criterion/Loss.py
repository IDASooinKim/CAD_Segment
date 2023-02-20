# -- coding: utf-8 -*-    

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdMSoftmaxLoss(nn.Module):

    r"""
        [Discription]
            Additive Margin Softmax Loss.
        [Attributes]
            embedding_dim: Dimension of the embedding vector
            n_classes: Number of classes to be embedded
            scale: Global scale factor
            margin: size of additive margin
    """

    def __init__(self,
        embedding_dim: int,
        n_classes: int,
        scale: float=30.0,
        margin: float=0.4
    ):
        super(AdMSoftmaxLoss, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin

        self.embedding = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=embedding_dim,
            max_norm=1
        )
        self.loss = nn.CrossEntropyLoss()        
    
    def forward(self,
        x: torch.Tensor,
        labels: torch.Tensor
    ):
        n, m = x.shape
        
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.n_classes

        x = F.normalize(
            input=x, 
            dim=1
        )
        w = self.embedding_weight
        cos_theta = torch.matmul(
            input=w, 
            other=x.T
        ).T
        psi = cos_theta - self.margin

        one_hot = F.one_hot(
            tensor=labels,
            num_classes=self.n_classes
        )
        logits = self.scale * torch.where(one_hot==1, psi, cos_theta)
        error = self.loss(logits, labels)

        return error, logits


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, 
            inputs: torch.tensor, 
            targets: torch.tensor, 
            smooth=1
        ):
        
        inputs = F.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice 