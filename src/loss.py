# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ContrastiveLoss(nn.Module):

    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance()
        loss = torch.mean((1- label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss





