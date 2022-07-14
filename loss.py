import torch
import torch.nn as nn


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)  # 利用预测值与标签相乘当作交集
        union = torch.sum(predict + target)
        dice = 1. - (2 * intersection + self.epsilon) / (union + self.epsilon)
        return dice



