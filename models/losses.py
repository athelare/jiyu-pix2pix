import torch
from torch.nn import functional as F
from torch import Tensor
import torch.nn as nn


class RL1Loss(nn.Module):
    def __init__(self, roll_dis: int):
        super(RL1Loss, self).__init__()
        self.roll_dis = roll_dis

    def forward(self, _input: Tensor, target: Tensor) -> Tensor:
        res_candies = Tensor()
        for i in range(-self.roll_dis,self.roll_dis+1):
            for j in range(-self.roll_dis,self.roll_dis+1):
                tmp_in = torch.roll(_input, (i, j), (0, 1))
                res_candies = torch.cat((res_candies, F.l1_loss(tmp_in, target, reduction='mean')))
        return torch.min(res_candies)
