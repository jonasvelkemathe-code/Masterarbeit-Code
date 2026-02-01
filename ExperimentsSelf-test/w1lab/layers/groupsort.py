import torch
import torch.nn as nn

class GroupSort(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        if D & 1:
            x = torch.cat((x, x.new_zeros(B, 1)), dim=1)
        x = x.view(B, -1, 2)
        x = torch.stack((x.max(-1).values, x.min(-1).values), dim=-1).view(B, -1)
        return x[:, :D]

class MaxMin(GroupSort):
    pass


