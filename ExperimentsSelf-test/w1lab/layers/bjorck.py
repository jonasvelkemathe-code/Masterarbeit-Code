import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


def power_iteration(W: torch.Tensor, iters: int = 10) -> torch.Tensor:
    m, n = W.shape
    u = F.normalize(W.new_empty(m).normal_(), dim=0)
    v = F.normalize(W.new_empty(n).normal_(), dim=0)
    for _ in range(iters):
        v = F.normalize(W.t().mv(u), dim=0)
        u = F.normalize(W.mv(v), dim=0)
    sigma = u @ (W @ v)
    return sigma.clamp_min(1e-8)


class BjorckParam(nn.Module):
    def __init__(self, iters: int = 3, beta: float = 0.5,
                 pre_scale: bool = True, pi_iters: int = 5):
        super().__init__()
        self.iters = iters
        self.beta = beta
        self.pre_scale = pre_scale
        self.pi_iters = pi_iters

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if self.pre_scale:
            s = power_iteration(W, iters=self.pi_iters)
            W = W / s

        m, n = W.shape
        if m >= n:
            for _ in range(self.iters):
                S = W.t() @ W
                S2 = S @ S
                I = torch.eye(n, device=W.device, dtype=W.dtype)
                M = (15.0 / 8.0) * I - (5.0 / 4.0) * S + (3.0 / 8.0) * S2
                W = W @ M
        else:
            for _ in range(self.iters):
                S = W @ W.t()
                S2 = S @ S
                I = torch.eye(m, device=W.device, dtype=W.dtype)
                M = (15.0 / 8.0) * I - (5.0 / 4.0) * S + (3.0 / 8.0) * S2
                W = M @ W

        return W


class LinearBjorck(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 iters=3, beta=0.5, pre_scale=True, pi_iters=5):
        super().__init__(in_features, out_features, bias=bias)
        parametrize.register_parametrization(
            self,
            "weight",
            BjorckParam(iters=iters, beta=beta, pre_scale=pre_scale, pi_iters=pi_iters)
        )
