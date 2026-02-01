import torch
import torch.nn as nn

@torch.no_grad()
def parseval_project_weight_(W: torch.Tensor, beta: float = 1e-4, rows: bool = True) -> torch.Tensor:
    if rows:
        WWt = W @ W.t()
        tmp = WWt @ W
    else:
        WtW = W.t() @ W
        tmp = W @ WtW
    W.mul_(1.0 + beta)
    W.add_(-beta * tmp)
    return W

@torch.no_grad()
def parseval_project_module_(module: nn.Module, beta: float = 1e-4, rows_if_out_ge_in: bool = True):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            W = m.weight
            out, in_ = W.shape
            rows = rows_if_out_ge_in if (out >= in_) else (not rows_if_out_ge_in)
            parseval_project_weight_(W, beta=beta, rows=rows)



