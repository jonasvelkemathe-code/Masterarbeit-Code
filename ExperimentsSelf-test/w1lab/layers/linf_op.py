import torch
import torch.nn as nn


@torch.no_grad()
def project_matrix_op_linf_(W, tau=1.0):
    m, n = W.shape


    for i in range(m):
        row = W[i]


        signs = torch.sign(row)
        y = row.abs()


        u, _ = torch.sort(y, descending=True)


        prefix = torch.cumsum(u, dim=0)


        rho = 1
        for k in range(1, n + 1):
            theta_k = (prefix[k - 1] - tau) / k


            if u[k - 1] > theta_k:
                rho = k


        theta = (prefix[rho - 1] - tau) / rho


        y_proj = torch.clamp(y - theta, min=0.0)
        W[i] = signs * y_proj

    return W

@torch.no_grad()
def project_module_op_linf_(module, tau_linear=1.0):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            project_matrix_op_linf_(m.weight, tau_linear)
