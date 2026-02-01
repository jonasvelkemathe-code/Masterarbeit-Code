import torch

class ConeDiracShellProvider:
    def __init__(self, d: int, device: str = "cpu"):
        self.d = d
        self.device = device

    @torch.no_grad()
    def sample(self, n: int):
        X = torch.zeros(n, self.d, device=self.device)

        Z = torch.randn(n, self.d, device=self.device)
        Y = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return X, Y

    def true_w1(self) -> float:
        return 1.0

def make_provider_cone_dirac_shell(d: int, device: str):
    return ConeDiracShellProvider(d, device)
