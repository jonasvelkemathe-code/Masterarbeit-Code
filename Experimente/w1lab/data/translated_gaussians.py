import torch


class GaussShiftProvider:
    def __init__(self, d: int, shift: float, device: str):
        self.d = d
        self.device = device
        self.a = torch.zeros(d, device=device)
        self.a[0] = shift

    @torch.no_grad()
    def sample(self, n: int):
        X = torch.randn(n, self.d, device=self.device)
        Y = X + self.a
        return X, Y

    def true_w1(self) -> float:
        return self.a.norm().item()


def make_provider_gauss_shift(d: int, shift: float, device: str):
    return GaussShiftProvider(d, shift, device)
