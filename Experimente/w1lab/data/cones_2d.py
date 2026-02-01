import torch

_CENTERS = torch.tensor([[-2.0, 0.0],
                         [ 0.0, 0.0],
                         [ 2.0, 0.0]])

class Cones2DProvider:
    def __init__(self, device: str = "cpu"):
        self.device = device

    @torch.no_grad()
    def sample(self, n: int):
        centers = _CENTERS.to(self.device)
        idx = torch.randint(0, 3, (n,), device=self.device)
        C = centers[idx]

        X = C.clone()

        theta = 2 * torch.pi * torch.rand(n, device=self.device)
        unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        Y = C + unit
        return X, Y

    def true_w1(self) -> float:
        return 1.0

def make_provider_cones2d(device: str):
    return Cones2DProvider(device)
