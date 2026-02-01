import torch.nn as nn
from w1lab.layers import GroupSort, LinearBjorck, SpectralLinear


def make_critic(d: int, width: int = 128, depth: int = 3,
                method: str = "bjorck", bjorck_iters: int = 3):
    layers = []
    act = GroupSort()

    def lin(i, o):
        if method == "bjorck":
            return LinearBjorck(i, o, iters=bjorck_iters)
        if method == "spectral":
            return SpectralLinear(i, o)
        return nn.Linear(i, o)

    feat = d
    for _ in range(depth - 1):
        layers += [lin(feat, width), act]
        feat = width
    layers += [lin(feat, 1)]
    return nn.Sequential(*layers)
