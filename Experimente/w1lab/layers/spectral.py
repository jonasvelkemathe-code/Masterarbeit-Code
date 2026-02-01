import torch.nn as nn
from torch.nn.utils import spectral_norm

def SpectralLinear(in_features, out_features, bias=True, n_power_iterations=1):
    lin = nn.Linear(in_features, out_features, bias=bias)
    spectral_norm(lin, n_power_iterations=n_power_iterations)
    return lin
