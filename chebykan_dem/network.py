"""Chebyshev polynomial KAN layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from chebykan_dem.config import conf
from chebykan_dem.utils import log


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.expanded_dim = in_features * (degree + 1)
        self.linear = nn.Linear(self.expanded_dim, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        t_list = [torch.ones_like(x_norm), x_norm]
        for k in range(2, self.degree + 1):
            t_next = 2 * x_norm * t_list[-1] - t_list[-2]
            t_list.append(t_next)
        t_concat = torch.cat(t_list, dim=-1)
        return self.linear(t_concat)


class ChebyKANNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int] | None = None, degree: int | None = None):
        super().__init__()
        layer_sizes = layer_sizes or conf.kan_layers
        degree = degree if degree is not None else conf.chebyshev_degree

        self.layer_sizes = layer_sizes
        self.degree = degree

        self.kan_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.kan_layers.append(ChebyKANLayer(layer_sizes[i], layer_sizes[i + 1], degree))

        total_params = sum(p.numel() for p in self.parameters())
        log(
            f"  ChebyKAN: {' → '.join(map(str, self.layer_sizes))}, "
            f"degree={self.degree}, params={total_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.kan_layers[:-1]):
            x = layer(x)
            x = torch.tanh(x)
        return self.kan_layers[-1](x)
