"""Hyperparameters and problem setup (notched plate, AT2 phase field)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    Lx: float = 1.0
    Ly: float = 1.0
    crack_x0: float = -0.5
    crack_y0: float = 0.0
    crack_x1: float = 0.0
    crack_y1: float = 0.0
    notch_half_width: float = 0.0075

    mat_E: float = 210000.0
    mat_nu: float = 0.3
    Gc: float = 2.7
    l0: float = 0.015

    mesh_size_tip: float = 0.003
    mesh_size_far: float = 0.008

    u_max: float = 0.008
    n_steps: int = 81

    pff_model_type: str = "AT2"
    tol_ir: float = 1e-2
    phi_c: float = 0.90
    k_residual: float = 1e-10

    kan_layers: list = field(default_factory=lambda: [2, 128, 128, 64, 3])
    chebyshev_degree: int = 3

    n_epochs_pretrain: int = 1000
    n_epochs_main: int = 3000
    lr: float = 5e-5
    weight_decay: float = 1e-6
    alpha_grad_penalty: float = 0.005
    rigid_body_penalty: float = 1.0

    seed: int = 1234
    output_dir: str = "results_sent_homogeneous"

    def __post_init__(self) -> None:
        self.lmbda = self.mat_E * self.mat_nu / ((1 + self.mat_nu) * (1 - 2 * self.mat_nu))
        self.mu = self.mat_E / (2 * (1 + self.mat_nu))
        self.disp_steps = np.linspace(0.0, self.u_max, self.n_steps)


# Global config used by physics/solver (single-threaded scripts).
conf = Config()


def get_config() -> Config:
    return conf


def set_output_dir(path: str) -> None:
    conf.output_dir = path


def save_config_json(
    output_dir: str,
    cfg: Config | None = None,
    *,
    case_id: str | None = None,
    results_schema_version: int | None = None,
) -> None:
    from chebykan_dem.naming import RESULTS_SCHEMA_VERSION

    cfg = cfg or conf
    d: dict[str, Any] = {
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "crack_x0": cfg.crack_x0,
        "crack_x1": cfg.crack_x1,
        "mat_E": cfg.mat_E,
        "mat_nu": cfg.mat_nu,
        "Gc": cfg.Gc,
        "l0": cfg.l0,
        "u_max": cfg.u_max,
        "n_steps": cfg.n_steps,
        "kan_layers": cfg.kan_layers,
        "chebyshev_degree": cfg.chebyshev_degree,
        "results_schema_version": results_schema_version if results_schema_version is not None else RESULTS_SCHEMA_VERSION,
    }
    if case_id is not None:
        d["case_id"] = case_id
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
