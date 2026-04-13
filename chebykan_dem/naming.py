"""
Output naming aligned with the paper / revisecode layout.

Each case directory typically contains::

    <output_dir>/
    ├── config.json
    ├── case_summary.json
    ├── shared_mesh.msh
    ├── mesh_coordinates.npz
    ├── quick_preview.png
    └── chebykan/
        ├── results.csv
        ├── alpha_profile_y0.csv
        ├── timing.csv
        ├── step_{NNN}.npz
        ├── frame_{NNN}.png
        ├── force_displacement.png
        ├── model.pth
        ├── chebyshev_coeffs.npz
        └── learned_activations.npz
"""

from __future__ import annotations

import json
import os
from typing import Any

# Subfolder for ChebyKAN-DEM outputs (same as revisecode).
METHOD_SUBDIR = "chebykan"

RESULTS_SCHEMA_VERSION = 1

# CSV column names (must match revisecode / manuscript tables).
RESULTS_CSV_HEADER = ["step", "disp", "force", "E_elastic", "E_fracture", "E_total", "alpha_max"]
TIMING_CSV_HEADER = ["step", "epochs", "time_step", "time_cumulative"]
ALPHA_PROFILE_HEADER = ["x", "alpha"]

# step_000.npz, frame_000.png — three-digit step index (same as Chebyunified / revisecode DEM).
STEP_NPZ_FMT = "step_{:03d}.npz"
FRAME_PNG_FMT = "frame_{:03d}.png"


def method_dir(output_root: str) -> str:
    return os.path.join(output_root, METHOD_SUBDIR)


def save_case_summary(output_dir: str, case_id: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "case_id": case_id,
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "chebykan_subdir": METHOD_SUBDIR,
    }
    if extra:
        payload.update(extra)
    path = os.path.join(output_dir, "case_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
