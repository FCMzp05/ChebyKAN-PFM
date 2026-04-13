#!/usr/bin/env python3
"""Run the three inclined cracks benchmark.

Default output directory:
    <repo>/results_three_inclined_cracks/

Schematic geometry (conceptual only):

    A rectangular specimen containing three pre-existing inclined crack
    segments. The exact crack angles, crack lengths, specimen dimensions,
    and loading boundaries are encoded in the benchmark mesh.

The benchmark geometry is read from ``shared_mesh.msh`` in the case directory.
Case-specific material settings can be adapted through the standard repository
input structure.
"""

from pathlib import Path

from chebykan_dem.cases import require_mesh_or_exit, resolve_output_dir
from chebykan_dem.config import set_output_dir
from chebykan_dem.solver import run_pipeline
from chebykan_dem.utils import log

_REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = "three_inclined_cracks"

if __name__ == "__main__":
    out = resolve_output_dir(CASE, parent=str(_REPO_ROOT))
    set_output_dir(out)
    log(f"Output directory: {out}")
    require_mesh_or_exit(out, CASE)
    run_pipeline(out, case_id=CASE, generate_mesh_if_missing=False)
