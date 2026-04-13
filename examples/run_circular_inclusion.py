#!/usr/bin/env python3
"""Run the circular inclusion benchmark.

Default output directory:
    <repo>/results_circular_inclusion/

Schematic geometry (conceptual only):

    +--------------------------------------+
    |                                      |
    |               matrix                 |
    |            (circular inclusion)      |
    |                                      |
    +--------------------------------------+

The benchmark geometry is read from ``shared_mesh.msh`` in the case directory.
Material properties and inclusion-specific settings can be adapted through the
standard repository workflow.
"""

from pathlib import Path

from chebykan_dem.cases import require_mesh_or_exit, resolve_output_dir
from chebykan_dem.config import set_output_dir
from chebykan_dem.solver import run_pipeline
from chebykan_dem.utils import log

_REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = "circular_inclusion"

if __name__ == "__main__":
    out = resolve_output_dir(CASE, parent=str(_REPO_ROOT))
    set_output_dir(out)
    log(f"Output directory: {out}")
    require_mesh_or_exit(out, CASE)
    run_pipeline(out, case_id=CASE, generate_mesh_if_missing=False)
