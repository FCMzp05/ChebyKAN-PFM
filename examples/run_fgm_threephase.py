#!/usr/bin/env python3
"""Run the three-phase FGM-style benchmark.

Default output directory:
    <repo>/results_fgm_threephase/

Schematic geometry (conceptual only):

    +--------------------------------------+
    |     phase 1     | phase 2 | phase 3 |
    |                 |         |         |
    +--------------------------------------+

The actual benchmark may use region boundaries, inclusions, or graded zones
that are more detailed than this sketch. The benchmark geometry is read from
``shared_mesh.msh`` in the case directory.

Material inputs can be adapted through the repository configuration and the
user-editable material workflow.
"""

from pathlib import Path

from chebykan_dem.cases import require_mesh_or_exit, resolve_output_dir
from chebykan_dem.config import set_output_dir
from chebykan_dem.solver import run_pipeline
from chebykan_dem.utils import log

_REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = "fgm_threephase"

if __name__ == "__main__":
    out = resolve_output_dir(CASE, parent=str(_REPO_ROOT))
    set_output_dir(out)
    log(f"Output directory: {out}")
    require_mesh_or_exit(out, CASE)
    run_pipeline(out, case_id=CASE, generate_mesh_if_missing=False)
