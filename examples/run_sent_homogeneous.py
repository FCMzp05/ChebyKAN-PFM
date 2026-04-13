#!/usr/bin/env python3
"""Run the self-contained homogeneous SENT benchmark.

This is the recommended first example for public users because it is the only
benchmark in this repository that generates its mesh internally.

Default output directory:
    <repo>/results_sent_homogeneous/

Schematic:

    Mode I opening

           y
           ^
           |      prescribed opening displacement
      _____|_______________________________
     |                                     |
     |                                     |
     |   single-edge notch                 |
     |   on the left side          ______  |
     |                            |      | |
     |____________________________|______|_|

Notes:
- No external ``shared_mesh.msh`` is needed.
- The script simply resolves the default output directory and launches the
  standard solver pipeline.
"""

from pathlib import Path

from chebykan_dem.cases import resolve_output_dir
from chebykan_dem.config import set_output_dir
from chebykan_dem.solver import run_pipeline
from chebykan_dem.utils import log

_REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out = resolve_output_dir("sent_homogeneous", parent=str(_REPO_ROOT))
    set_output_dir(out)
    log(f"Output directory: {out}")
    run_pipeline(out, case_id="sent_homogeneous", generate_mesh_if_missing=True)
