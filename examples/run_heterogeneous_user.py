#!/usr/bin/env python3
"""Create the placeholder input file for a user-defined heterogeneous case.

This script writes an editable material template to the default output
directory so that custom heterogeneous material layouts can be introduced in a
clean and inspectable way.

Default output directory:
    <repo>/results_heterogeneous_user/

Generated file:
    <repo>/results_heterogeneous_user/material_template.json

Concept:

    +--------------------------------------+
    |   user-defined regions, phases, or   |
    |   spatially varying material laws    |
    +--------------------------------------+

Use this entry point when you want a lightweight starting point for custom
heterogeneous material inputs.
"""

from pathlib import Path

from chebykan_dem.cases import resolve_output_dir, write_heterogeneous_material_template
from chebykan_dem.utils import log

_REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out = resolve_output_dir("heterogeneous_user", parent=str(_REPO_ROOT))
    write_heterogeneous_material_template(out)
    log(f"Output directory: {out}")
