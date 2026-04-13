"""Case registry and user-facing guidance for benchmark entry points."""

from __future__ import annotations

import os
import sys

from chebykan_dem.utils import log

# Case id -> default output folder name.
DEFAULT_OUTPUT_BY_CASE: dict[str, str] = {
    "sent_homogeneous": "results_sent_homogeneous",
    "layered_bimaterial": "results_layered_bimaterial",
    "circular_inclusion": "results_circular_inclusion",
    "three_inclined_cracks": "results_three_inclined_cracks",
    "fgm_threephase": "results_fgm_threephase",
    "heterogeneous_user": "results_heterogeneous_user",
}

CASE_HELP: dict[str, str] = {
    "sent_homogeneous": (
        "Homogeneous single-edge-notched tension specimen with in-repo Gmsh mesh generation."
    ),
    "layered_bimaterial": (
        "Layered bimaterial benchmark using the standard shared_mesh.msh geometry convention."
    ),
    "circular_inclusion": (
        "Circular inclusion benchmark using the standard shared_mesh.msh geometry convention."
    ),
    "three_inclined_cracks": (
        "Three pre-existing inclined cracks benchmark using the standard shared_mesh.msh geometry convention."
    ),
    "fgm_threephase": (
        "Three-phase FGM-style benchmark using the standard shared_mesh.msh geometry convention."
    ),
    "heterogeneous_user": (
        "Template entry point for user-defined heterogeneous materials; writes material_template.json under the case directory."
    ),
}


def default_output_dir(case_id: str) -> str:
    """Return the default output folder name for a given case id."""
    if case_id not in DEFAULT_OUTPUT_BY_CASE:
        raise KeyError(f"Unknown case_id={case_id!r}. Known: {list(DEFAULT_OUTPUT_BY_CASE)}")
    return DEFAULT_OUTPUT_BY_CASE[case_id]



def resolve_output_dir(case_id: str, parent: str | None = None) -> str:
    """Resolve ``<parent>/<default_output_dir(case_id)>``.

    If ``parent`` is omitted, the current working directory is used.
    """
    root = parent or os.getcwd()
    return os.path.join(root, default_output_dir(case_id))



def require_mesh_or_exit(output_dir: str, case_id: str) -> None:
    """Ensure that the benchmark mesh file exists in the case directory."""
    mesh_file = os.path.join(output_dir, "shared_mesh.msh")
    if os.path.isfile(mesh_file):
        return

    log(f"Missing mesh file: {mesh_file}")
    log(f"Case: {case_id} — {CASE_HELP.get(case_id, '')}")
    log(
        "This benchmark reads its geometry from 'shared_mesh.msh'. "
        "Please ensure that the file is present in the case output directory, then rerun."
    )
    sys.exit(1)



def write_heterogeneous_material_template(output_dir: str) -> None:
    """Write ``material_template.json`` under ``output_dir``.

    This file is intended as a lightweight starting point for user-defined
    heterogeneous material data.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "material_template.json")
    template = {
        "_comment": (
            "Editable material input template. Adapt these fields to your own "
            "heterogeneous material layout or spatial material law."
        ),
        "E": None,
        "nu": None,
        "Gc": None,
        "l0": None,
        "regions": None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    log(f"Wrote {path}")
