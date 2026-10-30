# ChebyKAN-DEM

PyTorch implementation of **Chebyshev KAN** networks for a **deep energy method (DEM)** solving **phase-field fracture** (AT2, spectral split) on **Gmsh** triangle meshes.

This repository is organized as a compact **research-code release** accompanying the paper. It exposes the model structure, training workflow, benchmark entry points, and result layout in a form that is easy to inspect and adapt.

## Repository Overview

The repository contains:
- the ChebyKAN-DEM solver implementation,
- a self-contained homogeneous SENT benchmark,
- several heterogeneous benchmark entry points,
- a template entry point for user-defined heterogeneous material layouts,
- result files consistent with the paper.

The homogeneous SENT example can generate its mesh directly in the repository. The heterogeneous benchmark examples read their geometry from a benchmark mesh file named:

```text
shared_mesh.msh
```

Material fields and heterogeneous layouts can be adapted through the case-specific input structure and the provided template utilities.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

A working **Gmsh** installation is required. The Python package `gmsh` wraps the Gmsh library, but the underlying Gmsh runtime must still be available on your machine.

## Quick Start

Run the homogeneous SENT benchmark first:

```bash
python examples/run_sent_homogeneous.py
```

or equivalently:

```bash
python -m chebykan_dem --case sent_homogeneous
chebykan-dem --case sent_homogeneous
```

This is the simplest entry point and is the recommended first check for a new environment.

## Geometry and Input Conventions

The heterogeneous benchmark examples use a shared convention:
- geometry is read from `shared_mesh.msh`,
- case results are written to a dedicated output directory,
- material information can be adapted through user-editable inputs or template files.

Typical command pattern:

```bash
chebykan-dem --case layered_bimaterial --output ./results_layered_bimaterial
```

Supported case ids:
- `sent_homogeneous`
- `layered_bimaterial`
- `circular_inclusion`
- `three_inclined_cracks`
- `fgm_threephase`
- `heterogeneous_user`

For `heterogeneous_user`, the CLI points you to `examples/run_heterogeneous_user.py`, which creates the placeholder material input file.

## Example Entry Points (`examples/`)

| Script | Role |
|---|---|
| `run_sent_homogeneous.py` | Homogeneous SENT benchmark with in-repo mesh generation. |
| `run_layered_bimaterial.py` | Layered bimaterial benchmark entry point using the standard benchmark mesh convention. |
| `run_circular_inclusion.py` | Circular inclusion benchmark entry point using the standard benchmark mesh convention. |
| `run_three_inclined_cracks.py` | Three inclined cracks benchmark entry point using the standard benchmark mesh convention. |
| `run_fgm_threephase.py` | Three-phase FGM-style benchmark entry point using the standard benchmark mesh convention. |
| `run_heterogeneous_user.py` | Writes `results_heterogeneous_user/material_template.json` as a starting point for custom heterogeneous material inputs. |

## Output Layout

Each case directory follows the naming convention used in the paper:

```text
<case_dir>/
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
```

`results.csv` columns:
- `step`
- `disp`
- `force`
- `E_elastic`
- `E_fracture`
- `E_total`
- `alpha_max`

`timing.csv` columns:
- `step`
- `epochs`
- `time_step`
- `time_cumulative`

Displacement written to CSV is in **metres**. Most figures convert the horizontal axis to **micrometres (μm)** for presentation.

## Configuration

The main runtime configuration lives in:

- `chebykan_dem/config.py`

This file controls geometry dimensions, loading schedule, network widths, Chebyshev degree, optimizer settings, and output locations.

## Notes For Users

- Start with `sent_homogeneous` to confirm your environment is working.
- Use the heterogeneous examples as benchmark-specific entry points following the same output and mesh conventions.
- If you want a custom heterogeneous setup, start from `examples/run_heterogeneous_user.py` and extend the material handling in the solver / physics layer.
- The repository is intentionally structured so that benchmark geometry, solver configuration, and material customization remain easy to inspect and modify.

## Citation

If you use this code, please cite:

> P. Zhang, K. Tang, B.-X. Xu. ChebyKAN-enhanced deep energy method for phase field fracture: From homogeneous to heterogeneous materials. *Journal of the Mechanics and Physics of Solids* 214, 106693 (2026). https://doi.org/10.1016/j.jmps.2026.106693

```bibtex
@article{Zhang2026ChebyKANDEM,
  author  = {Zhang, Peng and Tang, Keke and Xu, Bai-Xiang},
  title   = {{ChebyKAN}-enhanced deep energy method for phase field fracture: From homogeneous to heterogeneous materials},
  journal = {Journal of the Mechanics and Physics of Solids},
  volume  = {214},
  pages   = {106693},
  year    = {2026},
  doi     = {10.1016/j.jmps.2026.106693}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
