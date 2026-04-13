"""
ChebyKAN deep energy method (DEM) for phase-field fracture on triangular meshes.
"""

from chebykan_dem.cases import DEFAULT_OUTPUT_BY_CASE, default_output_dir, resolve_output_dir
from chebykan_dem.config import Config, get_config
from chebykan_dem.mesh import generate_shared_mesh, load_mesh_data, save_mesh_data
from chebykan_dem.naming import METHOD_SUBDIR, RESULTS_SCHEMA_VERSION
from chebykan_dem.network import ChebyKANLayer, ChebyKANNetwork
from chebykan_dem.physics import DEMSolverBase
from chebykan_dem.solver import ChebyKANDEMSolver, run_chebykan_dem, run_pipeline

__all__ = [
    "Config",
    "get_config",
    "DEFAULT_OUTPUT_BY_CASE",
    "default_output_dir",
    "resolve_output_dir",
    "METHOD_SUBDIR",
    "RESULTS_SCHEMA_VERSION",
    "ChebyKANLayer",
    "ChebyKANNetwork",
    "DEMSolverBase",
    "ChebyKANDEMSolver",
    "run_chebykan_dem",
    "run_pipeline",
    "generate_shared_mesh",
    "load_mesh_data",
    "save_mesh_data",
]

__version__ = "0.1.0"
