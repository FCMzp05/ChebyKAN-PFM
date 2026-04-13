"""Gmsh mesh generation and loading (no finite-element backend)."""

from __future__ import annotations

import os

import numpy as np

from chebykan_dem.config import conf
from chebykan_dem.utils import log


def generate_shared_mesh(output_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a notched plate mesh with Gmsh and write ``shared_mesh.msh``."""
    try:
        import gmsh
    except ImportError as e:
        raise ImportError("Gmsh Python API is required. Install with: pip install gmsh") from e

    os.makedirs(output_dir, exist_ok=True)
    mesh_file = os.path.join(output_dir, "shared_mesh.msh")

    log(">>> Generating Gmsh mesh (notched plate)...")

    gmsh.initialize()
    gmsh.model.add("plate_with_notch")

    Lx, Ly = conf.Lx, conf.Ly
    crack_x0, crack_x1 = conf.crack_x0, conf.crack_x1
    notch_half_width = conf.notch_half_width
    mesh_size_tip, mesh_size_far = conf.mesh_size_tip, conf.mesh_size_far

    p_bl = gmsh.model.geo.addPoint(-Lx / 2, -Ly / 2, 0, mesh_size_far)
    p_br = gmsh.model.geo.addPoint(Lx / 2, -Ly / 2, 0, mesh_size_far)
    p_tr = gmsh.model.geo.addPoint(Lx / 2, Ly / 2, 0, mesh_size_far)
    p_tl = gmsh.model.geo.addPoint(-Lx / 2, Ly / 2, 0, mesh_size_far)

    p_notch_left_top = gmsh.model.geo.addPoint(crack_x0, notch_half_width, 0, mesh_size_tip)
    p_notch_left_bottom = gmsh.model.geo.addPoint(crack_x0, -notch_half_width, 0, mesh_size_tip)
    p_notch_right_top = gmsh.model.geo.addPoint(crack_x1, notch_half_width, 0, mesh_size_tip)
    p_notch_right_bottom = gmsh.model.geo.addPoint(crack_x1, -notch_half_width, 0, mesh_size_tip)

    l_left_bottom = gmsh.model.geo.addLine(p_bl, p_notch_left_bottom)
    l_left_top = gmsh.model.geo.addLine(p_notch_left_top, p_tl)
    l_top = gmsh.model.geo.addLine(p_tl, p_tr)
    l_right = gmsh.model.geo.addLine(p_tr, p_br)
    l_bottom = gmsh.model.geo.addLine(p_br, p_bl)

    l_notch_bottom = gmsh.model.geo.addLine(p_notch_left_bottom, p_notch_right_bottom)
    l_notch_right = gmsh.model.geo.addLine(p_notch_right_bottom, p_notch_right_top)
    l_notch_top = gmsh.model.geo.addLine(p_notch_right_top, p_notch_left_top)

    curve_loop = gmsh.model.geo.addCurveLoop(
        [l_left_bottom, l_notch_bottom, l_notch_right, l_notch_top, l_left_top, l_top, l_right, l_bottom]
    )

    gmsh.model.geo.addPlaneSurface([curve_loop])
    gmsh.model.geo.synchronize()

    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", [l_notch_top, l_notch_right, l_notch_bottom])
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 100)

    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    gmsh.model.mesh.field.setNumber(f2, "SizeMin", mesh_size_tip)
    gmsh.model.mesh.field.setNumber(f2, "SizeMax", mesh_size_far)
    gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.3)

    gmsh.model.mesh.field.setAsBackgroundMesh(f2)
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_file)
    log(f"  Saved mesh: {mesh_file}")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

    elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_conn = None
    for i, e_type in enumerate(elem_types):
        if e_type == 2:
            tri_conn = elem_node_tags[i].reshape(-1, 3)
            break
    if tri_conn is None:
        gmsh.finalize()
        raise RuntimeError("No 2D triangle elements found in mesh.")

    node_map = {tag: i for i, tag in enumerate(node_tags)}
    conn_mapped = np.zeros_like(tri_conn, dtype=np.int64)
    for i in range(tri_conn.shape[0]):
        for j in range(3):
            conn_mapped[i, j] = node_map[tri_conn[i, j]]

    x, y = nodes[:, 0], nodes[:, 1]
    x1, y1 = x[conn_mapped[:, 0]], y[conn_mapped[:, 0]]
    x2, y2 = x[conn_mapped[:, 1]], y[conn_mapped[:, 1]]
    x3, y3 = x[conn_mapped[:, 2]], y[conn_mapped[:, 2]]
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    gmsh.finalize()

    log(f"  Mesh: {nodes.shape[0]} nodes, {conn_mapped.shape[0]} triangles")
    return nodes, conn_mapped, areas


def load_mesh_data(output_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load triangle mesh from ``shared_mesh.msh``."""
    try:
        import gmsh
    except ImportError as e:
        raise ImportError("Gmsh Python API is required.") from e

    mesh_file = os.path.join(output_dir, "shared_mesh.msh")
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    gmsh.initialize()
    gmsh.open(mesh_file)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

    elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_conn = None
    for i, e_type in enumerate(elem_types):
        if e_type == 2:
            tri_conn = elem_node_tags[i].reshape(-1, 3)
            break
    if tri_conn is None:
        gmsh.finalize()
        raise RuntimeError("No 2D triangle elements in mesh file.")

    node_map = {tag: i for i, tag in enumerate(node_tags)}
    conn_mapped = np.zeros_like(tri_conn, dtype=np.int64)
    for i in range(tri_conn.shape[0]):
        for j in range(3):
            conn_mapped[i, j] = node_map[tri_conn[i, j]]

    x, y = nodes[:, 0], nodes[:, 1]
    x1, y1 = x[conn_mapped[:, 0]], y[conn_mapped[:, 0]]
    x2, y2 = x[conn_mapped[:, 1]], y[conn_mapped[:, 1]]
    x3, y3 = x[conn_mapped[:, 2]], y[conn_mapped[:, 2]]
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    gmsh.finalize()
    return nodes, conn_mapped, areas


def save_mesh_data(nodes: np.ndarray, conn: np.ndarray, areas: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "mesh_coordinates.npz")
    np.savez(path, nodes=nodes, conn=conn, areas=areas)
