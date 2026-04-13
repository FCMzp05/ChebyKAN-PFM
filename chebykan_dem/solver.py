"""Training loop, outputs, and ChebyKAN interpretability exports."""

from __future__ import annotations

import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
import torch.optim as optim

from chebykan_dem.config import conf
from chebykan_dem.naming import (
    ALPHA_PROFILE_HEADER,
    FRAME_PNG_FMT,
    METHOD_SUBDIR,
    RESULTS_CSV_HEADER,
    STEP_NPZ_FMT,
    TIMING_CSV_HEADER,
    save_case_summary,
)
from chebykan_dem.network import ChebyKANNetwork
from chebykan_dem.physics import DEMSolverBase
from chebykan_dem.utils import ensure_dir, log


class ChebyKANDEMSolver(DEMSolverBase):
    def __init__(self, nodes: np.ndarray, conn: np.ndarray, areas: np.ndarray) -> None:
        network = ChebyKANNetwork().to(conf.device)
        super().__init__(network, nodes, conn, areas)


def extract_alpha_profile_y0(nodes: np.ndarray, alpha: np.ndarray, tol: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(nodes[:, 1]) < tol
    x_profile = nodes[mask, 0]
    alpha_profile = alpha[mask] if isinstance(alpha, np.ndarray) else alpha.cpu().numpy()[mask]
    sort_idx = np.argsort(x_profile)
    return x_profile[sort_idx], alpha_profile[sort_idx]


def save_method_results(
    data_log: list,
    nodes: np.ndarray,
    alpha_final: np.ndarray,
    output_dir: str,
    subdir: str = METHOD_SUBDIR,
) -> None:
    method_dir = os.path.join(output_dir, subdir)
    ensure_dir(method_dir)

    results_path = os.path.join(method_dir, "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RESULTS_CSV_HEADER)
        writer.writerows(data_log)

    x_prof, alpha_prof = extract_alpha_profile_y0(nodes, alpha_final)
    prof_path = os.path.join(method_dir, "alpha_profile_y0.csv")
    with open(prof_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ALPHA_PROFILE_HEADER)
        for x, a in zip(x_prof, alpha_prof):
            writer.writerow([x, a])


def _save_interpretability(solver: ChebyKANDEMSolver, method_dir: str) -> None:
    log("  Saving ChebyKAN interpretability (weights, coeffs, activation samples)...")

    model_path = os.path.join(method_dir, "model.pth")
    torch.save(solver.network.state_dict(), model_path)
    log(f"    model: {model_path}")

    coeffs_dict: dict[str, np.ndarray] = {}
    for layer_idx, layer in enumerate(solver.network.kan_layers):
        weight = layer.linear.weight.detach().cpu().numpy()
        out_f, in_f, deg = layer.out_features, layer.in_features, layer.degree
        coeffs = weight.reshape(out_f, in_f, deg + 1)
        coeffs_dict[f"coeffs_layer_{layer_idx}"] = coeffs

    coeffs_path = os.path.join(method_dir, "chebyshev_coeffs.npz")
    np.savez(coeffs_path, **coeffs_dict)
    log(f"    coeffs: {coeffs_path}")

    x_samples = np.linspace(-1, 1, 100)
    first = solver.network.kan_layers[0]
    out_f, in_f, deg = first.out_features, first.in_features, first.degree
    weight_first = first.linear.weight.detach().cpu().numpy()
    coeffs_first = weight_first.reshape(out_f, in_f, deg + 1)

    activations = np.zeros((out_f, in_f, len(x_samples)))
    for i in range(out_f):
        for j in range(in_f):
            c = coeffs_first[i, j, :]
            t = np.zeros((deg + 1, len(x_samples)))
            t[0, :] = 1.0
            if deg >= 1:
                t[1, :] = x_samples
            for k in range(2, deg + 1):
                t[k, :] = 2 * x_samples * t[k - 1, :] - t[k - 2, :]
            activations[i, j, :] = np.sum(c[:, np.newaxis] * t, axis=0)

    np.savez(os.path.join(method_dir, "learned_activations.npz"), x_samples=x_samples, activations=activations)
    log(f"    learned_activations: shape {activations.shape}")


def run_chebykan_dem(
    solver: ChebyKANDEMSolver,
    nodes: np.ndarray,
    output_dir: str,
    result_subdir: str = METHOD_SUBDIR,
) -> np.ndarray:
    log("\n>>> ChebyKAN-DEM training")
    method_dir = os.path.join(output_dir, result_subdir)
    ensure_dir(method_dir)

    optimizer = optim.Rprop(solver.network.parameters(), lr=conf.lr, step_sizes=(1e-10, 50))
    H_history = torch.zeros(solver.inp.shape[0], device=conf.device)

    log(f"  Pretraining ({conf.n_epochs_pretrain} epochs)...")
    for _epoch in range(conf.n_epochs_pretrain):
        optimizer.zero_grad()
        u, v, alpha = solver.compute_fields(0.0)
        loss = torch.mean(u**2 + v**2 + alpha**2)
        loss.backward()
        optimizer.step()

    data_log: list = []
    timing_log: list[list[float]] = []
    time_cumulative = 0.0
    triangles = solver.conn.cpu().numpy()
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)

    log(f"  Loading steps ({conf.n_steps})...")
    log(f"  Step |  Disp(μm) |  Force(N) | α_max  |   E_el(J)  |  E_frac(J) | E_total(J) | Time(s)")
    log("  " + "-" * 95)

    for step_idx, disp_val in enumerate(conf.disp_steps):
        start_t = time.time()

        for _epoch in range(conf.n_epochs_main):
            optimizer.zero_grad()
            loss = solver.loss_total(H_history, disp_val)
            loss += conf.weight_decay * sum(torch.sum(p**2) for p in solver.network.parameters())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            E_el, E_frac, E_pen = solver.get_total_energy(H_history, disp_val)
        E_el_val, E_frac_val = E_el.item(), E_frac.item()

        with torch.no_grad():
            eps_fd = 1e-6
            if disp_val < eps_fd:
                E_0 = E_el_val + E_frac_val + E_pen.item()
                E_p_el, E_p_frac, E_p_pen = solver.get_total_energy(H_history, disp_val + eps_fd)
                reaction_force = (E_p_el.item() + E_p_frac.item() + E_p_pen.item() - E_0) / eps_fd
            else:
                E_m_el, E_m_frac, E_m_pen = solver.get_total_energy(H_history, disp_val - eps_fd)
                E_p_el, E_p_frac, E_p_pen = solver.get_total_energy(H_history, disp_val + eps_fd)
                E_m = E_m_el.item() + E_m_frac.item() + E_m_pen.item()
                E_p = E_p_el.item() + E_p_frac.item() + E_p_pen.item()
                reaction_force = (E_p - E_m) / (2 * eps_fd)

        with torch.no_grad():
            u_curr, v_curr, alpha_curr = solver.compute_fields(disp_val)
            solver.update_alpha_history(disp_val)
            H_history = solver.update_history_field(H_history, disp_val)
            alpha_max = alpha_curr.max().item()

        elapsed = time.time() - start_t
        time_cumulative += elapsed
        timing_log.append([step_idx, float(conf.n_epochs_main), elapsed, time_cumulative])

        E_total = E_el_val + E_frac_val
        data_log.append([step_idx, disp_val, reaction_force, E_el_val, E_frac_val, E_total, alpha_max])

        log(
            f"  {step_idx:4d} | {disp_val * 1000:.2f}μm | {reaction_force:9.1f}N | {alpha_max:.4f} | "
            f"{E_el_val:10.6f} | {E_frac_val:10.6f} | {E_total:10.6f} | {elapsed:6.1f}s"
        )

        np.savez(
            os.path.join(method_dir, STEP_NPZ_FMT.format(step_idx)),
            u=u_curr.cpu().numpy(),
            v=v_curr.cpu().numpy(),
            alpha=alpha_curr.cpu().numpy(),
        )

        if step_idx % 5 == 0 or step_idx == conf.n_steps - 1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.tripcolor(triang, alpha_curr.cpu().numpy(), shading="gouraud", cmap="coolwarm", vmin=0, vmax=1)
            ax.set_aspect("equal")
            ax.set_title(f"ChebyKAN-DEM: D={disp_val * 1000:.2f}μm")
            plt.savefig(os.path.join(method_dir, FRAME_PNG_FMT.format(step_idx)), dpi=100)
            plt.close()

    alpha_final = alpha_curr.cpu().numpy()
    save_method_results(data_log, nodes, alpha_final, output_dir, subdir=result_subdir)

    timing_path = os.path.join(method_dir, "timing.csv")
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TIMING_CSV_HEADER)
        w.writerows(timing_log)
    log(f"  Wrote {timing_path}")

    data = np.array(data_log)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[:, 1] * 1e3, data[:, 2], "b-", lw=2)
    ax.set_xlabel("Displacement (μm)")
    ax.set_ylabel("Reaction force (N)")
    ax.set_title("ChebyKAN-DEM force–displacement")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fd_path = os.path.join(method_dir, "force_displacement.png")
    plt.savefig(fd_path, dpi=150)
    plt.close()
    log(f"  Wrote {fd_path}")

    _save_interpretability(solver, method_dir)

    log("  ChebyKAN-DEM finished.")
    return np.array(data_log)


def plot_quick_preview(output_dir: str, subdir: str = METHOD_SUBDIR) -> None:
    path = os.path.join(output_dir, subdir, "results.csv")
    if not os.path.exists(path):
        log(f"  No results at {path}, skip preview plot.")
        return
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # disp in results.csv is in metres; axis in μm (same convention as revisecode figures).
    axes[0].plot(d[:, 1] * 1e3, d[:, 2], "-", linewidth=1.5)
    axes[0].set_xlabel("Displacement (μm)")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Force–displacement (preview)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d[:, 1] * 1e3, d[:, 5], "-", linewidth=1.5)
    axes[1].set_xlabel("Displacement (μm)")
    axes[1].set_ylabel("Total energy (J)")
    axes[1].set_title("Energy (preview)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "quick_preview.png")
    plt.savefig(out, dpi=100)
    plt.close()
    log(f"  Wrote {out}")


def run_pipeline(
    output_dir: str | None = None,
    *,
    case_id: str = "sent_homogeneous",
    generate_mesh_if_missing: bool = True,
) -> None:
    """Generate mesh if allowed, train ChebyKAN-DEM, write case_summary and preview."""
    from chebykan_dem.config import save_config_json
    from chebykan_dem.mesh import generate_shared_mesh, load_mesh_data, save_mesh_data

    out = output_dir or conf.output_dir
    ensure_dir(out)
    save_config_json(out, case_id=case_id)
    save_case_summary(out, case_id)

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    mesh_path = os.path.join(out, "shared_mesh.msh")
    if not os.path.exists(mesh_path):
        if not generate_mesh_if_missing:
            log(f"Missing {mesh_path} and mesh generation disabled.")
            import sys

            sys.exit(1)
        nodes, conn, areas = generate_shared_mesh(out)
    else:
        log(f">>> Using existing mesh: {mesh_path}")
        nodes, conn, areas = load_mesh_data(out)

    save_mesh_data(nodes, conn, areas, out)
    log(f"  Mesh: {nodes.shape[0]} nodes, {conn.shape[0]} triangles")

    results_csv = os.path.join(out, METHOD_SUBDIR, "results.csv")
    if os.path.exists(results_csv):
        with open(results_csv, encoding="utf-8") as f:
            nlines = sum(1 for _ in f)
        if nlines >= conf.n_steps + 1:
            log(">>> ChebyKAN results already complete, skipping training.")
            plot_quick_preview(out)
            return

    solver = ChebyKANDEMSolver(nodes, conn, areas)
    run_chebykan_dem(solver, nodes, out)
    plot_quick_preview(out)
