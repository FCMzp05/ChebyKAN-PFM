"""Phase-field fracture DEM: energy, history field, and spectral split."""

from __future__ import annotations

import torch
import torch.nn as nn

from chebykan_dem.config import conf


class NonsmoothSigmoid(nn.Module):
    def __init__(self, support: float = 2.0, coeff: float = 1e-3):
        super().__init__()
        self.support = support
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x > self.support
        b = x < -self.support
        c = ~(a | b)
        return (
            a * (self.coeff * (x - self.support) + 1.0)
            + b * (self.coeff * (x + self.support))
            + c * (x / (2 * self.support) + 0.5)
        )


class IrreversiblePhaseField(nn.Module):
    def __init__(self, beta: float = 50.0):
        super().__init__()
        self.beta = beta

    def forward(self, alpha_net: torch.Tensor, alpha_history: torch.Tensor) -> torch.Tensor:
        diff = alpha_net - alpha_history
        alpha_increment = torch.nn.functional.softplus(diff, beta=self.beta)
        alpha_final = alpha_history + alpha_increment
        return torch.clamp(alpha_final, 0.0, 1.0)


class MaterialProperties:
    def __init__(self) -> None:
        self.E = torch.tensor(conf.mat_E, device=conf.device)
        self.nu = torch.tensor(conf.mat_nu, device=conf.device)
        self.Gc = torch.tensor(conf.Gc, device=conf.device)
        self.l0 = torch.tensor(conf.l0, device=conf.device)
        self.lmbda = torch.tensor(conf.lmbda, device=conf.device)
        self.mu = torch.tensor(conf.mu, device=conf.device)
        self.k = torch.tensor(conf.k_residual, device=conf.device)


class PFFModel:
    def degrade(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (1 - alpha) ** 2, 2 * (alpha - 1)

    def damage_fun(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return alpha**2, 2.0

    def irr_penalty_weight(self) -> float:
        return 1.0 / conf.tol_ir**2 - 1.0


class DEMSolverBase:
    """Deep energy method on a triangle mesh with one Gauss point per element."""

    def __init__(self, network: nn.Module, nodes: object, conn: object, areas: object) -> None:
        self.network = network
        self.alpha_constraint = NonsmoothSigmoid()
        self.irreversible = IrreversiblePhaseField(beta=50.0)
        self.mat = MaterialProperties()
        self.pff = PFFModel()

        self.inp = torch.tensor(nodes, dtype=torch.float32).to(conf.device)
        self.conn = torch.tensor(conn, dtype=torch.long).to(conf.device)
        self.area = torch.tensor(areas, dtype=torch.float32).to(conf.device)

        tol = 1e-5
        self.bottom_mask = torch.abs(self.inp[:, 1] - (-conf.Ly / 2)) < tol
        self.top_mask = torch.abs(self.inp[:, 1] - (conf.Ly / 2)) < tol

        bottom_nodes = self.inp[self.bottom_mask]
        if len(bottom_nodes) > 0:
            center_idx = torch.argmin(torch.abs(bottom_nodes[:, 0]))
            self.center_node_idx = torch.where(self.bottom_mask)[0][center_idx].item()
        else:
            self.center_node_idx = 0

        self.alpha_history = torch.zeros(self.inp.shape[0], device=conf.device)

        self.gauss_pts = torch.tensor([[1 / 3, 1 / 3]], dtype=torch.float32).to(conf.device)
        self.gauss_wts = torch.tensor([0.5], dtype=torch.float32).to(conf.device)

    def compute_fields(self, lmbda_val: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.network(self.inp)
        if not torch.is_tensor(lmbda_val):
            lmbda_val = torch.tensor(lmbda_val, device=conf.device, dtype=torch.float32)

        _x, y = self.inp[:, 0], self.inp[:, 1]
        y0, yL = -conf.Ly / 2, conf.Ly / 2
        bubble_y = (y - y0) * (yL - y)
        linear_y = (y - y0) / (yL - y0)

        u = bubble_y * out[:, 0] * lmbda_val
        v = (bubble_y * out[:, 1] + linear_y) * lmbda_val

        alpha_raw = out[:, 2]
        alpha_net = self.alpha_constraint(alpha_raw)
        alpha = self.irreversible(alpha_net, self.alpha_history)
        return u, v, alpha

    def update_alpha_history(self, lmbda_val: float | torch.Tensor) -> None:
        with torch.no_grad():
            _, _, alpha = self.compute_fields(lmbda_val)
            self.alpha_history = torch.maximum(self.alpha_history, alpha)

    def spectral_split(
        self, eps_11: torch.Tensor, eps_22: torch.Tensor, eps_12: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lmbda, mu = self.mat.lmbda, self.mat.mu
        trace = eps_11 + eps_22
        dev = (eps_11 - eps_22) / 2.0
        radius = torch.sqrt(dev**2 + eps_12**2)
        eps_1 = trace / 2.0 + radius
        eps_2 = trace / 2.0 - radius
        eps_1_plus, eps_2_plus = torch.relu(eps_1), torch.relu(eps_2)
        eps_1_minus, eps_2_minus = eps_1 - eps_1_plus, eps_2 - eps_2_plus
        trace_plus = eps_1_plus + eps_2_plus
        trace_minus = eps_1_minus + eps_2_minus
        psi_plus = 0.5 * lmbda * trace_plus**2 + mu * (eps_1_plus**2 + eps_2_plus**2)
        psi_minus = 0.5 * lmbda * trace_minus**2 + mu * (eps_1_minus**2 + eps_2_minus**2)
        return psi_plus, psi_minus

    def compute_fields_at_gauss_points(self, lmbda_val: float | torch.Tensor) -> dict[str, torch.Tensor]:
        node_coords = self.inp[self.conn]
        u, v, alpha = self.compute_fields(lmbda_val)
        u_nodes, v_nodes, alpha_nodes = u[self.conn], v[self.conn], alpha[self.conn]

        n_elem = self.conn.shape[0]
        xi, eta = self.gauss_pts[0, 0], self.gauss_pts[0, 1]
        n = torch.tensor([1 - xi - eta, xi, eta], device=conf.device)
        dN_dxi = torch.tensor([-1.0, 1.0, 0.0], device=conf.device)
        dN_deta = torch.tensor([-1.0, 0.0, 1.0], device=conf.device)

        dx_dxi = torch.einsum("i,ei->e", dN_dxi, node_coords[:, :, 0])
        dy_dxi = torch.einsum("i,ei->e", dN_dxi, node_coords[:, :, 1])
        dx_deta = torch.einsum("i,ei->e", dN_deta, node_coords[:, :, 0])
        dy_deta = torch.einsum("i,ei->e", dN_deta, node_coords[:, :, 1])
        det_j = dx_dxi * dy_deta - dy_dxi * dx_deta
        inv_det_j = 1.0 / det_j

        dN_dx = torch.zeros(n_elem, 3, device=conf.device)
        dN_dy = torch.zeros(n_elem, 3, device=conf.device)
        for i in range(3):
            dN_dx[:, i] = inv_det_j * (dy_deta * dN_dxi[i] - dy_dxi * dN_deta[i])
            dN_dy[:, i] = inv_det_j * (-dx_deta * dN_dxi[i] + dx_dxi * dN_deta[i])

        alpha_gp = torch.einsum("i,ei->e", n, alpha_nodes)
        du_dx = torch.einsum("ei,ei->e", dN_dx, u_nodes)
        du_dy = torch.einsum("ei,ei->e", dN_dy, u_nodes)
        dv_dx = torch.einsum("ei,ei->e", dN_dx, v_nodes)
        dv_dy = torch.einsum("ei,ei->e", dN_dy, v_nodes)
        dalpha_dx = torch.einsum("ei,ei->e", dN_dx, alpha_nodes)
        dalpha_dy = torch.einsum("ei,ei->e", dN_dy, alpha_nodes)

        return {
            "alpha": alpha_gp,
            "du_dx": du_dx,
            "du_dy": du_dy,
            "dv_dx": dv_dx,
            "dv_dy": dv_dy,
            "dalpha_dx": dalpha_dx,
            "dalpha_dy": dalpha_dy,
        }

    def get_total_energy(
        self, H_history: torch.Tensor, lmbda_val: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = self.compute_fields_at_gauss_points(lmbda_val)

        eps_11 = fields["du_dx"]
        eps_22 = fields["dv_dy"]
        eps_12 = 0.5 * (fields["du_dy"] + fields["dv_dx"])
        alpha_val = fields["alpha"]
        grad_alpha_x = fields["dalpha_dx"]
        grad_alpha_y = fields["dalpha_dy"]

        g_val, _ = self.pff.degrade(alpha_val)
        psi_plus, psi_minus = self.spectral_split(eps_11, eps_22, eps_12)

        E_el = ((g_val + self.mat.k) * psi_plus + psi_minus) * self.area

        w_alpha, cw = self.pff.damage_fun(alpha_val)
        Gc, l0 = self.mat.Gc, self.mat.l0
        E_frac = (Gc / (cw * l0)) * (w_alpha + l0**2 * (grad_alpha_x**2 + grad_alpha_y**2)) * self.area

        H_gp = H_history[self.conn].mean(dim=1)
        H_current = torch.maximum(H_gp, psi_plus)
        alpha_from_H = 2 * l0 * H_current / (Gc + 2 * l0 * H_current + 1e-12)
        alpha_from_H = torch.clamp(alpha_from_H, 0.0, 1.0)
        E_pen = (
            0.5
            * Gc
            * self.pff.irr_penalty_weight()
            * torch.relu(-(alpha_val - alpha_from_H * 0.9)) ** 2
            * self.area
        )

        return E_el.sum(), E_frac.sum(), E_pen.sum()

    def loss_total(self, H_history: torch.Tensor, lmbda_val: float | torch.Tensor) -> torch.Tensor:
        E_el, E_frac, E_pen = self.get_total_energy(H_history, lmbda_val)
        rigid = conf.rigid_body_penalty * self.compute_fields(lmbda_val)[0][self.center_node_idx] ** 2

        fields = self.compute_fields_at_gauss_points(lmbda_val)
        alpha_smooth = conf.alpha_grad_penalty * torch.mean(fields["dalpha_dx"] ** 2 + fields["dalpha_dy"] ** 2)

        return torch.log10(E_el + E_frac + E_pen + rigid + alpha_smooth + 1e-12)

    def update_history_field(self, H_history: torch.Tensor, lmbda_val: float | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fields = self.compute_fields_at_gauss_points(lmbda_val)
            eps_11 = fields["du_dx"]
            eps_22 = fields["dv_dy"]
            eps_12 = 0.5 * (fields["du_dy"] + fields["dv_dx"])
            psi_plus, _ = self.spectral_split(eps_11, eps_22, eps_12)

            n_nodes = self.inp.shape[0]
            psi_plus_nodes = torch.zeros(n_nodes, device=conf.device)
            count = torch.zeros(n_nodes, device=conf.device)

            for i in range(3):
                psi_plus_nodes.scatter_add_(0, self.conn[:, i], psi_plus)
                count.scatter_add_(0, self.conn[:, i], torch.ones_like(psi_plus))

            psi_plus_nodes = psi_plus_nodes / (count + 1e-12)
            return torch.maximum(H_history, psi_plus_nodes)
