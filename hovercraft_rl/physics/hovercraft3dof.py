"""Hovercraft subatuado de 3 graus de liberdade (Kong et al., 2022, Eq. 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import PhysicsModel


@dataclass
class Hovercraft3DOFParams:
    """Parâmetros do modelo. Veja ``physics/README.md``."""

    beta: float = 0.2                      # amortecimento de sway  (1/s)
    dt: float = 0.02                       # tempo de amostragem h  (s)
    tau_u_range: tuple[float, float] = (0.0, 2.0)    # aceleração de surge   (m/s²)
    tau_r_range: tuple[float, float] = (-1.0, 1.0)   # aceleração de guinada (rad/s²)
    integrator: str = "euler"              # "euler" (spec) ou "rk4"
    substeps: int = 1                      # sub-passos de integração por step

    def __post_init__(self) -> None:
        if self.integrator not in ("euler", "rk4"):
            raise ValueError(f"integrator inválido: {self.integrator!r}")
        if self.dt <= 0 or self.substeps < 1:
            raise ValueError("dt deve ser > 0 e substeps >= 1")


class Hovercraft3DOF(PhysicsModel):
    """Estado ``[x, y, psi, u, v, r]``; ação ``[tau_u, tau_r]``.

    ``psi`` não é normalizado (é contínuo, acumula voltas); use cos/sin.
    """

    state_names = ("x", "y", "psi", "u", "v", "r")
    action_names = ("tau_u", "tau_r")

    def __init__(self, params: Hovercraft3DOFParams | dict | None = None) -> None:
        if params is None:
            params = Hovercraft3DOFParams()
        elif isinstance(params, dict):
            params = Hovercraft3DOFParams(**params)
        self.params = params
        self._low = np.array([params.tau_u_range[0], params.tau_r_range[0]], dtype=np.float64)
        self._high = np.array([params.tau_u_range[1], params.tau_r_range[1]], dtype=np.float64)
        super().__init__()

    @property
    def dt(self) -> float:
        return self.params.dt

    @property
    def action_low(self) -> np.ndarray:
        return self._low.copy()

    @property
    def action_high(self) -> np.ndarray:
        return self._high.copy()

    def derivatives(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Lado direito das EDOs (Eq. 4 do artigo)."""
        _, _, psi, u, v, r = s
        tau_u, tau_r = a
        c, sn = np.cos(psi), np.sin(psi)
        return np.array([
            u * c - v * sn,                    # dx
            u * sn + v * c,                    # dy
            r,                                 # dpsi
            v * r + tau_u,                     # du
            -u * r - self.params.beta * v,     # dv
            tau_r,                             # dr
        ])

    def step(self, action: Sequence[float]) -> np.ndarray:
        a = self.clip_action(action)
        h = self.params.dt / self.params.substeps
        s = self._state
        for _ in range(self.params.substeps):
            if self.params.integrator == "euler":
                s = s + h * self.derivatives(s, a)
            else:
                k1 = self.derivatives(s, a)
                k2 = self.derivatives(s + 0.5 * h * k1, a)
                k3 = self.derivatives(s + 0.5 * h * k2, a)
                k4 = self.derivatives(s + h * k3, a)
                s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = s
        self.t += self.params.dt
        return self.state
