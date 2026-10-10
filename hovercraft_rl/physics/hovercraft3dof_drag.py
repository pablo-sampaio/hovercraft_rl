"""Variante do ``hovercraft3dof`` com amortecimento em surge e em guinada.

Com ``beta_u = beta_r = 0`` é idêntica ao modelo do artigo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hovercraft3dof import Hovercraft3DOF, Hovercraft3DOFParams


@dataclass
class Hovercraft3DOFDragParams(Hovercraft3DOFParams):
    beta_u: float = 0.0   # arrasto longitudinal (surge)      (1/s)
    beta_r: float = 0.0   # amortecimento de guinada (yaw)    (1/s)


class Hovercraft3DOFDrag(Hovercraft3DOF):
    """Eq. 4 do artigo + ``-beta_u*u`` em du e ``-beta_r*r`` em dr."""

    def __init__(self, params: Hovercraft3DOFDragParams | dict | None = None) -> None:
        if params is None:
            params = Hovercraft3DOFDragParams()
        elif isinstance(params, dict):
            params = Hovercraft3DOFDragParams(**params)
        super().__init__(params)

    def derivatives(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        d = super().derivatives(s, a)
        d[3] -= self.params.beta_u * s[3]
        d[5] -= self.params.beta_r * s[5]
        return d
