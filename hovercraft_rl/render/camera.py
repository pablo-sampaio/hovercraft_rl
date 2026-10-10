"""Câmera 2D: transformações mundo (m) -> tela (px)."""

from __future__ import annotations

import numpy as np

VIEW_LABELS = {
    "global": "Global (top-down)",
    "ego": "Egocêntrica (heading-up)",
}


class Camera:
    """Duas visões.

    - ``"global"``: mapa fixo, ajustado para mostrar a pista inteira (norte para cima).
    - ``"ego"``: segue o veículo e gira com ele (frente sempre para cima).
    """

    def __init__(self, width: int, height: int, view: str = "global",
                 ego_ppm: float = 18.0, ego_anchor: float = 0.62) -> None:
        if view not in VIEW_LABELS:
            raise ValueError(f"view deve ser uma de {list(VIEW_LABELS)}")
        self.width, self.height = width, height
        self.view = view
        self.ego_ppm = ego_ppm
        self.ego_anchor = ego_anchor        # fração da altura onde fica o veículo na visão ego
        self.global_ppm = 10.0
        self.global_center = np.zeros(2)
        self.center = np.zeros(2)
        self.angle = 0.0                    # rotação aplicada ao mundo (rad)
        self._zoom = {"global": 1.0, "ego": 1.0}

    @property
    def label(self) -> str:
        return VIEW_LABELS[self.view]

    @property
    def ppm(self) -> float:
        """Pixels por metro na visão corrente."""
        base = self.global_ppm if self.view == "global" else self.ego_ppm
        return base * self._zoom[self.view]

    def toggle(self) -> None:
        self.view = "ego" if self.view == "global" else "global"

    def zoom(self, factor: float) -> None:
        self._zoom[self.view] = float(np.clip(self._zoom[self.view] * factor, 0.2, 8.0))

    def fit(self, points: np.ndarray, margin_px: int = 30) -> None:
        """Ajusta a visão global para conter todos os ``points``."""
        lo, hi = points.min(axis=0), points.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        self.global_center = 0.5 * (lo + hi)
        self.global_ppm = float(min((self.width - 2 * margin_px) / span[0],
                                    (self.height - 2 * margin_px) / span[1]))

    def follow(self, x: float, y: float, psi: float) -> None:
        if self.view == "global":
            self.center = self.global_center
            self.angle = 0.0
        else:
            self.center = np.array([x, y], dtype=np.float64)
            self.angle = np.pi / 2 - psi

    @property
    def _origin(self) -> tuple[float, float]:
        if self.view == "global":
            return self.width / 2, self.height / 2
        return self.width / 2, self.height * self.ego_anchor

    def to_screen(self, pts) -> np.ndarray:
        """Converte pontos (N, 2) ou (2,) do mundo para pixels."""
        p = np.asarray(pts, dtype=np.float64)
        d = p - self.center
        c, s = np.cos(self.angle), np.sin(self.angle)
        rx = c * d[..., 0] - s * d[..., 1]
        ry = s * d[..., 0] + c * d[..., 1]
        ox, oy = self._origin
        return np.stack([ox + rx * self.ppm, oy - ry * self.ppm], axis=-1)

    def screen_angle(self, world_angle: float) -> float:
        """Ângulo (rad, anti-horário) de uma direção do mundo, já na tela."""
        return world_angle + self.angle

    def visible_radius(self) -> float:
        """Raio (m) que cobre a tela inteira a partir do centro."""
        return float(np.hypot(self.width, self.height) / self.ppm)
