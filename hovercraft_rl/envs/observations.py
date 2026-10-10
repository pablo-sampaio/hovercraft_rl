"""Construtores de observação (plugáveis no ambiente)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from .track_env import HovercraftTrackEnv


class ObservationBuilder(ABC):
    """Transforma o estado do ambiente em um vetor de observação.

    ``last_points`` (opcional) guarda pontos em coordenadas do mundo para o overlay de debug.
    """

    last_points: np.ndarray | None = None

    @abstractmethod
    def size(self, env: "HovercraftTrackEnv") -> int: ...

    @abstractmethod
    def __call__(self, env: "HovercraftTrackEnv") -> np.ndarray: ...

    def space(self, env: "HovercraftTrackEnv") -> spaces.Box:
        n = self.size(env)
        return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def names(self, env: "HovercraftTrackEnv") -> list[str]:
        return [f"o{i}" for i in range(self.size(env))]

    def _lookahead(self, env, n: int, spacing: float) -> np.ndarray:
        s = env.projection.s + spacing * np.arange(1, n + 1)
        pts = env.track.point_at(s).reshape(n, 2)
        self.last_points = pts
        return pts


class ReferenceErrorObservation(ObservationBuilder):
    """Observação proposta na especificação.

    Vetor: ``[x - x_ref, y - y_ref, cos psi, sin psi, u, v, r]``, opcionalmente seguido de:

    - ``neighbor_headings``: ``[cos, sin]`` da direção da pista **antes** e **depois** do ponto de
      referência. Usa os segmentos vizinhos ao de referência se ``neighbor_distance`` for ``None``;
      caso contrário, usa a direção em ``s ∓ neighbor_distance``.
    - ``n_lookahead`` pontos à frente na linha base, espaçados de ``lookahead_spacing`` metros
      de arco, como ``(x_i - x, y_i - y)`` (referencial do mundo).
    """

    def __init__(self, neighbor_headings: bool = True, neighbor_distance: float | None = None,
                 n_lookahead: int = 5, lookahead_spacing: float = 5.0) -> None:
        self.neighbor_headings = neighbor_headings
        self.neighbor_distance = neighbor_distance
        self.n_lookahead = n_lookahead
        self.lookahead_spacing = lookahead_spacing

    def size(self, env) -> int:
        return 7 + (4 if self.neighbor_headings else 0) + 2 * self.n_lookahead

    def names(self, env) -> list[str]:
        n = ["ex", "ey", "cos_psi", "sin_psi", "u", "v", "r"]
        if self.neighbor_headings:
            n += ["cos_prev", "sin_prev", "cos_next", "sin_next"]
        for i in range(1, self.n_lookahead + 1):
            n += [f"la{i}_dx", f"la{i}_dy"]
        return n

    def __call__(self, env) -> np.ndarray:
        st = env.physics.state_dict()
        pr = env.projection
        x, y, psi = st["x"], st["y"], st["psi"]
        obs = [x - pr.x_ref, y - pr.y_ref, np.cos(psi), np.sin(psi), st["u"], st["v"], st["r"]]
        if self.neighbor_headings:
            tr = env.track
            if self.neighbor_distance is None:
                h_prev = tr.headings[(pr.segment - 1) % tr.n_segments]
                h_next = tr.headings[(pr.segment + 1) % tr.n_segments]
            else:
                h_prev = float(tr.heading_at(pr.s - self.neighbor_distance))
                h_next = float(tr.heading_at(pr.s + self.neighbor_distance))
            obs += [np.cos(h_prev), np.sin(h_prev), np.cos(h_next), np.sin(h_next)]
        if self.n_lookahead:
            pts = self._lookahead(env, self.n_lookahead, self.lookahead_spacing)
            obs += list((pts - [x, y]).ravel())
        else:
            self.last_points = None
        return np.asarray(obs, dtype=np.float32)


class BodyFrameObservation(ObservationBuilder):
    """Alternativa invariante a rotação/translação da pista (tudo relativo ao veículo).

    Vetor: ``[e_lat / half_width, cos(psi - psi_ref), sin(psi - psi_ref), u, v, r]`` seguido de
    ``n_lookahead`` pontos à frente expressos no referencial do corpo (frente = +x, esquerda = +y).
    """

    def __init__(self, n_lookahead: int = 5, lookahead_spacing: float = 5.0) -> None:
        self.n_lookahead = n_lookahead
        self.lookahead_spacing = lookahead_spacing

    def size(self, env) -> int:
        return 6 + 2 * self.n_lookahead

    def names(self, env) -> list[str]:
        n = ["e_lat_norm", "cos_dpsi", "sin_dpsi", "u", "v", "r"]
        for i in range(1, self.n_lookahead + 1):
            n += [f"la{i}_fwd", f"la{i}_left"]
        return n

    def __call__(self, env) -> np.ndarray:
        st = env.physics.state_dict()
        pr = env.projection
        x, y, psi = st["x"], st["y"], st["psi"]
        dpsi = psi - pr.heading
        obs = [pr.lateral_error / env.track.half_width, np.cos(dpsi), np.sin(dpsi), st["u"], st["v"], st["r"]]
        if self.n_lookahead:
            d = self._lookahead(env, self.n_lookahead, self.lookahead_spacing) - [x, y]
            c, s = np.cos(psi), np.sin(psi)
            body = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
            obs += list(body.ravel())
        else:
            self.last_points = None
        return np.asarray(obs, dtype=np.float32)


OBSERVATIONS = {
    "reference_error": ReferenceErrorObservation,
    "body_frame": BodyFrameObservation,
}
