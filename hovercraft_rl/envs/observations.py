"""Construtores de observação (plugáveis no ambiente)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
import warnings

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


class _FrameObservation(ObservationBuilder):
    """Features comuns nos eixos do mundo ou nos eixos do hovercraft."""

    relative = False

    def __init__(self, n_lookahead: int | None = None, lookahead_spacing: float | None = None,
                 lookahead_distances: Sequence[float] | None = None, include_directions: bool = False,
                 include_hovercraft_heading: bool = False, include_lateral_error: bool = True) -> None:
        if lookahead_distances is not None:
            if n_lookahead is not None or lookahead_spacing is not None:
                raise ValueError("use lookahead_distances ou n_lookahead/lookahead_spacing, não ambos")
            distances = np.asarray(lookahead_distances, dtype=np.float64)
            if distances.ndim != 1:
                raise ValueError("lookahead_distances deve ser uma lista unidimensional")
        else:
            count = 5 if n_lookahead is None else n_lookahead
            spacing = 5.0 if lookahead_spacing is None else lookahead_spacing
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("n_lookahead deve ser um inteiro não negativo")
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("lookahead_spacing deve ser finito e positivo")
            distances = spacing * np.arange(1, count + 1, dtype=np.float64)
        if (not np.all(np.isfinite(distances)) or np.any(distances <= 0)
                or np.any(np.diff(distances) <= 0)):
            raise ValueError("distâncias de lookahead devem ser finitas, positivas e estritamente crescentes")
        self.distances = distances
        self.include_directions = include_directions
        self.include_hovercraft_heading = include_hovercraft_heading
        self.include_lateral_error = include_lateral_error
        self._warned_coincident = False

    def _validate_track(self, env: "HovercraftTrackEnv") -> None:
        if np.any(self.distances >= env.track.length):
            raise ValueError("cada distância de lookahead deve ser menor que o comprimento da pista")

    def names(self, env: "HovercraftTrackEnv") -> list[str]:
        self._validate_track(env)
        axes = ("fwd", "left") if self.relative else ("x", "y")
        names = [f"position_{axis}" for axis in axes]
        if self.include_lateral_error:
            names.append("e_lat_norm")
        if not self.relative or self.include_hovercraft_heading:
            names += ["hovercraft_cos", "hovercraft_sin"]
        names += ["track_cos", "track_sin", "u", "v", "r"]
        for i in range(len(self.distances)):
            names += [f"la{i + 1}_{axis}" for axis in axes]
        if self.include_directions:
            for i in range(max(len(self.distances) - 1, 0)):
                names += [f"la{i + 1}_to_la{i + 2}_cos", f"la{i + 1}_to_la{i + 2}_sin"]
        return names

    def size(self, env: "HovercraftTrackEnv") -> int:
        return len(self.names(env))

    def __call__(self, env: "HovercraftTrackEnv") -> np.ndarray:
        self._validate_track(env)
        state = env.physics.state_dict()
        projection = env.projection
        psi = state["psi"]
        frame_angle = psi if self.relative else 0.0
        c, s = np.cos(frame_angle), np.sin(frame_angle)
        rotation = np.array([[c, s], [-s, c]])
        position = rotation @ np.array([state["x"] - projection.x_ref, state["y"] - projection.y_ref])
        obs = [*position]
        if self.include_lateral_error:
            obs.append(projection.lateral_error / env.track.half_width)
        if not self.relative or self.include_hovercraft_heading:
            heading = psi - frame_angle
            obs += [np.cos(heading), np.sin(heading)]
        track_heading = projection.heading - frame_angle
        obs += [np.cos(track_heading), np.sin(track_heading), state["u"], state["v"], state["r"]]

        if len(self.distances):
            points = env.track.point_at(projection.s + self.distances).reshape(-1, 2)
            self.last_points = points
            ref = np.array([projection.x_ref, projection.y_ref])
            offsets = (points - ref) @ rotation.T
            obs.extend(offsets.ravel())
            if self.include_directions:
                vectors = np.diff(offsets, axis=0)
                lengths = np.linalg.norm(vectors, axis=1)
                coincident = lengths <= 1e-10
                if np.any(coincident) and not self._warned_coincident:
                    warnings.warn("pontos consecutivos de lookahead coincidem; direção definida como (0, 0)",
                                  RuntimeWarning, stacklevel=2)
                    self._warned_coincident = True
                directions = np.zeros_like(vectors)
                np.divide(vectors, lengths[:, None], out=directions, where=~coincident[:, None])
                obs.extend(directions.ravel())
        else:
            self.last_points = None
        return np.asarray(obs, dtype=np.float32)


class WorldFrameObservation(_FrameObservation):
    """Observação com origem na projeção e eixos do mundo."""


class HovercraftFrameObservation(_FrameObservation):
    """Observação com origem na projeção e eixos do hovercraft."""

    relative = True


OBSERVATIONS = {
    "reference_error": ReferenceErrorObservation,
    "body_frame": BodyFrameObservation,
    "world_frame": WorldFrameObservation,
    "hovercraft_frame": HovercraftFrameObservation,
}
