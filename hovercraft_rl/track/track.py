"""Geometria da pista: linha base (polilinha fechada) + meia-largura fixa."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Projection:
    """Resultado da projeção de um ponto (x, y) na linha base."""

    x_ref: float
    y_ref: float
    s: float               # arco ao longo da linha base, em [0, length)
    lateral_error: float   # distância com sinal (+ = à esquerda do sentido da pista)
    heading: float         # direção do segmento de referência (rad)
    segment: int           # índice do segmento de referência


def available_tracks() -> list[str]:
    """Nomes das pistas salvas em ``track/data``."""
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))


def chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    """Suavização de Chaikin para polígono fechado."""
    p = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        q = np.roll(p, -1, axis=0)
        p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], axis=1).reshape(-1, 2)
    return p


class Track:
    """Pista fechada.

    Parâmetros
    ----------
    points : (N, 2) vértices da linha base (o último liga ao primeiro; não repetir).
    half_width : distância lateral máxima permitida em torno da linha base (m).
    smooth_iterations : iterações de Chaikin aplicadas aos vértices.
    search_window : meia-janela (m) de busca do ponto de referência em torno do anterior.
    """

    def __init__(
        self,
        points: Sequence[Sequence[float]],
        half_width: float,
        name: str = "custom",
        smooth_iterations: int = 0,
        search_window: float = 15.0,
    ) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("points deve ter forma (N>=3, 2)")
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        pts = chaikin(pts, smooth_iterations)
        self.name = name
        self.half_width = float(half_width)
        self.search_window = float(search_window)

        self.points = pts                                      # (N, 2) início de cada segmento
        self.seg_vec = np.roll(pts, -1, axis=0) - pts           # (N, 2)
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)     # (N,)
        if np.any(self.seg_len <= 1e-12):
            raise ValueError("Pista contém vértices repetidos consecutivos")
        self.tangents = self.seg_vec / self.seg_len[:, None]
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)  # à esquerda
        self.headings = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len)])  # (N+1,)
        self.length = float(self.cum_s[-1])
        self._seg_mid_s = self.cum_s[:-1] + 0.5 * self.seg_len

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, name_or_path: str | Path, **overrides) -> "Track":
        """Carrega pelo nome (em ``track/data``) ou por caminho de arquivo JSON."""
        path = Path(name_or_path)
        if not path.suffix:
            path = DATA_DIR / f"{name_or_path}.json"
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        kwargs = dict(
            points=d["points"],
            half_width=d["half_width"],
            name=d.get("name", path.stem),
            smooth_iterations=d.get("smooth_iterations", 0),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "half_width": self.half_width,
            "smooth_iterations": 0,
            "points": np.round(self.points, 4).tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ------------------------------------------------------------ geometria
    @property
    def n_segments(self) -> int:
        return len(self.points)

    def wrap_s(self, s: float | np.ndarray) -> float | np.ndarray:
        return np.mod(s, self.length)

    def progress_delta(self, s_prev: float, s_new: float) -> float:
        """Avanço com sinal entre duas posições de arco, tratando a passagem pela largada."""
        d = (s_new - s_prev) % self.length
        return d - self.length if d > 0.5 * self.length else d

    def _segment_at(self, s: float | np.ndarray) -> np.ndarray:
        s = np.mod(s, self.length)
        return np.clip(np.searchsorted(self.cum_s, s, side="right") - 1, 0, self.n_segments - 1)

    def point_at(self, s: float | np.ndarray) -> np.ndarray:
        """Ponto(s) da linha base no arco ``s`` (aceita escalar ou vetor)."""
        s_arr = np.mod(np.asarray(s, dtype=np.float64), self.length)
        i = self._segment_at(s_arr)
        t = (s_arr - self.cum_s[i])[..., None]
        return self.points[i] + t * self.tangents[i]

    def heading_at(self, s: float | np.ndarray) -> np.ndarray:
        """Direção da linha base no arco ``s`` (do segmento que o contém)."""
        return self.headings[self._segment_at(s)]

    def project(self, x: float, y: float, hint_s: float | None = None) -> Projection:
        """Ponto mais próximo da linha base, calculado sobre os segmentos de reta.

        Se ``hint_s`` é dado, só considera segmentos a até ``search_window`` metros
        (de arco) dele, o que evita saltos para outros trechos da pista. Com
        ``hint_s=None``, a busca é global.
        """
        p = np.array([x, y], dtype=np.float64)
        if hint_s is None:
            idx = np.arange(self.n_segments)
        else:
            ds = np.abs((self._seg_mid_s - hint_s + 0.5 * self.length) % self.length - 0.5 * self.length)
            idx = np.nonzero(ds <= self.search_window + 0.5 * self.seg_len)[0]
            if len(idx) == 0:
                idx = np.array([self._segment_at(hint_s)])
        a = self.points[idx]
        d = self.seg_vec[idx]
        t = np.clip(np.einsum("ij,ij->i", p - a, d) / self.seg_len[idx] ** 2, 0.0, 1.0)
        closest = a + t[:, None] * d
        dist2 = np.sum((p - closest) ** 2, axis=1)
        k = int(np.argmin(dist2))
        i = int(idx[k])
        c = closest[k]
        cross = self.tangents[i, 0] * (p[1] - c[1]) - self.tangents[i, 1] * (p[0] - c[0])
        lat = float(np.sqrt(dist2[k])) * (1.0 if cross >= 0 else -1.0)
        s = float((self.cum_s[i] + t[k] * self.seg_len[i]) % self.length)
        return Projection(float(c[0]), float(c[1]), s, lat, float(self.headings[i]), i)

    def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Bordas esquerda e direita (vértices deslocados em ±half_width), para desenho."""
        n_prev = np.roll(self.normals, 1, axis=0)
        n_v = self.normals + n_prev
        n_v /= np.linalg.norm(n_v, axis=1, keepdims=True)
        scale = self.half_width / np.clip(np.einsum("ij,ij->i", n_v, self.normals), 0.3, None)
        left = self.points + n_v * scale[:, None]
        right = self.points - n_v * scale[:, None]
        return left, right

    def min_curvature_radius(self) -> float:
        """Menor raio de curvatura aproximado (deve ser > half_width para pista válida)."""
        dpsi = np.abs(np.angle(np.exp(1j * (self.headings - np.roll(self.headings, 1)))))
        ds = 0.5 * (self.seg_len + np.roll(self.seg_len, 1))
        with np.errstate(divide="ignore"):
            return float(np.min(ds / dpsi))

    def __repr__(self) -> str:
        return (f"Track(name={self.name!r}, length={self.length:.1f} m, "
                f"half_width={self.half_width} m, segments={self.n_segments})")
