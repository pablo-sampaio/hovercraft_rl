"""Desenho genérico da pista (independe do modelo físico)."""

from __future__ import annotations

import numpy as np
import pygame

from ..track import Track
from .camera import Camera

COLORS = {
    "ground": (28, 52, 40),
    "grid": (38, 66, 52),
    "asphalt": (58, 62, 68),
    "curb_a": (214, 69, 69),
    "curb_b": (236, 236, 236),
    "center": (230, 230, 230),
    "check_a": (245, 245, 245),
    "check_b": (20, 20, 20),
}


class TrackDrawer:
    """Pré-calcula a geometria da pista (em m) e a desenha via ``Camera``."""

    def __init__(self, track: Track, grid_spacing: float = 10.0,
                 curb_len: float = 2.0, dash_len: float = 2.0) -> None:
        self.track = track
        self.grid_spacing = grid_spacing
        self.left, self.right = track.boundaries()
        n = track.n_segments
        nxt = (np.arange(n) + 1) % n
        # quadriláteros do asfalto, um por segmento: (n, 4, 2)
        self.quads = np.stack([self.left, self.left[nxt], self.right[nxt], self.right], axis=1)
        mid_s = track.cum_s[:-1] + 0.5 * track.seg_len
        self.curb_parity = (np.floor(mid_s / curb_len).astype(int) % 2) == 0
        self.dash_mask = (np.floor(mid_s / dash_len).astype(int) % 2) == 0
        self.nxt = nxt
        self.start_cells = self._checker_cells()
        all_pts = np.vstack([self.left, self.right])
        self.bounds = (all_pts.min(axis=0), all_pts.max(axis=0))

    def _checker_cells(self, rows: int = 2, cols: int = 6, depth: float = 0.7) -> list:
        p0 = self.track.points[0]
        t = self.track.tangents[0]
        nrm = self.track.normals[0]
        hw = self.track.half_width
        cells = []
        for r in range(rows):
            for c in range(cols):
                a = -hw + 2 * hw * c / cols
                b = -hw + 2 * hw * (c + 1) / cols
                s0, s1 = (r - rows / 2) * depth, (r - rows / 2 + 1) * depth
                quad = np.array([p0 + nrm * a + t * s0, p0 + nrm * b + t * s0,
                                 p0 + nrm * b + t * s1, p0 + nrm * a + t * s1])
                cells.append((quad, (r + c) % 2 == 0))
        return cells

    def draw_ground(self, surf: pygame.Surface, cam: Camera) -> None:
        surf.fill(COLORS["ground"])
        g = self.grid_spacing
        if cam.view == "global":
            lo, hi = self.bounds[0] - 4 * g, self.bounds[1] + 4 * g
        else:
            rad = cam.visible_radius()
            lo, hi = cam.center - rad, cam.center + rad
        xs = np.arange(np.floor(lo[0] / g) * g, hi[0] + g, g)
        ys = np.arange(np.floor(lo[1] / g) * g, hi[1] + g, g)
        for x in xs:
            a, b = cam.to_screen([[x, lo[1]], [x, hi[1]]])
            pygame.draw.line(surf, COLORS["grid"], a, b, 1)
        for y in ys:
            a, b = cam.to_screen([[lo[0], y], [hi[0], y]])
            pygame.draw.line(surf, COLORS["grid"], a, b, 1)

    def draw(self, surf: pygame.Surface, cam: Camera) -> None:
        """Desenha a pista; na visão global (estática) reutiliza um cache."""
        if cam.view == "global":
            key = (cam.ppm, tuple(cam.center), surf.get_size())
            if getattr(self, "_cache_key", None) != key:
                self._cache = pygame.Surface(surf.get_size())
                self._draw(self._cache, cam)
                self._cache_key = key
            surf.blit(self._cache, (0, 0))
        else:
            self._draw(surf, cam)

    def _draw(self, surf: pygame.Surface, cam: Camera) -> None:
        self.draw_ground(surf, cam)
        if cam.view == "global":
            vis = np.arange(self.track.n_segments)
        else:  # só os segmentos próximos o bastante para aparecer na tela
            dist = np.linalg.norm(self.track.points - cam.center, axis=1)
            vis = np.nonzero(dist < 0.6 * cam.visible_radius() + 2 * self.track.half_width + 2.0)[0]
        quads = cam.to_screen(self.quads[vis])
        for q in quads:
            pygame.draw.polygon(surf, COLORS["asphalt"], q)
        # zebras nas bordas
        w = max(2, int(round(0.45 * cam.ppm)))
        for edge in (self.left, self.right):
            e = cam.to_screen(edge[vis])
            en = cam.to_screen(edge[self.nxt[vis]])
            for j, i in enumerate(vis):
                col = COLORS["curb_a"] if self.curb_parity[i] else COLORS["curb_b"]
                pygame.draw.line(surf, col, e[j], en[j], w)
        # linha central tracejada
        dv = vis[self.dash_mask[vis]]
        c = cam.to_screen(self.track.points[dv])
        cn = cam.to_screen(self.track.points[self.nxt[dv]])
        cw = max(1, int(round(0.12 * cam.ppm)))
        for a, b in zip(c, cn):
            pygame.draw.line(surf, COLORS["center"], a, b, cw)
        # largada quadriculada
        for quad, light in self.start_cells:
            pygame.draw.polygon(surf, COLORS["check_a"] if light else COLORS["check_b"], cam.to_screen(quad))


class MiniMap:
    """Mapa pequeno (norte para cima), usado na visão egocêntrica."""

    def __init__(self, track: Track, size: int = 170, margin: int = 10) -> None:
        self.size, self.margin = size, margin
        pts = track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.lo = lo
        self.scale = (size - 2 * margin) / float(np.max(hi - lo))
        self.offset = margin + 0.5 * ((size - 2 * margin) - (hi - lo) * self.scale)
        self.surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self.surface.fill((10, 14, 20, 170))
        pygame.draw.lines(self.surface, (200, 200, 200), True, self._map(pts).tolist(), 3)
        pygame.draw.circle(self.surface, (245, 245, 245), self._map(pts[0]), 3)

    def _map(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        q = (p - self.lo) * self.scale + self.offset
        q[..., 1] = self.size - q[..., 1]
        return q

    def draw(self, surf: pygame.Surface, x: float, y: float, psi: float, color) -> None:
        pos = (surf.get_width() - self.size - 10, 10)
        surf.blit(self.surface, pos)
        c = self._map([x, y]) + pos
        tip = c + 9 * np.array([np.cos(psi), -np.sin(psi)])
        pygame.draw.line(surf, color, c, tip, 2)
        pygame.draw.circle(surf, color, c, 4)
