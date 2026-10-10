"""Renderizador específico do modelo ``3dof+drag``."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from .base import ACCENT, BaseTrackRenderer

SKIRT = (34, 36, 42)
DECK = (242, 165, 65)
DECK_EDGE = (150, 95, 30)
CABIN = (95, 168, 211)
FAN = (80, 84, 92)
JET = (190, 235, 255)
NOSE = (255, 245, 225)
TAU_U_COLOR = (120, 220, 140)
TAU_R_COLOR = (255, 170, 80)


def _superellipse(a: float, b: float, n: float = 4.0, k: int = 40) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / n), b * np.sign(s) * np.abs(s) ** (2 / n)], axis=1)


class HovercraftRenderer(BaseTrackRenderer):
    """Desenha o hovercraft (casco, saia, cabine, ventiladores, lemes, jato) + vetor velocidade.

    Parâmetros extras
    -----------------
    action_low, action_high : limites de ``[tau_u, tau_r]`` para escalar as barras/jato.
    length, beam : dimensões do casco desenhado (m).
    min_pixels : comprimento mínimo do desenho na tela (aumenta o desenho na visão global).
    """

    def __init__(self, track, action_low=(0.0, -1.0), action_high=(2.0, 1.0),
                 length: float = 2.6, beam: float = 1.5, min_pixels: float = 30.0, **kwargs) -> None:
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.L, self.B = length, beam
        self.min_pixels = min_pixels
        L, B = length, beam
        self._skirt = _superellipse(L / 2 + 0.12, B / 2 + 0.12)
        self._deck = _superellipse(L / 2 - 0.05, B / 2 - 0.08)
        self._cabin = np.array([[0.75 * L / 2, 0.30 * B], [0.75 * L / 2, -0.30 * B],
                                [0.10, -0.33 * B], [0.10, 0.33 * B]])
        self._nose = np.array([[L / 2 + 0.1, 0.0], [L / 2 - 0.35, 0.22 * B], [L / 2 - 0.35, -0.22 * B]])
        self._fans = [np.array([-0.30 * L, y]) for y in (0.24 * B, -0.24 * B)]
        self._fan_r = 0.20 * B
        super().__init__(track, **kwargs)

    # ------------------------------------------------------------------
    def _scale(self) -> float:
        return max(1.0, self.min_pixels / (self.L * self.camera.ppm))

    def _body_to_screen(self, pts: np.ndarray, x: float, y: float, psi: float, k: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        p = np.asarray(pts) * k
        world = np.stack([x + c * p[..., 0] - s * p[..., 1], y + s * p[..., 0] + c * p[..., 1]], axis=-1)
        return self.camera.to_screen(world)

    def _frac(self, action, i: int) -> float:
        if action is None:
            return 0.0
        lo, hi = self.action_low[i], self.action_high[i]
        a = float(np.clip(action[i], lo, hi))
        return a / hi if a >= 0 else -a / lo if lo < 0 else 0.0

    def draw_vehicle(self, surf: pygame.Surface, state: Mapping[str, float],
                     action: Sequence[float] | None) -> None:
        x, y, psi = float(state["x"]), float(state["y"]), float(state["psi"])
        u, v = float(state.get("u", 0.0)), float(state.get("v", 0.0))
        k = self._scale()
        tf = lambda p: self._body_to_screen(p, x, y, psi, k)
        L = self.L
        thrust = max(0.0, self._frac(action, 0))
        yaw = self._frac(action, 1)

        # jato de ar dos ventiladores (proporcional a tau_u)
        if thrust > 0.02:
            for f in self._fans:
                ln = 0.4 + 1.6 * thrust
                jet = np.array([f + [-self._fan_r * 0.6, self._fan_r * 0.8],
                                f + [-self._fan_r * 0.6 - ln, self._fan_r * 0.35],
                                f + [-self._fan_r * 0.6 - ln, -self._fan_r * 0.35],
                                f + [-self._fan_r * 0.6, -self._fan_r * 0.8]])
                pygame.draw.polygon(surf, JET, tf(jet))
        # saia, convés, borda
        pygame.draw.polygon(surf, SKIRT, tf(self._skirt))
        deck = tf(self._deck)
        pygame.draw.polygon(surf, DECK, deck)
        pygame.draw.aalines(surf, DECK_EDGE, True, deck.tolist())
        pygame.draw.polygon(surf, CABIN, tf(self._cabin))
        pygame.draw.polygon(surf, NOSE, tf(self._nose))
        # ventiladores e lemes (lemes defletem com tau_r)
        rpx = max(2, int(self._fan_r * k * self.camera.ppm))
        delta = -0.6 * yaw
        rud = np.array([[0.0, 0.0], [-0.45 * L * 0.5, 0.0]])
        cr, sr = np.cos(delta), np.sin(delta)
        rud_rot = rud @ np.array([[cr, sr], [-sr, cr]])
        for f in self._fans:
            c = tf(f)
            pygame.draw.circle(surf, FAN, c, rpx)
            pygame.draw.circle(surf, SKIRT, c, rpx, max(1, rpx // 4))
            hinge = f + [-self._fan_r - 0.05, 0.0]
            seg = tf(hinge + rud_rot)
            pygame.draw.line(surf, (230, 230, 230), seg[0], seg[1], max(2, rpx // 3))

        # vetor velocidade no mundo (0.5 s à frente)
        c0 = self.camera.to_screen([x, y])
        vx = u * np.cos(psi) - v * np.sin(psi)
        vy = u * np.sin(psi) + v * np.cos(psi)
        if np.hypot(vx, vy) > 0.05:
            tip = self.camera.to_screen([x + 0.5 * vx, y + 0.5 * vy])
            self._arrow(surf, c0, tip, ACCENT)
        pygame.draw.circle(surf, (20, 20, 20), c0, 3)

    @staticmethod
    def _arrow(surf: pygame.Surface, a, b, color, width: int = 3) -> None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = np.hypot(*d)
        if n < 1:
            return
        d /= n
        pygame.draw.line(surf, color, a, b, width)
        h = min(12.0, 0.4 * n)
        left = b - h * d + 0.5 * h * np.array([-d[1], d[0]])
        right = b - h * d - 0.5 * h * np.array([-d[1], d[0]])
        pygame.draw.polygon(surf, color, [b, left, right])

    # ------------------------------------------------------------------
    def hud_lines(self, state, action, info: Mapping[str, Any]) -> list[str]:
        u, v, r = (float(state.get(n, 0.0)) for n in ("u", "v", "r"))
        psi = float(state["psi"])
        speed = np.hypot(u, v)
        lines = [f"t {info.get('t', 0.0):7.2f} s | s {100 * info.get('s', 0.0) / self.track.length:5.1f} %"]
        lines.append(f"|V| {speed:5.2f} m/s  u {u:+6.2f}  v {v:+6.2f}")
        lines.append(f"r {r:+6.2f} rad/s   psi {np.degrees(np.angle(np.exp(1j * psi))):+7.1f} deg")
        if "lateral_error" in info:
            lines.append(f"e_lat {info['lateral_error']:+5.2f} m (máx ±{self.track.half_width:.1f})")
        return lines

    def draw_hud_extra(self, surf, state, action, info, y0: int) -> int:
        a = np.zeros(2) if action is None else np.asarray(action, dtype=np.float64)
        self.draw_bar(surf, 16, y0, 370, f"tau_u {a[0]:4.2f}", a[0],
                      self.action_low[0], self.action_high[0], TAU_U_COLOR)
        self.draw_bar(surf, 16, y0 + 20, 370, f"tau_r {a[1]:+4.2f}", a[1],
                      self.action_low[1], self.action_high[1], TAU_R_COLOR)
        return y0 + 40
