"""Renderizador base: janela/superfície, câmera, pista, rastro, HUD e overlays.

Subclasses implementam ``draw_vehicle`` e ``hud_lines`` para um modelo físico específico.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from ..track import Track
from .camera import Camera
from .track_drawer import MiniMap, TrackDrawer

HUD_TEXT = (235, 235, 235)
HUD_DIM = (160, 170, 180)
ACCENT = (95, 200, 255)
WARN = (235, 70, 70)
REF = (255, 214, 64)
LOOKAHEAD = (226, 112, 255)
TRAIL = (95, 200, 255)


class BaseTrackRenderer:
    """Renderiza pista + veículo.

    Parâmetros
    ----------
    track : pista a desenhar.
    render_mode : ``"human"`` (janela pygame) ou ``"rgb_array"`` (retorna array HxWx3).
    view : ``"global"`` ou ``"ego"``.
    fps : taxa máxima de quadros no modo ``"human"`` (``None`` = sem limite).
    show_debug : desenha o ponto de referência e os pontos de lookahead.
    process_events : no modo ``"human"``, consome os eventos da janela (fechar, V, G, +/-).
        Use ``False`` quando outro módulo (ex.: o jogo) tratar os eventos.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, track: Track, render_mode: str = "human", view: str = "global",
                 width: int = 960, height: int = 720, fps: float | None = 50.0,
                 show_debug: bool = False, show_hud: bool = True, trail_length: int = 400,
                 process_events: bool = True, title: str = "Hovercraft RL") -> None:
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode inválido: {render_mode!r}")
        self.track = track
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.fps = fps
        self.show_debug = show_debug
        self.show_hud = show_hud
        self.process_events = process_events
        self.title = title
        self.camera = Camera(width, height, view=view)
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self.messages: list[str] = []      # linhas fixas no rodapé (ex.: ajuda do jogo)
        self._lap = 0
        self._lap_start_t = 0.0
        self._last_lap_time: float | None = None
        self._best_lap_time: float | None = None
        self._window = None
        self._clock = None
        self._surface = None
        self.closed = False
        self._init_pygame()
        self.track_drawer = TrackDrawer(track)
        self.camera.fit(np.vstack(track.boundaries()))
        self.minimap = MiniMap(track)

    # ----------------------------------------------------------- pygame
    def _init_pygame(self) -> None:
        if self.render_mode == "rgb_array" and "SDL_VIDEODRIVER" not in os.environ and not pygame.display.get_init():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)
            self._clock = pygame.time.Clock()
        self._surface = pygame.Surface((self.width, self.height))
        names = "consolas,dejavusansmono,menlo,couriernew"
        self.font = pygame.font.SysFont(names, 16)
        self.font_small = pygame.font.SysFont(names, 13)
        self.font_big = pygame.font.SysFont("segoeui,dejavusans,arial", 40, bold=True)

    def close(self) -> None:
        if not self.closed:
            if self._window is not None:
                pygame.display.quit()
            self.closed = True

    # ------------------------------------------------------- controle
    def reset(self) -> None:
        """Limpa rastro e cronômetro de voltas (chamar a cada episódio)."""
        self.trail.clear()
        self._lap = 0
        self._lap_start_t = 0.0
        self._last_lap_time = None

    def toggle_view(self) -> None:
        self.camera.toggle()

    def handle_event(self, event: pygame.event.Event) -> None:
        """Atalhos de visualização: V (visão), G (debug), H (HUD), +/- (zoom), fechar janela."""
        if event.type == pygame.QUIT:
            self.close()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                self.toggle_view()
            elif event.key == pygame.K_g:
                self.show_debug = not self.show_debug
            elif event.key == pygame.K_h:
                self.show_hud = not self.show_hud
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.camera.zoom(1.25)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.camera.zoom(0.8)

    def observe(self, state: Mapping[str, float]) -> None:
        """Registra a posição no rastro sem desenhar (o ambiente chama a cada passo)."""
        x, y, _ = self.vehicle_pose(state)
        if not self.trail or np.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.25:
            self.trail.append((x, y))

    # ---------------------------------------------- pontos de extensão
    def vehicle_pose(self, state: Mapping[str, float]) -> tuple[float, float, float]:
        return float(state["x"]), float(state["y"]), float(state["psi"])

    def draw_vehicle(self, surf: pygame.Surface, state: Mapping[str, float],
                     action: Sequence[float] | None) -> None:
        x, y, psi = self.vehicle_pose(state)
        c = self.camera.to_screen([x, y])
        pygame.draw.circle(surf, ACCENT, c, 6)

    def hud_lines(self, state: Mapping[str, float], action, info: Mapping[str, Any]) -> list[str]:
        return []

    def draw_hud_extra(self, surf: pygame.Surface, state, action, info, y0: int) -> int:
        return y0

    # ---------------------------------------------------------- render
    def render(self, state: Mapping[str, float], action: Sequence[float] | None = None,
               info: Mapping[str, Any] | None = None) -> np.ndarray | None:
        if self.closed:
            return None
        info = info or {}
        if self.render_mode == "human" and self.process_events:
            for event in pygame.event.get():
                self.handle_event(event)
            if self.closed:
                return None
        x, y, psi = self.vehicle_pose(state)
        self._update_laps(info)
        self.observe(state)

        surf = self._surface
        self.camera.follow(x, y, psi)
        self.track_drawer.draw(surf, self.camera)
        self._draw_trail(surf)
        if self.show_debug:
            self._draw_debug(surf, x, y, info)
        self.draw_vehicle(surf, state, action)
        if self.camera.view == "ego":
            self.minimap.draw(surf, x, y, psi, ACCENT)
        if self.show_hud:
            self._draw_hud(surf, state, action, info)
        if info.get("off_track"):
            self._banner(surf, "FORA DA PISTA", WARN)
        msgs = list(self.messages) + list(info.get("messages", ()))
        for i, msg in enumerate(msgs):
            t = self.font_small.render(msg, True, HUD_DIM)
            surf.blit(t, (10, self.height - 20 - 16 * (len(msgs) - 1 - i)))

        if self.render_mode == "human":
            self._window.blit(surf, (0, 0))
            pygame.display.flip()
            if self.fps:
                self._clock.tick(self.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()

    # ------------------------------------------------------- internos
    def _update_laps(self, info: Mapping[str, Any]) -> None:
        lap = int(info.get("lap", 0))
        t = float(info.get("t", 0.0))
        if lap > self._lap:
            self._last_lap_time = t - self._lap_start_t
            if self._best_lap_time is None or self._last_lap_time < self._best_lap_time:
                self._best_lap_time = self._last_lap_time
            self._lap_start_t = t
        elif lap < self._lap:
            self._lap_start_t = t
        self._lap = lap

    def _draw_trail(self, surf: pygame.Surface) -> None:
        if len(self.trail) < 2:
            return
        pts = self.camera.to_screen(np.array(self.trail))
        n = len(pts)
        for i in range(1, n):
            a = i / n
            col = tuple(int(c * a + g * (1 - a)) for c, g in zip(TRAIL, (58, 62, 68)))
            pygame.draw.line(surf, col, pts[i - 1], pts[i], 2)

    def _draw_debug(self, surf: pygame.Surface, x: float, y: float, info: Mapping[str, Any]) -> None:
        if "x_ref" in info:
            c = self.camera.to_screen([x, y])
            r = self.camera.to_screen([info["x_ref"], info["y_ref"]])
            pygame.draw.line(surf, REF, c, r, 2)
            pygame.draw.circle(surf, REF, r, 5)
        la = info.get("lookahead_points")
        if la is not None and len(la):
            pts = self.camera.to_screen(np.asarray(la))
            for p in pts:
                pygame.draw.circle(surf, LOOKAHEAD, p, 4, 2)

    def _draw_hud(self, surf: pygame.Surface, state, action, info: Mapping[str, Any]) -> None:
        lines = [f"Visão: {self.camera.label}"]
        lines += self.hud_lines(state, action, info)
        lap_txt = f"volta {info.get('lap', 0)}"
        if self._last_lap_time is not None:
            lap_txt += f" | última {self._last_lap_time:6.2f}s"
        if self._best_lap_time is not None:
            lap_txt += f" | melhor {self._best_lap_time:6.2f}s"
        lines.append(lap_txt)
        if "reward" in info:
            lines.append(f"recompensa {info['reward']:+7.3f} | retorno {info.get('episode_return', 0.0):8.2f}")

        panel_h = 14 + 20 * len(lines) + 60
        panel = pygame.Surface((390, panel_h), pygame.SRCALPHA)
        panel.fill((10, 14, 20, 175))
        surf.blit(panel, (8, 8))
        y = 15
        for i, ln in enumerate(lines):
            surf.blit(self.font.render(ln, True, ACCENT if i == 0 else HUD_TEXT), (16, y))
            y += 20
        self.draw_hud_extra(surf, state, action, info, y + 4)

    def _banner(self, surf: pygame.Surface, text: str, color) -> None:
        t = self.font_big.render(text, True, (255, 255, 255))
        w, h = t.get_width() + 40, t.get_height() + 16
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((*color, 210))
        pos = ((self.width - w) // 2, int(self.height * 0.18))
        surf.blit(box, pos)
        surf.blit(t, (pos[0] + 20, pos[1] + 8))

    def draw_bar(self, surf: pygame.Surface, x: int, y: int, w: int, label: str,
                 value: float, lo: float, hi: float, color) -> None:
        """Barra horizontal com zero marcado (para ações)."""
        surf.blit(self.font_small.render(label, True, HUD_DIM), (x, y))
        bx, bw, bh = x + 95, w - 95, 12
        pygame.draw.rect(surf, (60, 66, 76), (bx, y + 2, bw, bh), border_radius=3)
        frac = lambda v: (np.clip(v, lo, hi) - lo) / (hi - lo)
        x0 = bx + bw * frac(0.0 if lo < 0 < hi else lo)
        x1 = bx + bw * frac(value)
        pygame.draw.rect(surf, color, (min(x0, x1), y + 2, max(abs(x1 - x0), 1), bh), border_radius=3)
        if lo < 0 < hi:
            pygame.draw.line(surf, HUD_TEXT, (x0, y), (x0, y + bh + 3), 1)
