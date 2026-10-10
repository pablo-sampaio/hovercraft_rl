"""Jogar com o simulador pelo teclado.

Uso: ``python -m hovercraft_rl.game --track gp --view ego``

Para calibrar a física e a pista, edite o bloco PARÂMETROS logo abaixo.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from ..envs import HovercraftTrackEnv
from ..physics import Hovercraft3DOFDrag, Hovercraft3DOFDragParams
from ..track import Track, available_tracks

# =====================================================================================
# PARÂMETROS (edite aqui).
# Física do artigo: BETA_V = 0.2, BETA_U = BETA_R = 0. Os valores abaixo são uma calibração sugerida.
# =====================================================================================
TRACK = "oval"          # circle, oval, rounded_rect, kidney, gp (ou caminho .json); --track sobrescreve
HALF_WIDTH = 3.5        # meia-largura da pista (m); None = valor do arquivo (3.0).
                        #   Deve ser < raio mín. da pista: gp 3.9, kidney 7.4, rounded_rect 9.9, oval 20, circle 30

DT = 0.02               # passo de simulação h (s); o jogo roda em tempo real a 1/DT quadros/s
BETA_V = 1.0            # amortecimento lateral (sway), 1/s           dv = -u r - BETA_V v
                        #   (é o beta do artigo); derrapagem decai em ~1/BETA_V s       (0.8 -> ~1.2 s)
BETA_U = 0.15           # arrasto longitudinal (surge), 1/s           du = v r + tau_u - BETA_U u
                        #   velocidade terminal com propulsão máxima = TAU_U_MAX / BETA_U  (0.15 -> ~13 m/s)
BETA_R = 0.7            # amortecimento de guinada, 1/s               dr = tau_r - BETA_R r
                        #   taxa de guinada terminal = TAU_R_MAX / BETA_R                  (0.5 -> 2 rad/s)
TAU_U_MIN, TAU_U_MAX = 0.0, 2.0     # aceleração de propulsão (m/s²); TAU_U_MIN < 0 permite ré/freio
TAU_R_MIN, TAU_R_MAX = -1.0, 1.0    # aceleração angular de guinada (rad/s²)
INITIAL_SPEED = 0.0     # surge inicial u0 (m/s)
# =====================================================================================

HELP = "W/Seta cima: propulsão (S: ré, se habilitada) | A/D ou setas: guinada | Shift 50% | V visão | G debug | H HUD | +/- zoom | P pausa | R reset | Esc sair"


def make_physics() -> Hovercraft3DOFDrag:
    """Física do jogo a partir do bloco PARÂMETROS."""
    return Hovercraft3DOFDrag(Hovercraft3DOFDragParams(
        beta=BETA_V, beta_u=BETA_U, beta_r=BETA_R, dt=DT,
        tau_u_range=(TAU_U_MIN, TAU_U_MAX), tau_r_range=(TAU_R_MIN, TAU_R_MAX)))


def make_track(name: str) -> Track:
    """Pista do jogo; aplica HALF_WIDTH se definido."""
    track = Track.load(name) if HALF_WIDTH is None else Track.load(name, half_width=HALF_WIDTH)
    if track.min_curvature_radius() <= track.half_width:
        print(f"ATENÇÃO: meia-largura {track.half_width} m >= raio mínimo da pista "
              f"({track.min_curvature_radius():.1f} m); a borda interna vai se cruzar.")
    return track


def keyboard_action(keys, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Converte teclas pressionadas em ``[tau_u, tau_r]`` (máximo, ou 50% com Shift)."""
    gain = 0.5 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
    fwd = keys[pygame.K_w] or keys[pygame.K_UP]
    back = keys[pygame.K_s] or keys[pygame.K_DOWN]          # só tem efeito se TAU_U_MIN < 0
    tau_u = high[0] * gain if fwd else (low[0] * gain if back and low[0] < 0 else max(low[0], 0.0))
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    tau_r = gain * (high[1] * left + low[1] * right) if (left != right) else 0.0
    return np.array([tau_u, tau_r], dtype=np.float32)


def run_game(track: str = TRACK, view: str = "global", debug: bool = False, auto_reset: bool = False) -> None:
    tr = make_track(track)
    env = HovercraftTrackEnv(track=tr, physics=make_physics(), initial_speed=INITIAL_SPEED,
                             render_mode="human", render_view=view,
                             render_kwargs={"process_events": False, "show_debug": debug,
                                            "title": f"Hovercraft RL — {tr.name}"})
    env.reset()
    r = env.renderer
    low, high = env.action_space.low, env.action_space.high
    paused, crashed, running = False, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    crashed = False
                elif event.key == pygame.K_p:
                    paused = not paused
                else:
                    r.handle_event(event)
        if not running:
            break
        r.messages = [HELP] + (["PAUSADO"] if paused else []) + \
            (["Saiu da pista - pressione R para recomeçar"] if crashed else [])
        if paused or crashed:
            env.render()
            continue
        action = keyboard_action(pygame.key.get_pressed(), low, high)
        _, _, terminated, _, info = env.step(action)
        if terminated:
            print(f"Fora da pista: t = {info['t']:.2f} s, voltas = {info['lap']}, "
                  f"progresso = {info['progress']:.1f} m, retorno = {info['episode_return']:.1f}")
            if auto_reset:
                env.reset()
            else:
                crashed = True
    env.close()
    pygame.quit()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pilote o hovercraft pelo teclado (parâmetros no topo de game/play.py).")
    ap.add_argument("--track", default=TRACK, help=f"pista: {', '.join(available_tracks())} ou caminho .json")
    ap.add_argument("--view", default="global", choices=["global", "ego"])
    ap.add_argument("--debug", action="store_true", help="mostra ponto de referência e lookahead")
    ap.add_argument("--auto-reset", action="store_true", help="reinicia sozinho ao sair da pista")
    args = ap.parse_args(argv)
    run_game(args.track, args.view, args.debug, args.auto_reset)


if __name__ == "__main__":
    main()
