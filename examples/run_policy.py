"""Roda uma política simples (aleatória ou heurística) no ambiente.

Exemplos:
    python examples/run_policy.py --policy heuristic --track gp --render human
    python examples/run_policy.py --policy random --render rgb_array --save-frames frames/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hovercraft_rl.envs import BodyFrameObservation, make_env


class HeuristicPolicy:
    """Controlador de referência (não é RL), usado como linha de base e para testar o ambiente.

    Ideia: como quase não há força lateral (só ``-beta*v``), o hovercraft vira apontando o nariz
    para a aceleração desejada e usando o empuxo, como uma nave do *Asteroids*.

    1. Velocidade desejada: ``v_target`` na direção de um ponto ``lookahead`` m à frente na pista.
    2. Calcula a aceleração desejada no referencial do veículo, compensando o arrasto de sway.
    3. Aponta o nariz para essa aceleração e só aplica empuxo para frente.

    O estado e a geometria usados pelo controlador vêm exclusivamente de ``BodyFrameObservation``.

    Com os padrões, completa os 5000 passos em todas as pistas incluídas (~4 m/s).
    """

    def __init__(self, env, lookahead: float = 10.0, v_target: float = 4.0,
                 k_v: float = 1.0, k_p: float = 4.0, k_d: float = 3.0) -> None:
        self.env = env.unwrapped
        self.lookahead, self.v_target = lookahead, v_target
        self.k_v, self.k_p, self.k_d = k_v, k_p, k_d
        builder = self.env.observation_builder
        if not isinstance(builder, BodyFrameObservation):
            raise ValueError("HeuristicPolicy requer BodyFrameObservation")
        if builder.n_lookahead < 1 or builder.lookahead_spacing <= 0:
            raise ValueError("BodyFrameObservation precisa fornecer pontos de lookahead válidos")
        if lookahead <= 0 or lookahead > builder.n_lookahead * builder.lookahead_spacing:
            raise ValueError("lookahead deve estar coberto pelos pontos de BodyFrameObservation")
        self.n_lookahead = builder.n_lookahead
        self.lookahead_spacing = builder.lookahead_spacing
        self.beta = self.env.physics.params.beta
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        expected_size = 6 + 2 * self.n_lookahead
        if obs.shape != (expected_size,):
            raise ValueError(f"observação esperada com shape ({expected_size},), recebida {obs.shape}")

        _, _, _, u, v, r = obs[:6]
        points = obs[6:].reshape(self.n_lookahead, 2)
        steps = self.lookahead / self.lookahead_spacing
        nearest_step = round(steps)
        if np.isclose(steps, nearest_step):
            steps = float(nearest_step)
        lower_step = int(np.floor(steps))
        fraction = steps - lower_step
        lower_point = np.zeros(2) if lower_step == 0 else points[lower_step - 1]
        direction = lower_point if fraction == 0 else (
            lower_point + fraction * (points[lower_step] - lower_point))
        direction /= np.linalg.norm(direction)

        # In body coordinates, sway drag contributes [0, -beta*v].
        a_cmd = self.k_v * (self.v_target * direction - [u, v]) + [0.0, self.beta * v]
        err = np.arctan2(a_cmd[1], a_cmd[0])
        tau_u = max(0.0, float(np.linalg.norm(a_cmd) * np.cos(err)))
        tau_r = self.k_p * err - self.k_d * r
        return np.clip([tau_u, tau_r], self.action_low, self.action_high).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "heuristic"], default="heuristic")
    ap.add_argument("--track", default="oval")
    ap.add_argument("--render", choices=["none", "human", "rgb_array"], default="human")
    ap.add_argument("--view", choices=["global", "ego"], default="global")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--v-target", type=float, default=4.0)
    ap.add_argument("--save-frames", type=Path, default=None, help="salva PNG a cada --frame-every passos")
    ap.add_argument("--frame-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    observation = "body_frame" if args.policy == "heuristic" else None
    env = make_env(args.track, observation=observation, max_episode_steps=args.max_steps,
                   render_mode=None if args.render == "none" else args.render,
                   render_view=args.view, render_kwargs={"show_debug": True})
    env.action_space.seed(args.seed)
    policy = HeuristicPolicy(env, v_target=args.v_target) if args.policy == "heuristic" \
        else (lambda obs: env.action_space.sample())
    if args.save_frames:
        import matplotlib.pyplot as plt
        args.save_frames.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done, k = False, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
            if args.render == "rgb_array" and args.save_frames and k % args.frame_every == 0:
                plt.imsave(args.save_frames / f"ep{ep}_{k:05d}.png", env.render())
            k += 1
        print(f"episódio {ep}: passos={k} t={info['t']:.1f}s voltas={info['lap']} "
              f"progresso={info['progress']:.1f}m retorno={info['episode_return']:.1f} "
              f"{'(saiu da pista)' if terminated else '(truncado)'}")
    env.close()


if __name__ == "__main__":
    main()
