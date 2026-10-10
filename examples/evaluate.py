"""Avalia um modelo treinado por ``train_sb3.py``, com visualização.

Exemplo:
    python examples/evaluate.py runs/experiments/sac_gp_body_frame_bv1_bu0.15_br0.7_20261010_185004_323956_s0 --view ego --track gp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hovercraft_rl.envs import make_env

sys.path.insert(0, str(Path(__file__).parent))
from train_sb3 import ALGOS, env_kwargs_from  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="pasta com model.zip, vecnormalize.pkl e config.json")
    ap.add_argument("--model", default="model", help="arquivo do modelo dentro da pasta (sem .zip)")
    ap.add_argument("--track", default=None, help="padrão: a pista do treino")
    ap.add_argument("--view", choices=["global", "ego"], default="global")
    ap.add_argument("--render", choices=["human", "none"], default="human")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--start", choices=["fixed", "random"], default="fixed")
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    kw = env_kwargs_from(cfg)
    kw["start"] = args.start
    if args.track:
        kw["track"] = args.track
    render_mode = None if args.render == "none" else "human"
    venv = DummyVecEnv([lambda: make_env(**kw, render_mode=render_mode, render_view=args.view,
                                         render_kwargs={"show_debug": True})])
    venv = VecNormalize.load(str(args.run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    model = ALGOS[cfg["algo"]].load(args.run / args.model, env=venv)

    for ep in range(args.episodes):
        obs_v = venv.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs_v, deterministic=True)
            obs_v, reward, dones, infos = venv.step(action)
            ret += float(reward[0])
            steps += 1
            done = bool(dones[0])
        info = infos[0]
        why = "truncado" if info.get("TimeLimit.truncated") else "saiu da pista"
        print(f"episódio {ep}: passos={steps} retorno={ret:.1f} voltas={info['lap']} "
              f"progresso={info['progress']:.1f} m vel. média={info['progress'] / info['t']:.2f} m/s ({why})")
    venv.close()


if __name__ == "__main__":
    main()
