"""Treino de exemplo com Stable-Baselines3.

Algoritmos x ações:
    sac            -> continuous
    ppo, a2c       -> continuous, discrete, multidiscrete
    dqn            -> discrete

Exemplos:
    python examples/train_sb3.py --algo sac --track oval --steps 300000
    python examples/train_sb3.py --algo ppo --obs body_frame --n-envs 8 --steps 2000000
    python examples/train_sb3.py --algo dqn --actions discrete --tau-u-levels 3 --tau-r-levels 5
    python examples/train_sb3.py --algo ppo --actions multidiscrete --tau-r-levels 0 -0.5 0.5 -1 1
    tensorboard --logdir runs/tb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from hovercraft_rl.envs import make_env

ALGOS = {"sac": SAC, "ppo": PPO, "a2c": A2C, "dqn": DQN}
SUPPORTED = {
    "sac": {"continuous"},
    "ppo": {"continuous", "discrete", "multidiscrete"},
    "a2c": {"continuous", "discrete", "multidiscrete"},
    "dqn": {"discrete"},
}


def levels(values: list[float]):
    """Um único inteiro = nº de níveis; vários valores = níveis explícitos."""
    return int(values[0]) if len(values) == 1 else [float(v) for v in values]


def env_kwargs_from(cfg: dict) -> dict:
    """Argumentos de ``make_env`` a partir da configuração do treino (reusado por evaluate.py)."""
    return dict(track=cfg["track"], observation=cfg["obs"], start=cfg["start"],
                max_episode_steps=int(cfg["max_steps"]), action_mode=cfg["actions"],
                tau_u_levels=levels(cfg["tau_u_levels"]), tau_r_levels=levels(cfg["tau_r_levels"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), default="sac")
    ap.add_argument("--actions", choices=["continuous", "discrete", "multidiscrete"], default=None,
                    help="padrão: continuous (dqn: discrete)")
    ap.add_argument("--tau-u-levels", type=float, nargs="+", default=[3],
                    help="nº de níveis de tau_u ou lista de valores (ações discretas)")
    ap.add_argument("--tau-r-levels", type=float, nargs="+", default=[3],
                    help="nº de níveis de tau_r ou lista de valores (ações discretas)")
    ap.add_argument("--track", default="oval")
    ap.add_argument("--obs", choices=["reference_error", "body_frame"], default="reference_error")
    ap.add_argument("--start", choices=["fixed", "random"], default="random",
                    help="largada aleatória ajuda a explorar a pista inteira")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--checkpoint-every", type=int, default=50_000)
    args = ap.parse_args()
    if args.actions is None:
        args.actions = "discrete" if args.algo == "dqn" else "continuous"
    if args.actions not in SUPPORTED[args.algo]:
        ap.error(f"{args.algo} não suporta ações {args.actions}; use {sorted(SUPPORTED[args.algo])}")

    run = args.out / f"{args.algo}_{args.actions}_{args.track}_{args.obs}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (run / "config.json").write_text(json.dumps(cfg, indent=2))

    venv = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs_from(cfg))
    # u pode crescer bastante (não há arrasto em surge): normalizar observações é importante.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=args.algo in ("ppo", "a2c"), gamma=args.gamma)

    model = ALGOS[args.algo]("MlpPolicy", venv, gamma=args.gamma, seed=args.seed, verbose=1,
                             tensorboard_log=str(args.out / "tb"))
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every // args.n_envs, 1),
                              save_path=str(run / "checkpoints"), save_vecnormalize=True)
    model.learn(total_timesteps=args.steps, callback=ckpt, tb_log_name=run.name)
    model.save(run / "model")
    venv.save(str(run / "vecnormalize.pkl"))
    print(f"Modelo salvo em {run}")


if __name__ == "__main__":
    main()
