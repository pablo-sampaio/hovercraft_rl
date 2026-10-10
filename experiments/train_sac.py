"""Experimento SAC standalone; edite o bloco de configuração e execute pelo VS Code."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from hovercraft_rl.envs import make_env

# --------------------------------------------------------------------------------------
# Configuração do experimento (edite aqui para testar outros regimes de treinamento).
TRACK = "gp" #"oval"
OBSERVATION = "body_frame"
START_MODE = "random"
SEED = 0
N_ENVS = 1
TOTAL_TIMESTEPS = 300_000
MAX_EPISODE_STEPS = 5_000
CHECKPOINT_EVERY = 50_000

PHYSICS = "3dof+drag"
BETA_V = 1.00
BETA_U = 0.15
BETA_R = 0.70

SAC_KWARGS = {
    "gamma": 0.99,
    "verbose": 1,
}

RUNS_DIR = PROJECT_ROOT / "runs" / "experiments"
# --------------------------------------------------------------------------------------


def main() -> None:
    run_name = (
        f"sac_{TRACK}_{OBSERVATION}_bv{BETA_V:g}_bu{BETA_U:g}_br{BETA_R:g}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}_s{SEED}"
    )
    run_dir = RUNS_DIR / run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "algo": "sac",
        "actions": "continuous",
        "tau_u_levels": [3],
        "tau_r_levels": [3],
        "track": TRACK,
        "obs": OBSERVATION,
        "start": START_MODE,
        "steps": TOTAL_TIMESTEPS,
        "n_envs": N_ENVS,
        "max_steps": MAX_EPISODE_STEPS,
        "gamma": SAC_KWARGS["gamma"],
        "checkpoint_every": CHECKPOINT_EVERY,
        "seed": SEED,
        "physics": PHYSICS,
        "beta_v": BETA_V,
        "beta_u": BETA_U,
        "beta_r": BETA_R,
        "sac_kwargs": SAC_KWARGS,
        "out": str(RUNS_DIR),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    env_kwargs = {
        "track": TRACK,
        "physics": PHYSICS,
        "physics_params": {"beta_v": BETA_V, "beta_u": BETA_U, "beta_r": BETA_R},
        "observation": OBSERVATION,
        "start": START_MODE,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "action_mode": "continuous",
    }
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=SAC_KWARGS["gamma"])
    try:
        model = SAC(
            "MlpPolicy",
            vec_env,
            seed=SEED,
            tensorboard_log=str(RUNS_DIR / "tb"),
            **SAC_KWARGS,
        )
        checkpoint = CheckpointCallback(
            save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
            save_path=str(checkpoints_dir),
            save_vecnormalize=True,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint, tb_log_name=run_name)
        model.save(str(run_dir / "model"))
        vec_env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        vec_env.close()

    print(f"Treino concluído. Artefatos salvos em: {run_dir}")
    print(f"Avalie com: python examples/evaluate.py {run_dir}")


if __name__ == "__main__":
    main()
