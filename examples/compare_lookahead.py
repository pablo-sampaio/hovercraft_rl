"""Compara SAC e PPO com 2, 4 e 5 pontos de lookahead em hovercraft_frame.

Exemplo:
    conda run -n hovercraft-rl python examples/compare_lookahead.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hovercraft_rl.envs import make_env

ALGORITHMS = {"SAC": SAC, "PPO": PPO}
LOOKAHEAD_COUNTS = (2, 4, 5)
TRACK = "gp"
MAX_EPISODE_STEPS = 5_000
LOOKAHEAD_SPACING = 5.0


class EpisodeRewardCallback(BaseCallback):
    """Guarda retornos brutos dos episódios concluídos durante o treino."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episodes: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episodes.append({
                    "steps": self.num_timesteps,
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                })
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150_000,
                        help="passos de ambiente por treino (padrão: 150000)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="episódios estocásticos de avaliação por modelo (padrão: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="semente usada em todos os treinos para comparação pareada")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="pasta de saída (padrão: examples/runs/lookahead_<data-hora>)")
    return parser.parse_args()


def make_raw_env(n_lookahead: int, seed: int, start: str):
    env_kwargs = {
        "track": TRACK,
        "physics": "3dof+drag",
        "physics_params": {"beta_v": 0.2, "beta_u": 0.0, "beta_r": 0.0},
        "observation": "hovercraft_frame",
        "observation_params": {
            "n_lookahead": n_lookahead,
            "lookahead_spacing": LOOKAHEAD_SPACING,
        },
        "start": start,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "action_mode": "continuous",
    }
    return make_vec_env(make_env, n_envs=1, seed=seed, env_kwargs=env_kwargs)


def save_episode_log(path: Path, episodes: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["steps", "return", "length"])
        writer.writeheader()
        writer.writerows(episodes)


def save_evaluation_log(path: Path, returns: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "return"])
        writer.writeheader()
        writer.writerows(
            {"episode": index, "return": value}
            for index, value in enumerate(returns, start=1)
        )


def train_one(algorithm: str, n_lookahead: int, args: argparse.Namespace,
              output_dir: Path) -> dict:
    run_name = f"{algorithm.lower()}_{n_lookahead}_points"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_env = VecNormalize(
        make_raw_env(n_lookahead, args.seed, start="random"),
        norm_obs=True,
        norm_reward=False,
    )
    callback = EpisodeRewardCallback()
    model_class = ALGORITHMS[algorithm]
    model = model_class(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        gamma=0.99,
        seed=args.seed,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
        device="auto",
    )

    print(f"[TREINO] {algorithm}, {n_lookahead} pontos, {args.steps} passos")
    try:
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        model.save(str(run_dir / "model"))
        train_env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        train_env.close()

    save_episode_log(run_dir / "training_episodes.csv", callback.episodes)

    set_random_seed(args.seed + 10_000)
    eval_env = VecNormalize.load(
        str(run_dir / "vecnormalize.pkl"),
        make_raw_env(n_lookahead, args.seed + 10_000, start="fixed"),
    )
    eval_env.training = False
    eval_env.norm_reward = False

    returns = []
    try:
        for episode_index in range(args.eval_episodes):
            observations = eval_env.reset()
            done = False
            episode_return = 0.0
            while not done:
                actions, _ = model.predict(observations, deterministic=False)
                observations, rewards, dones, _ = eval_env.step(actions)
                episode_return += float(rewards[0])
                done = bool(dones[0])
            returns.append(episode_return)
            print(f"  [AVALIAÇÃO] episódio {episode_index + 1}: retorno={episode_return:.2f}")
    finally:
        eval_env.close()

    save_evaluation_log(run_dir / "evaluation_episodes.csv", returns)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    print(f"[RESULTADO] {algorithm}, {n_lookahead} pontos: "
          f"média={mean_return:.2f}, desvio-padrão={std_return:.2f}")

    return {
        "algorithm": algorithm,
        "n_lookahead": n_lookahead,
        "training_episodes": callback.episodes,
        "evaluation_returns": returns,
        "mean_return": mean_return,
        "std_return": std_return,
        "run_dir": str(run_dir),
    }


def plot_training_curves(results: list[dict], output_path: Path) -> None:
    colors = {2: "tab:blue", 4: "tab:orange", 5: "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for axis, algorithm in zip(axes, ALGORITHMS):
        has_curve = False
        for n_lookahead in LOOKAHEAD_COUNTS:
            result = next(
                item for item in results
                if item["algorithm"] == algorithm and item["n_lookahead"] == n_lookahead
            )
            episodes = result["training_episodes"]
            if not episodes:
                continue
            steps = np.asarray([episode["steps"] for episode in episodes])
            returns = np.asarray([episode["return"] for episode in episodes])
            window = min(10, len(returns))
            smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
            axis.plot(
                steps[window - 1:],
                smoothed,
                color=colors[n_lookahead],
                label=f"{n_lookahead} pontos",
            )
            has_curve = True
        axis.set_title(algorithm)
        axis.set_xlabel("Passos de ambiente")
        axis.grid(alpha=0.25)
        if has_curve:
            axis.legend()

    axes[0].set_ylabel("Retorno bruto (média móvel de até 10 episódios)")
    fig.suptitle("Treinamento em GP com hovercraft_frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_evaluation(results: list[dict], output_path: Path) -> None:
    positions = np.arange(len(LOOKAHEAD_COUNTS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 5))

    for offset, algorithm in zip((-width / 2, width / 2), ALGORITHMS):
        selected = [
            next(
                item for item in results
                if item["algorithm"] == algorithm and item["n_lookahead"] == count
            )
            for count in LOOKAHEAD_COUNTS
        ]
        means = [item["mean_return"] for item in selected]
        deviations = [item["std_return"] for item in selected]
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=deviations,
            capsize=4,
            label=algorithm,
        )

    axis.set_xticks(positions, [str(count) for count in LOOKAHEAD_COUNTS])
    axis.set_xlabel("Quantidade de pontos à frente")
    axis.set_ylabel("Retorno médio em avaliação")
    axis.set_title("Avaliação estocástica: média e desvio-padrão")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_episodes <= 0:
        raise SystemExit("--steps e --eval-episodes devem ser positivos")

    torch.set_num_threads(1)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "runs" / f"lookahead_{timestamp}"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    set_random_seed(args.seed)
    results = [
        train_one(algorithm, n_lookahead, args, output_dir)
        for algorithm in ALGORITHMS
        for n_lookahead in LOOKAHEAD_COUNTS
    ]

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["algorithm", "n_lookahead", "mean_return", "std_return"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})

    config = {
        "track": TRACK,
        "algorithms": list(ALGORITHMS),
        "lookahead_counts": LOOKAHEAD_COUNTS,
        "lookahead_spacing": LOOKAHEAD_SPACING,
        "steps": args.steps,
        "evaluation_episodes": args.eval_episodes,
        "seed": args.seed,
        "training_start": "random",
        "evaluation_start": "fixed",
        "normalization": {"observations": True, "rewards": False},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    plot_training_curves(results, output_dir / "training_curves.png")
    plot_evaluation(results, output_dir / "evaluation_summary.png")
    print(f"\nResultados e gráficos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
