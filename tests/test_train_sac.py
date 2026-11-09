import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("tensorboard")

from examples import train_sac
from examples.train import env_kwargs_from
from stable_baselines3 import SAC


def test_training_saves_explicit_algorithm_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(train_sac, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(train_sac, "TOTAL_TIMESTEPS", 4)
    monkeypatch.setattr(train_sac, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(train_sac, "MAX_EPISODE_STEPS", 5)
    monkeypatch.setattr(train_sac, "GAMMA", 0.9)
    monkeypatch.setattr(train_sac, "LEARNING_RATE", 0.0007)
    monkeypatch.setattr(train_sac, "BATCH_SIZE", 32)
    monkeypatch.setattr(train_sac, "ACTOR_HIDDEN", [16, 8])
    monkeypatch.setattr(train_sac, "CRITIC_HIDDEN", [32, 16])

    train_sac.main()

    run_dir = next(path for path in tmp_path.iterdir() if path.name.startswith("sac_"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    kwargs = config["algo_kwargs"]
    assert config["algo"] == "sac"
    assert env_kwargs_from(config) == config["env_kwargs"]
    assert kwargs == {
        "policy": "MlpPolicy",
        "seed": train_sac.SEED,
        "gamma": 0.9,
        "learning_rate": 0.0007,
        "batch_size": 32,
        "policy_kwargs": {"net_arch": {"pi": [16, 8], "qf": [32, 16]}},
        "tensorboard_log": str(tmp_path / "tb"),
    }

    model = SAC.load(run_dir / "model.zip")
    assert model.gamma == kwargs["gamma"]
    assert model.learning_rate == kwargs["learning_rate"]
    assert model.batch_size == kwargs["batch_size"]
    assert model.policy_kwargs["net_arch"] == kwargs["policy_kwargs"]["net_arch"]
    assert model.seed == kwargs["seed"]
    assert (run_dir / "vecnormalize.pkl").is_file()

    evaluate = Path(train_sac.__file__).with_name("evaluate.py")
    result = subprocess.run(
        [sys.executable, str(evaluate), str(run_dir), "--render", "none", "--episodes", "1"],
        capture_output=True,
        check=True,
    )
    assert b"passos=" in result.stdout
