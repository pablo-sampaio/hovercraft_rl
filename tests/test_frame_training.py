import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("tensorboard")

from examples import train_sac
from examples.train import env_kwargs_from
from hovercraft_rl.envs import make_env


@pytest.mark.parametrize("observation", ["world_frame", "hovercraft_frame"])
def test_training_and_evaluation_reconstruct_observation(tmp_path, monkeypatch, observation):
    params = {"lookahead_distances": [2, 5, 12], "include_directions": True,
              "include_hovercraft_heading": True, "include_lateral_error": False}
    monkeypatch.setattr(train_sac, "OBSERVATION", observation)
    monkeypatch.setattr(train_sac, "OBSERVATION_PARAMS", params)
    monkeypatch.setattr(train_sac, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(train_sac, "TOTAL_TIMESTEPS", 4)
    monkeypatch.setattr(train_sac, "MAX_EPISODE_STEPS", 5)
    monkeypatch.setattr(train_sac, "CHECKPOINT_EVERY", 2)
    train_sac.main()

    run = next(path for path in tmp_path.iterdir() if path.name.startswith("sac_"))
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg["env_kwargs"]["observation"] == observation
    assert cfg["env_kwargs"]["observation_params"] == params
    first = make_env(**cfg["env_kwargs"])
    second = make_env(**env_kwargs_from(cfg))
    np.testing.assert_array_equal(first.reset(seed=42)[0], second.reset(seed=42)[0])
    first.close()
    second.close()

    evaluate = Path(train_sac.__file__).with_name("evaluate.py")
    result = subprocess.run(
        [sys.executable, str(evaluate), str(run), "--render", "none", "--episodes", "1",
         "--track", "oval"],
        capture_output=True, check=True,
    )
    assert b"passos=" in result.stdout

    short_track = tmp_path / "short.json"
    short_track.write_text(json.dumps({"points": [[0, 0], [2, 0], [2, 2], [0, 2]],
                                       "half_width": 1}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(evaluate), str(run), "--render", "none", "--episodes", "1",
         "--track", str(short_track)],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"comprimento da pista" in result.stderr
