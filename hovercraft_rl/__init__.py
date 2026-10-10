"""Toolkit para experimentos de RL com um hovercraft subatuado seguindo uma pista."""

from gymnasium.envs.registration import register, registry

ENV_ID = "HovercraftTrack-v0"                 # ações contínuas (Box)
ENV_ID_DISCRETE = "HovercraftTrackDiscrete-v0"  # ações Discrete(n_u * n_r), padrão 3 x 3

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="hovercraft_rl.envs.track_env:HovercraftTrackEnv",
             max_episode_steps=5000)
if ENV_ID_DISCRETE not in registry:
    register(id=ENV_ID_DISCRETE, entry_point="hovercraft_rl.envs.track_env:make_env",
             kwargs={"action_mode": "discrete", "max_episode_steps": None},
             max_episode_steps=5000)

__version__ = "0.1.0"
