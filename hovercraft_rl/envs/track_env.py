"""Ambiente Gymnasium: hovercraft seguindo uma pista fechada."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from ..physics import PhysicsModel, make_physics
from ..track import Projection, Track
from .observations import OBSERVATIONS, ObservationBuilder, ReferenceErrorObservation
from .rewards import RewardTerm, default_reward


class HovercraftTrackEnv(gym.Env):
    """Ambiente contínuo (sem truncamento interno). Termina se o veículo sai da pista.

    Parâmetros
    ----------
    track : nome de pista salva, caminho JSON ou ``Track``.
    physics : nome registrado em ``physics.PHYSICS_MODELS`` ou instância de ``PhysicsModel``.
    physics_params : parâmetros do modelo (dataclass ou dict), se ``physics`` for um nome.
    observation : ``ObservationBuilder`` ou nome em ``OBSERVATIONS`` (padrão: ``ReferenceErrorObservation()``).
    reward : ``RewardTerm`` (padrão: ``default_reward()`` = progresso + penalidade por sair).
    start : ``"fixed"`` (largada em ``start_s``) ou ``"random"`` (``s`` uniforme na pista).
    start_s : arco da largada fixa (m).
    initial_speed : surge inicial ``u0`` (m/s).
    render_mode : ``None``, ``"human"`` ou ``"rgb_array"``.
    render_view : ``"global"`` ou ``"ego"``.
    render_kwargs : argumentos extras do renderizador (ex.: ``show_debug=True``).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, track: str | Track = "oval", physics: str | PhysicsModel = "hovercraft3dof",
                 physics_params: Any = None, observation: ObservationBuilder | str | None = None,
                 reward: RewardTerm | None = None, start: str = "fixed", start_s: float = 0.0,
                 initial_speed: float = 0.0, render_mode: str | None = None,
                 render_view: str = "global", render_kwargs: dict | None = None) -> None:
        super().__init__()
        self.track = track if isinstance(track, Track) else Track.load(track)
        self.physics = make_physics(physics, physics_params)
        if observation is None:
            observation = ReferenceErrorObservation()
        elif isinstance(observation, str):
            observation = OBSERVATIONS[observation]()
        self.observation_builder = observation
        self.reward_fn = reward if reward is not None else default_reward()
        if start not in ("fixed", "random"):
            raise ValueError("start deve ser 'fixed' ou 'random'")
        self.start = start
        self.start_s = float(start_s)
        self.initial_speed = float(initial_speed)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode inválido: {render_mode!r}")
        self.render_mode = render_mode
        self.render_view = render_view
        self.render_kwargs = dict(render_kwargs or {})
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.physics.dt))}
        self._renderer = None

        self.action_space = spaces.Box(self.physics.action_low.astype(np.float32),
                                       self.physics.action_high.astype(np.float32), dtype=np.float32)
        # estado inicial provisório para dimensionar a observação
        self._place(self.start_s, self.initial_speed)
        self.observation_space = self.observation_builder.space(self)

        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._info: dict[str, Any] = {}
        self.episode_return = 0.0

    # ------------------------------------------------------------------
    def _place(self, s0: float, u0: float, state: dict | None = None) -> None:
        if state is None:
            p = self.track.point_at(s0)
            psi = float(self.track.heading_at(s0))
            state = {"x": p[0], "y": p[1], "psi": psi, "u": u0}
        self.physics.reset(state)
        st = self.physics.state_dict()
        self.projection: Projection = self.track.project(st["x"], st["y"])
        self.progress = 0.0           # avanço líquido acumulado desde a largada (m)
        self.lap = 0

    def _make_info(self, delta: float) -> dict[str, Any]:
        pr = self.projection
        off = (not np.all(np.isfinite(self.physics.state))) or abs(pr.lateral_error) > self.track.half_width
        info = {
            "t": self.physics.t,
            "state": self.physics.state_dict(),
            "s": pr.s,
            "x_ref": pr.x_ref,
            "y_ref": pr.y_ref,
            "ref_heading": pr.heading,
            "lateral_error": pr.lateral_error,
            "progress_delta": delta,
            "progress": self.progress,
            "lap": self.lap,
            "off_track": bool(off),
        }
        pts = getattr(self.observation_builder, "last_points", None)
        if pts is not None:
            info["lookahead_points"] = pts
        return info

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "start_s" in options:
            s0 = float(options["start_s"])
        elif self.start == "random":
            s0 = float(self.np_random.uniform(0.0, self.track.length))
        else:
            s0 = self.start_s
        self._place(s0, float(options.get("initial_speed", self.initial_speed)), options.get("state"))
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.episode_return = 0.0
        obs = self.observation_builder(self)
        self._info = self._make_info(0.0)
        self.reward_fn.reset(self, self._info)
        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return obs, dict(self._info)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        self.physics.step(a)
        self.last_action = np.clip(a, self.action_space.low, self.action_space.high).astype(np.float32)
        st = self.physics.state_dict()
        prev_info = self._info
        prev_s = self.projection.s
        if np.all(np.isfinite(self.physics.state)):
            self.projection = self.track.project(st["x"], st["y"], hint_s=prev_s)
        delta = self.track.progress_delta(prev_s, self.projection.s)
        self.progress += delta
        self.lap = max(0, int(np.floor(self.progress / self.track.length)))

        obs = self.observation_builder(self)
        info = self._make_info(delta)
        terminated = info["off_track"]
        reward = float(self.reward_fn(self, prev_info, info, self.last_action, terminated))
        self.episode_return += reward
        info["reward"] = reward
        info["episode_return"] = self.episode_return
        if hasattr(self.reward_fn, "last_terms"):
            info["reward_terms"] = dict(self.reward_fn.last_terms)
        self._info = info
        if self.render_mode == "human":
            self.render()
        elif self.render_mode is not None:
            self.renderer.observe(info["state"])
        return obs, reward, terminated, False, dict(info)

    # ------------------------------------------------------------------
    @property
    def renderer(self):
        """Renderizador (criado sob demanda)."""
        if self._renderer is None:
            from ..render import make_renderer
            kw = dict(render_mode=self.render_mode or "rgb_array", view=self.render_view,
                      fps=self.metadata["render_fps"])
            kw.update(self.render_kwargs)
            self._renderer = make_renderer(self.physics, self.track, **kw)
        return self._renderer

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() chamado sem render_mode; use render_mode='human' ou 'rgb_array'.")
            return None
        info = dict(self._info)
        info.setdefault("episode_return", self.episode_return)
        return self.renderer.render(self.physics.state_dict(), self.last_action, info)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(track: str | Track = "oval", max_episode_steps: int | None = 5000,
             action_mode: str = "continuous", tau_u_levels=3, tau_r_levels=3, **kwargs) -> gym.Env:
    """Cria o ambiente com ``TimeLimit`` (padrão: 5000 passos = 100 s com h = 0.02 s).

    - ``max_episode_steps=None``: ambiente contínuo puro (sem truncamento).
    - ``action_mode``: ``"continuous"`` (Box), ``"discrete"`` (Discrete(n_u*n_r)) ou
      ``"multidiscrete"`` (MultiDiscrete([n_u, n_r])); níveis em ``tau_u_levels``/``tau_r_levels``
      (nº de níveis ou lista de valores).
    """
    env = HovercraftTrackEnv(track=track, **kwargs)
    if action_mode != "continuous":
        from .wrappers import ACTION_WRAPPERS
        if action_mode not in ACTION_WRAPPERS:
            raise ValueError(f"action_mode deve ser 'continuous' ou um de {list(ACTION_WRAPPERS)}")
        env = ACTION_WRAPPERS[action_mode](env, tau_u_levels, tau_r_levels)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
