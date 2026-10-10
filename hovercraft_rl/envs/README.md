# `envs` — ambiente Gymnasium

```python
from hovercraft_rl.envs import make_env
env = make_env("gp")                    # TimeLimit(5000) por padrão
env = make_env("gp", max_episode_steps=None)   # contínuo puro
env = make_env("gp", action_mode="discrete")   # ações discretas (ver abaixo)
# ou: import hovercraft_rl; gym.make("HovercraftTrack-v0", track="gp")
```

## `HovercraftTrackEnv` — parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `track` | `"oval"` | nome em `track/data`, caminho `.json` ou `Track` |
| `physics` | `"hovercraft3dof"` | nome registrado ou instância de `PhysicsModel` |
| `physics_params` | `None` | parâmetros do modelo (dict ou dataclass) |
| `observation` | `ReferenceErrorObservation()` | `ObservationBuilder` ou nome (`"reference_error"`, `"body_frame"`) |
| `reward` | `default_reward()` | `RewardTerm` (geralmente `CompositeReward`) |
| `start` | `"fixed"` | `"fixed"` (em `start_s`) ou `"random"` (s uniforme) |
| `start_s` | 0.0 | arco da largada fixa (m) |
| `initial_speed` | 0.0 | $u_0$ (m/s); o veículo parte sobre a linha base, alinhado a ela, com $v=r=0$ |
| `render_mode` | `None` | `"human"` ou `"rgb_array"` |
| `render_view` | `"global"` | `"global"` ou `"ego"` |
| `render_kwargs` | `{}` | repassado ao renderizador (ex.: `show_debug=True`) |

`reset(options=...)` aceita `start_s`, `initial_speed` ou `state` (dict com o estado completo ou parcial).

## Ação

O ambiente base é **contínuo**: `Box([tau_u_min, tau_r_min], [tau_u_max, tau_r_max])`. Por padrão, de `[0, -1]` a `[2, 1]`. Valores fora dos limites são saturados pela física.

Há variantes **discretizadas** (wrappers em `wrappers.py`), escolhidas pelo `action_mode` de `make_env`:

| `action_mode` | Espaço | Mapeamento |
|---|---|---|
| `"continuous"` (padrão) | `Box(2)` | direto |
| `"discrete"` | `Discrete(n_u · n_r)` | índice → par $(\tau_u, \tau_r)$ da grade; útil para DQN |
| `"multidiscrete"` | `MultiDiscrete([n_u, n_r])` | um índice por componente |

Os níveis são definidos por `tau_u_levels` e `tau_r_levels`:
- um inteiro gera *n* valores igualmente espaçados nos limites da física;
- uma lista define os valores explicitamente;
- o padrão é 3 × 3.

```python
make_env("gp", action_mode="discrete", tau_u_levels=3, tau_r_levels=5)        # 15 ações
make_env("gp", action_mode="multidiscrete", tau_r_levels=[-1, -0.3, 0, 0.3, 1])
gym.make("HovercraftTrackDiscrete-v0", track="gp")                            # Discrete(9)
```

- A tabela de ações fica em `env.get_wrapper_attr("table")` (discrete) e `u_levels`/`r_levels`.
- O passo de tempo continua `h` = 0.02 s. A ação escolhida vale por um passo, como no caso contínuo.

## Episódio

- **Contínuo:** não há fim por tempo dentro do ambiente.
- **Termina** (`terminated=True`, falha) quando $|e_{lat}| >$ `half_width` (saiu da pista) ou quando o estado deixa de ser finito.
- **Trunca** (`truncated=True`) só via `TimeLimit`. O padrão é 5000 passos = 100 s.

## Observações

### `ReferenceErrorObservation` (padrão, conforme a especificação)

O ponto de referência $(x_{ref}, y_{ref})$ é a projeção ortogonal do veículo sobre o segmento mais próximo da linha base (ver `track/README.md`).

| Índices | Conteúdo |
|---|---|
| 0–1 | $x-x_{ref}$, $y-y_{ref}$ (m, referencial do mundo) |
| 2–3 | $\cos\psi$, $\sin\psi$ |
| 4–6 | $u$, $v$ (m/s), $r$ (rad/s) |
| 7–10 | `neighbor_headings`: $\cos,\sin$ da direção da pista antes e depois da referência |
| 11– | `n_lookahead` pontos à frente, $(x_i - x,\ y_i - y)$, espaçados de `lookahead_spacing` m de arco |

Padrões:
- `neighbor_headings=True`, `neighbor_distance=None`. Com `None`, usa os segmentos vizinhos; com um valor, usa a direção em $s \mp d$.
- `n_lookahead=5`, `lookahead_spacing=5.0`.

### `BodyFrameObservation` (alternativa)

`[e_lat/half_width, cos(ψ-ψ_ref), sin(ψ-ψ_ref), u, v, r, (frente_i, esquerda_i) × n_lookahead]`.

- Os pontos à frente ficam no referencial do corpo.
- É invariante à posição e à orientação absolutas da pista, o que tende a generalizar melhor entre pistas.

Para criar outra observação, herde `ObservationBuilder` e implemente `size(env)` e `__call__(env)`. Dentro deles, use:
- `env.physics.state_dict()`;
- `env.projection`;
- `env.track`.

## Recompensa

`CompositeReward([(termo, peso), ...])` soma termos independentes. O valor de cada termo vai para `info["reward_terms"]`.

**Padrão** (`default_reward()`):

$$
r_t = \underbrace{\Delta s_t}_{\text{progresso}} \; - \; 10 \cdot \mathbb{1}[\text{saiu da pista}]
$$

- $\Delta s_t$ é o avanço ao longo da linha base no passo, em metros.
  - É negativo se o veículo andar para trás.
  - A passagem pela largada é tratada.
  - Andar rápido *ao longo da pista* rende mais; correr de lado ou em zigue-zague, não.
  - A busca do ponto de referência em janela impede ganhar progresso cortando caminho.
- A penalidade (`OffTrackPenalty(penalty)`) só é aplicada no passo terminal.

Termos extras, desligados por padrão:
- `LateralErrorPenalty`: $-(e/w)^2$;
- `ControlEffortPenalty(weights)`: $-\sum w_i a_i^2$.

Para criar um termo novo, herde `RewardTerm` e implemente `__call__(env, prev_info, info, action, terminated)`.

## `info`

| Chave | Conteúdo |
|---|---|
| `t` | tempo simulado (s) |
| `state` | estado completo da física (dict) |
| `s` | arco do ponto de referência (m) |
| `x_ref`, `y_ref`, `ref_heading` | ponto de referência e direção do seu segmento |
| `lateral_error` | erro lateral com sinal (+ = à esquerda) |
| `progress_delta` | progresso no passo (m) |
| `progress` | progresso acumulado (m) |
| `lap` | voltas completas |
| `off_track` | saiu da pista |
| `reward`, `episode_return`, `reward_terms` | recompensa do passo, retorno acumulado e valor de cada termo |
| `lookahead_points` | pontos à frente, no mundo (para debug) |
