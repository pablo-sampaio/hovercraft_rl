# `game` — pilotar pelo teclado

```bash
python -m hovercraft_rl.game --track gp --view ego
hovercraft-play --track oval --debug          # mesmo comando, instalado pelo pip
```

| Opção | Descrição |
|---|---|
| `--track` | `circle`, `oval`, `rounded_rect`, `kidney`, `gp` ou caminho `.json` (padrão: `TRACK`) |
| `--view` | `global` ou `ego` |
| `--debug` | mostra o ponto de referência e o lookahead |
| `--auto-reset` | reinicia sozinho ao sair da pista |

## Calibração: bloco `PARÂMETROS` no topo de `play.py`

| Constante | Padrão | Efeito |
|---|---|---|
| `TRACK` | `"oval"` | pista padrão |
| `HALF_WIDTH` | 3.5 (`None` = 3.0 do arquivo) | meia-largura da pista (m); deve ser menor que o raio mínimo da pista |
| `DT` | 0.02 | passo $h$ (s); o jogo roda a $1/h$ quadros/s |
| `BETA_V` | 0.8 (artigo: 0.2) | amortecimento lateral (o $\beta$ do artigo): $\dot v = -ur - \beta_v v$ |
| `BETA_U` | 0.15 (artigo: 0) | arrasto longitudinal: $\dot u = vr + \tau_u - \beta_u u$; velocidade terminal = `TAU_U_MAX / BETA_U` ≈ 13 m/s |
| `BETA_R` | 0.5 (artigo: 0) | amortecimento de guinada: $\dot r = \tau_r - \beta_r r$; taxa terminal = `TAU_R_MAX / BETA_R` = 2 rad/s |
| `TAU_U_MIN`, `TAU_U_MAX` | 0, 2 | limites de $\tau_u$ (m/s²); com mínimo < 0, a tecla **S** dá ré/freio |
| `TAU_R_MIN`, `TAU_R_MAX` | −1, 1 | limites de $\tau_r$ (rad/s²) |
| `INITIAL_SPEED` | 0 | surge inicial (m/s) |

- O jogo usa o modelo `hovercraft3dof_drag` (`physics/hovercraft3dof_drag.py`). Com `BETA_V = 0.2` e `BETA_U = BETA_R = 0`, ele é idêntico ao do artigo.
- Os valores padrão são uma calibração sugerida, não do artigo. Com eles, o controlador heurístico completa oval e kidney mirando até 10 m/s, e a derrapagem típica fica em v/u ≈ 0.15–0.35.
- Os parâmetros calibrados podem ser levados ao ambiente de RL com `make_env(..., physics="hovercraft3dof_drag", physics_params={...})`, ou passando `make_physics()` deste módulo.

## Teclas

| Tecla | Ação |
|---|---|
| **W** / ↑ | $\tau_u$ máximo; solto = 0 |
| **S** / ↓ | $\tau_u$ mínimo (só se `TAU_U_MIN < 0`) |
| **A** / ← | $\tau_r$ máximo positivo (gira à esquerda) |
| **D** / → | $\tau_r$ máximo negativo (gira à direita) |
| **Shift** (segurado) | 50% da intensidade |
| **V** / **G** / **H** | alterna visão / debug / HUD |
| **+** / **−** | zoom |
| **P** | pausa |
| **R** | reinicia |
| **Esc** | sai |

A simulação roda em tempo real. O jogo usa o próprio `HovercraftTrackEnv` (sem `TimeLimit`), então o HUD mostra a mesma recompensa que o agente de RL receberia.
