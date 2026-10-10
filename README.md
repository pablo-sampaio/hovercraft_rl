# hovercraft-rl

Toolkit para experimentar controle de um **hovercraft subatuado** por Reinforcement Learning. A tarefa é percorrer uma pista fechada o mais rápido possível sem sair dela.

O modelo físico é o de Kong et al. (2022), *Trajectory Tracking Control for Under-Actuated Hovercraft…*, J Syst Sci Complex 35:502–521 (ver `papers/`).

## Instalação (conda)

```bash
conda env create -f environment.yml --solver=libmamba   # o solver "classic" pode levar muito tempo
conda activate hovercraft-rl
pip install -e .
pytest                                                  # 31 testes
```

Versões testadas (Windows 11, out/2026):

| Pacote | Versão | Uso |
|---|---|---|
| python | 3.11.17 | |
| numpy | 2.4.6 (com **OpenBLAS**) | física e pista |
| gymnasium | 1.4.0 | ambiente |
| pygame | 2.6.1 | visualização e teclado |
| stable-baselines3 | 2.9.0 | treino de exemplo |
| torch | 2.14.1+cpu | treino de exemplo |
| scipy, matplotlib, pytest | 1.17 / 3.11 / 9.1 | utilidades e testes |

> O `environment.yml` fixa `libblas=*=*openblas` para evitar um erro com produto de matrizes encontrado no desenvolvimento (erro do Windows `0xc06d007f`).

## Uso rápido

```bash
python -m hovercraft_rl.game --track gp --view ego                 # jogar pelo teclado
python examples/run_policy.py --policy heuristic --track gp        # controlador de referência, com render
python examples/train_sb3.py --algo sac --track oval --steps 300000 --beta-v 1.0 --beta-u 0.15 --beta-r 0.70
python experiments/train_sac.py                              # experimento standalone, config editável no topo
python examples/train_sb3.py --algo dqn --actions discrete --tau-r-levels 5 --steps 500000
python examples/evaluate.py runs/sac_continuous_oval_body_frame_3dof+drag_bv1_bu0.15_br0.7_s0 --view ego
tensorboard --logdir runs/tb
```

```python
import hovercraft_rl, gymnasium as gym
env = gym.make("HovercraftTrack-v0", track="kidney", render_mode="human")   # contínuo, TimeLimit 5000
env = gym.make("HovercraftTrackDiscrete-v0", track="kidney")                # Discrete(9)
```

### Treino (`examples/train_sb3.py`)

| Algoritmo | Ações aceitas |
|---|---|
| `sac` | `continuous` |
| `ppo`, `a2c` | `continuous`, `discrete`, `multidiscrete` |
| `dqn` | `discrete` |

Outras opções:
- `--obs reference_error|body_frame`;
- `--physics` (padrão: `3dof+drag`), `--beta-v`, `--beta-u`, `--beta-r`;
- `--start random|fixed`;
- `--n-envs`, `--gamma`, `--seed`;
- `--tau-u-levels` / `--tau-r-levels`: nº de níveis ou lista de valores.

Cada execução grava em `runs/<algo>_<ações>_<pista>_<obs>_<motor>_<betas>_s<seed>/`:
- `config.json`;
- `model.zip`;
- `vecnormalize.pkl`;
- `checkpoints/`.

`evaluate.py` reconstrói o ambiente a partir do `config.json`, e `--track` permite testar em outra pista.
O motor físico pode ser trocado ou estendido registrando novas subclasses de `PhysicsModel` em
`PHYSICS_MODELS`. O script `experiments/train_sac.py` é uma alternativa standalone: seus parâmetros
de física e treino são constantes editáveis no início do arquivo, e a execução salva modelo,
normalização, configuração e checkpoints sob `runs/experiments/`.

O controlador heurístico de `run_policy.py` completa os 5000 passos em todas as pistas a ~4 m/s. Ele serve como linha de base a ser superada.

## Estrutura

Cada camada depende só das anteriores. Para trocar uma peça, basta passar outro objeto ou nome.

```
hovercraft_rl/
  physics/   simulador puro (estado + ação), modelos registrados por nome  -> physics/README.md
  track/     pistas (JSON), projeção na linha base, progresso              -> track/README.md
  render/    visualizador pygame (visão global e egocêntrica, HUD)        -> render/README.md
  envs/      ambiente Gymnasium, observações, recompensas e ações discretas -> envs/README.md
  game/      pilotar pelo teclado (usa o ambiente e o visualizador)       -> game/README.md
examples/    run_policy.py (aleatório/heurístico), train_sb3.py, evaluate.py
experiments/ train_sac.py (experimento SAC standalone, configuração editável diretamente no código)
tests/       física, pista, ambiente
```

| Para trocar… | Faça… |
|---|---|
| modelo físico | subclasse de `PhysicsModel` + registro em `PHYSICS_MODELS`; renderizador em `RENDERERS` |
| pista | novo JSON em `hovercraft_rl/track/data/` (ou `Track(...)`) |
| observação | subclasse de `ObservationBuilder` → `observation=` |
| recompensa | `CompositeReward([...])` com `RewardTerm`s → `reward=` |
| ações | `action_mode="continuous" \| "discrete" \| "multidiscrete"` em `make_env` |
| visualização | subclasse de `BaseTrackRenderer` |
