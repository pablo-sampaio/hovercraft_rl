# `physics` — simulador físico

Simulador puro: recebe ações e devolve o estado. **Não** sabe nada de pista, recompensa ou falha.

## Interface (`base.PhysicsModel`)

| Membro | Descrição |
|---|---|
| `state_names`, `action_names` | nomes das variáveis |
| `action_low`, `action_high` | limites do atuador (a ação é saturada) |
| `dt` | tempo de amostragem (s) |
| `reset(state)` | `state` = dict parcial (faltantes = 0), vetor completo ou `None` (zeros) |
| `step(action) -> np.ndarray` | avança `dt` e retorna o estado |
| `state`, `state_dict()`, `t` | estado atual (cópia), como dict, tempo simulado |

Para criar outro modelo, herde `PhysicsModel` e registre-o em `PHYSICS_MODELS` (`physics/__init__.py`).
Depois, use `make_physics("nome", params)` ou passe `physics="nome"` ao ambiente.

## Modelo padrão `3dof+drag` (Kong et al., 2022, Eq. 4 + amortecimento)

Estado $s=[x,\,y,\,\psi,\,u,\,v,\,r]$:
- $x, y$: posição no referencial da Terra (m);
- $\psi$: guinada (rad), **não normalizada**;
- $u$: surge e $v$: sway, no referencial do corpo (m/s);
- $r$: taxa de guinada (rad/s).

Ação $a=[\tau_u,\,\tau_r]$.

$$
\begin{aligned}
\dot x &= u\cos\psi - v\sin\psi \\
\dot y &= u\sin\psi + v\cos\psi \\
\dot\psi &= r \\
\dot u &= v\,r + \tau_u - \beta_u u \\
\dot v &= -u\,r - \beta_v v \\
\dot r &= \tau_r - \beta_r r
\end{aligned}
$$

Com `beta_v=0.2` e `beta_u=beta_r=0`, o modelo se reduz ao modelo do artigo. A discretização
padrão é Euler explícito: $s_{k+1} = s_k + h\,f(s_k, a_k)$.

### Parâmetros (`Hovercraft3DOFParams`)

| Parâmetro | Padrão | Significado |
|---|---|---|
| `beta_v` | 0.2 | amortecimento de sway $\beta_v = d_2/m_1$ (1/s) |
| `beta_u` | 0.0 | amortecimento longitudinal (surge) (1/s) |
| `beta_r` | 0.0 | amortecimento de guinada (yaw) (1/s) |
| `dt` | 0.02 | tempo de amostragem $h$ (s) — mesmo valor do artigo |
| `tau_u_range` | (0, 2) | limites de $\tau_u$ (m/s²) |
| `tau_r_range` | (-1, 1) | limites de $\tau_r$ (rad/s²) |
| `integrator` | `"euler"` | `"euler"` ou `"rk4"` |
| `substeps` | 1 | sub-passos de integração por `step` |

O motor padrão é `Hovercraft3DOFDrag`, registrado com o nome `"3dof+drag"`. Para definir os
parâmetros ao criar um ambiente:

```python
env = make_env(
    "oval",
    physics="3dof+drag",
    physics_params={"beta_v": 1.0, "beta_u": 0.15, "beta_r": 0.7},
)
```

Para adicionar outro motor, implemente uma subclasse de `PhysicsModel` e registre-a em
`PHYSICS_MODELS` com um novo nome.

### Relação com o artigo de Kong et al. (2022) 

O artigo se chama "Trajectory Tracking Control for Under-Actuated Hovercraft Using Differential Flatness and ReinforcementLearning-Based Active Disturbance Rejection Control" (Kong et al., 2022).

- O artigo despreza o amortecimento em surge e guinada, e mantém o amortecimento em sway com
  `beta_v=0.2`. Para reproduzi-lo, use `beta_u=0` e `beta_r=0`.
- **Nenhum limite de estado.** O artigo apenas *supõe* que $u, v, r$ e $\tau$ são limitados (Remark 2.2). Aqui, só a ação é saturada (limite do atuador); o estado não sofre clipping.
- **Restrição de $\psi$.** A restrição $-\pi/2<\psi<\pi/2$ do artigo vem do `arctan` usado na prova de *flatness* e **não** se aplica à simulação.
- **Erros tipográficos no artigo.** O artigo escreve $\tau_r=\tau_3/m_3$ e $\beta=d_2/m_2$; pelo modelo (1)–(2), o correto seria $\tau_r=\tau_3/m_2$ e $\beta=d_2/m_1$. Isso é irrelevante aqui, pois usamos diretamente $\tau_u, \tau_r, \beta_v$.

### Consequências práticas

- **Energia:** $\frac{d}{dt}\tfrac12(u^2+v^2) = u\,\tau_u - \beta_u u^2 - \beta_v v^2$.
  - Com `beta_u > 0`, surge perde energia mesmo sem derrapar; com `beta_v > 0`, sway também dissipa.
- **Guinada:** $r$ decai naturalmente quando `beta_r > 0`; com `beta_r = 0`, para parar de girar é
  preciso aplicar $\tau_r$ contrário.
- **Derrapagem forte:** com `beta_v=0.2`, o sway em regime é $v \approx -u r/\beta_v$, ou seja, o veículo desliza muito, como no gelo.
  - O único termo de amortecimento lateral é $-\beta_v v$.
  - Para curvar, é preciso apontar o nariz para dentro da curva e usar o próprio empuxo $\tau_u$, como uma nave do *Asteroids*.
  - No mundo: $\dot{\mathbf{v}} = R(\psi)\,[\tau_u-\beta_u u,\ -\beta_v v]^T$.
- **Fig. 3 do artigo:** ela **não** é reproduzida com os valores declarados ($\tau_u=0.8$, $\tau_r=0.2$).
  - Com $\tau_r = 0.2$ constante, $\psi(10.5\,s)=11$ rad, enquanto a figura mostra $\psi$ indo de 0 a ≈ π.
  - O perfil de $\psi$ da figura é compatível com $\tau_r \approx 0.056$.
  - Isso não afeta o modelo; só não use a figura como teste.

### Convenção de eixos

- $x$ para a direita, $y$ para cima, $\psi$ anti-horário a partir de $+x$ (padrão matemático).
- O artigo desenha um referencial NED, mas as equações planas são idênticas; muda apenas a visualização.
