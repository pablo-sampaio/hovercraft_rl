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

## Modelo `hovercraft3dof` (Kong et al., 2022, Eq. 4)

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
\dot u &= v\,r + \tau_u \\
\dot v &= -u\,r - \beta\,v \\
\dot r &= \tau_r
\end{aligned}
$$

A discretização padrão é Euler explícito: $s_{k+1} = s_k + h\,f(s_k, a_k)$.

### Parâmetros (`Hovercraft3DOFParams`)

| Parâmetro | Padrão | Significado |
|---|---|---|
| `beta` | 0.2 | amortecimento de sway $\beta = d_2/m_1$ (1/s) |
| `dt` | 0.02 | tempo de amostragem $h$ (s) — mesmo valor do artigo |
| `tau_u_range` | (0, 2) | limites de $\tau_u$ (m/s²) |
| `tau_r_range` | (-1, 1) | limites de $\tau_r$ (rad/s²) |
| `integrator` | `"euler"` | `"euler"` ou `"rk4"` |
| `substeps` | 1 | sub-passos de integração por `step` |

### Tratamento dado pelo artigo (replicado aqui)

- **Sem amortecimento em surge e guinada.** O artigo despreza $d_1$ e $d_3$ por serem pequenos frente a $d_2$ (Remark 2.1). A única dissipação é a do sway, $-\beta v$.
- **Nenhum limite de estado.** O artigo apenas *supõe* que $u, v, r$ e $\tau$ são limitados (Remark 2.2). Aqui, só a ação é saturada (limite do atuador); o estado não sofre clipping.
- **Restrição de $\psi$.** A restrição $-\pi/2<\psi<\pi/2$ do artigo vem do `arctan` usado na prova de *flatness* e **não** se aplica à simulação.
- **Erros tipográficos no artigo.** O artigo escreve $\tau_r=\tau_3/m_3$ e $\beta=d_2/m_2$; pelo modelo (1)–(2), o correto seria $\tau_r=\tau_3/m_2$ e $\beta=d_2/m_1$. Isso é irrelevante aqui, pois usamos diretamente $\tau_u, \tau_r, \beta$.

### Consequências práticas

- **Energia:** $\frac{d}{dt}\tfrac12(u^2+v^2) = u\,\tau_u - \beta v^2$.
  - Com $\tau_u\ge 0$, a velocidade só diminui por derrapagem (sway). "Frear" exige girar e derrapar.
  - Sem derrapar, a velocidade nunca diminui.
- **Guinada:** $r$ não decai sozinho. Para parar de girar, é preciso aplicar $\tau_r$ contrário.
- **Derrapagem forte:** com $\beta=0.2$, o sway em regime é $v \approx -u r/\beta$, ou seja, o veículo desliza muito, como no gelo.
  - A única força lateral é $-\beta v$.
  - Para curvar, é preciso apontar o nariz para dentro da curva e usar o próprio empuxo $\tau_u$, como uma nave do *Asteroids*.
  - No mundo: $\dot{\mathbf{v}} = R(\psi)\,[\tau_u,\ -\beta v]^T$.
- **Fig. 3 do artigo:** ela **não** é reproduzida com os valores declarados ($\tau_u=0.8$, $\tau_r=0.2$).
  - Com $\tau_r = 0.2$ constante, $\psi(10.5\,s)=11$ rad, enquanto a figura mostra $\psi$ indo de 0 a ≈ π.
  - O perfil de $\psi$ da figura é compatível com $\tau_r \approx 0.056$.
  - Isso não afeta o modelo; só não use a figura como teste.

### Convenção de eixos

- $x$ para a direita, $y$ para cima, $\psi$ anti-horário a partir de $+x$ (padrão matemático).
- O artigo desenha um referencial NED, mas as equações planas são idênticas; muda apenas a visualização.
